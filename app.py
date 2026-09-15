import datetime
import io
import uuid
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# 全局页面配置
# --------------------------------------------------
st.set_page_config(
    page_title="深层卤水钾盐动-静综合资源评价系统",
    page_icon="🧂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 各专题任务配额（严格锁定）
TOPIC_TARGETS = {
    "川中专题": 20_000_000.0,    # 2000万吨
    "川东北专题": 15_000_000.0,  # 1500万吨
    "川西专题": 10_000_000.0,    # 1000万吨
    "川南专题": 5_000_000.0,     # 500万吨
}

# 专题独立存储容器
if "topic_archives" not in st.session_state:
    st.session_state.topic_archives = {
        "川中专题": [],
        "川东北专题": [],
        "川西专题": [],
        "川南专题": [],
    }

# 缓存当前正在核算的单单元结果
if "temp_single_record" not in st.session_state:
    st.session_state.temp_single_record = None
if "temp_mc_samples" not in st.session_state:
    st.session_state.temp_mc_samples = None

# --------------------------------------------------
# 侧边栏：免密切换与独立进度监控
# --------------------------------------------------
with st.sidebar:
    st.title("🧂 专题独立控制台")
    st.caption("四川盆地深层富钾卤水储量评价系统")

    active_topic = st.selectbox(
        "请选择您所属的专题：",
        ["川中专题", "川东北专题", "川西专题", "川南专题"],
        index=0,
    )

    my_target = TOPIC_TARGETS[active_topic]
    my_records = st.session_state.topic_archives[active_topic]

    my_total_kcl = sum([r["最终核定KCl储量(万吨)"] for r in my_records])
    my_progress = min(1.0, (my_total_kcl * 1e4) / my_target) if my_target > 0 else 0.0

    st.markdown("---")
    st.subheader(f"🎯 【{active_topic}】增储进度")
    st.metric(
        label="本专题已核定 KCl 储量",
        value=f"{my_total_kcl:,.2f} 万吨",
        delta=(
            f"距目标还差: {(my_target/1e4 - my_total_kcl):,.2f} 万吨"
            if (my_total_kcl * 1e4) < my_target
            else "🎉 本专题增储目标已达成！"
        ),
    )
    st.progress(my_progress)
    st.caption(
        f"目标配额：{my_target/1e4:.0f} 万吨 | 完成率：{my_progress * 100:.2f}%"
    )

    st.markdown("---")
    st.info(
        f"🔒 **独立隔离保护**：\n"
        f"当前仅展示并操作【{active_topic}】归档数据，其他专题成果在此互不可见。"
    )

    if st.button("🗑️ 清空本专题归档记录", use_container_width=True):
        st.session_state.topic_archives[active_topic] = []
        st.session_state.temp_single_record = None
        st.session_state.temp_mc_samples = None
        st.rerun()


# --------------------------------------------------
# 核心自适应计算引擎（动态校准静态参数模式）
# --------------------------------------------------
def evaluate_and_archive_unit(
    row_data, assigned_topic, default_dyn_method="方法一：物质平衡校准有效孔隙度",
    run_mc=True, mc_params=None
):
    if mc_params is None:
        mc_params = {
            "n_sim": 5000,
            "phi_err": 0.20,
            "c_err": 0.15,
            "sw_err": 0.08,
            "s_err": 0.15
        }

    r = {str(k).strip(): v for k, v in row_data.items() if pd.notna(v)}
    calc_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unit_id = f"UNIT-{uuid.uuid4().hex[:8].upper()}"

    zone_name = str(r.get("构造带", r.get("构造名称", r.get("计算单元", "未命名单元"))))
    res_type = str(r.get("储层类型", r.get("类型", "油气同层型"))).strip()

    # 1. 静态基础参数
    C = float(r.get("C", r.get("KCl品位", r.get("品位", 0.018))))
    A_km2 = float(r.get("A", r.get("Aw", r.get("面积", 0.0))))
    A_m2 = A_km2 * 1e6
    h_init = float(r.get("h", r.get("厚度", r.get("有效厚度", 0.0))))

    phi_init = float(r.get("phi", r.get("ϕ", r.get("孔隙度", r.get("孔隙率", 0.08)))))
    if phi_init > 1.0:
        phi_init = phi_init / 100.0

    has_seismic_vol = ("V" in r) or ("雕刻体积" in r) or ("储层体积" in r)
    if has_seismic_vol:
        V_raw = float(r.get("V", r.get("雕刻体积", r.get("储层体积", 0.0))))
        V_m3 = V_raw * 1e6 if V_raw < 1e5 else V_raw
        geom_method = "三维地震精细雕刻体积(V)"
    else:
        V_m3 = A_m2 * h_init
        geom_method = "平面面积乘厚度估算(A×h)"

    # 初始静态参数记录
    phi_adj = phi_init
    h_adj = h_init
    Sw_init = float(r.get("Sw", r.get("含水饱和度", 0.65)))
    if Sw_init > 1.0: Sw_init /= 100.0
    Bw_init = float(r.get("Bw", r.get("体积系数", 1.02)))
    S_init = float(r.get("S", r.get("弹性储水系数", 0.0005)))
    head_h = float(r.get("承压水头标高", r.get("h_head", 350.0)))
    top_H = float(r.get("储层顶面标高", r.get("H_top", -2200.0)))
    elastic_head = max(0.0, head_h - top_H)

    # 2. 原始未校准静态计算
    if "非油气" in res_type:
        calc_model = "非油气同层型"
        Q_static_raw = (phi_init * V_m3) + (S_init * elastic_head * A_m2)
    else:
        calc_model = "油气同层型"
        Q_static_raw = (V_m3 * phi_init * Sw_init) / Bw_init if has_seismic_vol else (A_m2 * h_init * phi_init * Sw_init) / Bw_init
    P_static_raw = Q_static_raw * C

    # 3. 动态约束：校准静态参数到合理区间
    dyn_choice = str(r.get("动态方法", default_dyn_method))
    dyn_method_applied = "纯静态（未校准）"
    dyn_process = "未录入动态参数，直接沿用原始静态解释参数"
    param_adjust_note = "参数未发生校准变更"

    has_m1 = ("Wp" in r or "排卤量" in r or "累计产水量" in r) and ("dp" in r or "压降" in r or "Δp" in r)
    has_m2 = (("qw" in r or "日产水量" in r) and ("dp_flow" in r or "流压差" in r)) or ("Re" in r or "波及半径" in r)

    if "方法一" in dyn_choice and has_m1:
        Wp = float(r.get("Wp", r.get("排卤量", 0.0)))
        dp = float(r.get("dp", r.get("压降", 1.0)))
        ct_raw = float(r.get("ct", r.get("Ct", 8.5)))
        ct = ct_raw * 1e-4 if ct_raw > 1e-2 else ct_raw

        if dp > 0 and ct > 0 and V_m3 > 0:
            # 动态弹性压释机理反演等效连通流动孔隙度: phi_dyn = Wp / (ct * dp * V_total)
            # 在实际地质中，往往结合连通效率因子加权校正原始孔隙度
            phi_inferred = Wp / (ct * dp * V_m3)
            # 设定合理物性上下限（避免单井测试规模小导致过度折减，权重加权融合）
            # 取 70% 静态解释基准 + 30% 动态等效特征，或者根据动态连通率合理微调
            calib_ratio = np.clip(phi_inferred / (phi_init + 1e-6), 0.65, 1.25)
            phi_adj = phi_init * calib_ratio

            dyn_method_applied = "方法一：物质平衡校准有效孔隙度"
            dyn_process = (
                f"由累计排卤 Wp={Wp:.0f}m³、压降 Δp={dp:.2f}MPa 反求动态孔隙特征；"
                f"有效孔隙连通校正系数={calib_ratio:.3f}"
            )
            param_adjust_note = f"孔隙度由原始静态 {phi_init*100:.2f}% 校准为 {phi_adj*100:.2f}%"

    elif "方法二" in dyn_choice and has_m2:
        # 方法二：渗流流动压差反演校准有效储卤厚度 h
        qw = float(r.get("qw", r.get("日产水量", 180.0)))
        dp_flow = float(r.get("dp_flow", r.get("流压差", 4.5)))
        k_perm = float(r.get("k", r.get("渗透率", 2.5)))
        mu = float(r.get("mu", r.get("卤水黏度", 1.15)))

        if dp_flow > 0 and k_perm > 0:
            # 拟稳态径向流动产水剖面反求有效出水厚度: h_dyn = (qw * mu * ln(Re/rw)) / (2*pi*k*dp)
            # 工程简化形式: h_dyn 反算
            h_inferred = (qw * mu * 3.5) / (0.00708 * k_perm * dp_flow + 1e-6)
            h_ratio = np.clip(h_inferred / (h_init + 1e-6), 0.70, 1.30)
            h_adj = h_init * h_ratio

            dyn_method_applied = "方法二：渗流压差反演校准有效厚度"
            dyn_process = (
                f"由日产水 qw={qw:.1f}m³/d、生产压差 Δp={dp_flow:.2f}MPa 反求动态出水厚度；"
                f"厚度校准系数={h_ratio:.3f}"
            )
            param_adjust_note = f"储卤有效厚度由原始静态 {h_init:.2f}m 校准为 {h_adj:.2f}m"

    # 4. 用【动态校准后的合理参数】重新执行容积法
    if "非油气" in res_type:
        V_final = V_m3 if has_seismic_vol else (A_m2 * h_adj)
        Q_final = (phi_adj * V_final) + (S_init * elastic_head * A_m2)
        formula_final = (
            f"Q = ϕ_adj·V + S·(h-H)·A = {phi_adj:.4f}×{V_final:.2e} + "
            f"{S_init:.5f}×({head_h}-{top_H})×{A_m2:.2e}"
        )
    else:
        if has_seismic_vol:
            Q_final = (V_m3 * phi_adj * Sw_init) / Bw_init
            formula_final = f"Q = (V·ϕ_adj·Sw)/Bw = ({V_m3:.2e}×{phi_adj:.4f}×{Sw_init:.2f})/{Bw_init:.2f}"
        else:
            Q_final = (A_m2 * h_adj * phi_adj * Sw_init) / Bw_init
            formula_final = f"Q = (Aw·h_adj·ϕ_adj·Sw)/Bw = ({A_m2:.2e}×{h_adj:.2f}×{phi_adj:.4f}×{Sw_init:.2f})/{Bw_init:.2f}"

    P_final = Q_final * C

    # 5. 蒙特卡洛随机不确定性模拟（基于校准后的合理均值）
    p90_val, p50_val, p10_val = np.nan, np.nan, np.nan
    samples = None
    if run_mc:
        np.random.seed(42)
        n_sim = int(mc_params.get("n_sim", 5000))
        p_err = float(mc_params.get("phi_err", 0.20))
        c_err = float(mc_params.get("c_err", 0.15))

        phi_sim = np.random.triangular(phi_adj * (1 - p_err), phi_adj, phi_adj * (1 + p_err), n_sim)
        c_sim = np.random.triangular(C * (1 - c_err), C, C * (1 + c_err), n_sim)

        if "非油气" in res_type:
            s_err = float(mc_params.get("s_err", 0.15))
            s_sim = np.random.triangular(S_init * (1 - s_err), S_init, S_init * (1 + s_err), n_sim)
            V_used = V_m3 if has_seismic_vol else (A_m2 * h_adj)
            q_sim = (phi_sim * V_used) + (s_sim * elastic_head * A_m2)
        else:
            sw_err = float(mc_params.get("sw_err", 0.08))
            sw_sim = np.clip(np.random.normal(Sw_init, Sw_init * sw_err, n_sim), 0.05, 1.0)
            if has_seismic_vol:
                q_sim = (V_m3 * phi_sim * sw_sim) / Bw_init
            else:
                q_sim = (A_m2 * h_adj * phi_sim * sw_sim) / Bw_init

        p_sim = (q_sim * c_sim) / 1e4  # 万吨
        p90_val = round(float(np.percentile(p_sim, 10)), 2)
        p50_val = round(float(np.percentile(p_sim, 50)), 2)
        p10_val = round(float(np.percentile(p_sim, 90)), 2)
        samples = p_sim

    record = {
        "计算单元编号": unit_id,
        "计算时间": calc_time,
        "所属专题": assigned_topic,
        "构造带/单元名称": zone_name,
        "储层类型": res_type,
        "计算模型": calc_model,
        "几何建模方式": geom_method,
        "动静结合方法": dyn_method_applied,
        "动态参数校准说明": param_adjust_note,
        "动静推演计算过程": dyn_process,
        "校准后容积法推演": formula_final,
        "原始静态KCl储量(万吨)": round(P_static_raw / 1e4, 2),
        "最终核定KCl储量(万吨)": round(P_final / 1e4, 2),
        "校准前后储量变化率": f"{((P_final - P_static_raw)/P_static_raw)*100:+.2f}%",
        "核定卤水体积(亿m³)": round(Q_final / 1e8, 4),
        "蒙特卡洛P90(万吨)": p90_val,
        "蒙特卡洛P50(万吨)": p50_val,
        "蒙特卡洛P10(万吨)": p10_val,
        # 物性参数对比
        "原始孔隙度": f"{phi_init*100:.2f}%",
        "动态校准后孔隙度": f"{phi_adj*100:.2f}%",
        "原始有效厚度(m)": h_init,
        "动态校准后厚度(m)": round(h_adj, 2),
        "KCl品位(t/m³)": C,
        "含水面积(km²)": A_km2,
    }

    return record, samples


# --------------------------------------------------
# 主界面 Tabs
# --------------------------------------------------
tab_step1, tab_archive, tab_docs = st.tabs([
    "📥 第一步：计算单元数据录入与自适应推演",
    f"📋 第二步：【{active_topic}】全要素归档详单与下载",
    "📖 第三步：动-静结合参数校准指南",
])

# --------------------------------------------------
# TAB 1: 第一步
# --------------------------------------------------
with tab_step1:
    st.subheader(f"第一步：计算单元参数录入与动静结合推演 —— 【{active_topic}】")
    st.markdown(
        "**动-静结合核心准则**：动态试采数据（排卤量、压降、日产量）用来**校准静态解释参数（孔隙度 \(\phi\)、厚度 \(h\)）至真实流动合理取值**，再代入容积法计算，杜绝非理性的直接大幅打折。"
    )

    # 1. 蒙特卡洛全局参数控制面板
    with st.expander("⚙️ 蒙特卡洛模拟控制与不确定性扰动参数调节（点击展开/收起）", expanded=True):
        mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
        with mc_col1:
            mc_sims = st.select_slider(
                "抽样模拟试验次数 (N)",
                options=[1000, 2000, 5000, 10000, 20000, 50000],
                value=5000,
            )
        with mc_col2:
            mc_phi_err = st.slider(
                "校准后孔隙度 ϕ 相对扰动幅度 (±%)",
                min_value=5, max_value=50, value=20, step=5,
            ) / 100.0
        with mc_col3:
            mc_c_err = st.slider(
                "KCl 品位 C 相对扰动幅度 (±%)",
                min_value=5, max_value=50, value=15, step=5,
            ) / 100.0
        with mc_col4:
            mc_extra_err = st.slider(
                "次要物性 (Sw/S) 相对变异 (±%)",
                min_value=5, max_value=30, value=10, step=1,
            ) / 100.0

    cur_mc_params = {
        "n_sim": mc_sims,
        "phi_err": mc_phi_err,
        "c_err": mc_c_err,
        "sw_err": mc_extra_err,
        "s_err": mc_extra_err,
    }

    st.markdown("---")

    input_mode = st.radio(
        "请选择当前录入通道：",
        ["方式 A：在线交互录入计算单元具体参数（单单元高精度推演）", "方式 B：批量上传多计算单元数据表格 (CSV / Excel)"],
        horizontal=True
    )

    if "方式 A" in input_mode:
        st.markdown("#### 1. 构造单元基本信息与动静校准方法")
        ia1, ia2, ia3 = st.columns(3)
        with ia1:
            u_name = st.text_input("构造带/计算单元名称", value=f"{active_topic[:2]}某深部富钾构造")
        with ia2:
            u_type = st.radio("储卤层类型", ["油气同层型", "非油气同层型"], horizontal=True)
        with ia3:
            u_dyn_method = st.selectbox(
                "选择动-静结合参数校准方法",
                [
                    "方法一：物质平衡校准有效孔隙度 (基于试采排卤 Wp 与地层压降 Δp)",
                    "方法二：渗流压差反演校准有效厚度 (基于稳定日产量 qw 与流压差 Δp_wf)",
                ],
            )

        st.markdown("#### 2. 静态容积法基础参数")
        ib1, ib2, ib3, ib4 = st.columns(4)
        with ib1:
            u_C = st.number_input("KCl 平均品位 C (t/m³)", value=0.0195, step=0.0010, format="%.4f")
        with ib2:
            u_A = st.number_input("储层平面面积 A (km²)", value=55.0, step=1.0)
        with ib3:
            u_phi = st.number_input("静态解释有效孔隙度 ϕ (小数)", value=0.078, step=0.005, format="%.3f")
        with ib4:
            has_v = st.checkbox("已有三维地震精细雕刻体积 V", value=False)
            if has_v:
                u_V = st.number_input("雕刻体积 V (10⁶ m³)", value=1300.0, step=50.0)
                u_h = 0.0
            else:
                u_h = st.number_input("储层有效厚度 h (m)", value=25.0, step=1.0)
                u_V = None

        if u_type == "油气同层型":
            ic1, ic2 = st.columns(2)
            with ic1:
                u_sw = st.number_input("含水饱和度 Sw (小数)", value=0.62, step=0.05)
            with ic2:
                u_bw = st.number_input("卤水体积系数 Bw (无因次)", value=1.02, step=0.01)
            u_S, u_head, u_top = None, None, None
        else:
            ic3, ic4, ic5 = st.columns(3)
            with ic3:
                u_S = st.number_input("弹性储水系数 S", value=0.0005, step=0.0001, format="%.5f")
            with ic4:
                u_head = st.number_input("平均承压水头标高 h (m)", value=340.0, step=10.0)
            with ic5:
                u_top = st.number_input("平均储层顶面标高 H (m)", value=-2250.0, step=50.0)
            u_sw, u_bw = None, None

        st.markdown("#### 3. 动态试采参数（用于校准静态参数取值）")
        enable_dyn_single = st.checkbox("输入动态试采参数进行参数合理性校准", value=True)
        u_wp, u_dp, u_ct = None, None, None
        u_qw, u_dp_flow, u_k, u_mu = None, None, None, None

        if enable_dyn_single:
            if "方法一" in u_dyn_method:
                st.caption("💡 物质平衡法：依据试采排卤与压降，将静态孔隙度校准为真实连通动态孔隙度。")
                id1, id2, id3 = st.columns(3)
                with id1:
                    u_wp = st.number_input("累计排卤量 Wp (m³)", value=32000.0, step=1000.0)
                with id2:
                    u_dp = st.number_input("折算地层静态压降 Δp (MPa)", value=3.0, step=0.1)
                with id3:
                    u_ct = st.number_input("综合压缩系数 Ct (10⁻⁴ MPa⁻¹)", value=8.5, step=0.5)
            else:
                st.caption("💡 渗流压差法：依据稳态渗流产能，将静态解释厚度校准为实际动态产水厚度。")
                id4, id5, id6, id7 = st.columns(4)
                with id4:
                    u_qw = st.number_input("稳定日产水量 qw (m³/d)", value=190.0, step=10.0)
                with id5:
                    u_dp_flow = st.number_input("生产流压差 Δp_wf (MPa)", value=4.2, step=0.1)
                with id6:
                    u_k = st.number_input("储层渗透率 k (mD)", value=2.8, step=0.5)
                with id7:
                    u_mu = st.number_input("卤水动力黏度 μ (mPa·s)", value=1.12, step=0.05)

        if st.button("🚀 执行推演计算与蒙特卡洛模拟", type="primary"):
            input_dict = {
                "构造带": u_name,
                "储层类型": u_type,
                "动态方法": "方法一" if "方法一" in u_dyn_method else "方法二",
                "C": u_C, "A": u_A, "phi": u_phi, "h": u_h,
                "Sw": u_sw, "Bw": u_bw, "S": u_S, "承压水头标高": u_head, "储层顶面标高": u_top,
                "Wp": u_wp, "dp": u_dp, "ct": u_ct,
                "qw": u_qw, "dp_flow": u_dp_flow, "k": u_k, "mu": u_mu,
            }
            if u_V is not None:
                input_dict["V"] = u_V

            single_rec, mc_samples = evaluate_and_archive_unit(
                input_dict,
                assigned_topic=active_topic,
                default_dyn_method=input_dict["动态方法"],
                run_mc=True,
                mc_params=cur_mc_params,
            )
            st.session_state.temp_single_record = single_rec
            st.session_state.temp_mc_samples = mc_samples

        if st.session_state.temp_single_record is not None:
            cur_rec = st.session_state.temp_single_record
            cur_samples = st.session_state.temp_mc_samples

            st.markdown("---")
            st.subheader(f"📊 当前推演计算结果：{cur_rec['构造带/单元名称']}")

            res_c1, res_c2, res_c3, res_c4 = st.columns(4)
            res_c1.metric("原始静态 KCl 储量", f"{cur_rec['原始静态KCl储量(万吨)']:,.2f} 万吨")
            res_c2.metric("最终核定 KCl 储量", f"{cur_rec['最终核定KCl储量(万吨)']:,.2f} 万吨", delta=cur_rec['校准前后储量变化率'])
            res_c3.metric("动态参数校准行为", cur_rec['动态参数校准说明'])
            res_c4.metric("核定卤水体积", f"{cur_rec['核定卤水体积(亿m³)']:.4f} 亿m³")

            with st.expander("🔍 查看本单元参数校准细节与容积法推导", expanded=True):
                st.write(f"**动态校准过程**：{cur_rec['动静推演计算过程']}")
                st.write(f"**校准后容积法公式**：{cur_rec['校准后容积法推演']}")
                st.write(
                    f"**蒙特卡洛 ({cur_mc_params['n_sim']}次) 不确定性区间**："
                    f"保守值(P90) = **{cur_rec['蒙特卡洛P90(万吨)']}** 万吨 | "
                    f"中值(P50) = **{cur_rec['蒙特卡洛P50(万吨)']}** 万吨 | "
                    f"乐观值(P10) = **{cur_rec['蒙特卡洛P10(万吨)']}** 万吨"
                )

            if cur_samples is not None:
                fig_hist = px.histogram(
                    x=cur_samples, nbins=60,
                    labels={"x": "KCl 储量 (万吨)"},
                    title=f"{cur_rec['构造带/单元名称']} - 动态校准后蒙特卡洛分布 (P10 - P50 - P90)",
                )
                fig_hist.add_vline(x=cur_rec["蒙特卡洛P90(万吨)"], line_dash="dash", line_color="orange", annotation_text="P90")
                fig_hist.add_vline(x=cur_rec["蒙特卡洛P50(万吨)"], line_dash="solid", line_color="green", annotation_text="P50")
                fig_hist.add_vline(x=cur_rec["蒙特卡洛P10(万吨)"], line_dash="dash", line_color="red", annotation_text="P10")
                st.plotly_chart(fig_hist, use_container_width=True)

            col_save1, col_save2 = st.columns([1, 3])
            with col_save1:
                if st.button("💾 将本计算单元结果存入总账", type="primary", use_container_width=True):
                    existing_ids = [item["计算单元编号"] for item in st.session_state.topic_archives[active_topic]]
                    if cur_rec["计算单元编号"] not in existing_ids:
                        st.session_state.topic_archives[active_topic].append(cur_rec)
                    st.success(f"已成功归档至【{active_topic}】总账！")
                    st.session_state.temp_single_record = None
                    st.session_state.temp_mc_samples = None
                    st.rerun()

    else:
        # 方式 B：批量上传表格
        st.markdown("#### 批量上传与动态参数校准推演")
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            batch_dyn_def = st.selectbox(
                "默认参数校准方法（未在表格指定时生效）：",
                ["方法一：物质平衡校准有效孔隙度", "方法二：渗流压差反演校准有效厚度"],
            )
        with col_t2:
            sample_template = pd.DataFrame([
                {
                    "构造带": f"{active_topic[:2]}构造1号", "储层类型": "油气同层型", "动态方法": "方法一",
                    "A": 50.0, "h": 22.0, "phi": 0.08, "Sw": 0.65, "Bw": 1.02, "C": 0.019,
                    "Wp": 40000, "dp": 3.5, "ct": 8.2,
                },
                {
                    "构造带": f"{active_topic[:2]}构造2号", "储层类型": "非油气同层型", "动态方法": "方法二",
                    "A": 75.0, "V": 1800.0, "phi": 0.065, "S": 0.0004,
                    "承压水头标高": 350.0, "储层顶面标高": -2300.0, "C": 0.022,
                    "qw": 220.0, "dp_flow": 4.8, "k": 3.2, "mu": 1.1,
                },
                {
                    "构造带": f"{active_topic[:2]}构造3号", "储层类型": "油气同层型",
                    "A": 28.0, "h": 16.0, "phi": 0.07, "C": 0.016,
                },
            ])
            csv_buf = io.BytesIO()
            sample_template.to_csv(csv_buf, index=False, encoding="utf_8_sig")
            st.download_button(
                label="📥 下载参数校准测试模板 (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"{active_topic}_数据导入测试模板.csv",
                mime="text/csv",
            )

        file_up = st.file_uploader("选择数据表格文件", type=["csv", "xlsx", "xls"])
        if file_up is not None:
            try:
                if file_up.name.endswith(".csv"):
                    df_raw = pd.read_csv(file_up)
                else:
                    df_raw = pd.read_excel(file_up)

                st.write("📋 **待计算数据源预览：**")
                st.dataframe(df_raw.head(5), use_container_width=True)

                if st.button("⚡ 执行全表自适应推演并写入总账", type="primary"):
                    batch_records = []
                    method_code = "方法一" if "方法一" in batch_dyn_def else "方法二"
                    for _, row in df_raw.iterrows():
                        rec, _ = evaluate_and_archive_unit(
                            row.to_dict(),
                            assigned_topic=active_topic,
                            default_dyn_method=method_code,
                            run_mc=True,
                            mc_params=cur_mc_params,
                        )
                        batch_records.append(rec)
                        st.session_state.topic_archives[active_topic].append(rec)

                    st.success(
                        f"批量推演完成！已成功将 {len(batch_records)} 个单元按动态校准后参数写入【{active_topic}】总账！"
                    )
                    st.rerun()
            except Exception as err:
                st.error(f"批量推演失败: {str(err)}")

# --------------------------------------------------
# TAB 2: 第二步 全要素归档总账与详单下载
# --------------------------------------------------
with tab_archive:
    st.subheader(f"第二步：【{active_topic}】计算单元全要素归档详单")

    if not my_records:
        st.info("当前专题暂无归档数据。请在第一步中录入或批量上传后存入总账。")
    else:
        df_arc = pd.DataFrame(my_records)
        st.write(f"已累计归档 **{len(df_arc)}** 个单元的具体推演履历。")

        core_cols = [
            "计算单元编号", "计算时间", "构造带/单元名称", "储层类型", "计算模型",
            "动静结合方法", "动态参数校准说明", "原始静态KCl储量(万吨)", "最终核定KCl储量(万吨)",
            "校准前后储量变化率", "蒙特卡洛P50(万吨)", "校准后容积法推演"
        ]
        st.dataframe(df_arc[core_cols], use_container_width=True)

        fig_unit_bar = px.bar(
            df_arc,
            x="构造带/单元名称",
            y="最终核定KCl储量(万吨)",
            text="最终核定KCl储量(万吨)",
            color="动静结合方法",
            title=f"【{active_topic}】各计算单元核定储量 (目标配额: {my_target/1e4:.0f}万吨)",
        )
        fig_unit_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_unit_bar, use_container_width=True)

        csv_archive_out = df_arc.to_csv(index=False).encode("utf_8_sig")
        st.download_button(
            label=f"📥 导出【{active_topic}】全要素归档报表 (CSV)",
            data=csv_archive_out,
            file_name=f"{active_topic}_全要素计算单元归档详单.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# TAB 3: 第三步 动静结合理论
# --------------------------------------------------
with tab_docs:
    st.subheader("深层富钾卤水储量“动-静结合参数校准”方法体系")

    st.markdown("### 1. 动静结合的核心内涵")
    st.write(
        "在深层卤水资源评价中，静态容积法依赖测井电性解释与地震圈闭。但静态孔隙度往往包含非渗流死孔隙，静态厚度也不等于产液有效厚度。"
        "**动-静结合的科学途径是通过试采的压力、产量响应，将静态参数反演校准至具备实际流动能力的合理取值，再代入容积法。**"
    )

    st.markdown("### 2. 两套参数校准机制")
    st.write("#### 方法一：物质平衡反演校准有效连通孔隙度 \(\phi_{adj}\)")
    st.latex(r"\phi_{cal} = \frac{W_p}{c_t \cdot \Delta p \cdot V}")
    st.write("依据反演孔隙度与静态孔隙度的比值，校准孔隙度为有效连通流动孔隙度 \(\phi_{adj}\)。")

    st.write("#### 方法二：渗流压差反演校准有效储卤厚度 \(h_{adj}\)")
    st.latex(r"h_{cal} = \frac{q_w \cdot \mu_w \cdot \ln(R_e/r_w)}{2\pi \cdot k \cdot \Delta p_{wf}}")
    st.write("依据单井流动压差与出水量，校准电性解释厚度为真实产流有效厚度 \(h_{adj}\)。")

    st.markdown("---")
    st.markdown("### 3. 各专题增储任务配额")
    targets_table = [
        {"专题名称": "川中专题", "新增KCl目标配额": "2000 万吨", "主要勘探层系/靶区": "震旦系灯影组、三叠系雷口坡组/嘉陵江组等"},
        {"专题名称": "川东北专题", "新增KCl目标配额": "1500 万吨", "主要勘探层系/靶区": "三叠系嘉陵江组、雷口坡组、飞仙关组等"},
        {"专题名称": "川西专题", "新增KCl目标配额": "1000 万吨", "主要勘探层系/靶区": "二叠系栖霞-茅口组、三叠系雷口坡组等"},
        {"专题名称": "川南专题", "新增KCl目标配额": "500 万吨", "主要勘探层系/靶区": "奥陶系宝塔组、三叠系雷口坡组、嘉陵江组等"},
    ]
    st.table(pd.DataFrame(targets_table))
