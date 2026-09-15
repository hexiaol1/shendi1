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
    page_title="深层卤水钾盐动-静综合资源评价自适应系统",
    page_icon="🧂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 各专题目标配额（严格按照任务书设定）
TOPIC_TARGETS = {
    "川东北专题": 15_000_000.0,  # 1500万吨
    "川西专题": 10_000_000.0,    # 1000万吨
    "川中专题": 20_000_000.0,    # 2000万吨
    "川南专题": 5_000_000.0,     # 500万吨
}

# 专题独立存储（彼此互不相见）
if "topic_archives" not in st.session_state:
    st.session_state.topic_archives = {
        "川东北专题": [],
        "川西专题": [],
        "川中专题": [],
        "川南专题": [],
    }

# --------------------------------------------------
# 侧边栏：免密专题切换与独立指标监控
# --------------------------------------------------
with st.sidebar:
    st.title("🧂 专题独立控制台")
    st.caption("四川盆地深层富钾卤水储量评价系统")

    active_topic = st.selectbox(
        "请选择您所属的专题：",
        ["川东北专题", "川西专题", "川中专题", "川南专题"],
        index=0,
    )

    my_target = TOPIC_TARGETS[active_topic]
    my_records = st.session_state.topic_archives[active_topic]

    # 汇总计算当前专题的累计最终储量
    my_total_kcl = sum([r["最终核定KCl储量(万吨)"] for r in my_records])
    my_progress = min(1.0, (my_total_kcl * 1e4) / my_target) if my_target > 0 else 0.0

    st.markdown("---")
    st.subheader(f"🎯 【{active_topic}】增储进度")
    st.metric(
        label="本专题已核算 KCl 储量",
        value=f"{my_total_kcl:,.2f} 万吨",
        delta=(
            f"距目标差: {(my_target/1e4 - my_total_kcl):,.2f} 万吨"
            if (my_total_kcl * 1e4) < my_target
            else "🎉 本专题增储目标已达成！"
        ),
    )
    st.progress(my_progress)
    st.caption(
        f"目标配额：{my_target/1e4:.0f} 万吨 | 当前达成率：{my_progress * 100:.2f}%"
    )

    st.markdown("---")
    st.info(
        f"🔒 **数据隔离提示**：\n"
        f"当前工作区仅加载【{active_topic}】的计算归档与成果。其他专题的数据完全独立隔离，在此处不可见。"
    )

    if st.button("🗑️ 清空本专题所有归档记录", use_container_width=True):
        st.session_state.topic_archives[active_topic] = []
        st.rerun()


# --------------------------------------------------
# 自适应评价核心引擎（带全过程追溯生成）
# --------------------------------------------------
def evaluate_and_archive_unit(
    row_data, assigned_topic, default_dyn_method="方法一：弹性压释物质平衡法", run_mc=True, n_sim=5000
):
    r = {str(k).strip(): v for k, v in row_data.items() if pd.notna(v)}
    calc_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unit_id = f"UNIT-{uuid.uuid4().hex[:8].upper()}"

    zone_name = str(r.get("构造带", r.get("构造名称", r.get("计算单元", "未命名单元"))))
    res_type = str(r.get("储层类型", r.get("类型", "油气同层型"))).strip()

    # 1. 基础静态参数
    C = float(r.get("C", r.get("KCl品位", r.get("品位", 0.018))))
    A_km2 = float(r.get("A", r.get("Aw", r.get("面积", 0.0))))
    A_m2 = A_km2 * 1e6
    h = float(r.get("h", r.get("厚度", r.get("有效厚度", 0.0))))

    phi = float(r.get("phi", r.get("ϕ", r.get("孔隙度", r.get("孔隙率", 0.08)))))
    if phi > 1.0:
        phi = phi / 100.0

    # 地震雕刻判定
    has_seismic_vol = ("V" in r) or ("雕刻体积" in r) or ("储层体积" in r)
    if has_seismic_vol:
        V_raw = float(r.get("V", r.get("雕刻体积", r.get("储层体积", 0.0))))
        V_m3 = V_raw * 1e6 if V_raw < 1e5 else V_raw
        geom_method = "三维地震精细雕刻体积(V)"
    else:
        V_m3 = A_m2 * h
        geom_method = "平面面积乘厚度估算(A×h)"

    # 2. 静态容积法推导过程记录
    if "非油气" in res_type:
        calc_model = "非油气同层型容积法"
        S = float(r.get("S", r.get("弹性储水系数", 0.0005)))
        head_h = float(r.get("承压水头标高", r.get("h_head", 350.0)))
        top_H = float(r.get("储层顶面标高", r.get("H_top", -2200.0)))
        elastic_head = max(0.0, head_h - top_H)

        Q_static = (phi * V_m3) + (S * elastic_head * A_m2)
        formula_desc = (
            f"Q = ϕ·V + S·(h-H)·A = {phi:.4f}×{V_m3:.2e} + "
            f"{S:.5f}×({head_h}-{top_H})×{A_m2:.2e}"
        )
        Sw_val, Bw_val = np.nan, np.nan
        S_val, head_val, top_val = S, head_h, top_H
    else:
        calc_model = "油气同层型容积法"
        Sw = float(r.get("Sw", r.get("含水饱和度", 0.65)))
        if Sw > 1.0:
            Sw /= 100.0
        Bw = float(r.get("Bw", r.get("体积系数", 1.02)))

        if has_seismic_vol:
            Q_static = (V_m3 * phi * Sw) / Bw
            formula_desc = f"Q = (V·ϕ·Sw)/Bw = ({V_m3:.2e}×{phi:.4f}×{Sw:.2f})/{Bw:.2f}"
        else:
            Q_static = (A_m2 * h * phi * Sw) / Bw
            formula_desc = (
                f"Q = (Aw·h·ϕ·Sw)/Bw = ({A_m2:.2e}×{h:.2f}×{phi:.4f}×{Sw:.2f})/{Bw:.2f}"
            )
        Sw_val, Bw_val = Sw, Bw
        S_val, head_val, top_val = np.nan, np.nan, np.nan

    P_static = Q_static * C  # 吨

    # 3. 动态约束方法识别与计算
    dyn_choice = str(r.get("动态方法", default_dyn_method))
    alpha = 1.0
    dyn_process = "未录入有效动态参数，系统按静态容积法基准输出"
    dyn_method_applied = "纯静态（未启用动态约束）"

    has_m1 = ("Wp" in r or "排卤量" in r or "累计产水量" in r) and (
        "dp" in r or "压降" in r or "Δp" in r
    )
    has_m2 = (
        ("qw" in r or "日产水量" in r) and ("dp_flow" in r or "流压差" in r)
    ) or ("Re" in r or "波及半径" in r)

    if "方法一" in dyn_choice and has_m1:
        Wp = float(r.get("Wp", r.get("排卤量", r.get("累计产水量", 0.0))))
        dp = float(r.get("dp", r.get("压降", r.get("Δp", 1.0))))
        ct_raw = float(r.get("ct", r.get("Ct", r.get("压缩系数", 8.5))))
        ct = ct_raw * 1e-4 if ct_raw > 1e-2 else ct_raw

        if dp > 0 and ct > 0:
            Q_dyn = Wp / (ct * dp)
            alpha = min(1.0, max(0.05, Q_dyn / Q_static)) if Q_static > 0 else 1.0
            dyn_method_applied = "方法一：弹性压释物质平衡法"
            dyn_process = (
                f"动态水体 Q_dyn = Wp/(ct·Δp) = {Wp:.0f}/({ct:.2e}×{dp:.2f}) = {Q_dyn/1e8:.4f} 亿m³; "
                f"动静比 α = min(1.0, Q_dyn/Q_static) = {alpha:.4f}"
            )
    elif "方法二" in dyn_choice and has_m2:
        if "Re" in r or "波及半径" in r:
            Re = float(r.get("Re", r.get("波及半径", 1000.0)))
            calc_re_note = f"直接采用实测波及半径 Re={Re:.1f}m"
        else:
            qw = float(r.get("qw", r.get("日产水量", 150.0)))
            dp_flow = float(r.get("dp_flow", r.get("流压差", 5.0)))
            k_perm = float(r.get("k", r.get("渗透率", 2.0)))
            mu = float(r.get("mu", r.get("卤水黏度", 1.2)))
            Re = max(
                100.0,
                np.sqrt((qw * mu) / (0.00708 * k_perm * max(1.0, h) * dp_flow + 1e-6)) * 100.0,
            )
            calc_re_note = f"基于渗流公式反求波及半径 Re={Re:.1f}m"

        A_dyn_m2 = np.pi * (Re**2)
        if "非油气" in res_type:
            Q_dyn = (phi * A_dyn_m2 * h) + (S_val * elastic_head * A_dyn_m2)
        else:
            Q_dyn = (A_dyn_m2 * h * phi * Sw_val) / Bw_val

        alpha = min(1.0, max(0.05, Q_dyn / Q_static)) if Q_static > 0 else 1.0
        dyn_method_applied = "方法二：渗流压力波及漏斗法"
        dyn_process = (
            f"{calc_re_note}; 波及面积 A_dyn = {A_dyn_m2/1e6:.2f} km²; "
            f"动态控水 Q_dyn = {Q_dyn/1e8:.4f} 亿m³; 动静约束比 α = {alpha:.4f}"
        )
    elif has_m1:
        # 自适应保底执行方法一
        Wp = float(r.get("Wp", 0.0))
        dp = float(r.get("dp", 1.0))
        ct = float(r.get("ct", 8.5)) * 1e-4
        Q_dyn = Wp / (ct * dp)
        alpha = min(1.0, max(0.05, Q_dyn / Q_static)) if Q_static > 0 else 1.0
        dyn_method_applied = "方法一：弹性压释物质平衡法(自适应降级触发)"
        dyn_process = f"Q_dyn = {Q_dyn/1e8:.4f} 亿m³, 约束比 α = {alpha:.4f}"

    Q_final = Q_static * alpha
    P_final = P_static * alpha

    # 4. 蒙特卡洛模拟
    p90_val, p50_val, p10_val = np.nan, np.nan, np.nan
    samples = None
    if run_mc:
        np.random.seed(42)
        phi_err = float(r.get("phi_err", 0.20))
        c_err = float(r.get("c_err", 0.15))

        phi_sim = np.random.triangular(phi * (1 - phi_err), phi, phi * (1 + phi_err), n_sim)
        c_sim = np.random.triangular(C * (1 - c_err), C, C * (1 + c_err), n_sim)

        if "非油气" in res_type:
            q_sim = (phi_sim * V_m3) + (S_val * elastic_head * A_m2)
        else:
            if has_seismic_vol:
                q_sim = (V_m3 * phi_sim * Sw_val) / Bw_val
            else:
                q_sim = (A_m2 * h * phi_sim * Sw_val) / Bw_val

        p_sim = (q_sim * c_sim * alpha) / 1e4  # 万吨
        p90_val = round(float(np.percentile(p_sim, 10)), 2)
        p50_val = round(float(np.percentile(p_sim, 50)), 2)
        p10_val = round(float(np.percentile(p_sim, 90)), 2)
        samples = p_sim

    # 5. 完整详单字典归档（包含输入参数、计算方式、折减过程与成果）
    record = {
        "计算单元编号": unit_id,
        "计算时间": calc_time,
        "所属专题": assigned_topic,
        "构造带/单元名称": zone_name,
        "储层类型": res_type,
        "计算模型": calc_model,
        "几何建模方式": geom_method,
        "容积法公式推导过程": formula_desc,
        "动静结合方法": dyn_method_applied,
        "动静结合推导过程及参数": dyn_process,
        "动静有效性系数(α)": round(alpha, 4),
        "静态卤水体积(亿m³)": round(Q_static / 1e8, 4),
        "静态KCl储量(万吨)": round(P_static / 1e4, 2),
        "最终核定卤水体积(亿m³)": round(Q_final / 1e8, 4),
        "最终核定KCl储量(万吨)": round(P_final / 1e4, 2),
        "蒙特卡洛P90(万吨)": p90_val,
        "蒙特卡洛P50(万吨)": p50_val,
        "蒙特卡洛P10(万吨)": p10_val,
        # 详细原始参数
        "KCl品位(t/m³)": C,
        "含水面积(km²)": A_km2,
        "有效厚度(m)": h,
        "储层体积(10⁶m³)": round(V_m3 / 1e6, 2),
        "孔隙度(小数)": phi,
        "含水饱和度(小数)": Sw_val,
        "卤水体积系数": Bw_val,
        "弹性储水系数": S_val,
        "承压水头标高(m)": head_val,
        "储层顶面标高(m)": top_val,
    }

    return record, samples


# --------------------------------------------------
# 主界面 Tabs
# --------------------------------------------------
tab_upload, tab_single, tab_archive, tab_docs = st.tabs([
    "📁 批量上传与自适应计算",
    "✍️ 单单元交互计算与归档",
    f"📋 【{active_topic}】全要素归档详单与下载",
    "📖 双动静结合方法体系指南",
])

# --------------------------------------------------
# TAB 1: 批量数据上传计算并归档
# --------------------------------------------------
with tab_upload:
    st.subheader(f"批量上传【{active_topic}】多源数据表并自动归档")
    st.write("各专题可上传任意格式的 CSV / Excel 表格。系统将自动自适应匹配字段，逐个单元推演计算并写入专属归档总账。")

    default_dyn = st.radio(
        "默认动-静结合方法（若表格内未指定“动态方法”列，则执行此项）：",
        [
            "方法一：弹性压释物质平衡法 (需要累计排卤量 Wp 与折算压降 dp)",
            "方法二：渗流压力波及漏斗法 (需要产水速度 qw、压差 dp_flow 或 波及半径 Re)",
        ],
        horizontal=False,
    )

    sample_template = pd.DataFrame([
        {
            "构造带": f"{active_topic[:3]}构造A井区",
            "储层类型": "油气同层型",
            "动态方法": "方法一",
            "A": 50.0, "h": 22.0, "phi": 0.08, "Sw": 0.65, "Bw": 1.02, "C": 0.019,
            "Wp": 40000, "dp": 3.5, "ct": 8.2,
        },
        {
            "构造带": f"{active_topic[:3]}深层承压B区",
            "储层类型": "非油气同层型",
            "动态方法": "方法二",
            "A": 75.0, "V": 1800.0, "phi": 0.065, "S": 0.0004,
            "承压水头标高": 350.0, "储层顶面标高": -2300.0, "C": 0.022,
            "qw": 220.0, "dp_flow": 4.8, "k": 3.2, "mu": 1.1,
        },
        {
            "构造带": f"{active_topic[:3]}新层系早期单元C",
            "储层类型": "油气同层型",
            "A": 28.0, "h": 16.0, "phi": 0.07, "C": 0.016,
        },
    ])
    csv_buf = io.BytesIO()
    sample_template.to_csv(csv_buf, index=False, encoding="utf_8_sig")
    st.download_button(
        label="📥 下载多单元批量测试模板 (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"{active_topic}_数据导入测试模板.csv",
        mime="text/csv",
    )

    file_up = st.file_uploader("选择要导入并计算的数据文件", type=["csv", "xlsx", "xls"])
    if file_up is not None:
        try:
            if file_up.name.endswith(".csv"):
                df_raw = pd.read_csv(file_up)
            else:
                df_raw = pd.read_excel(file_up)

            st.write("📋 **待计算数据源预览：**")
            st.dataframe(df_raw.head(5), use_container_width=True)

            if st.button("⚡ 执行全表自适应推演计算并写入总账", type="primary"):
                batch_records = []
                method_code = "方法一" if "方法一" in default_dyn else "方法二"
                for _, row in df_raw.iterrows():
                    rec, _ = evaluate_and_archive_unit(
                        row.to_dict(),
                        assigned_topic=active_topic,
                        default_dyn_method=method_code,
                        run_mc=True,
                        n_sim=2000,
                    )
                    batch_records.append(rec)
                    st.session_state.topic_archives[active_topic].append(rec)

                st.success(
                    f"计算成功！已将 {len(batch_records)} 个计算单元写入【{active_topic}】专属归档！"
                )
                df_batch_show = pd.DataFrame(batch_records)[[
                    "计算单元编号", "构造带/单元名称", "储层类型", "几何建模方式",
                    "动静结合方法", "动静有效性系数(α)", "最终核定KCl储量(万吨)", "蒙特卡洛P50(万吨)"
                ]]
                st.dataframe(df_batch_show, use_container_width=True)
        except Exception as err:
            st.error(f"批量解析推演失败: {str(err)}")

# --------------------------------------------------
# TAB 2: 单单元交互计算与归档
# --------------------------------------------------
with tab_single:
    st.subheader(f"单计算单元参数录入与详细推演 —— 【{active_topic}】")

    c1, c2, c3 = st.columns(3)
    with c1:
        u_name = st.text_input("构造带/圈闭/计算单元名称", value=f"{active_topic[:3]}某先导试验区")
    with c2:
        u_type = st.radio("储卤层类型", ["油气同层型", "非油气同层型"], horizontal=True)
    with c3:
        u_dyn_method = st.selectbox(
            "选择动-静结合评价方法",
            [
                "方法一：弹性压释物质平衡法 (Wp - Δp)",
                "方法二：渗流压力波及漏斗法 (qw - Δp_wf - k)",
            ],
        )

    st.markdown("##### 1. 静态容积法参数")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        u_C = st.number_input("KCl平均品位 C (t/m³)", value=0.0195, step=0.0010, format="%.4f")
    with g2:
        u_A = st.number_input("储层平面面积 A (km²)", value=55.0, step=1.0)
    with g3:
        u_phi = st.number_input("有效孔隙度 ϕ (小数)", value=0.078, step=0.005, format="%.3f")
    with g4:
        has_v = st.checkbox("已有三维地震精细雕刻体积 V", value=False)
        if has_v:
            u_V = st.number_input("雕刻体积 V (10⁶ m³)", value=1300.0, step=50.0)
            u_h = 0.0
        else:
            u_h = st.number_input("储层有效厚度 h (m)", value=25.0, step=1.0)
            u_V = None

    if u_type == "油气同层型":
        st.markdown("##### 2. 油气同层参数")
        o1, o2 = st.columns(2)
        with o1:
            u_sw = st.number_input("含水饱和度 Sw (小数)", value=0.62, step=0.05)
        with o2:
            u_bw = st.number_input("卤水体积系数 Bw", value=1.02, step=0.01)
        u_S, u_head, u_top = None, None, None
    else:
        st.markdown("##### 2. 非油气同层承压水参数")
        n1, n2, n3 = st.columns(3)
        with n1:
            u_S = st.number_input("弹性储水系数 S", value=0.0005, step=0.0001, format="%.5f")
        with n2:
            u_head = st.number_input("平均承压水头标高 h (m)", value=340.0, step=10.0)
        with n3:
            u_top = st.number_input("平均储层顶面标高 H (m)", value=-2250.0, step=50.0)
        u_sw, u_bw = None, None

    st.markdown("##### 3. 动态校准参数（可选，留空则执行纯静态计算）")
    enable_dyn_single = st.checkbox("输入动态试采参数进行动静结合校准", value=True)
    u_wp, u_dp, u_ct = None, None, None
    u_qw, u_dp_flow, u_k, u_mu, u_re = None, None, None, None, None

    if enable_dyn_single:
        if "方法一" in u_dyn_method:
            d1, d2, d3 = st.columns(3)
            with d1:
                u_wp = st.number_input("累计排卤量 Wp (m³)", value=32000.0, step=1000.0)
            with d2:
                u_dp = st.number_input("折算地层压降 Δp (MPa)", value=3.0, step=0.1)
            with d3:
                u_ct = st.number_input("综合压缩系数 Ct (10⁻⁴ MPa⁻¹)", value=8.5, step=0.5)
        else:
            d4, d5, d6, d7 = st.columns(4)
            with d4:
                u_qw = st.number_input("稳定日产水量 qw (m³/d)", value=190.0, step=10.0)
            with d5:
                u_dp_flow = st.number_input("生产流压差 Δp_wf (MPa)", value=4.2, step=0.1)
            with d6:
                u_k = st.number_input("储层渗透率 k (mD)", value=2.8, step=0.5)
            with d7:
                u_mu = st.number_input("卤水动力黏度 μ (mPa·s)", value=1.12, step=0.05)

    if st.button("🚀 开始核算该单元并生成详细追溯报告", type="primary"):
        input_data = {
            "构造带": u_name,
            "储层类型": u_type,
            "动态方法": "方法一" if "方法一" in u_dyn_method else "方法二",
            "C": u_C, "A": u_A, "phi": u_phi, "h": u_h,
            "Sw": u_sw, "Bw": u_bw, "S": u_S, "承压水头标高": u_head, "储层顶面标高": u_top,
            "Wp": u_wp, "dp": u_dp, "ct": u_ct,
            "qw": u_qw, "dp_flow": u_dp_flow, "k": u_k, "mu": u_mu, "Re": u_re,
        }
        if u_V is not None:
            input_data["V"] = u_V

        single_rec, mc_samples = evaluate_and_archive_unit(
            input_data,
            assigned_topic=active_topic,
            default_dyn_method=input_data["动态方法"],
            run_mc=True,
            n_sim=5000,
        )

        st.success(f"计算完成！归档编号：{single_rec['计算单元编号']}")
        
        # 结果与推导说明卡片
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("静态 KCl 资源量", f"{single_rec['静态KCl储量(万吨)']:,.2f} 万吨")
        col_res2.metric("动静连通系数 (α)", f"{single_rec['动静有效性系数(α)']:.4f}")
        col_res3.metric("最终核定 KCl 储量", f"{single_rec['最终核定KCl储量(万吨)']:,.2f} 万吨")

        with st.expander("🔍 查看本单元详细推演过程与计算公式", expanded=True):
            st.write(f"**容积法推导**：{single_rec['容积法公式推导过程']}")
            st.write(f"**动静约束逻辑**：{single_rec['动静结合推导过程及参数']}")
            st.write(f"**不确定性区间**：P90={single_rec['蒙特卡洛P90(万吨)']} 万吨 | P50={single_rec['蒙特卡洛P50(万吨)']} 万吨 | P10={single_rec['蒙特卡洛P10(万吨)']} 万吨")

        if mc_samples is not None:
            fig_hist = px.histogram(
                x=mc_samples, nbins=50,
                labels={"x": "KCl 储量 (万吨)"},
                title=f"{u_name} - 蒙特卡洛概率分布 (P10 - P50 - P90)",
            )
            fig_hist.add_vline(x=single_rec["蒙特卡洛P90(万吨)"], line_dash="dash", line_color="orange", annotation_text="P90")
            fig_hist.add_vline(x=single_rec["蒙特卡洛P50(万吨)"], line_dash="solid", line_color="green", annotation_text="P50")
            fig_hist.add_vline(x=single_rec["蒙特卡洛P10(万吨)"], line_dash="dash", line_color="red", annotation_text="P10")
            st.plotly_chart(fig_hist, use_container_width=True)

        if st.button("💾 将本次详细推演成果写入总账归档"):
            st.session_state.topic_archives[active_topic].append(single_rec)
            st.success("已成功归档！可切换至“全要素归档详单”查看与导出。")

# --------------------------------------------------
# TAB 3: 本专题全要素归档总账与详单下载
# --------------------------------------------------
with tab_archive:
    st.subheader(f"📋 【{active_topic}】计算单元全要素归档详单")

    if not my_records:
        st.info("当前专题暂无归档数据。请在 Tab 1 或 Tab 2 中执行计算后写入总账。")
    else:
        df_arc = pd.DataFrame(my_records)
        st.write(f"已累计归档 **{len(df_arc)}** 个计算单元的完整参数及推演履历。")

        # 核心关键字段预览展示
        core_cols = [
            "计算单元编号", "计算时间", "构造带/单元名称", "储层类型", "计算模型",
            "动静结合方法", "动静有效性系数(α)", "静态KCl储量(万吨)", "最终核定KCl储量(万吨)",
            "蒙特卡洛P50(万吨)", "容积法公式推导过程", "动静结合推导过程及参数"
        ]
        st.dataframe(df_arc[core_cols], use_container_width=True)

        # 柱状分布图
        fig_unit_bar = px.bar(
            df_arc,
            x="构造带/单元名称",
            y="最终核定KCl储量(万吨)",
            text="最终核定KCl储量(万吨)",
            color="动静结合方法",
            title=f"【{active_topic}】各计算单元 KCl 核定储量分布 (目标: {my_target/1e4:.0f}万吨)",
        )
        fig_unit_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_unit_bar, use_container_width=True)

        # 完整详单导出（含全部输入参数和推导文本）
        csv_archive_out = df_arc.to_csv(index=False).encode("utf_8_sig")
        st.download_button(
            label=f"📥 导出【{active_topic}】全要素计算归档总报表 (包含完整推导过程 CSV)",
            data=csv_archive_out,
            file_name=f"{active_topic}_全要素计算单元归档详单.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# TAB 4: 双动静结合方法体系与公式说明书
# --------------------------------------------------
with tab_docs:
    st.subheader("深层卤水钾盐储量动-静结合双方法技术规范")

    st.markdown("### 1. 静态容积法评价模型")
    st.write("**（1）油气同层型卤水：**")
    st.latex(r"Q_w = \frac{A_w \cdot h \cdot \phi \cdot S_w}{B_w}, \quad P_A = Q_w \cdot C")
    st.caption("注：当具备三维地震精细雕刻体积 V 时，公式自动自适应替代为 Q = (V · ϕ · Sw) / Bw。")

    st.write("**（2）非油气同层型卤水：**")
    st.latex(r"Q_{ws} = \phi \cdot V + S \cdot (h - H) \cdot A, \quad P_A = Q_{ws} \cdot C")
    st.caption("注：若缺乏精细雕刻体 V，系统自动采用几何体积 V = A · h 替代。")

    st.markdown("---")
    st.markdown("### 2. 双套动-静结合方法原理与公式")

    st.write("#### 方法一：地层弹性压释 / 物质平衡法")
    st.write("基于试采流压衰减与累计采卤量，反算弹性水体连通体积并折减静态储量：")
    st.latex(r"Q_{dyn} = \frac{W_p}{c_t \cdot \Delta p}")
    st.latex(r"\alpha = \min\left(1.0, \; \frac{Q_{dyn}}{Q_{stat}}\right), \quad P_{final} = P_{stat} \cdot \alpha")

    st.write("#### 方法二：渗流压力波及漏斗 / 拟稳态有效半径法")
    st.write("基于拟稳态径向渗流机理，反求单井实际波及泄流半径 Re 与动态控制面积：")
    st.latex(r"R_e = \sqrt{\frac{q_w \cdot \mu_w}{0.00708 \cdot k \cdot h \cdot \Delta p_{wf}}}")
    st.latex(r"A_{dyn} = \pi R_e^2, \quad \alpha = \min\left(1.0, \; \frac{A_{dyn}}{A_{stat}}\right), \quad P_{final} = P_{stat} \cdot \alpha")

    st.markdown("---")
    st.markdown("### 3. 各专题增储配额一览表")
    targets_display = [
        {"专题名称": "川中专题", "新增KCl目标配额": "2000 万吨", "主要勘探层系/靶区": "震旦系灯影组、三叠系雷口坡组/嘉陵江组等"},
        {"专题名称": "川东北专题", "新增KCl目标配额": "1500 万吨", "主要勘探层系/靶区": "三叠系嘉陵江组、雷口坡组、飞仙关组等"},
        {"专题名称": "川西专题", "新增KCl目标配额": "1000 万吨", "主要勘探层系/靶区": "二叠系栖霞-茅口组、三叠系雷口坡组等"},
        {"专题名称": "川南专题", "新增KCl目标配额": "500 万吨", "主要勘探层系/靶区": "奥陶系宝塔组、三叠系雷口坡组、嘉陵江组等"},
        {"专题名称": "全盆地总目标", "新增KCl目标配额": "5000 万吨", "主要勘探层系/靶区": "动-静结合综合评价新增储量"},
    ]
    st.table(pd.DataFrame(targets_display))
