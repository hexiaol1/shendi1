import io
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

# 各专题独立鉴权与增储目标配额
AUTH_CREDENTIALS = {
    "川中专题": {"key": "cz2026", "target": 12_500_000.0},
    "川西专题": {"key": "cx2026", "target": 12_500_000.0},
    "川东北专题": {"key": "cdb2026", "target": 12_500_000.0},
    "川南专题": {"key": "cn2026", "target": 12_500_000.0},
}

# 独立数据隔离仓储
if "isolated_storage" not in st.session_state:
    st.session_state.isolated_storage = {
        "川中专题": [],
        "川西专题": [],
        "川东北专题": [],
        "川南专题": [],
    }

if "current_user_topic" not in st.session_state:
    st.session_state.current_user_topic = None


# --------------------------------------------------
# 自适应评价核心引擎（支持两套动静结合方法）
# --------------------------------------------------
def evaluate_brine_record(
    row_data, assigned_topic, dyn_method_choice="方法一：弹性压释物质平衡法", run_mc=True, n_sim=5000
):
    res = {}
    r = {str(k).strip(): v for k, v in row_data.items() if pd.notna(v)}

    res["构造带"] = str(r.get("构造带", r.get("构造名称", "未命名构造")))
    res["专题"] = assigned_topic
    res["储层类型"] = str(r.get("储层类型", r.get("类型", "油气同层型"))).strip()

    # 1. 基础参数提取与自适应清洗
    C = float(r.get("C", r.get("KCl品位", r.get("品位", 0.018))))
    res["KCl品位(t/m³)"] = C

    A_km2 = float(r.get("A", r.get("Aw", r.get("面积", 0.0))))
    A_m2 = A_km2 * 1e6
    h = float(r.get("h", r.get("厚度", r.get("有效厚度", 0.0))))

    phi = float(r.get("phi", r.get("ϕ", r.get("孔隙度", r.get("孔隙率", 0.08)))))
    if phi > 1.0:
        phi = phi / 100.0

    # 判断是否具备地震精细雕刻体积 V (10^6 m3)
    has_seismic_vol = ("V" in r) or ("雕刻体积" in r) or ("储层体积" in r)
    if has_seismic_vol:
        V_raw = float(r.get("V", r.get("雕刻体积", r.get("储层体积", 0.0))))
        V_m3 = V_raw * 1e6 if V_raw < 1e5 else V_raw
        res["几何建模方式"] = "地震精细雕刻体积 V"
    else:
        V_m3 = A_m2 * h
        res["几何建模方式"] = "面积×厚度估算 (A×h)"

    # 2. 静态容积法基准计算
    if "非油气" in res["储层类型"]:
        res["计算模型"] = "非油气同层型"
        S = float(r.get("S", r.get("弹性储水系数", 0.0005)))
        head_h = float(r.get("承压水头标高", r.get("h_head", 350.0)))
        top_H = float(r.get("储层顶面标高", r.get("H_top", -2200.0)))
        elastic_head = max(0.0, head_h - top_H)
        Q_static = (phi * V_m3) + (S * elastic_head * A_m2)
    else:
        res["计算模型"] = "油气同层型"
        Sw = float(r.get("Sw", r.get("含水饱和度", 0.65)))
        if Sw > 1.0:
            Sw /= 100.0
        Bw = float(r.get("Bw", r.get("体积系数", 1.02)))
        if has_seismic_vol:
            Q_static = (V_m3 * phi * Sw) / Bw
        else:
            Q_static = (A_m2 * h * phi * Sw) / Bw

    P_static = Q_static * C
    res["静态卤水体积(亿m³)"] = round(Q_static / 1e8, 4)
    res["静态KCl储量(万吨)"] = round(P_static / 1e4, 2)

    # 3. 动态约束选择与执行
    # 如果上传的数据表中有指定列，则以数据行为主；否则遵循全局选择
    chosen_dyn = str(r.get("动态方法", dyn_method_choice))
    alpha = 1.0
    dynamic_status = "无动态数据(执行纯静态)"

    # 判断是否存在动态方法一所需参数 (Wp, dp)
    has_method1_params = ("Wp" in r or "排卤量" in r or "累计产水量" in r) and (
        "dp" in r or "压降" in r or "Δp" in r
    )
    # 判断是否存在动态方法二所需参数 (qw, dp_flow, k 或 直接Re/波及半径)
    has_method2_params = (
        ("qw" in r or "日产水量" in r or "产水速度" in r)
        and ("dp_flow" in r or "流压差" in r or "生产压差" in r)
    ) or ("Re" in r or "波及半径" in r)

    if "方法一" in chosen_dyn and has_method1_params:
        Wp = float(r.get("Wp", r.get("排卤量", r.get("累计产水量", 0.0))))
        dp = float(r.get("dp", r.get("压降", r.get("Δp", 1.0))))
        ct_raw = float(r.get("ct", r.get("Ct", r.get("压缩系数", 8.5))))
        ct = ct_raw * 1e-4 if ct_raw > 1e-2 else ct_raw

        if dp > 0 and ct > 0:
            Q_dyn = Wp / (ct * dp)
            alpha = min(1.0, max(0.05, Q_dyn / Q_static)) if Q_static > 0 else 1.0
            dynamic_status = f"方法一(弹性物质平衡): 约束比 α={alpha:.3f}"
    elif "方法二" in chosen_dyn and has_method2_params:
        # 方法二：渗流压力波及漏斗 / 动态控制半径约束
        if "Re" in r or "波及半径" in r:
            Re = float(r.get("Re", r.get("波及半径", 1000.0)))
        else:
            # 拟稳态径向流动经验反算波及有效阻抗半径
            qw = float(r.get("qw", r.get("日产水量", 150.0)))  # m3/d
            dp_flow = float(r.get("dp_flow", r.get("流压差", 5.0)))  # MPa
            k_perm = float(r.get("k", r.get("渗透率", 2.0)))  # mD
            mu = float(r.get("mu", r.get("卤水黏度", 1.2)))  # mPa·s
            # 渗流波及当量半径 Re (米)
            Re = max(100.0, np.sqrt((qw * mu) / (0.00708 * k_perm * max(1.0, h) * dp_flow + 1e-6)) * 100.0)

        A_dyn_m2 = np.pi * (Re**2)
        # 动态控制水体积
        if "非油气" in res["储层类型"]:
            Q_dyn = (phi * A_dyn_m2 * h) + (S * elastic_head * A_dyn_m2)
        else:
            Q_dyn = (A_dyn_m2 * h * phi * Sw) / Bw

        alpha = min(1.0, max(0.05, Q_dyn / Q_static)) if Q_static > 0 else 1.0
        dynamic_status = f"方法二(渗流压降漏斗): Re={Re:.0f}m, α={alpha:.3f}"
    elif has_method1_params:
        # 自适应容错：选了方法二但只有方法一的数据时自动降级应用方法一
        Wp = float(r.get("Wp", 0.0))
        dp = float(r.get("dp", 1.0))
        ct = float(r.get("ct", 8.5)) * 1e-4
        Q_dyn = Wp / (ct * dp)
        alpha = min(1.0, max(0.05, Q_dyn / Q_static)) if Q_static > 0 else 1.0
        dynamic_status = f"自适应切换为方法一: α={alpha:.3f}"

    res["动态评价模式"] = chosen_dyn
    res["动态约束状态"] = dynamic_status
    Q_final = Q_static * alpha
    P_final = P_static * alpha

    res["最终核定卤水体积(亿m³)"] = round(Q_final / 1e8, 4)
    res["最终核定KCl储量(万吨)"] = round(P_final / 1e4, 2)
    res["P_final_raw"] = P_final

    # 4. 蒙特卡洛模拟
    mc_results = {}
    if run_mc:
        np.random.seed(42)
        phi_err = float(r.get("phi_err", r.get("孔隙度相对误差", 0.20)))
        c_err = float(r.get("c_err", r.get("品位相对误差", 0.15)))

        phi_samples = np.random.triangular(
            phi * (1 - phi_err), phi, phi * (1 + phi_err), n_sim
        )
        c_samples = np.random.triangular(
            C * (1 - c_err), C, C * (1 + c_err), n_sim
        )

        if "非油气" in res["储层类型"]:
            q_samples = (phi_samples * V_m3) + (S * elastic_head * A_m2)
        else:
            if has_seismic_vol:
                q_samples = (V_m3 * phi_samples * Sw) / Bw
            else:
                q_samples = (A_m2 * h * phi_samples * Sw) / Bw

        p_samples = (q_samples * c_samples * alpha) / 1e4  # 万吨
        mc_results = {
            "P90(万吨)": round(float(np.percentile(p_samples, 10)), 2),
            "P50(万吨)": round(float(np.percentile(p_samples, 50)), 2),
            "P10(万吨)": round(float(np.percentile(p_samples, 90)), 2),
            "samples": p_samples,
        }
        res["P90保守储量(万吨)"] = mc_results["P90(万吨)"]
        res["P50中值储量(万吨)"] = mc_results["P50(万吨)"]
        res["P10乐观储量(万吨)"] = mc_results["P10(万吨)"]

    return res, mc_results


# --------------------------------------------------
# 独立专题登录与会话隔离
# --------------------------------------------------
if st.session_state.current_user_topic is None:
    st.title("🔒 四川盆地深层卤水钾盐资源评价系统 - 专题独立通道")
    st.markdown("各专题（川中、川西、川东北、川南）数据完全隔离，请选择所属专题并输入授权密钥。")

    col_l1, col_l2, _ = st.columns([1, 1, 1])
    with col_l1:
        topic_input = st.selectbox(
            "选择您所属的专题",
            ["川中专题", "川西专题", "川东北专题", "川南专题"],
        )
    with col_l2:
        key_input = st.text_input("输入专题授权密钥", type="password")

    if st.button("进入专题工作台", type="primary"):
        expected_key = AUTH_CREDENTIALS[topic_input]["key"]
        if key_input == expected_key:
            st.session_state.current_user_topic = topic_input
            st.success(f"已成功进入【{topic_input}】独立工作空间！")
            st.rerun()
        else:
            st.error("密钥错误，请联系项目管理组获取授权密钥。")
    st.stop()

# --------------------------------------------------
# 当前专题上下文环境
# --------------------------------------------------
active_topic = st.session_state.current_user_topic
my_data = st.session_state.isolated_storage[active_topic]
my_target = AUTH_CREDENTIALS[active_topic]["target"]

with st.sidebar:
    st.title(f"🧂 {active_topic}")
    st.caption("独立评估总台账（数据物理隔离保护）")

    my_total_kcl = sum([item["P_final_raw"] for item in my_data])
    my_progress = min(1.0, my_total_kcl / my_target) if my_target > 0 else 0.0

    st.metric(
        label=f"{active_topic} 核定 KCl 储量",
        value=f"{my_total_kcl / 1e4:,.2f} 万吨",
        delta=(
            f"距离目标还差: {(my_target - my_total_kcl)/1e4:,.2f} 万吨"
            if my_total_kcl < my_target
            else "🎉 本专题增储目标达成！"
        ),
    )
    st.progress(my_progress)
    st.caption(
        f"专题目标配额：{my_target/1e4:.0f} 万吨 | 完成度：{my_progress * 100:.2f}%"
    )

    st.markdown("---")
    st.info(
        f"🔐 **隔离机制提示**：\n"
        f"您当前仅能查阅与导出【{active_topic}】的数据，系统不会展示其他专题的任何进度或结果。"
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("退出当前专题", use_container_width=True):
            st.session_state.current_user_topic = None
            st.rerun()
    with col_btn2:
        if st.button("清空本专题数据", use_container_width=True):
            st.session_state.isolated_storage[active_topic] = []
            st.rerun()

# --------------------------------------------------
# 主界面 Tabs
# --------------------------------------------------
tab_upload, tab_single, tab_summary, tab_manual = st.tabs([
    "📁 批量上传任意数据自适应计算",
    "✍️ 单构造带灵活输入测算",
    f"📊 【{active_topic}】独立成果总账",
    "📖 双套动静结合方法手册与规范",
])

# --------------------------------------------------
# TAB 1: 批量上传计算
# --------------------------------------------------
with tab_upload:
    st.subheader(f"批量上传【{active_topic}】多源数据表 (CSV / Excel)")
    st.write("系统支持两套动静结合方法自动切换与字段自适应识别。")

    batch_dyn_method = st.radio(
        "选择默认动静结合约束方法（若上传表格中未指定“动态方法”列，则按此执行）：",
        [
            "方法一：弹性压释物质平衡法（需累计排液量 Wp 与折算压降 dp）",
            "方法二：渗流压力波及漏斗法（需日产水量 qw、生产压差 dp_flow 或 波及半径 Re）",
        ],
        horizontal=False,
    )

    sample_df = pd.DataFrame([
        {
            "构造带": f"{active_topic[:2]}构造1号",
            "储层类型": "油气同层型",
            "动态方法": "方法一：弹性压释物质平衡法",
            "A": 45, "h": 25, "phi": 0.08, "Sw": 0.65, "Bw": 1.02, "C": 0.018,
            "Wp": 35000, "dp": 3.2, "ct": 8.5,
        },
        {
            "构造带": f"{active_topic[:2]}构造2号",
            "储层类型": "非油气同层型",
            "动态方法": "方法二：渗流压力波及漏斗法",
            "A": 60, "V": 1500, "phi": 0.065, "S": 0.0004, "承压水头标高": 300, "储层顶面标高": -2500, "C": 0.021,
            "qw": 200, "dp_flow": 4.5, "k": 3.5, "mu": 1.1,
        },
        {
            "构造带": f"{active_topic[:2]}未试水圈闭3号",
            "储层类型": "油气同层型",
            "A": 30, "h": 18, "phi": 0.07, "C": 0.015,
        },
    ])
    csv_buffer = io.BytesIO()
    sample_df.to_csv(csv_buffer, index=False, encoding="utf_8_sig")
    st.download_button(
        label="📥 下载含两套动静结合字段的综合测试模板 (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"{active_topic}_深层卤水双方法储量模板.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "选择要上传并核算的数据文件", type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)

            st.write("📋 **待测算数据预览：**")
            st.dataframe(df_input.head(5), use_container_width=True)

            if st.button("⚡ 一键自适应智能核算并入库", type="primary"):
                batch_results = []
                method_tag = "方法一" if "方法一" in batch_dyn_method else "方法二"
                for idx, row in df_input.iterrows():
                    calc_res, _ = evaluate_brine_record(
                        row.to_dict(),
                        assigned_topic=active_topic,
                        dyn_method_choice=method_tag,
                        run_mc=True,
                        n_sim=2000,
                    )
                    batch_results.append(calc_res)
                    st.session_state.isolated_storage[active_topic].append(calc_res)

                st.success(f"核算完成！已入库 {len(batch_results)} 个构造带至【{active_topic}】总账！")
                df_batch_show = pd.DataFrame(batch_results).drop(columns=["P_final_raw"])
                st.dataframe(df_batch_show, use_container_width=True)
        except Exception as e:
            st.error(f"核算失败: {str(e)}")

# --------------------------------------------------
# TAB 2: 单构造带手动输入测算
# --------------------------------------------------
with tab_single:
    st.subheader(f"单构造带交互式核算 —— 【{active_topic}】")

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        s_zone = st.text_input("构造带/圈闭名称", value=f"{active_topic[:2]}某深层富钾构造")
    with c_m2:
        s_type = st.radio("储卤层类型", ["油气同层型", "非油气同层型"], horizontal=True)
    with c_m3:
        dyn_selection = st.selectbox(
            "选择动-静结合综合评价方法",
            [
                "方法一：弹性压释物质平衡法 (Wp - Δp)",
                "方法二：渗流压力波及漏斗法 (qw - Δp_wf - k)",
            ],
        )

    st.markdown("##### 1. 静态容积法参数")
    cg1, cg2, cg3, cg4 = st.columns(4)
    with cg1:
        s_C = st.number_input("KCl平均品位 C (t/m³)", value=0.0190, step=0.0010, format="%.4f")
    with cg2:
        s_A = st.number_input("储层平面面积 A (km²)", value=50.0, step=1.0)
    with cg3:
        s_phi = st.number_input("有效孔隙度 ϕ (小数)", value=0.075, step=0.005, format="%.3f")
    with cg4:
        use_seismic = st.checkbox("已有三维地震精细雕刻体积 V", value=False)
        if use_seismic:
            s_V = st.number_input("雕刻体积 V (10⁶ m³)", value=1200.0, step=50.0)
            s_h = 0.0
        else:
            s_h = st.number_input("储层有效厚度 h (m)", value=24.0, step=1.0)
            s_V = None

    if s_type == "油气同层型":
        st.markdown("##### 2. 油气同层专属参数")
        co1, co2 = st.columns(2)
        with co1:
            s_sw = st.number_input("含水饱和度 Sw (小数)", value=0.60, step=0.05)
        with co2:
            s_bw = st.number_input("卤水体积系数 Bw (无因次)", value=1.02, step=0.01)
        s_S, s_head, s_top = None, None, None
    else:
        st.markdown("##### 2. 非油气同层专属参数")
        cn1, cn2, cn3 = st.columns(3)
        with cn1:
            s_S = st.number_input("弹性储水系数 S", value=0.0005, step=0.0001, format="%.5f")
        with cn2:
            s_head = st.number_input("平均承压水头标高 h (m)", value=320.0, step=10.0)
        with cn3:
            s_top = st.number_input("平均储层顶面标高 H (m)", value=-2100.0, step=50.0)
        s_sw, s_bw = None, None

    st.markdown("##### 3. 动态约束参数（可选，留空则执行纯静态计算）")
    enable_dyn = st.checkbox("启用当前选定的动-静结合方法进行校准", value=True)

    s_wp, s_dp, s_ct = None, None, None
    s_qw, s_dp_flow, s_k, s_mu, s_re = None, None, None, None, None

    if enable_dyn:
        if "方法一" in dyn_selection:
            st.info("💡 **方法一参数输入**：基于试采累计排卤量与地层综合压降计算弹性水体释水体积。")
            cd1, cd2, cd3 = st.columns(3)
            with cd1:
                s_wp = st.number_input("试采阶段累计排卤量 Wp (m³)", value=28000.0, step=1000.0)
            with cd2:
                s_dp = st.number_input("折算地层静态压降 Δp (MPa)", value=2.8, step=0.1)
            with cd3:
                s_ct = st.number_input("综合压缩系数 Ct (10⁻⁴ MPa⁻¹)", value=8.0, step=0.5)
        else:
            st.info("💡 **方法二参数输入**：基于拟稳态渗流方程或直接输入压力恢复波及半径约束有效泄流面积。")
            cm_mode = st.radio("方法二约束模式", ["通过渗流参数反算波及漏斗", "直接指定实测波及半径 Re"], horizontal=True)
            if cm_mode == "通过渗流参数反算波及漏斗":
                cd4, cd5, cd6, cd7 = st.columns(4)
                with cd4:
                    s_qw = st.number_input("稳定日产水量 qw (m³/d)", value=180.0, step=10.0)
                with cd5:
                    s_dp_flow = st.number_input("生产流动压差 Δp_wf (MPa)", value=4.5, step=0.1)
                with cd6:
                    s_k = st.number_input("储层渗透率 k (mD)", value=2.5, step=0.5)
                with cd7:
                    s_mu = st.number_input("卤水动力黏度 μ (mPa·s)", value=1.15, step=0.05)
            else:
                s_re = st.number_input("实测有效压力波及半径 Re (m)", value=1200.0, step=50.0)

    if st.button("🚀 计算本构造带并生成概率分布", type="primary"):
        input_dict = {
            "构造带": s_zone,
            "储层类型": s_type,
            "动态方法": "方法一" if "方法一" in dyn_selection else "方法二",
            "C": s_C, "A": s_A, "phi": s_phi, "h": s_h,
            "Sw": s_sw, "Bw": s_bw, "S": s_S, "承压水头标高": s_head, "储层顶面标高": s_top,
            "Wp": s_wp, "dp": s_dp, "ct": s_ct,
            "qw": s_qw, "dp_flow": s_dp_flow, "k": s_k, "mu": s_mu, "Re": s_re,
        }
        if s_V is not None:
            input_dict["V"] = s_V

        res_calc, mc_res = evaluate_brine_record(
            input_dict,
            assigned_topic=active_topic,
            dyn_method_choice=input_dict["动态方法"],
            run_mc=True,
            n_sim=5000,
        )

        r_c1, r_c2, r_c3, r_c4 = st.columns(4)
        r_c1.metric("建模模式", res_calc["几何建模方式"])
        r_c2.metric("动态约束结果", res_calc["动态约束状态"])
        r_c3.metric("核定卤水量", f"{res_calc['最终核定卤水体积(亿m³)']:.4f} 亿m³")
        r_c4.metric("核定 KCl 储量", f"{res_calc['最终核定KCl储量(万吨)']:,.2f} 万吨")

        if mc_res:
            fig_hist = px.histogram(
                x=mc_res["samples"],
                nbins=50,
                labels={"x": "KCl 储量 (万吨)"},
                title=f"{s_zone} - 蒙特卡洛不确定性模拟分布 (P10 - P50 - P90)",
            )
            fig_hist.add_vline(x=mc_res["P90(万吨)"], line_dash="dash", line_color="orange", annotation_text=f"P90: {mc_res['P90(万吨)']}万吨")
            fig_hist.add_vline(x=mc_res["P50(万吨)"], line_dash="solid", line_color="green", annotation_text=f"P50: {mc_res['P50(万吨)']}万吨")
            fig_hist.add_vline(x=mc_res["P10(万吨)"], line_dash="dash", line_color="red", annotation_text=f"P10: {mc_res['P10(万吨)']}万吨")
            st.plotly_chart(fig_hist, use_container_width=True)

        if st.button("💾 确认录入该结果到本专题总账"):
            st.session_state.isolated_storage[active_topic].append(res_calc)
            st.success(f"已成功录入 {s_zone}，请前往独立总账查看！")

# --------------------------------------------------
# TAB 3: 本专题专属总账看板 (完全隔离)
# --------------------------------------------------
with tab_summary:
    st.subheader(f"📋 【{active_topic}】独立成果总账")

    if not my_data:
        st.info(f"当前【{active_topic}】暂无已保存的计算成果。请先通过数据上传或单构造带输入进行添加。")
    else:
        df_my = pd.DataFrame(my_data)
        display_df = (
            df_my.drop(columns=["P_final_raw"])
            if "P_final_raw" in df_my.columns
            else df_my
        )
        st.dataframe(display_df, use_container_width=True)

        fig_bar = px.bar(
            df_my,
            x="构造带",
            y="最终核定KCl储量(万吨)",
            text="最终核定KCl储量(万吨)",
            title=f"【{active_topic}】各构造带 KCl 储存量核定分布",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        csv_out = display_df.to_csv(index=False).encode("utf_8_sig")
        st.download_button(
            label=f"📥 导出【{active_topic}】综合评价成果总报表 (CSV)",
            data=csv_out,
            file_name=f"{active_topic}_深层卤水钾盐储量成果总账.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# TAB 4: 双套动静结合方法手册与规范
# --------------------------------------------------
with tab_manual:
    st.subheader("深层富钾卤水储量评价双套“动-静结合”方法理论指南")

    st.markdown("### 一、 静态容积法计算模型")
    st.write("**1. 油气同层型卤水：**")
    st.latex(r"Q_w = \frac{A_w \cdot h \cdot \phi \cdot S_w}{B_w}, \quad P_A = Q_w \cdot C")
    st.caption("注：若地震精细雕刻提供了三维储集体积 V，则 Q = (V · ϕ · Sw) / Bw。")

    st.write("**2. 非油气同层型卤水：**")
    st.latex(r"Q_{ws} = \phi \cdot V + S \cdot (h - H) \cdot A, \quad P_A = Q_{ws} \cdot C")
    st.caption("注：若缺乏精细雕刻体 V，系统自动采用几何体积 V = A · h 替代。")

    st.markdown("---")
    st.markdown("### 二、 两套动-静结合综合评价方法原理")

    st.write("#### 方法一：地层弹性压释 / 物质平衡法")
    st.write("适用于封闭弹性储卤构造。依据实测生产压降与累计采出液量反求动态控水体积：")
    st.latex(r"Q_{dyn} = \frac{W_p}{c_t \cdot \Delta p}")
    st.write("定义动-静连通有效性系数 α 并约束静态储量：")
    st.latex(r"\alpha = \min\left(1.0, \; \frac{Q_{dyn}}{Q_{stat}}\right), \quad P_{final} = P_{stat} \cdot \alpha")

    st.write("#### 方法二：渗流压力波及漏斗 / 拟稳态有效半径法")
    st.write("适用于具备单井试井产能测试或不稳定压力恢复资料的储层。基于平面径向拟稳态渗流方程推导有效泄流波及半径：")
    st.latex(r"R_e = \sqrt{\frac{q_w \cdot \mu_w}{0.00708 \cdot k \cdot h \cdot \Delta p_{wf}}}")
    st.write("进而反算波及有效动态控制体积，对静态大面积外推圈闭进行合理收敛折减：")
    st.latex(r"A_{dyn} = \pi R_e^2, \quad \alpha = \min\left(1.0, \; \frac{A_{dyn}}{A_{stat}}\right)")

    st.markdown("---")
    st.markdown("### 三、 支持上传的数据字段全景表")
    fields_data = [
        {"参数类别": "通用基础", "标准物理量": "构造带名称", "推荐列名": "构造带", "备用列名": "构造名称, 圈闭"},
        {"参数类别": "通用基础", "标准物理量": "储层类型", "推荐列名": "储层类型", "备用列名": "类型 (油气同层型/非油气同层型)"},
        {"参数类别": "通用基础", "标准物理量": "动态方法指定", "推荐列名": "动态方法", "备用列名": "方法一 / 方法二"},
        {"参数类别": "静态物性", "标准物理量": "KCl 品位", "推荐列名": "C", "备用列名": "KCl品位, 品位 (t/m³)"},
        {"参数类别": "静态物性", "标准物理量": "含水面积", "推荐列名": "A", "备用列名": "Aw, 面积 (km²)"},
        {"参数类别": "静态物性", "标准物理量": "有效厚度", "推荐列名": "h", "备用列名": "厚度 (m)"},
        {"参数类别": "静态物性", "标准物理量": "三维精细体积", "推荐列名": "V", "备用列名": "雕刻体积 (10⁶ m³)"},
        {"参数类别": "静态物性", "标准物理量": "孔隙度", "推荐列名": "phi", "备用列名": "ϕ, 孔隙度 (小数或%)"},
        {"参数类别": "静态物性", "标准物理量": "含水饱和度", "推荐列名": "Sw", "备用列名": "含水饱和度 (小数)"},
        {"参数类别": "方法一参数", "标准物理量": "累计排卤量", "推荐列名": "Wp", "备用列名": "排卤量, 累计产水量 (m³)"},
        {"参数类别": "方法一参数", "标准物理量": "静态折算压降", "推荐列名": "dp", "备用列名": "压降, Δp (MPa)"},
        {"参数类别": "方法一参数", "标准物理量": "综合压缩系数", "推荐列名": "ct", "备用列名": "Ct, 压缩系数 (10⁻⁴ MPa⁻¹)"},
        {"参数类别": "方法二参数", "标准物理量": "稳定日产水量", "推荐列名": "qw", "备用列名": "日产水量 (m³/d)"},
        {"参数类别": "方法二参数", "标准物理量": "生产流动压差", "推荐列名": "dp_flow", "备用列名": "流压差, 生产压差 (MPa)"},
        {"参数类别": "方法二参数", "标准物理量": "储层渗透率", "推荐列名": "k", "备用列名": "渗透率 (mD)"},
        {"参数类别": "方法二参数", "标准物理量": "有效波及半径", "推荐列名": "Re", "备用列名": "波及半径 (m)"},
    ]
    st.table(pd.DataFrame(fields_data))
