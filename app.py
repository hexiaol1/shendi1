import datetime
import io
import uuid
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
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

# 流程各步骤在 Session 中的临时流水线数据
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = {}

# --------------------------------------------------
# 侧边栏：专题免密切换与独立进度监控
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

    my_total_kcl = sum([float(r.get("最终核定KCl储量(万吨)", 0.0)) for r in my_records])
    my_progress = min(1.0, (my_total_kcl * 1e4) / my_target) if my_target > 0 else 0.0

    st.markdown("---")
    st.subheader(f"🎯 【{active_topic}】增储进度")
    st.metric(
        label="本专题已核定 KCl 储量",
        value=f"{my_total_kcl:,.2f} 万吨",
        delta=(
            f"距目标差: {(my_target/1e4 - my_total_kcl):,.2f} 万吨"
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
        st.session_state.pipeline_data = {}
        st.rerun()


# --------------------------------------------------
# 辅助函数：样本解析与全参数蒙特卡洛抽样生成器
# --------------------------------------------------
def parse_sample_values(raw_input):
    if isinstance(raw_input, (list, np.ndarray, pd.Series)):
        arr = np.array(raw_input, dtype=float)
        return arr[~np.isnan(arr)]
    if pd.isna(raw_input):
        return np.array([], dtype=float)
    s = str(raw_input).replace("，", ",").replace(";", ",").replace("\n", ",")
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    vals = []
    for t in tokens:
        try:
            v = float(t)
            vals.append(v)
        except ValueError:
            continue
    return np.array(vals, dtype=float)


def generate_variable_distribution(
    mc_mode="单点/少井先验扰动模式",
    samples_arr=None,
    base_val=1.0,
    err_ratio=0.15,
    dist_type="自动优选",
    n_sim=10000,
    clip_range=(0.0, None)
):
    if samples_arr is None:
        samples_arr = np.array([])
    valid_samples = samples_arr[samples_arr > 0]
    fit_desc = ""

    if "多井" in mc_mode and len(valid_samples) >= 3:
        mean_val = float(np.mean(valid_samples))
        std_val = float(np.std(valid_samples, ddof=1))
        skewness = stats.skew(valid_samples)
        chosen = "lognorm" if (dist_type == "对数正态分布 (Lognormal)" or (dist_type == "自动优选" and skewness > 0.3)) else "norm"

        if chosen == "lognorm":
            shape, loc, scale = stats.lognorm.fit(valid_samples, floc=0)
            draws = stats.lognorm.rvs(shape, loc=loc, scale=scale, size=n_sim)
            fit_desc = f"对数正态拟合(μ={mean_val:.4f}, σ={std_val:.4f}, n={len(valid_samples)})"
        else:
            loc, scale = stats.norm.fit(valid_samples)
            draws = stats.norm.rvs(loc=loc, scale=scale, size=n_sim)
            fit_desc = f"正态拟合(μ={mean_val:.4f}, σ={std_val:.4f}, n={len(valid_samples)})"
    else:
        mean_val = valid_samples[0] if len(valid_samples) > 0 else base_val
        low = max(clip_range[0], mean_val * (1.0 - err_ratio))
        high = mean_val * (1.0 + err_ratio)
        draws = np.random.triangular(low, mean_val, high, n_sim)
        fit_desc = f"先验扰动(基准={mean_val:.4f}, ±{err_ratio*100:.0f}%)"

    low_clip = clip_range[0]
    high_clip = clip_range[1]
    draws = np.clip(draws, low_clip, high_clip) if high_clip is not None else np.clip(draws, low_clip, None)
    return draws, mean_val, fit_desc, valid_samples


# --------------------------------------------------
# 主界面 Tabs（流程化分步导航）
# --------------------------------------------------
tab_step1, tab_step2, tab_step3, tab_step4, tab_step5 = st.tabs([
    "1️⃣ 静态参数与台账录入",
    "2️⃣ 全参数蒙特卡洛模拟",
    "3️⃣ 多物理场动态校准",
    "4️⃣ 核定成果与总账归档",
    "📖 方法体系指南",
])

# --------------------------------------------------
# 步骤 1: 静态参数与台账录入
# --------------------------------------------------
with tab_step1:
    st.subheader(f"第一步：构造单元基本信息与静态物性录入 —— 【{active_topic}】")
    st.caption("支持单单元交互录入或下载标准多井/多构造台账模板进行上传，数据将自动流转至下一步。")

    # 提供全新规范的多井综合台账模板
    col_dl, col_blank = st.columns([2, 2])
    with col_dl:
        sample_well_template = pd.DataFrame([
            {
                "构造带": f"{active_topic[:2]}构造1号", "井号": "Well-1", "储层类型": "油气同层型",
                "含水面积(km2)": 55.0, "有效厚度(m)": 24.5, "有效孔隙度(%)": 7.8,
                "含水饱和度(小数)": 0.65, "卤水体积系数": 1.02, "KCl品位(t/m3)": 0.0195,
                "累计排卤量Wp(m3)": 35000, "地层压降dp(MPa)": 3.2, "日产水量qw(m3/d)": 180, "流压差dp_flow(MPa)": 4.5
            },
            {
                "构造带": f"{active_topic[:2]}构造1号", "井号": "Well-2", "储层类型": "油气同层型",
                "含水面积(km2)": 55.0, "有效厚度(m)": 26.0, "有效孔隙度(%)": 8.5,
                "含水饱和度(小数)": 0.62, "卤水体积系数": 1.02, "KCl品位(t/m3)": 0.0205,
                "累计排卤量Wp(m3)": 40000, "地层压降dp(MPa)": 3.0, "日产水量qw(m3/d)": 210, "流压差dp_flow(MPa)": 4.2
            },
            {
                "构造带": f"{active_topic[:2]}深层承压区", "井号": "Well-3", "储层类型": "非油气同层型",
                "含水面积(km2)": 70.0, "有效厚度(m)": 30.0, "有效孔隙度(%)": 6.9,
                "弹性储水系数": 0.00045, "承压水头标高(m)": 350.0, "储层顶面标高(m)": -2200.0, "KCl品位(t/m3)": 0.0215,
                "累计排卤量Wp(m3)": 28000, "地层压降dp(MPa)": 2.8, "日产水量qw(m3/d)": 160, "流压差dp_flow(MPa)": 4.0
            }
        ])
        buf_t = io.BytesIO()
        sample_well_template.to_csv(buf_t, index=False, encoding="utf_8_sig")
        st.download_button(
            label="📥 下载更新版多井实测与储量综合台账模板 (CSV)",
            data=buf_t.getvalue(),
            file_name=f"{active_topic}_更新版深层富钾卤水综合台账模板.csv",
            mime="text/csv",
        )

    st.markdown("---")
    input_channel = st.radio("请选择当前数据录入通道：", ["方式 A：在线交互录入（推荐）", "方式 B：上传综合台账文件 (CSV/Excel)"], horizontal=True)

    if "方式 A" in input_channel:
        c1, c2 = st.columns(2)
        with c1:
            u_name = st.text_input("构造带/计算单元名称", value=f"{active_topic[:2]}某深部富钾构造")
        with c2:
            u_type = st.radio("储层类型", ["油气同层型", "非油气同层型"], horizontal=True)

        st.markdown("##### 储层几何与物性参数")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            u_A = st.number_input("含水面积 A (km²)", value=55.0, step=1.0)
        with p2:
            use_v = st.checkbox("具备三维地震雕刻体积 V", value=False)
            if use_v:
                u_V = st.number_input("雕刻体积 V (10⁶ m³)", value=1300.0, step=50.0)
                u_h = 0.0
            else:
                u_h = st.number_input("储层有效厚度 h (m)", value=25.0, step=1.0)
                u_V = None
        with p3:
            u_phi = st.number_input("基准有效孔隙度 ϕ (小数)", value=0.078, step=0.005, format="%.3f")
        with p4:
            u_C = st.number_input("基准 KCl 平均品位 C (t/m³)", value=0.0195, step=0.0010, format="%.4f")

        if u_type == "油气同层型":
            o1, o2 = st.columns(2)
            with o1:
                u_sw = st.number_input("含水饱和度 Sw (小数)", value=0.62, step=0.05)
            with o2:
                u_bw = st.number_input("卤水体积系数 Bw", value=1.02, step=0.01)
            u_S, u_head, u_top = 0.0, 0.0, 0.0
        else:
            n1, n2, n3 = st.columns(3)
            with n1:
                u_S = st.number_input("弹性储水系数 S", value=0.0005, step=0.0001, format="%.5f")
            with n2:
                u_head = st.number_input("平均承压水头标高 h (m)", value=340.0, step=10.0)
            with n3:
                u_top = st.number_input("平均储层顶面标高 H (m)", value=-2250.0, step=50.0)
            u_sw, u_bw = 0.0, 1.0

        st.caption("💡 若有该构造带多口井的孔隙度或品位实测序列，可在此粘贴（选填，用于后续拟合）：")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            u_phi_str = st.text_input("多井孔隙度序列（逗号分隔）", value="0.065, 0.082, 0.071, 0.095, 0.078, 0.088")
        with m_col2:
            u_c_str = st.text_input("多井品位序列（逗号分隔）", value="0.0175, 0.0192, 0.0210, 0.0185")

        if st.button("下一步：确认静态参数并进入蒙特卡洛设置 ➡️", type="primary"):
            st.session_state.pipeline_data = {
                "构造带": u_name,
                "储层类型": u_type,
                "A": u_A, "h": u_h, "V": u_V,
                "phi": u_phi, "C": u_C,
                "Sw": u_sw, "Bw": u_bw,
                "S": u_S, "承压水头标高": u_head, "储层顶面标高": u_top,
                "多井孔隙度序列": u_phi_str,
                "多井品位序列": u_c_str,
            }
            st.success(f"已暂存【{u_name}】静态参数！请点击上方切换至「2️⃣ 全参数蒙特卡洛模拟」。")

    else:
        file_up = st.file_uploader("上传规范台账文件", type=["csv", "xlsx", "xls"])
        if file_up is not None:
            try:
                if file_up.name.endswith(".csv"):
                    df_up = pd.read_csv(file_up)
                else:
                    df_up = pd.read_excel(file_up)
                st.write("已解析台账前5行：")
                st.dataframe(df_up.head(5), use_container_width=True)

                if st.button("将表格首行作为当前流转单元"):
                    row0 = df_up.iloc[0].to_dict()
                    phi_val = float(row0.get("有效孔隙度(%)", row0.get("phi", 7.8)))
                    if phi_val > 1.0: phi_val /= 100.0
                    st.session_state.pipeline_data = {
                        "构造带": str(row0.get("构造带", "台账构造1号")),
                        "储层类型": str(row0.get("储层类型", "油气同层型")),
                        "A": float(row0.get("含水面积(km2)", row0.get("A", 50.0))),
                        "h": float(row0.get("有效厚度(m)", row0.get("h", 25.0))),
                        "V": None,
                        "phi": phi_val,
                        "C": float(row0.get("KCl品位(t/m3)", row0.get("C", 0.0195))),
                        "Sw": float(row0.get("含水饱和度(小数)", row0.get("Sw", 0.65))),
                        "Bw": float(row0.get("卤水体积系数", row0.get("Bw", 1.02))),
                        "S": float(row0.get("弹性储水系数", row0.get("S", 0.0005))),
                        "承压水头标高": float(row0.get("承压水头标高(m)", 350.0)),
                        "储层顶面标高": float(row0.get("储层顶面标高(m)", -2200.0)),
                        "Wp": float(row0.get("累计排卤量Wp(m3)", row0.get("Wp", 35000))),
                        "dp": float(row0.get("地层压降dp(MPa)", row0.get("dp", 3.0))),
                        "qw": float(row0.get("日产水量qw(m3/d)", row0.get("qw", 180))),
                        "dp_flow": float(row0.get("流压差dp_flow(MPa)", row0.get("dp_flow", 4.2))),
                        "多井孔隙度序列": ", ".join([str(v) for v in df_up["有效孔隙度(%)"].dropna().values]) if "有效孔隙度(%)" in df_up.columns else "",
                        "多井品位序列": ", ".join([str(v) for v in df_up["KCl品位(t/m3)"].dropna().values]) if "KCl品位(t/m3)" in df_up.columns else "",
                    }
                    st.success("已成功从表格中流转参数！请前往下一步。")
            except Exception as e:
                st.error(f"解析出错: {e}")

# --------------------------------------------------
# 步骤 2: 全参数蒙特卡洛模拟
# --------------------------------------------------
with tab_step2:
    st.subheader("第二步：全参数蒙特卡洛随机模拟配置")
    if not st.session_state.pipeline_data:
        st.warning("⚠️ 请先在「1️⃣ 静态参数与台账录入」中录入并确认基础参数！")
    else:
        p_data = st.session_state.pipeline_data
        st.info(f"当前流转单元：**【{p_data['构造带']}】** | 储层类型：**{p_data['储层类型']}**")

        m1, m2 = st.columns(2)
        with m1:
            mc_mode = st.radio("蒙特卡洛建模模式：", ["模式一：多井实测样本拟合模式 (井数≥3)", "模式二：单点/少井先验扰动模式 (探井少/单点代表值)"], index=0)
        with m2:
            n_sim = st.select_slider("抽样模拟次数 (N)", options=[1000, 2000, 5000, 10000, 20000], value=10000)

        dist_type = "自动优选"
        if "模式一" in mc_mode:
            dist_type = st.selectbox("多井概率拟合分布：", ["自动优选 (按偏度判别)", "对数正态分布 (Lognormal)", "正态分布 (Normal)"])

        st.markdown("##### 🔧 各参数先验不确定性变异范围配置 (±%)")
        u1, u2, u3, u4, u5, u6 = st.columns(6)
        with u1: err_A = st.slider("面积 A (±%)", 5, 40, 15, 5) / 100.0
        with u2: err_h = st.slider("厚度 h (±%)", 5, 40, 15, 5) / 100.0
        with u3: err_phi = st.slider("孔隙度 ϕ (±%)", 5, 50, 20, 5) / 100.0
        with u4: err_C = st.slider("品位 C (±%)", 5, 40, 15, 5) / 100.0
        with u5: err_Sw = st.slider("饱和度 Sw (±%)", 5, 30, 10, 5) / 100.0
        with u6: err_S = st.slider("储水系数 S (±%)", 5, 40, 15, 5) / 100.0

        if st.button("运行蒙特卡洛抽样并查看分布 🎲", type="primary"):
            phi_arr = parse_sample_values(p_data.get("多井孔隙度序列", ""))
            c_arr = parse_sample_values(p_data.get("多井品位序列", ""))

            A_m2 = p_data["A"] * 1e6
            h_val = p_data["h"]

            phi_draws, phi_mean, phi_desc, phi_raw_valid = generate_variable_distribution(
                mc_mode, phi_arr, p_data["phi"], err_phi, dist_type, n_sim, (0.001, 0.45)
            )
            c_draws, c_mean, c_desc, c_raw_valid = generate_variable_distribution(
                mc_mode, c_arr, p_data["C"], err_C, dist_type, n_sim, (0.0001, 0.20)
            )
            A_draws, A_mean, _, _ = generate_variable_distribution(
                "单点/少井先验扰动模式", np.array([]), A_m2, err_A, "三角分布", n_sim, (1e5, None)
            )
            h_draws, h_mean, _, _ = generate_variable_distribution(
                "单点/少井先验扰动模式", np.array([]), h_val, err_h, "三角分布", n_sim, (1.0, 500.0)
            )
            Sw_draws, Sw_mean, _, _ = generate_variable_distribution(
                "单点/少井先验扰动模式", np.array([]), p_data["Sw"], err_Sw, "三角分布", n_sim, (0.05, 1.0)
            )
            S_draws, S_mean, _, _ = generate_variable_distribution(
                "单点/少井先验扰动模式", np.array([]), p_data["S"], err_S, "三角分布", n_sim, (1e-6, 0.1)
            )

            # 暂存抽样数据至 session
            st.session_state.pipeline_data.update({
                "mc_mode": mc_mode, "n_sim": n_sim,
                "phi_draws": phi_draws, "phi_mean": phi_mean, "phi_desc": phi_desc, "phi_raw_valid": phi_raw_valid,
                "c_draws": c_draws, "c_mean": c_mean, "c_desc": c_desc, "c_raw_valid": c_raw_valid,
                "A_draws": A_draws, "A_mean": A_mean,
                "h_draws": h_draws, "h_mean": h_mean,
                "Sw_draws": Sw_draws, "Sw_mean": Sw_mean,
                "S_draws": S_draws, "S_mean": S_mean,
            })
            st.success("蒙特卡洛抽样已完成！请查看下方拟合分布，随后进入「3️⃣ 多物理场动态校准」。")

        if "phi_draws" in st.session_state.pipeline_data:
            draw_data = st.session_state.pipeline_data
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                fig_phi = px.histogram(
                    x=draw_data["phi_draws"] * 100, nbins=50, histnorm='probability density',
                    labels={"x": "孔隙度 (%)"}, title=f"孔隙度概率分布 ({draw_data['phi_desc']})"
                )
                if len(draw_data["phi_raw_valid"]) > 0:
                    for v in draw_data["phi_raw_valid"]:
                        fig_phi.add_vline(x=v * 100, line_dash="dot", line_color="orange")
                st.plotly_chart(fig_phi, use_container_width=True)
            with f_col2:
                fig_c = px.histogram(
                    x=draw_data["c_draws"], nbins=50, histnorm='probability density',
                    labels={"x": "品位 (t/m³)"}, title=f"品位概率分布 ({draw_data['c_desc']})"
                )
                if len(draw_data["c_raw_valid"]) > 0:
                    for v in draw_data["c_raw_valid"]:
                        fig_c.add_vline(x=v, line_dash="dot", line_color="orange")
                st.plotly_chart(fig_c, use_container_width=True)

# --------------------------------------------------
# 步骤 3: 多物理场动态校准
# --------------------------------------------------
with tab_step3:
    st.subheader("第三步：多物理场动态试采数据与参数校准")
    if "phi_draws" not in st.session_state.pipeline_data:
        st.warning("⚠️ 请先在「2️⃣ 全参数蒙特卡洛模拟」中完成变量抽样！")
    else:
        p_data = st.session_state.pipeline_data
        st.markdown(
            "动态测试数据（排卤量、压降、日产量、流压差等）将作为反演依据，"
            "**定向校准选定的静态地质参数至实际流动合理区间**，然后再代入容积法计算。"
        )

        calib_choice = st.selectbox(
            "请指定本次动态试采数据校准的参数对象：",
            [
                "校准有效孔隙度 ϕ (基于弹性物质平衡 Wp - Δp)",
                "校准有效储卤厚度 h (基于产水剖面渗流压差 qw - Δp_wf)",
                "校准动态连通面积 A (基于试井压力波及漏斗 Re)",
                "校准有效含水饱和度 Sw (基于两相渗流产能比)",
                "校准弹性储水系数 S (基于承压扬程释水方程)",
                "不进行动态校准 (纯静态蒙特卡洛输出)"
            ]
        )

        st.markdown("##### 录入动态试水与试采测试数据")
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            in_wp = st.number_input("累计排卤量 Wp (m³)", value=float(p_data.get("Wp", 35000.0)), step=1000.0)
        with d2:
            in_dp = st.number_input("地层静压降 Δp (MPa)", value=float(p_data.get("dp", 3.0)), step=0.1)
        with d3:
            in_qw = st.number_input("稳定日产水量 qw (m³/d)", value=float(p_data.get("qw", 180.0)), step=10.0)
        with d4:
            in_dp_flow = st.number_input("生产流压差 Δp_wf (MPa)", value=float(p_data.get("dp_flow", 4.2)), step=0.1)
        with d5:
            in_k = st.number_input("储层渗透率 k (mD)", value=2.8, step=0.5)

        if st.button("执行动态反演校准与最终储量核定 🚀", type="primary"):
            calib_factor = 1.0
            dyn_applied = "纯静态（未校准）"
            dyn_process = "未执行动态校准"

            V_base = (p_data["V"] * 1e6) if p_data.get("V") is not None else (p_data["A_mean"] * p_data["h_mean"])
            ct = 8.5e-4
            mu = 1.15

            if "校准有效孔隙度" in calib_choice and in_wp > 0 and in_dp > 0:
                phi_cal = in_wp / (ct * in_dp * V_base)
                calib_factor = np.clip(phi_cal / (p_data["phi_mean"] + 1e-6), 0.65, 1.35)
                dyn_applied = "弹性物质平衡校准有效孔隙度 ϕ"
                dyn_process = f"Wp={in_wp:.0f}m³, Δp={in_dp:.2f}MPa 反求孔隙度；校准系数={calib_factor:.3f}"
            elif "校准有效储卤厚度" in calib_choice and in_qw > 0 and in_dp_flow > 0:
                h_cal = (in_qw * mu * 3.5) / (0.00708 * in_k * in_dp_flow + 1e-6)
                calib_factor = np.clip(h_cal / (p_data["h_mean"] + 1e-6), 0.65, 1.35)
                dyn_applied = "渗流压差反演校准有效厚度 h"
                dyn_process = f"qw={in_qw:.1f}m³/d, Δp_wf={in_dp_flow:.2f}MPa 反求有效厚度；校准系数={calib_factor:.3f}"
            elif "校准动态连通面积" in calib_choice:
                Re = max(200.0, np.sqrt((in_qw * mu) / (0.00708 * in_k * max(1.0, p_data["h_mean"]) * max(0.1, in_dp_flow) + 1e-6)) * 100.0)
                A_dyn_m2 = np.pi * (Re**2)
                calib_factor = np.clip(A_dyn_m2 / (p_data["A_mean"] + 1e-6), 0.70, 1.25)
                dyn_applied = "试井压力波及漏斗校准连通面积 A"
                dyn_process = f"反求供液半径 Re={Re:.0f}m, 面积={A_dyn_m2/1e6:.2f}km²；校准系数={calib_factor:.3f}"
            elif "校准有效含水饱和度" in calib_choice:
                calib_factor = np.clip(1.0 - (in_dp_flow / (in_dp_flow + 5.0)) * 0.15, 0.80, 1.15)
                dyn_applied = "两相渗流产能校准可动水饱和度 Sw"
                dyn_process = f"剔除束缚水影响；校准系数={calib_factor:.3f}"
            elif "校准弹性储水系数" in calib_choice:
                S_cal = in_wp / (in_dp * 100.0 * p_data["A_mean"] + 1e-6)
                calib_factor = np.clip(S_cal / (p_data["S_mean"] + 1e-6), 0.65, 1.35)
                dyn_applied = "承压扬程释水校准储水系数 S"
                dyn_process = f"排卤量与扬程降反求储水系数；校准系数={calib_factor:.3f}"

            # 最终抽样与储量计算
            phi_f_draws = p_data["phi_draws"] * (calib_factor if "孔隙度" in calib_choice else 1.0)
            h_f_draws = p_data["h_draws"] * (calib_factor if "厚度" in calib_choice else 1.0)
            A_f_draws = p_data["A_draws"] * (calib_factor if "面积" in calib_choice else 1.0)
            Sw_f_draws = p_data["Sw_draws"] * (calib_factor if "含水饱和度" in calib_choice else 1.0)
            S_f_draws = p_data["S_draws"] * (calib_factor if "储水系数" in calib_choice else 1.0)

            phi_f_mean = p_data["phi_mean"] * (calib_factor if "孔隙度" in calib_choice else 1.0)
            h_f_mean = p_data["h_mean"] * (calib_factor if "厚度" in calib_choice else 1.0)
            A_f_mean = p_data["A_mean"] * (calib_factor if "面积" in calib_choice else 1.0)
            Sw_f_mean = p_data["Sw_mean"] * (calib_factor if "含水饱和度" in calib_choice else 1.0)
            S_f_mean = p_data["S_mean"] * (calib_factor if "储水系数" in calib_choice else 1.0)

            elastic_head = max(0.0, p_data["承压水头标高"] - p_data["储层顶面标高"])

            if "非油气" in p_data["储层类型"]:
                V_f_draws = V_base if p_data.get("V") is not None else (A_f_draws * h_f_draws)
                V_f_mean = V_base if p_data.get("V") is not None else (A_f_mean * h_f_mean)
                Q_draws = (phi_f_draws * V_f_draws) + (S_f_draws * elastic_head * A_f_draws)
                Q_mean = (phi_f_mean * V_f_mean) + (S_f_mean * elastic_head * A_f_mean)
                formula_str = f"Q = ϕ·V + S·(h-H)·A = {phi_f_mean:.4f}×{V_f_mean:.2e} + {S_f_mean:.5f}×{elastic_head:.1f}×{A_f_mean:.2e}"
            else:
                Bw_val = p_data.get("Bw", 1.02)
                if p_data.get("V") is not None:
                    Q_draws = (V_base * phi_f_draws * Sw_f_draws) / Bw_val
                    Q_mean = (V_base * phi_f_mean * Sw_f_mean) / Bw_val
                    formula_str = f"Q = (V·ϕ·Sw)/Bw = ({V_base:.2e}×{phi_f_mean:.4f}×{Sw_f_mean:.2f})/{Bw_val:.2f}"
                else:
                    Q_draws = (A_f_draws * h_f_draws * phi_f_draws * Sw_f_draws) / Bw_val
                    Q_mean = (A_f_mean * h_f_mean * phi_f_mean * Sw_f_mean) / Bw_val
                    formula_str = f"Q = (A·h·ϕ·Sw)/Bw = ({A_f_mean:.2e}×{h_f_mean:.2f}×{phi_f_mean:.4f}×{Sw_f_mean:.2f})/{Bw_val:.2f}"

            P_draws = (Q_draws * p_data["c_draws"]) / 1e4  # 万吨
            P_mean = Q_mean * p_data["c_mean"]

            # 纯静态基准
            P_static_base = ((V_base * p_data["phi_mean"] * p_data["Sw_mean"]) / p_data.get("Bw", 1.02)) * p_data["c_mean"] if "非油气" not in p_data["储层类型"] else ((p_data["phi_mean"] * V_base) + (p_data["S_mean"] * elastic_head * p_data["A_mean"])) * p_data["c_mean"]

            res_record = {
                "计算单元编号": f"UNIT-{uuid.uuid4().hex[:8].upper()}",
                "计算时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "所属专题": active_topic,
                "构造带/单元名称": p_data["构造带"],
                "储层类型": p_data["储层类型"],
                "蒙特卡洛模式": p_data["mc_mode"],
                "动态校准对象参数": calib_choice,
                "动态校准方法类型": dyn_applied,
                "动态校准推演明细": dyn_process,
                "动态校准系数": round(calib_factor, 4),
                "原始静态KCl储量(万吨)": round(P_static_base / 1e4, 2),
                "最终核定KCl储量(万吨)": round(P_mean / 1e4, 2),
                "校准前后储量变化率": f"{((P_mean - P_static_base)/(P_static_base+1e-6))*100:+.2f}%",
                "核定卤水体积(亿m³)": round(Q_mean / 1e8, 4),
                "蒙特卡洛P90(万吨)": round(float(np.percentile(P_draws, 10)), 2),
                "蒙特卡洛P50(万吨)": round(float(np.percentile(P_draws, 50)), 2),
                "蒙特卡洛P10(万吨)": round(float(np.percentile(P_draws, 90)), 2),
                "核定孔隙度": f"{phi_f_mean*100:.2f}%",
                "核定厚度(m)": round(h_f_mean, 2),
                "核定面积(km²)": round(A_f_mean / 1e6, 2),
                "平均品位(t/m³)": p_data["c_mean"],
                "校准后容积法推演": formula_str,
            }

            st.session_state.temp_single_record = res_record
            st.session_state.temp_mc_artifacts = {"P_draws": P_draws}
            st.success("核定成功！请前往「4️⃣ 核定成果与总账归档」查阅详情并写入总账！")

# --------------------------------------------------
# 步骤 4: 核定成果与总账归档
# --------------------------------------------------
with tab_step4:
    st.subheader(f"第四步：【{active_topic}】综合成果核定与全要素总账归档")

    if st.session_state.temp_single_record is not None:
        rec = st.session_state.temp_single_record
        arts = st.session_state.temp_mc_artifacts
        st.markdown(f"#### 构造单元即时核定看板：{rec['构造带/单元名称']}")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("核定孔隙度", rec["核定孔隙度"])
        r2.metric("核定储卤厚度", f"{rec['核定厚度(m)']} m")
        r3.metric("最终核定 KCl 储量", f"{rec['最终核定KCl储量(万吨)']:,.2f} 万吨", delta=rec["校准前后储量变化率"])
        r4.metric("核定卤水总量", f"{rec['核定卤水体积(亿m³)']:.4f} 亿m³")

        st.info(f"📌 **动态校准评价总结**：{rec['动态校准推演明细']}")

        # 概率直方图
        fig_hist = px.histogram(
            x=arts["P_draws"], nbins=60, labels={"x": "KCl 储量 (万吨)"},
            title=f"{rec['构造带/单元名称']} - 全参数蒙特卡洛最终储量概率分布 (P10 - P50 - P90)"
        )
        fig_hist.add_vline(x=rec["蒙特卡洛P90(万吨)"], line_dash="dash", line_color="orange", annotation_text=f"P90: {rec['蒙特卡洛P90(万吨)']}万吨")
        fig_hist.add_vline(x=rec["蒙特卡洛P50(万吨)"], line_dash="solid", line_color="green", annotation_text=f"P50: {rec['蒙特卡洛P50(万吨)']}万吨")
        fig_hist.add_vline(x=rec["蒙特卡洛P10(万吨)"], line_dash="dash", line_color="red", annotation_text=f"P10: {rec['蒙特卡洛P10(万吨)']}万吨")
        st.plotly_chart(fig_hist, use_container_width=True)

        if st.button("💾 将本单元核定成果正式写入专属总账", type="primary"):
            existing_ids = [item.get("计算单元编号") for item in st.session_state.topic_archives[active_topic]]
            if rec["计算单元编号"] not in existing_ids:
                st.session_state.topic_archives[active_topic].append(rec)
            st.success(f"已成功归档至【{active_topic}】总账！")
            st.session_state.temp_single_record = None
            st.session_state.temp_mc_artifacts = None
            st.rerun()

    st.markdown("---")
    st.markdown(f"#### 📋 【{active_topic}】历史归档详单与成果下载")

    if not my_records:
        st.info("当前专题总账尚无归档数据，完成上方计算后点击“写入专属总账”即可保存。")
    else:
        df_arc = pd.DataFrame(my_records)
        st.write(f"已累计归档 **{len(df_arc)}** 个单元的具体核定推演履历。")

        preferred_cols = [
            "计算单元编号", "计算时间", "构造带/单元名称", "储层类型",
            "动态校准对象参数", "动态校准方法类型", "原始静态KCl储量(万吨)", "最终核定KCl储量(万吨)",
            "校准前后储量变化率", "蒙特卡洛P90(万吨)", "蒙特卡洛P50(万吨)", "蒙特卡洛P10(万吨)",
            "核定孔隙度", "核定厚度(m)", "核定面积(km²)", "校准后容积法推演"
        ]
        valid_cols = [col for col in preferred_cols if col in df_arc.columns]
        st.dataframe(df_arc[valid_cols], use_container_width=True)

        fig_unit_bar = px.bar(
            df_arc, x="构造带/单元名称", y="最终核定KCl储量(万吨)", text="最终核定KCl储量(万吨)",
            color="动态校准对象参数" if "动态校准对象参数" in df_arc.columns else None,
            title=f"【{active_topic}】各构造带核定储量柱状图 (目标配额: {my_target/1e4:.0f}万吨)",
        )
        fig_unit_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_unit_bar, use_container_width=True)

        csv_archive_out = df_arc.to_csv(index=False).encode("utf_8_sig")
        st.download_button(
            label=f"📥 导出【{active_topic}】全要素归档总报表 (CSV)",
            data=csv_archive_out,
            file_name=f"{active_topic}_全要素动态校准储量归档详单.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# 步骤 5: 方法体系指南
# --------------------------------------------------
with tab_step5:
    st.subheader("深层富钾卤水储量评价方法体系与操作指南")

    st.markdown("### 1. 容积法双模型标准公式")
    st.write("**（1）油气同层型卤水：**")
    st.latex(r"Q_w = \frac{A_w \cdot h \cdot \phi \cdot S_w}{B_w}, \quad P_A = Q_w \cdot C")
    st.write("**（2）非油气同层型卤水：**")
    st.latex(r"Q_{ws} = \phi \cdot V + S \cdot (h - H) \cdot A, \quad P_A = Q_{ws} \cdot C")

    st.markdown("---")
    st.markdown("### 2. 多物理场动态校准矩阵")
    calib_guide_df = pd.DataFrame([
        {"动态测试数据": "累计排卤 Wp + 地层压降 Δp", "渗流/物理机理": "弹性水体物质平衡", "适宜校准参数": "有效连通孔隙度 ϕ / 弹性储水系数 S", "地质意义": "剔除非流动死孔隙，还原真实连通流动孔隙"},
        {"动态测试数据": "日产水量 qw + 生产流压差 Δp_wf", "渗流/物理机理": "单井拟稳态径向流动", "适宜校准参数": "有效产液厚度 h", "地质意义": "修正电性划分过宽的非均质出水剖面厚度"},
        {"动态测试数据": "压力恢复探测边界 Re", "渗流/物理机理": "压力波扩散与边界效应", "适宜校准参数": "有效动水连通面积 A / 体积 V", "地质意义": "以实际流体连通域约束静态构造圈闭面积"},
        {"动态测试数据": "两相产流比变化", "渗流/物理机理": "相渗透率分异", "适宜校准参数": "可动含水饱和度 Sw", "地质意义": "剥离不可流动束缚水，精准核算产卤相饱和度"},
    ])
    st.table(calib_guide_df)

    st.markdown("---")
    st.markdown("### 3. 四大专题增储任务配额")
    targets_table = [
        {"专题名称": "川中专题", "新增KCl目标配额": "2000 万吨", "主要勘探层系/靶区": "震旦系灯影组、三叠系雷口坡组/嘉陵江组等"},
        {"专题名称": "川东北专题", "新增KCl目标配额": "1500 万吨", "主要勘探层系/靶区": "三叠系嘉陵江组、雷口坡组、飞仙关组等"},
        {"专题名称": "川西专题", "新增KCl目标配额": "1000 万吨", "主要勘探层系/靶区": "二叠系栖霞-茅口组、三叠系雷口坡组等"},
        {"专题名称": "川南专题", "新增KCl目标配额": "500 万吨", "主要勘探层系/靶区": "奥陶系宝塔组、三叠系雷口坡组、嘉陵江组等"},
    ]
    st.table(pd.DataFrame(targets_table))
