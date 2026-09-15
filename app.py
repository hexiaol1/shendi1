import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# -----------------------------------------------------------------------------
# 1. 页面配置与专业工程样式注入
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="深层富钾卤水储量动-静结合评估系统 (Sichuan Basin KCl Evaluator)",
    page_icon="🧂",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: 700;
        color: #1E3A8A;
        padding-bottom: 8px;
        border-bottom: 2px solid #CBD5E1;
        margin-bottom: 16px;
    }
    .metric-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .highlight-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin: 10px 0;
        border-radius: 2px 6px 6px 2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 四大构造带典型地质与试采测试参数数据库
# -----------------------------------------------------------------------------
DEFAULT_REGION_DATA = {
    "川中构造带 (高石梯-磨溪/震旦系-三叠系)": {
        "reg_code": "CZ",
        "type": "油气同层型",
        "A": 450.0, "h": 22.0, "phi": 0.055, "S": 0.42, "B": 1.025,
        "V": 0.0, "H": 0.0, "C": 18.5,
        "Wp": 125000.0, "dp": 5.8, "Cw": 4.2e-4, "Cf": 6.8e-4, "sigma_p": 0.35,
        "A_std": 0.12, "h_std": 0.15, "phi_std": 0.15, "C_std": 0.10, "S_std": 0.12,
        "description": "气顶/凝析气水同层产出，承压高，具备完整试井关井压恢测试数据。"
    },
    "川西构造带 (平落坝-鸭母河/雷口坡-须家河组)": {
        "reg_code": "CX",
        "type": "非油气同层型",
        "A": 320.0, "h": 35.0, "phi": 0.048, "S": 0.0008, "B": 1.0,
        "V": 10500.0, "H": 15.0, "C": 22.0,
        "Wp": 45000.0, "dp": 3.2, "Cw": 4.0e-4, "Cf": 7.0e-4, "sigma_p": 0.45,
        "A_std": 0.15, "h_std": 0.18, "phi_std": 0.18, "C_std": 0.10, "S_std": 0.20,
        "description": "富裂缝-溶孔储层，水体厚度大，裂缝给水与微承压弹性复合驱动。"
    },
    "川东北构造带 (铁山坡-渡口河/飞仙关组-嘉陵江组)": {
        "reg_code": "CDB",
        "type": "油气同层型",
        "A": 520.0, "h": 28.0, "phi": 0.062, "S": 0.38, "B": 1.030,
        "V": 0.0, "H": 0.0, "C": 25.4,
        "Wp": 210000.0, "dp": 7.2, "Cw": 4.1e-4, "Cf": 6.5e-4, "sigma_p": 0.40,
        "A_std": 0.10, "h_std": 0.12, "phi_std": 0.14, "C_std": 0.08, "S_std": 0.10,
        "description": "海相鲕粒滩相储集体，溶蚀孔隙发育，富气富钾，生产动用测试充分。"
    },
    "川南构造带 (自贡-威远/嘉陵江组-雷口坡组)": {
        "reg_code": "CN",
        "type": "非油气同层型",
        "A": 380.0, "h": 18.0, "phi": 0.051, "S": 0.0005, "B": 1.0,
        "V": 6500.0, "H": 8.0, "C": 31.2,
        "Wp": 85000.0, "dp": 4.6, "Cw": 3.9e-4, "Cf": 7.2e-4, "sigma_p": 0.30,
        "A_std": 0.12, "h_std": 0.15, "phi_std": 0.15, "C_std": 0.12, "S_std": 0.25,
        "description": "高矿化度纯承压卤水构造，承压释水与局部构造疏干开采机制显著。"
    }
}

# -----------------------------------------------------------------------------
# 3. 会话状态（Session State）初始化
# -----------------------------------------------------------------------------
if "evaluated_zones" not in st.session_state:
    st.session_state["evaluated_zones"] = {}

# -----------------------------------------------------------------------------
# 4. 物理数学模型计算引擎
# -----------------------------------------------------------------------------
def calc_volumetric_Q_static(params, model_type):
    """
    静态容积法单点确定性计算
    返回值: Q (地面可动用/赋存卤水体积, 10^4 m^3)
    """
    if model_type == "油气同层型":
        # Q = (A * h * phi * S) / B
        # A 单位 km^2 = 10^6 m^2; h 为 m; 乘积为 m^3; 折算 10^4 m^3 需乘 100
        A_m2 = params['A'] * 1e6
        h_m = params['h']
        phi = params['phi']
        S = params['S']
        B = params['B'] if params['B'] > 0 else 1.0
        V_m3 = (A_m2 * h_m * phi * S) / B
        return V_m3 / 1e4
    else:
        # 非油气同层型: Q = phi * V + S * (h - H) * A
        # V: 10^4 m^3; (h - H)*A: m * 10^6 m^2 = 10^6 m^3 = 100 * 10^4 m^3
        V_elastic = params['phi'] * params['V']
        dh = max(0.0, params['h'] - params['H'])
        V_drainage = params['S'] * dh * (params['A'] * 100.0)
        return V_elastic + V_drainage

def run_monte_carlo_simulation(base_params, model_type, n_samples=30000):
    """
    蒙特卡洛随机模拟：构建地质参数先验概率密度
    """
    np.random.seed(42)
    # 面积 A (截断正态分布)
    A_samples = np.random.normal(base_params['A'], base_params['A'] * base_params.get('A_std', 0.12), n_samples)
    A_samples = np.clip(A_samples, base_params['A'] * 0.5, base_params['A'] * 1.8)

    # 净厚度 h (对数正态分布趋势)
    h_mu = np.log(base_params['h'])
    h_sigma = base_params.get('h_std', 0.15)
    h_samples = np.random.lognormal(h_mu, h_sigma, n_samples)
    h_samples = np.clip(h_samples, base_params['h'] * 0.4, base_params['h'] * 2.0)

    # 孔隙度 phi (截断正态)
    phi_samples = np.random.normal(base_params['phi'], base_params['phi'] * base_params.get('phi_std', 0.15), n_samples)
    phi_samples = np.clip(phi_samples, 0.005, 0.35)

    # 浓度 C (正态分布)
    C_samples = np.random.normal(base_params['C'], base_params['C'] * base_params.get('C_std', 0.10), n_samples)
    C_samples = np.clip(C_samples, 2.0, 180.0)

    if model_type == "油气同层型":
        S_samples = np.random.normal(base_params['S'], base_params['S'] * base_params.get('S_std', 0.12), n_samples)
        S_samples = np.clip(S_samples, 0.05, 0.98)

        B_samples = np.random.normal(base_params['B'], 0.015, n_samples)
        B_samples = np.clip(B_samples, 1.001, 1.25)

        Q_samples = ((A_samples * 1e6) * h_samples * phi_samples * S_samples / B_samples) / 1e4
    else:
        V_samples = np.random.normal(base_params['V'], base_params['V'] * 0.15, n_samples)
        V_samples = np.clip(V_samples, base_params['V'] * 0.4, base_params['V'] * 2.2)

        S_samples = np.random.normal(base_params['S'], base_params['S'] * base_params.get('S_std', 0.20), n_samples)
        S_samples = np.clip(S_samples, 0.00005, 0.05)

        H_samples = np.random.normal(base_params['H'], base_params['H'] * 0.15 + 0.1, n_samples)
        H_samples = np.clip(H_samples, 0.0, h_samples)

        dh_samples = np.maximum(0.0, h_samples - H_samples)
        Q_samples = (phi_samples * V_samples) + (S_samples * dh_samples * (A_samples * 100.0))

    # KCl 资源量 P (万吨) = Q (10^4 m^3) * C (kg/m^3 或 g/L) * 10^4 kg / 10^7 kg = Q * C / 1000
    P_samples = (Q_samples * C_samples) / 1000.0

    return pd.DataFrame({
        "A": A_samples, "h": h_samples, "phi": phi_samples,
        "Q": Q_samples, "C": C_samples, "P_KCl": P_samples
    })

def apply_bayesian_dynamic_constraint(df_prior, dyn_params):
    """
    基于深层封闭承压水弹性物质平衡方程构建似然，进行贝叶斯重要性重采样
    Wp = Vw_dyn * Ct * dp
    反演理论压降: dp_pred = Wp / (V_res_m3 * Ct)
    """
    Wp = dyn_params['Wp']         # 累积产卤量 (m^3)
    dp_obs = dyn_params['dp']     # 实测平均压降 (MPa)
    Cw = dyn_params['Cw']         # 水压缩系数 (MPa^-1)
    Cf = dyn_params['Cf']         # 岩石骨架压缩系数 (MPa^-1)
    sigma_p = dyn_params['sigma_p']  # 观测误差 (MPa)

    # 综合压缩系数 Ct = Cw + Cf / phi
    Ct_samples = Cw + (Cf / np.maximum(df_prior['phi'].values, 0.005))

    # 地下储层水体积 (m^3) = Q (10^4 m^3) * 1e4
    V_res_m3 = df_prior['Q'].values * 1e4

    # 正演理论压降
    dp_pred = Wp / (V_res_m3 * Ct_samples)

    # 高斯对数似然 (Log-Likelihood)
    residuals = dp_pred - dp_obs
    log_likelihood = -0.5 * ((residuals / sigma_p) ** 2)

    # 规避数值溢出，实施重要性归一化权重计算
    max_log = np.max(log_likelihood)
    weights = np.exp(log_likelihood - max_log)
    weights_sum = np.sum(weights)

    if weights_sum == 0 or np.isnan(weights_sum):
        weights = np.ones(len(df_prior)) / len(df_prior)
    else:
        weights /= weights_sum

    # 重采样 (Resampling)
    indices = np.random.choice(len(df_prior), size=len(df_prior), p=weights, replace=True)
    df_posterior = df_prior.iloc[indices].copy().reset_index(drop=True)
    df_posterior['dp_pred'] = dp_pred[indices]

    # 动态连通动用折减系数
    Vw_dyn_median = Wp / (np.median(Ct_samples) * dp_obs)
    vol_reduction_factor = Vw_dyn_median / np.median(V_res_m3)

    return df_posterior, vol_reduction_factor

# -----------------------------------------------------------------------------
# 5. 侧边栏交互控制面板
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ 评估控制中心")

selected_region = st.sidebar.selectbox(
    "1. 选择目标评价构造带",
    list(DEFAULT_REGION_DATA.keys())
)

region_info = DEFAULT_REGION_DATA[selected_region]

eval_level = st.sidebar.radio(
    "2. 勘探资料丰富度与评价级别",
    [
        "级别 I: 资料匮乏 (基础容积法 - 确定性计算)",
        "级别 II: 资料中等 (容积法 + 蒙特卡洛随机模拟)",
        "级别 III: 资料丰富 (容积法 + 蒙特卡洛 + 试采动态物质平衡约束)"
    ],
    index=2
)

model_type = st.sidebar.selectbox(
    "3. 矿床赋存产状模型",
    ["油气同层型", "非油气同层型"],
    index=0 if region_info["type"] == "油气同层型" else 1
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**片区简要概况：**\n*{region_info['description']}*")

# -----------------------------------------------------------------------------
# 6. 主界面顶部：全局攻关目标看板 (5000 万吨 KCl)
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">深层富钾卤水钾盐（KCl）储量动-静结合评估系统</div>', unsafe_allow_html=True)

all_zones = st.session_state["evaluated_zones"]
total_kcl = sum([val["P_KCl_eval"] for val in all_zones.values()])
target_kcl = 5000.0  # 战略攻关目标：5000 万吨
progress_ratio = min(1.0, total_kcl / target_kcl)

col_k1, col_k2, col_k3, col_k4 = st.columns([2, 1, 1, 1])
with col_k1:
    st.write(f"**全盆地新增 5000 万吨 KCl 储量攻关进度条** (已核准: **{total_kcl:.2f}** 万吨 / {target_kcl:.0f} 万吨)")
    st.progress(progress_ratio)
with col_k2:
    st.metric(label="当前完成率", value=f"{progress_ratio * 100:.1f} %")
with col_k3:
    gap = target_kcl - total_kcl
    st.metric(label="剩余任务缺口", value=f"{max(0.0, gap):.2f} 万吨", delta=f"{-gap:.2f}" if gap > 0 else "已超额达标", delta_color="inverse")
with col_k4:
    st.metric(label="已提交台账单元", value=f"{len(all_zones)} / 4 个")

# -----------------------------------------------------------------------------
# 7. 参数录入与地质-工程联动配置箱
# -----------------------------------------------------------------------------
st.markdown("### 1. 储层地质静态与测试参数录入")

with st.expander(f"⚙️ 当前评估单元：{selected_region} - 参数配置箱", expanded=True):
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    with col_p1:
        st.markdown("**【地质圈闭与储层规模】**")
        val_A = st.number_input("含卤含气面积 A (km²)", min_value=0.5, max_value=10000.0, value=float(region_info["A"]), step=5.0)
        val_h = st.number_input("有效储层净厚度 h (m)", min_value=0.2, max_value=500.0, value=float(region_info["h"]), step=1.0)
        val_phi = st.number_input("有效孔隙度 ϕ (小数)", min_value=0.001, max_value=0.40, value=float(region_info["phi"]), step=0.005, format="%.3f")

    with col_p2:
        st.markdown("**【赋存特征与品位参数】**")
        val_C = st.number_input("卤水KCl折算浓度 C (g/L 或 kg/m³)", min_value=0.5, max_value=250.0, value=float(region_info["C"]), step=0.5)
        if model_type == "油气同层型":
            val_S = st.number_input("含水饱和度 S (小数)", min_value=0.01, max_value=1.0, value=float(region_info["S"]), step=0.02)
            val_B = st.number_input("地层卤水体积系数 B", min_value=0.95, max_value=1.35, value=float(region_info["B"]), step=0.005)
            val_V = 0.0
            val_H = 0.0
        else:
            val_V = st.number_input("含水层总几何体积 V (10⁴ m³)", min_value=1.0, max_value=200000.0, value=float(region_info["V"]), step=100.0)
            val_S = st.number_input("弹性给水度/释水系数 S (无量纲)", min_value=0.00001, max_value=0.10, value=float(region_info["S"]), step=0.0002, format="%.5f")
            val_H = st.number_input("计划开采疏干水头 H (m)", min_value=0.0, max_value=500.0, value=float(region_info["H"]), step=1.0)
            val_B = 1.0

    with col_p3:
        st.markdown("**【蒙特卡洛不确定度标准差】**")
        std_A = st.slider("面积变异系数 σ_A", 0.05, 0.40, float(region_info.get("A_std", 0.12)), 0.01)
        std_h = st.slider("厚度变异系数 σ_h", 0.05, 0.40, float(region_info.get("h_std", 0.15)), 0.01)
        std_phi = st.slider("孔隙度变异系数 σ_ϕ", 0.05, 0.40, float(region_info.get("phi_std", 0.15)), 0.01)
        std_C = st.slider("浓度变异系数 σ_C", 0.02, 0.30, float(region_info.get("C_std", 0.10)), 0.01)

    with col_p4:
        st.markdown("**【试采动态排卤工程参数】**")
        is_dynamic = "级别 III" in eval_level
        val_Wp = st.number_input("累积排卤采出量 Wp (m³)", min_value=10.0, max_value=1e8, value=float(region_info["Wp"]), disabled=not is_dynamic)
        val_dp = st.number_input("实测地层有效压降 Δp (MPa)", min_value=0.01, max_value=80.0, value=float(region_info["dp"]), disabled=not is_dynamic)
        val_sigma = st.number_input("测压观测标准误差 σ_p (MPa)", min_value=0.01, max_value=5.0, value=float(region_info["sigma_p"]), disabled=not is_dynamic)
        val_Cw = st.number_input("卤水压缩系数 Cw (10⁻⁴ MPa⁻¹)", min_value=1.0, max_value=10.0, value=float(region_info["Cw"] * 1e4), disabled=not is_dynamic) * 1e-4
        val_Cf = st.number_input("岩石孔隙压缩系数 Cf (10⁻⁴ MPa⁻¹)", min_value=1.0, max_value=20.0, value=float(region_info["Cf"] * 1e4), disabled=not is_dynamic) * 1e-4

input_params = {
    "A": val_A, "h": val_h, "phi": val_phi, "S": val_S, "B": val_B,
    "V": val_V, "H": val_H, "C": val_C,
    "A_std": std_A, "h_std": std_h, "phi_std": std_phi, "C_std": std_C, "S_std": 0.15
}

dyn_params = {
    "Wp": val_Wp, "dp": val_dp, "sigma_p": val_sigma,
    "Cw": val_Cw, "Cf": val_Cf
}

# -----------------------------------------------------------------------------
# 8. 多标签页解算、对比与工程指南呈现
# -----------------------------------------------------------------------------
tab_calc, tab_benchmark, tab_docs = st.tabs([
    "🚀 当前构造单元综合解算",
    "📊 四大构造带台账对比与攻关监控",
    "📖 技术规范与数学物理推导"
])

# --------------------------- TAB 1: 综合解算 ----------------------------------
with tab_calc:
    # 1. 基础确定性容积法解算
    Q_det_10k_m3 = calc_volumetric_Q_static(input_params, model_type)
    P_det_kcl_10k_ton = (Q_det_10k_m3 * input_params['C']) / 1000.0

    if "级别 I" in eval_level:
        st.subheader("【级别 I】确定性容积法计算结果 (资料匮乏)")
        st.info("适用于探井较少、地震未雕刻、仅具备离散试样测试资料的早期普查阶段。")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("可采/赋存卤水体积 Q", f"{Q_det_10k_m3:,.2f} 万方")
        with col2:
            st.metric("折算 KCl 地质储量 P", f"{P_det_kcl_10k_ton:,.2f} 万吨")
        with col3:
            if st.button("💾 锁定并提交当前构造带储量 (级别 I)", type="primary"):
                st.session_state["evaluated_zones"][selected_region] = {
                    "method": "级别 I: 确定性容积法",
                    "model_type": model_type,
                    "Q_eval": Q_det_10k_m3,
                    "P_KCl_eval": P_det_kcl_10k_ton,
                    "P10": P_det_kcl_10k_ton,
                    "P50": P_det_kcl_10k_ton,
                    "P90": P_det_kcl_10k_ton,
                    "dyn_constrained": False,
                    "reduction_factor": 1.0
                }
                st.success(f"已成功固化提交：{selected_region}")
                st.rerun()

    elif "级别 II" in eval_level:
        st.subheader("【级别 II】蒙特卡洛概率估算结果 (资料中等)")
        st.info("圈闭已经初步落实，测井/岩心孔隙度与有效厚度具备区间统计特征。执行 30,000 次地质先验抽样。")

        with st.spinner("正在执行先验蒙特卡洛模拟..."):
            df_mc = run_monte_carlo_simulation(input_params, model_type, n_samples=30000)
            p10 = np.percentile(df_mc['P_KCl'], 90)
            p50 = np.percentile(df_mc['P_KCl'], 50)
            p90 = np.percentile(df_mc['P_KCl'], 10)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("单点确定性参考值", f"{P_det_kcl_10k_ton:,.2f} 万吨")
        c2.metric("P90 保守/证实储量", f"{p90:,.2f} 万吨")
        c3.metric("P50 中值/概算储量", f"{p50:,.2f} 万吨")
        c4.metric("P10 乐观/可能储量", f"{p10:,.2f} 万吨")

        fig_mc = px.histogram(
            df_mc, x="P_KCl", nbins=70, marginal="box",
            title=f"{selected_region} - 静态蒙特卡洛储量概率密度分布 (KCl)",
            labels={"P_KCl": "KCl 地质储量 (万吨)"},
            color_discrete_sequence=["#3B82F6"]
        )
        fig_mc.add_vline(x=p50, line_dash="dash", line_color="red", annotation_text=f"P50: {p50:.1f}万吨")
        st.plotly_chart(fig_mc, use_container_width=True)

        if st.button("💾 锁定并提交当前构造带储量 (以 P50 估算量入库)", type="primary"):
            st.session_state["evaluated_zones"][selected_region] = {
                "method": "级别 II: 蒙特卡洛模拟法",
                "model_type": model_type,
                "Q_eval": np.percentile(df_mc['Q'], 50),
                "P_KCl_eval": p50,
                "P10": p10, "P50": p50, "P90": p90,
                "dyn_constrained": False,
                "reduction_factor": 1.0
            }
            st.success(f"已成功固化提交：{selected_region}")
            st.rerun()

    else:
        st.subheader("【级别 III】动-静耦合（蒙特卡洛 + 试采物质平衡贝叶斯更新）")
        st.info("地质静态建模与现场试采排卤压降响应深度闭环，剔除非连通储层无效体积。")

        with st.spinner("执行先验蒙特卡洛采样并根据实测压降进行贝叶斯似然更新..."):
            df_prior = run_monte_carlo_simulation(input_params, model_type, n_samples=30000)
            df_post, red_factor = apply_bayesian_dynamic_constraint(df_prior, dyn_params)

            prior_p50 = np.percentile(df_prior['P_KCl'], 50)
            post_p10 = np.percentile(df_post['P_KCl'], 90)
            post_p50 = np.percentile(df_post['P_KCl'], 50)
            post_p90 = np.percentile(df_post['P_KCl'], 10)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("纯静态 P50 中值", f"{prior_p50:,.2f} 万吨")
        c2.metric("动态约束后 P50 储量", f"{post_p50:,.2f} 万吨", delta=f"{(post_p50 - prior_p50):,.2f} 万吨")
        c3.metric("动态连通动用系数", f"{red_factor:.3f}", help="试采压降反映的有效动用体积与静态地质圈定体积之比")
        c4.metric("动态置信区间 [P90 - P10]", f"{post_p90:.1f} ~ {post_p10:.1f} 万吨")

        fig_bayes = go.Figure()
        fig_bayes.add_trace(go.Histogram(
            x=df_prior['P_KCl'], nbinsx=70, name='静态先验概率 (Monte Carlo)',
            marker_color='#93C5FD', opacity=0.55
        ))
        fig_bayes.add_trace(go.Histogram(
            x=df_post['P_KCl'], nbinsx=70, name='动态物质平衡后验 (Bayesian Updated)',
            marker_color='#1E40AF', opacity=0.75
        ))
        fig_bayes.update_layout(
            barmode='overlay',
            title=f"{selected_region} - 动静结合校正前后储量概率密度对比",
            xaxis_title="KCl 储量 (万吨)",
            yaxis_title="频数 / 概率加权",
            legend=dict(x=0.70, y=0.95)
        )
        st.plotly_chart(fig_bayes, use_container_width=True)

        if st.button("💾 锁定并提交当前构造带储量 (动静结合更新值入库)", type="primary"):
            st.session_state["evaluated_zones"][selected_region] = {
                "method": "级别 III: 动静结合贝叶斯反演",
                "model_type": model_type,
                "Q_eval": np.percentile(df_post['Q'], 50),
                "P_KCl_eval": post_p50,
                "P10": post_p10, "P50": post_p50, "P90": post_p90,
                "dyn_constrained": True,
                "reduction_factor": red_factor
            }
            st.success(f"已成功固化动静结合成果：{selected_region}")
            st.rerun()

# --------------------------- TAB 2: 台账对比 ----------------------------------
with tab_benchmark:
    st.subheader("全盆地四大构造带储量汇总台账")
    if len(st.session_state["evaluated_zones"]) == 0:
        st.warning("当前尚未锁定任何构造带数据。请在第 1 标签页中完成计算并点击【💾 锁定并提交】按钮。")
    else:
        rows = []
        for name, data in st.session_state["evaluated_zones"].items():
            rows.append({
                "构造单元名称": name,
                "地质产状模型": data["model_type"],
                "采用评价方法": data["method"],
                "有效卤水体积 Q (万方)": f"{data['Q_eval']:,.1f}",
                "P90 保守储量 (万吨)": f"{data['P90']:,.2f}",
                "P50 核定储量 (万吨)": data['P_KCl_eval'],
                "P10 乐观储量 (万吨)": f"{data['P10']:,.2f}",
                "动用折减系数": f"{data['reduction_factor']:.3f}",
                "动态约束状态": "已实施" if data["dyn_constrained"] else "未约束"
            })
        df_summary = pd.DataFrame(rows)
        st.dataframe(df_summary, use_container_width=True)

        cg1, cg2 = st.columns(2)
        with cg1:
            fig_bar = px.bar(
                df_summary, x="构造单元名称", y="P50 核定储量 (万吨)",
                color="动态约束状态", text="P50 核定储量 (万吨)",
                title="各构造单元新增 KCl 储量对比 (万吨)",
                color_discrete_map={"已实施": "#10B981", "未约束": "#F59E0B"}
            )
            fig_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with cg2:
            fig_pie = px.pie(
                df_summary, names="构造单元名称", values="P50 核定储量 (万吨)",
                title="全盆地新增储量贡献比例",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        csv_data = df_summary.to_csv(index=False).encode('utf_8_sig')
        st.download_button(
            "📥 导出四川盆地深层富钾卤水储量台账报告 (CSV)",
            csv_data,
            "四川盆地深层富钾卤水KCl储量动静结合评估台账.csv",
            "text/csv"
        )

# --------------------------- TAB 3: 技术规范与推导 ----------------------------
with tab_docs:
    st.markdown(r"""
    ### 📖 算法数学模型推导与专题操作规范

    ---
    #### 一、 容积法双模式物理方程
    根据深层卤水与气藏的伴生赋存机理差异，系统内建两套标准化公式：

    **1. 模式 A：油气同层型（气顶/凝析气水同层）**
    $$Q = \frac{A \cdot h \cdot \phi \cdot S}{B}, \quad P = Q \cdot C \times 10^{-3}$$
    * $A$：含水圈闭面积 ($\text{km}^2$)
    * $h$：储层有效富钾厚度 ($\text{m}$)
    * $\phi$：有效孔隙度 (小数)
    * $S$：含水饱和度 (小数)
    * $B$：地层卤水体积系数 (通常为 $1.01 \sim 1.05$)
    * $Q$：地面条件下的有效赋存卤水体积 ($10^4\ \text{m}^3$)
    * $C$：卤水 $\text{KCl}$ 折算质量浓度 ($\text{g/L}$ 或 $\text{kg/m}^3$)
    * $P$：$\text{KCl}$ 地质储量 (万吨)

    **2. 模式 B：非油气同层型（纯卤承压或微承压含水层）**
    $$Q = \phi \cdot V + S \cdot (h - H) \cdot A \times 100, \quad P = Q \cdot C \times 10^{-3}$$
    * $V$：含水层总几何体积 ($10^4\ \text{m}^3$)
    * $S$：重力给水度或弹性释水系数 (无量纲)
    * $H$：开采降落水头或计划疏干深度 ($\text{m}$)

    ---
    #### 二、 动态物质平衡与贝叶斯约束反演机制
    在深层封闭承压储集体中，开采排卤释放的能量由卤水自身的弹性膨胀与储层岩石骨架压缩弹性释水共同提供：
    $$W_p = V_{w, \text{dyn}} \cdot C_t \cdot \Delta p$$
    其中综合压缩系数 $C_t$ 满足：
    $$C_t = C_w + \frac{C_f}{\phi}$$
    * $W_p$：测试累积排卤采出量 ($\text{m}^3$)
    * $\Delta p$：地层平均有效压降 ($\text{MPa}$)
    * $C_w$：高矿化度卤水压缩系数 ($\text{MPa}^{-1}$)
    * $C_f$：储层岩石孔隙压缩系数 ($\text{MPa}^{-1}$)

    **贝叶斯后验重采样算法流程：**
    1. 蒙特卡洛模块生成先验地质样本集 $\{A^{(k)}, h^{(k)}, \phi^{(k)}, Q^{(k)}\}_{k=1}^N$。
    2. 计算每个样本对应的理论正演压降：$\Delta p_{\text{pred}}^{(k)} = \frac{W_p}{Q^{(k)} \cdot 10^4 \cdot C_t^{(k)}}$。
    3. 构建基于实测测压误差 $\sigma_p$ 的高斯似然函数：
       $$\mathcal{L}_k = \exp\left( -\frac{(\Delta p_{\text{pred}}^{(k)} - \Delta p_{\text{obs}})^2}{2\sigma_p^2} \right)$$
    4. 归一化权重实施重要性重采样，剔除与动态生产响应冲突的非连通假象体积，输出更契合实际生产能力的后验 P10、P50、P90 储量。
    """)
