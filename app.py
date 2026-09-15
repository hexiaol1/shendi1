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

if "topic_archives" not in st.session_state:
    st.session_state.topic_archives = {
        "川中专题": [],
        "川东北专题": [],
        "川西专题": [],
        "川南专题": [],
    }

if "temp_single_record" not in st.session_state:
    st.session_state.temp_single_record = None
if "temp_mc_artifacts" not in st.session_state:
    st.session_state.temp_mc_artifacts = None

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
        st.session_state.temp_single_record = None
        st.session_state.temp_mc_artifacts = None
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


def extract_samples_from_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        phi_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["por", "phi", "孔隙度", "孔隙率"])]
        c_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["kcl", "品位", "浓度", "grade"])]
        
        phi_samples = df[phi_cols[0]].dropna().values if phi_cols else np.array([])
        c_samples = df[c_cols[0]].dropna().values if c_cols else np.array([])
        
        phi_arr = np.array(phi_samples, dtype=float)
        if len(phi_arr) > 0 and np.mean(phi_arr) > 1.0:
            phi_arr = phi_arr / 100.0
            
        c_arr = np.array(c_samples, dtype=float)
        return phi_arr, c_arr, df
    except Exception:
        return np.array([]), np.array([]), None


def generate_variable_distribution(
    mc_mode="单点/少井先验扰动模式",
    samples_arr=None,
    base_val=1.0,
    err_ratio=0.15,
    dist_type="自动优选",
    n_sim=10000,
    clip_range=(0.0, None)
):
    """
    通用参数抽样函数：支持多井分布拟合抽样与单值先验三角分布抽样
    """
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
# 全参数蒙特卡洛与多物理场动态校准核心推演引擎
# --------------------------------------------------
def evaluate_and_archive_unit(
    row_data, assigned_topic,
    dyn_target_param="校准有效孔隙度 ϕ",
    dyn_method_type="方法一：物质平衡法 (Wp - Δp)",
    mc_mode="单点/少井先验扰动模式", n_sim=10000, dist_type="自动优选",
    uncertainty_cfg=None
):
    if uncertainty_cfg is None:
        uncertainty_cfg = {"A": 0.15, "h": 0.15, "phi": 0.20, "Sw": 0.10, "S": 0.15, "C": 0.15}

    r = {str(k).strip(): v for k, v in row_data.items() if pd.notna(v)}
    calc_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unit_id = f"UNIT-{uuid.uuid4().hex[:8].upper()}"

    zone_name = str(r.get("构造带", r.get("构造名称", r.get("计算单元", "未命名单元"))))
    res_type = str(r.get("储层类型", r.get("类型", "油气同层型"))).strip()

    # 1. 基础物理量基准解析
    A_km2 = float(r.get("A", r.get("Aw", r.get("面积", 50.0))))
    A_m2_base = A_km2 * 1e6
    h_base = float(r.get("h", r.get("厚度", r.get("有效厚度", 25.0))))

    has_seismic_vol = ("V" in r) or ("雕刻体积" in r) or ("储层体积" in r)
    if has_seismic_vol:
        V_raw = float(r.get("V", r.get("雕刻体积", r.get("储层体积", 0.0))))
        V_m3_base = V_raw * 1e6 if V_raw < 1e5 else V_raw
        geom_method = "三维地震精细雕刻体积(V)"
    else:
        V_m3_base = A_m2_base * h_base
        geom_method = "平面面积乘厚度估算(A×h)"

    phi_single = float(r.get("phi", r.get("ϕ", r.get("孔隙度", 0.08))))
    if phi_single > 1.0: phi_single /= 100.0
    phi_samples = parse_sample_values(r.get("多井孔隙度序列", r.get("phi_samples", "")))
    if len(phi_samples) > 0 and np.mean(phi_samples) > 1.0: phi_samples /= 100.0

    c_single = float(r.get("C", r.get("KCl品位", r.get("品位", 0.018))))
    c_samples = parse_sample_values(r.get("多井品位序列", r.get("C_samples", "")))

    Sw_base = float(r.get("Sw", r.get("含水饱和度", 0.65)))
    if Sw_base > 1.0: Sw_base /= 100.0
    Bw_base = float(r.get("Bw", r.get("体积系数", 1.02)))
    S_base = float(r.get("S", r.get("弹性储水系数", 0.0005)))

    head_h = float(r.get("承压水头标高", r.get("h_head", 350.0)))
    top_H = float(r.get("储层顶面标高", r.get("H_top", -2200.0)))
    elastic_head = max(0.0, head_h - top_H)

    # 2. 全参数蒙特卡洛抽样生成 (6大关键参数)
    phi_draws, phi_mean, phi_desc, phi_raw_valid = generate_variable_distribution(
        mc_mode, phi_samples, phi_single, uncertainty_cfg.get("phi", 0.20), dist_type, n_sim, (0.001, 0.45)
    )
    c_draws, c_mean, c_desc, c_raw_valid = generate_variable_distribution(
        mc_mode, c_samples, c_single, uncertainty_cfg.get("C", 0.15), dist_type, n_sim, (0.0001, 0.20)
    )
    A_draws, A_mean, A_desc, _ = generate_variable_distribution(
        "单点/少井先验扰动模式", np.array([]), A_m2_base, uncertainty_cfg.get("A", 0.15), "三角分布", n_sim, (1e5, None)
    )
    h_draws, h_mean, h_desc, _ = generate_variable_distribution(
        "单点/少井先验扰动模式", np.array([]), h_base, uncertainty_cfg.get("h", 0.15), "三角分布", n_sim, (1.0, 500.0)
    )
    Sw_draws, Sw_mean, Sw_desc, _ = generate_variable_distribution(
        "单点/少井先验扰动模式", np.array([]), Sw_base, uncertainty_cfg.get("Sw", 0.10), "三角分布", n_sim, (0.05, 1.0)
    )
    S_draws, S_mean, S_desc, _ = generate_variable_distribution(
        "单点/少井先验扰动模式", np.array([]), S_base, uncertainty_cfg.get("S", 0.15), "三角分布", n_sim, (1e-6, 0.1)
    )

    # 3. 原始静态容积法基准计算
    if "非油气" in res_type:
        calc_model = "非油气同层型"
        Q_static_raw = (phi_mean * V_m3_base) + (S_mean * elastic_head * A_mean)
    else:
        calc_model = "油气同层型"
        Q_static_raw = (V_m3_base * phi_mean * Sw_mean) / Bw_base if has_seismic_vol else (A_mean * h_mean * phi_mean * Sw_mean) / Bw_base
    P_static_raw = Q_static_raw * c_mean

    # 4. 多物理场动态试采校准（可定向校准 ϕ, h, A/V, Sw, S）
    calib_factor = 1.0
    calib_obj = dyn_target_param
    dyn_applied = "纯静态（未校准）"
    dyn_process = "未录入动态参数，沿用静态蒙特卡洛均值基准"

    Wp = float(r.get("Wp", r.get("排卤量", 0.0)))
    dp = float(r.get("dp", r.get("压降", 0.0)))
    qw = float(r.get("qw", r.get("日产水量", 0.0)))
    dp_flow = float(r.get("dp_flow", r.get("流压差", 0.0)))
    k_perm = float(r.get("k", r.get("渗透率", 2.5)))
    mu = float(r.get("mu", r.get("卤水黏度", 1.15)))
    ct = float(r.get("ct", r.get("Ct", 8.5))) * (1e-4 if float(r.get("ct", 8.5)) > 1e-2 else 1.0)

    has_balance_data = (Wp > 0 and dp > 0)
    has_flow_data = (qw > 0 and dp_flow > 0)

    if "校准有效孔隙度" in calib_obj and has_balance_data:
        # 物质平衡反演连通流动孔隙度
        phi_cal = Wp / (ct * dp * V_m3_base)
        calib_factor = np.clip(phi_cal / (phi_mean + 1e-6), 0.65, 1.35)
        dyn_applied = "弹性物质平衡校准有效孔隙度 ϕ"
        dyn_process = f"依据累计排卤 Wp={Wp:.0f}m³、压降 Δp={dp:.2f}MPa 反求孔隙度；校准比={calib_factor:.3f}"

    elif "校准有效储卤厚度" in calib_obj and has_flow_data:
        # 渗流压差反演开启产流厚度
        h_cal = (qw * mu * 3.5) / (0.00708 * k_perm * dp_flow + 1e-6)
        calib_factor = np.clip(h_cal / (h_mean + 1e-6), 0.65, 1.35)
        dyn_applied = "渗流压差反演校准有效厚度 h"
        dyn_process = f"依据稳定产水 qw={qw:.1f}m³/d、生产压差 Δp={dp_flow:.2f}MPa 反求有效出水厚度；校准比={calib_factor:.3f}"

    elif "校准动态连通面积" in calib_obj:
        # 试井压力波及漏斗反演有效供液面积 A
        if "Re" in r or "波及半径" in r:
            Re = float(r.get("Re", r.get("波及半径", 1200.0)))
        elif has_flow_data:
            Re = max(200.0, np.sqrt((qw * mu) / (0.00708 * k_perm * max(1.0, h_mean) * dp_flow + 1e-6)) * 100.0)
        else:
            Re = 1200.0
        A_dyn_m2 = np.pi * (Re**2)
        calib_factor = np.clip(A_dyn_m2 / (A_mean + 1e-6), 0.70, 1.25)
        dyn_applied = "试井压力波及漏斗校准连通面积 A"
        dyn_process = f"波及漏斗反求控制半径 Re={Re:.0f}m，动态控制面积={A_dyn_m2/1e6:.2f}km²；面积校准比={calib_factor:.3f}"

    elif "校准有效含水饱和度" in calib_obj and has_flow_data:
        # 产能动态产水率校准可动水饱和度
        calib_factor = np.clip(1.0 - (dp_flow / (dp_flow + 5.0)) * 0.15, 0.80, 1.15)
        dyn_applied = "两相渗流产能校准可动含水饱和度 Sw"
        dyn_process = f"依据流压衰减斜率剔除束缚水影响；可动水饱和度校准比={calib_factor:.3f}"

    elif "校准弹性储水系数" in calib_obj and has_balance_data:
        # 承压含水层水头压降反演储水系数
        delta_head = dp * 100.0  # MPa 转承压水头米
        S_cal = Wp / (delta_head * A_mean + 1e-6)
        calib_factor = np.clip(S_cal / (S_mean + 1e-6), 0.65, 1.35)
        dyn_applied = "物质平衡扬程释水校准储水系数 S"
        dyn_process = f"依据排卤量 Wp 与扬程降 Δh 反求弹性释水系数；校准比={calib_factor:.3f}"

    # 5. 将动态校准因子赋予选定的参数，代入全参数蒙特卡洛联合抽样
    phi_final_draws = phi_draws * (calib_factor if "孔隙度" in calib_obj else 1.0)
    h_final_draws = h_draws * (calib_factor if "厚度" in calib_obj else 1.0)
    A_final_draws = A_draws * (calib_factor if "面积" in calib_obj else 1.0)
    Sw_final_draws = Sw_draws * (calib_factor if "含水饱和度" in calib_obj else 1.0)
    S_final_draws = S_draws * (calib_factor if "储水系数" in calib_obj else 1.0)

    phi_final_mean = phi_mean * (calib_factor if "孔隙度" in calib_obj else 1.0)
    h_final_mean = h_mean * (calib_factor if "厚度" in calib_obj else 1.0)
    A_final_mean = A_mean * (calib_factor if "面积" in calib_obj else 1.0)
    Sw_final_mean = Sw_mean * (calib_factor if "含水饱和度" in calib_obj else 1.0)
    S_final_mean = S_mean * (calib_factor if "储水系数" in calib_obj else 1.0)

    # 容积法抽样与期望值推导
    if "非油气" in res_type:
        V_final_draws = V_m3_base * (calib_factor if "面积" in calib_obj else 1.0) if has_seismic_vol else (A_final_draws * h_final_draws)
        V_final_mean = V_m3_base * (calib_factor if "面积" in calib_obj else 1.0) if has_seismic_vol else (A_final_mean * h_final_mean)
        Q_final_draws = (phi_final_draws * V_final_draws) + (S_final_draws * elastic_head * A_final_draws)
        Q_final_mean = (phi_final_mean * V_final_mean) + (S_final_mean * elastic_head * A_final_mean)
        formula_final = (
            f"Q = ϕ·V + S·(h-H)·A = {phi_final_mean:.4f}×{V_final_mean:.2e} + "
            f"{S_final_mean:.5f}×({head_h}-{top_H})×{A_final_mean:.2e}"
        )
    else:
        if has_seismic_vol:
            V_final_draws = V_m3_base * (calib_factor if "面积" in calib_obj else 1.0)
            V_final_mean = V_m3_base * (calib_factor if "面积" in calib_obj else 1.0)
            Q_final_draws = (V_final_draws * phi_final_draws * Sw_final_draws) / Bw_base
            Q_final_mean = (V_final_mean * phi_final_mean * Sw_final_mean) / Bw_base
            formula_final = f"Q = (V·ϕ·Sw)/Bw = ({V_final_mean:.2e}×{phi_final_mean:.4f}×{Sw_final_mean:.2f})/{Bw_base:.2f}"
        else:
            Q_final_draws = (A_final_draws * h_final_draws * phi_final_draws * Sw_final_draws) / Bw_base
            Q_final_mean = (A_final_mean * h_final_mean * phi_final_mean * Sw_final_mean) / Bw_base
            formula_final = f"Q = (A·h·ϕ·Sw)/Bw = ({A_final_mean:.2e}×{h_final_mean:.2f}×{phi_final_mean:.4f}×{Sw_final_mean:.2f})/{Bw_base:.2f}"

    P_draws_kcl_wan = (Q_final_draws * c_draws) / 1e4
    P_final_mean = Q_final_mean * c_mean

    p90_val = round(float(np.percentile(P_draws_kcl_wan, 10)), 2)
    p50_val = round(float(np.percentile(P_draws_kcl_wan, 50)), 2)
    p10_val = round(float(np.percentile(P_draws_kcl_wan, 90)), 2)

    record = {
        "计算单元编号": unit_id,
        "计算时间": calc_time,
        "所属专题": assigned_topic,
        "构造带/单元名称": zone_name,
        "储层类型": res_type,
        "计算模型": calc_model,
        "几何建模方式": geom_method,
        "蒙特卡洛模式": mc_mode,
        "动态校准对象参数": calib_obj,
        "动态校准方法类型": dyn_applied,
        "动态校准推演明细": dyn_process,
        "动态校准系数": round(calib_factor, 4),
        "校准后容积法推演": formula_final,
        "原始静态KCl储量(万吨)": round(P_static_raw / 1e4, 2),
        "最终核定KCl储量(万吨)": round(P_final_mean / 1e4, 2),
        "校准前后储量变化率": f"{((P_final_mean - P_static_raw)/(P_static_raw+1e-6))*100:+.2f}%",
        "核定卤水体积(亿m³)": round(Q_final_mean / 1e8, 4),
        "蒙特卡洛P90(万吨)": p90_val,
        "蒙特卡洛P50(万吨)": p50_val,
        "蒙特卡洛P10(万吨)": p10_val,
        # 详细核定后的参数期望值
        "核定孔隙度": f"{phi_final_mean*100:.2f}%",
        "核定厚度(m)": round(h_final_mean, 2),
        "核定面积(km²)": round(A_final_mean / 1e6, 2),
        "平均品位(t/m³)": c_mean,
        "含水饱和度": round(Sw_final_mean, 3),
        "弹性储水系数": round(S_final_mean, 6),
    }

    mc_artifacts = {
        "P_draws": P_draws_kcl_wan,
        "phi_draws": phi_final_draws,
        "c_draws": c_draws,
        "phi_samples": phi_raw_valid,
        "c_samples": c_raw_valid,
    }

    return record, mc_artifacts


# --------------------------------------------------
# 主界面 Tabs
# --------------------------------------------------
tab_step1, tab_archive, tab_docs = st.tabs([
    "📥 第一步：全参数蒙特卡洛与多维动态校准",
    f"📋 第二步：【{active_topic}】全要素归档详单与下载",
    "📖 第三步：全参数蒙特卡洛与多物理场校准体系",
])

# --------------------------------------------------
# TAB 1: 第一步
# --------------------------------------------------
with tab_step1:
    st.subheader(f"第一步：计算单元全要素参数推演 —— 【{active_topic}】")
    st.markdown(
        "**全面扩展体系**：蒙特卡洛覆盖面积 \(A\)、厚度 \(h\)、孔隙度 \(\phi\)、饱和度 \(S_w\)、储水系数 \(S\) 及品位 \(C\)；"
        "动态试采数据可定向校准孔隙度、厚度、连通面积、饱和度或储水系数。"
    )

    # 1. 蒙特卡洛与全参数不确定性配置面板
    with st.expander("⚙️ 蒙特卡洛全参数不确定性扰动幅度配置（点击展开/收起）", expanded=True):
        m1, m2, m3 = st.columns(3)
        with m1:
            global_mc_mode = st.radio(
                "蒙特卡洛样本建模模式：",
                ["模式一：多井实测样本统计拟合模式", "模式二：单点/少井先验扰动模式"],
                index=0
            )
        with m2:
            global_n_sim = st.select_slider(
                "抽样模拟试验次数 (N)",
                options=[1000, 2000, 5000, 10000, 20000],
                value=10000,
            )
        with m3:
            global_dist_choice = st.selectbox(
                "多井概率密度拟合函数优选：",
                ["自动优选 (按偏度判别)", "对数正态分布 (Lognormal)", "正态分布 (Normal)"]
            )

        st.caption("🔧 **全参数先验不确定性变异系数（±%）设定**：")
        u_c1, u_c2, u_c3, u_c4, u_c5, u_c6 = st.columns(6)
        with u_c1:
            err_A = st.slider("含水面积 A (±%)", 5, 40, 15, 5) / 100.0
        with u_c2:
            err_h = st.slider("有效厚度 h (±%)", 5, 40, 15, 5) / 100.0
        with u_c3:
            err_phi = st.slider("孔隙度 ϕ (±%)", 5, 50, 20, 5) / 100.0
        with u_c4:
            err_C = st.slider("KCl 品位 C (±%)", 5, 40, 15, 5) / 100.0
        with u_c5:
            err_Sw = st.slider("饱和度 Sw (±%)", 5, 30, 10, 5) / 100.0
        with u_c6:
            err_S = st.slider("储水系数 S (±%)", 5, 40, 15, 5) / 100.0

        current_uncertainty_cfg = {
            "A": err_A, "h": err_h, "phi": err_phi, "C": err_C, "Sw": err_Sw, "S": err_S
        }

    st.markdown("---")

    input_channel = st.radio(
        "请选择录入通道：",
        ["方式 A：在线交互录入参数（推荐单单元高精度核算）", "方式 B：批量上传多构造带汇总数据表 (CSV / Excel)"],
        horizontal=True
    )

    if "方式 A" in input_channel:
        st.markdown("#### 1. 构造基本信息与动态校准目标设定")
        ia1, ia2, ia3 = st.columns(3)
        with ia1:
            u_name = st.text_input("构造带/计算单元名称", value=f"{active_topic[:2]}某深部富钾构造")
        with ia2:
            u_type = st.radio("储卤层类型", ["油气同层型", "非油气同层型"], horizontal=True)
        with ia3:
            # 动态校准对象自由选择
            u_dyn_target = st.selectbox(
                "选择本次动态试采数据重点校准的静态参数对象：",
                [
                    "校准有效孔隙度 ϕ (基于弹性物质平衡 Wp - Δp)",
                    "校准有效储卤厚度 h (基于产水剖面渗流压差 qw - Δp_wf)",
                    "校准动态连通面积 A (基于试井压力波及漏斗 Re)",
                    "校准有效含水饱和度 Sw (基于两相渗流产能比)",
                    "校准弹性储水系数 S (基于承压扬程释水方程)",
                ]
            )

        st.markdown("#### 2. 全参数静态物性与几何录入")
        if "模式一" in global_mc_mode:
            st.caption("🔍 **模式一：多井样本获取**（支持直接粘贴文本或上传单井测井解释台账文件）：")
            mc_input_type = st.radio("多井数据源：", ["上传本构造带多井台账文件 (Excel/CSV)", "文本框粘贴序列"], horizontal=True)

            u_phi_str = ""
            u_c_str = ""
            u_phi_single = 0.08
            u_c_single = 0.0185

            if "上传" in mc_input_type:
                col_dl_well, col_up_well = st.columns([1, 3])
                with col_dl_well:
                    well_sample_df = pd.DataFrame({
                        "井号": ["Well-1", "Well-2", "Well-3", "Well-4", "Well-5", "Well-6"],
                        "层段": ["T3l-1", "T3l-1", "T3l-2", "T3l-1", "T3l-2", "T3l-1"],
                        "有效孔隙度(%)": [6.8, 8.2, 7.5, 9.4, 7.9, 8.6],
                        "KCl品位(t/m3)": [0.0182, 0.0195, 0.0210, 0.0178, 0.0190, 0.0205]
                    })
                    w_buf = io.BytesIO()
                    well_sample_df.to_csv(w_buf, index=False, encoding="utf_8_sig")
                    st.download_button(
                        label="📥 下载多井台账模板 (CSV)",
                        data=w_buf.getvalue(),
                        file_name=f"{active_topic}_多井实测台账模板.csv",
                        mime="text/csv",
                    )
                with col_up_well:
                    well_file = st.file_uploader("上传多井台账文件", type=["csv", "xlsx", "xls"], key="well_table_up")
                    if well_file is not None:
                        f_phi, f_c, df_well_preview = extract_samples_from_file(well_file)
                        if df_well_preview is not None:
                            st.success(f"已识别文件！读取到 {len(f_phi)} 个孔隙度样本，{len(f_c)} 个品位样本。")
                            st.dataframe(df_well_preview.head(4), use_container_width=True)
                            u_phi_str = ", ".join([str(v) for v in f_phi])
                            u_c_str = ", ".join([str(v) for v in f_c])
            else:
                ib1, ib2 = st.columns(2)
                with ib1:
                    u_phi_str = st.text_area("多井孔隙度序列 ϕ", value="0.065, 0.082, 0.071, 0.095, 0.078, 0.088", height=65)
                with ib2:
                    u_c_str = st.text_area("多井品位序列 C (t/m³)", value="0.0175, 0.0192, 0.0210, 0.0185", height=65)
        else:
            st.caption("🔍 **模式二：单点/少井参数录入**：输入解释代表值：")
            ib3, ib4 = st.columns(2)
            with ib3:
                u_phi_single = st.number_input("储层平均有效孔隙度 ϕ", value=0.078, step=0.005, format="%.3f")
                u_phi_str = ""
            with ib4:
                u_c_single = st.number_input("KCl 平均品位 C (t/m³)", value=0.0190, step=0.0010, format="%.4f")
                u_c_str = ""

        ig1, ig2, ig3 = st.columns(3)
        with ig1:
            u_A = st.number_input("储层平面面积 A (km²)", value=55.0, step=1.0)
        with ig2:
            has_v = st.checkbox("已有三维地震精细雕刻体积 V", value=False)
            if has_v:
                u_V = st.number_input("雕刻体积 V (10⁶ m³)", value=1300.0, step=50.0)
                u_h = 0.0
            else:
                u_h = st.number_input("储层有效厚度 h (m)", value=25.0, step=1.0)
                u_V = None
        with ig3:
            if u_type == "油气同层型":
                u_sw = st.number_input("含水饱和度 Sw (小数)", value=0.62, step=0.05)
                u_bw = st.number_input("卤水体积系数 Bw", value=1.02, step=0.01)
                u_S, u_head, u_top = None, None, None
            else:
                u_S = st.number_input("弹性储水系数 S", value=0.0005, step=0.0001, format="%.5f")
                u_head = st.number_input("平均承压水头标高 h (m)", value=340.0, step=10.0)
                u_top = st.number_input("储层顶面标高 H (m)", value=-2250.0, step=50.0)
                u_sw, u_bw = None, None

        st.markdown("#### 3. 动态测试与试采参数录入")
        enable_dyn_single = st.checkbox("输入动态试采参数进行多物理场校准", value=True)
        u_wp, u_dp, u_ct = None, None, None
        u_qw, u_dp_flow, u_k, u_mu, u_re = None, None, None, None, None

        if enable_dyn_single:
            id1, id2, id3, id4, id5 = st.columns(5)
            with id1:
                u_wp = st.number_input("累计排卤量 Wp (m³)", value=32000.0, step=1000.0)
            with id2:
                u_dp = st.number_input("地层压降 Δp (MPa)", value=3.0, step=0.1)
            with id3:
                u_qw = st.number_input("稳定日产水量 qw (m³/d)", value=190.0, step=10.0)
            with id4:
                u_dp_flow = st.number_input("流压差 Δp_wf (MPa)", value=4.2, step=0.1)
            with id5:
                u_k = st.number_input("储层渗透率 k (mD)", value=2.8, step=0.5)

        if st.button("🚀 执行全参数蒙特卡洛与动态校准推演", type="primary"):
            input_dict = {
                "构造带": u_name,
                "储层类型": u_type,
                "动态校准目标": u_dyn_target,
                "多井孔隙度序列": u_phi_str,
                "多井品位序列": u_c_str,
                "phi": u_phi_single,
                "C": u_c_single,
                "A": u_A, "h": u_h, "Sw": u_sw, "Bw": u_bw,
                "S": u_S, "承压水头标高": u_head, "储层顶面标高": u_top,
                "Wp": u_wp, "dp": u_dp, "qw": u_qw, "dp_flow": u_dp_flow, "k": u_k,
            }
            if u_V is not None:
                input_dict["V"] = u_V

            single_rec, mc_artifacts = evaluate_and_archive_unit(
                input_dict,
                assigned_topic=active_topic,
                dyn_target_param=u_dyn_target,
                mc_mode=global_mc_mode,
                n_sim=global_n_sim,
                dist_type=global_dist_choice,
                uncertainty_cfg=current_uncertainty_cfg
            )
            st.session_state.temp_single_record = single_rec
            st.session_state.temp_mc_artifacts = mc_artifacts

        # 结果展示与图表可视化
        if st.session_state.temp_single_record is not None:
            cur_rec = st.session_state.temp_single_record
            cur_art = st.session_state.temp_mc_artifacts

            st.markdown("---")
            st.subheader(f"📊 当前推演核定结果：{cur_rec['构造带/单元名称']}")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("核定孔隙度 ϕ", cur_rec['核定孔隙度'])
            r2.metric("核定厚度 h", f"{cur_rec['核定厚度(m)']} m")
            r3.metric("最终核定 KCl 储量", f"{cur_rec['最终核定KCl储量(万吨)']:,.2f} 万吨", delta=cur_rec['校准前后储量变化率'])
            r4.metric("核定卤水体积", f"{cur_rec['核定卤水体积(亿m³)']:.4f} 亿m³")

            st.info(
                f"🎯 **动态校准详情**：校准对象为【{cur_rec['动态校准对象参数']}】，校准方法：`{cur_rec['动态校准方法类型']}`，"
                f"校准有效性乘子 = **{cur_rec['动态校准系数']}**。\n"
                f"推演明细：{cur_rec['动态校准推演明细']}"
            )

            plot_c1, plot_c2 = st.columns(2)
            with plot_c1:
                # 孔隙度抽样直方图
                fig_fit = px.histogram(
                    x=cur_art["phi_draws"] * 100, nbins=50, histnorm='probability density',
                    labels={"x": "孔隙度 (%)"},
                    title="孔隙度不确定性抽样分布 (含动态校准偏移)"
                )
                if len(cur_art["phi_samples"]) > 0:
                    for p_s in cur_art["phi_samples"]:
                        fig_fit.add_vline(x=p_s * 100, line_dash="dot", line_color="orange")
                st.plotly_chart(fig_fit, use_container_width=True)

            with plot_c2:
                # 全参数蒙特卡洛最终储量概率分布
                fig_p = px.histogram(
                    x=cur_art["P_draws"], nbins=60,
                    labels={"x": "KCl 储量 (万吨)"},
                    title="全参数联合蒙特卡洛储量分布 (P10 - P50 - P90)"
                )
                fig_p.add_vline(x=cur_rec["蒙特卡洛P90(万吨)"], line_dash="dash", line_color="orange", annotation_text=f"P90: {cur_rec['蒙特卡洛P90(万吨)']}万吨")
                fig_p.add_vline(x=cur_rec["蒙特卡洛P50(万吨)"], line_dash="solid", line_color="green", annotation_text=f"P50: {cur_rec['蒙特卡洛P50(万吨)']}万吨")
                fig_p.add_vline(x=cur_rec["蒙特卡洛P10(万吨)"], line_dash="dash", line_color="red", annotation_text=f"P10: {cur_rec['蒙特卡洛P10(万吨)']}万吨")
                st.plotly_chart(fig_p, use_container_width=True)

            with st.expander("🔍 查看本单元校准后容积法推导公式与核定参数全貌", expanded=True):
                st.write(f"**容积法推演方程**：`{cur_rec['校准后容积法推演']}`")
                st.write(
                    f"**核定物性矩阵**：面积 \(A\) = {cur_rec['核定面积(km²)']} km² | 厚度 \(h\) = {cur_rec['核定厚度(m)']} m | "
                    f"孔隙度 \(\phi\) = {cur_rec['核定孔隙度']} | 饱和度 \(S_w\) = {cur_rec['含水饱和度']} | 品位 \(C\) = {cur_rec['平均品位(t/m³)']}"
                )

            col_save1, _ = st.columns([1, 3])
            with col_save1:
                if st.button("💾 将本计算单元结果存入总账", type="primary", use_container_width=True):
                    existing_ids = [item["计算单元编号"] for item in st.session_state.topic_archives[active_topic]]
                    if cur_rec["计算单元编号"] not in existing_ids:
                        st.session_state.topic_archives[active_topic].append(cur_rec)
                    st.success(f"已成功归档至【{active_topic}】总账！")
                    st.session_state.temp_single_record = None
                    st.session_state.temp_mc_artifacts = None
                    st.rerun()

    else:
        # 方式 B：批量上传表格
        st.markdown("#### 批量上传多构造带数据表格")
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            batch_dyn_def = st.selectbox(
                "默认动态校准目标对象：",
                [
                    "校准有效孔隙度 ϕ (基于弹性物质平衡 Wp - Δp)",
                    "校准有效储卤厚度 h (基于产水剖面渗流压差 qw - Δp_wf)",
                    "校准动态连通面积 A (基于试井压力波及漏斗 Re)",
                ],
            )
        with col_t2:
            sample_template = pd.DataFrame([
                {
                    "构造带": f"{active_topic[:2]}构造1号", "储层类型": "油气同层型",
                    "动态校准目标": "校准有效孔隙度 ϕ",
                    "多井孔隙度序列": "0.065, 0.082, 0.071, 0.095, 0.078",
                    "多井品位序列": "0.0175, 0.0192, 0.0210, 0.0185",
                    "A": 50.0, "h": 22.0, "Sw": 0.65, "Bw": 1.02,
                    "Wp": 40000, "dp": 3.5, "ct": 8.2,
                },
                {
                    "构造带": f"{active_topic[:2]}构造2号", "储层类型": "非油气同层型",
                    "动态校准目标": "校准有效储卤厚度 h",
                    "phi": 0.068, "C": 0.021, "A": 75.0, "V": 1800.0, "S": 0.0004,
                    "承压水头标高": 350.0, "储层顶面标高": -2300.0,
                    "qw": 220.0, "dp_flow": 4.8, "k": 3.2,
                },
            ])
            csv_buf = io.BytesIO()
            sample_template.to_csv(csv_buf, index=False, encoding="utf_8_sig")
            st.download_button(
                label="📥 下载全参数批量测试模板 (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"{active_topic}_全参数批量测试模板.csv",
                mime="text/csv",
            )

        file_up = st.file_uploader("选择批量数据表格文件", type=["csv", "xlsx", "xls"])
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
                    for _, row in df_raw.iterrows():
                        row_dict = row.to_dict()
                        target_obj = str(row_dict.get("动态校准目标", batch_dyn_def))
                        auto_mode = "模式一：多井实测样本统计拟合模式" if pd.notna(row_dict.get("多井孔隙度序列")) else "模式二：单点/少井先验扰动模式"
                        rec, _ = evaluate_and_archive_unit(
                            row_dict,
                            assigned_topic=active_topic,
                            dyn_target_param=target_obj,
                            mc_mode=auto_mode,
                            n_sim=global_n_sim,
                            dist_type=global_dist_choice,
                            uncertainty_cfg=current_uncertainty_cfg
                        )
                        batch_records.append(rec)
                        st.session_state.topic_archives[active_topic].append(rec)

                    st.success(f"批量推演完成！已成功将 {len(batch_records)} 个单元按全参数动态校准写入【{active_topic}】总账！")
                    st.rerun()
            except Exception as err:
                st.error(f"批量推演失败: {str(err)}")

# --------------------------------------------------
# TAB 2: 第二步 全要素归档总账与详单下载（安全切片保护）
# --------------------------------------------------
with tab_archive:
    st.subheader(f"第二步：【{active_topic}】计算单元全要素归档详单")

    if not my_records:
        st.info("当前专题暂无归档数据。请在第一步中录入或批量上传后存入总账。")
    else:
        df_arc = pd.DataFrame(my_records)
        st.write(f"已累计归档 **{len(df_arc)}** 个单元的具体推演履历。")

        preferred_cols = [
            "计算单元编号", "计算时间", "构造带/单元名称", "储层类型", "计算模型",
            "蒙特卡洛模式", "动态校准对象参数", "动态校准方法类型", "动态校准系数",
            "原始静态KCl储量(万吨)", "最终核定KCl储量(万吨)", "校准前后储量变化率",
            "蒙特卡洛P90(万吨)", "蒙特卡洛P50(万吨)", "蒙特卡洛P10(万吨)",
            "核定孔隙度", "核定厚度(m)", "核定面积(km²)", "平均品位(t/m³)",
            "校准后容积法推演"
        ]
        
        valid_cols = [col for col in preferred_cols if col in df_arc.columns]
        if not valid_cols:
            valid_cols = list(df_arc.columns)
            
        st.dataframe(df_arc[valid_cols], use_container_width=True)

        x_col = "构造带/单元名称" if "构造带/单元名称" in df_arc.columns else df_arc.columns[0]
        y_col = "最终核定KCl储量(万吨)" if "最终核定KCl储量(万吨)" in df_arc.columns else df_arc.columns[1]
        color_col = "动态校准对象参数" if "动态校准对象参数" in df_arc.columns else None

        fig_unit_bar = px.bar(
            df_arc,
            x=x_col,
            y=y_col,
            text=y_col,
            color=color_col,
            title=f"【{active_topic}】各计算单元核定储量分布 (目标配额: {my_target/1e4:.0f}万吨)",
        )
        fig_unit_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_unit_bar, use_container_width=True)

        csv_archive_out = df_arc.to_csv(index=False).encode("utf_8_sig")
        st.download_button(
            label=f"📥 导出【{active_topic}】全要素归档总报表 (全参数及校准记录 CSV)",
            data=csv_archive_out,
            file_name=f"{active_topic}_全参数动态校准储量归档详单.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# TAB 3: 第三步 理论指南
# --------------------------------------------------
with tab_docs:
    st.subheader("全参数蒙特卡洛与多物理场动态校准方法指南")

    st.markdown("### 1. 全参数蒙特卡洛联合抽样原理")
    st.write(
        "深层卤水钾盐储量计算涉及 6 大核心变量：\(A\)、\(h\)、\(\phi\)、\(S_w\)、\(S\) 和 \(C\)。"
        "系统放弃了单一针对孔隙度和品位的局部扰动，改为全要素多维概率超立方抽样。"
        "对于每一个随机试验样本组，均独立代入容积法方程，从而得到真实反映地质综合不确定性的累积概率分布曲线（CDF）。"
    )

    st.markdown("### 2. 多物理场动态校准工程矩阵")
    calib_guide_df = pd.DataFrame([
        {"动态测试数据": "累计排卤 Wp + 地层压降 Δp", "渗流/物理机理": "弹性水体物质平衡", "适宜校准参数": "有效连通孔隙度 ϕ / 弹性储水系数 S", "地质意义": "剔除非流动死孔隙，还原实际流动连通体积"},
        {"动态测试数据": "日产水量 qw + 生产流压差 Δp_wf", "渗流/物理机理": "单井拟稳态径向流动", "适宜校准参数": "有效产液厚度 h", "地质意义": "根据真实渗流剖面修正电性划分的过宽厚度"},
        {"动态测试数据": "压力恢复曲线探测边界 Re", "渗流/物理机理": "压力波扩散与边界效应", "适宜校准参数": "有效动水连通面积 A / 体积 V", "地质意义": "以实际流体流动连通域约束静态构造圈闭面积"},
        {"动态测试数据": "两相产流比变化", "渗流/物理机理": "相渗透率分异", "适宜校准参数": "可动含水饱和度 Sw", "地质意义": "剥离不可流动束缚水，精准核算产卤水饱和度"},
    ])
    st.table(calib_guide_df)

    st.markdown("---")
    st.markdown("### 3. 各专题增储任务配额")
    targets_table = [
        {"专题名称": "川中专题", "新增KCl目标配额": "2000 万吨", "主要勘探层系/靶区": "震旦系灯影组、三叠系雷口坡组/嘉陵江组等"},
        {"专题名称": "川东北专题", "新增KCl目标配额": "1500 万吨", "主要勘探层系/靶区": "三叠系嘉陵江组、雷口坡组、飞仙关组等"},
        {"专题名称": "川西专题", "新增KCl目标配额": "1000 万吨", "主要勘探层系/靶区": "二叠系栖霞-茅口组、三叠系雷口坡组等"},
        {"专题名称": "川南专题", "新增KCl目标配额": "500 万吨", "主要勘探层系/靶区": "奥陶系宝塔组、三叠系雷口坡组、嘉陵江组等"},
    ]
    st.table(pd.DataFrame(targets_table))
