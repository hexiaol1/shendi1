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

# 缓存当前正在核算的单单元结果
if "temp_single_record" not in st.session_state:
    st.session_state.temp_single_record = None
if "temp_mc_artifacts" not in st.session_state:
    st.session_state.temp_mc_artifacts = None

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
# 辅助函数：多井离散样本解析与自适应抽样
# --------------------------------------------------
def parse_sample_values(raw_input):
    """解析以逗号、空格或换行分隔的测试样本序列"""
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
    """从上传的多井单项台账中自动匹配提取孔隙度和品位序列"""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        phi_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["por", "phi", "孔隙度", "孔隙率"])]
        c_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["kcl", "品位", "浓度", "grade"])]
        
        phi_samples = df[phi_cols[0]].dropna().values if phi_cols else np.array([])
        c_samples = df[c_cols[0]].dropna().values if c_cols else np.array([])
        
        # 兼容百分比与小数
        phi_arr = np.array(phi_samples, dtype=float)
        if len(phi_arr) > 0 and np.mean(phi_arr) > 1.0:
            phi_arr = phi_arr / 100.0
            
        c_arr = np.array(c_samples, dtype=float)
        return phi_arr, c_arr, df
    except Exception as e:
        return np.array([]), np.array([]), None


def generate_parameter_draws(
    mc_mode="单点/少井先验扰动模式",
    samples_arr=None,
    base_val=0.08,
    err_ratio=0.20,
    dist_type="自动优选",
    n_sim=10000,
    val_type="phi"
):
    if samples_arr is None:
        samples_arr = np.array([])
    valid_samples = samples_arr[samples_arr > 0]

    fit_info = {}

    if "多井" in mc_mode and len(valid_samples) >= 3:
        mean_val = float(np.mean(valid_samples))
        std_val = float(np.std(valid_samples, ddof=1))

        if dist_type == "对数正态分布 (Lognormal)":
            chosen = "lognorm"
        elif dist_type == "正态分布 (Normal)":
            chosen = "norm"
        else:
            skewness = stats.skew(valid_samples)
            chosen = "lognorm" if skewness > 0.3 else "norm"

        if chosen == "lognorm":
            shape, loc, scale = stats.lognorm.fit(valid_samples, floc=0)
            draws = stats.lognorm.rvs(shape, loc=loc, scale=scale, size=n_sim)
            fit_name = "对数正态分布 (多井拟合)"
        else:
            loc, scale = stats.norm.fit(valid_samples)
            draws = stats.norm.rvs(loc=loc, scale=scale, size=n_sim)
            fit_name = "正态分布 (多井拟合)"

        fit_info = {
            "mode": "多井实测样本拟合模式",
            "type": fit_name,
            "desc": f"{fit_name}(μ={mean_val:.4f}, σ={std_val:.4f}, n={len(valid_samples)})",
            "mean": mean_val,
            "std": std_val,
            "n_samples": len(valid_samples),
            "samples": valid_samples,
        }
    else:
        mean_val = valid_samples[0] if len(valid_samples) > 0 else base_val
        low = mean_val * (1.0 - err_ratio)
        high = mean_val * (1.0 + err_ratio)
        draws = np.random.triangular(low, mean_val, high, n_sim)
        fit_info = {
            "mode": "单点/少井先验扰动模式",
            "type": "三角分布 (先验扰动)",
            "desc": f"先验三角分布(基准={mean_val:.4f}, 扰动±{err_ratio*100:.0f}%)",
            "mean": mean_val,
            "std": float(np.std(draws)),
            "n_samples": len(valid_samples),
            "samples": valid_samples,
        }

    if val_type == "phi":
        draws = np.clip(draws, 0.001, 0.45)
    elif val_type == "C":
        draws = np.clip(draws, 0.0001, 0.15)
    elif val_type == "Sw":
        draws = np.clip(draws, 0.05, 1.0)
    else:
        draws = np.clip(draws, 0.00001, 1.0)

    return draws, fit_info


# --------------------------------------------------
# 核心自适应计算引擎（双模式蒙特卡洛 + 动静校准）
# --------------------------------------------------
def evaluate_and_archive_unit(
    row_data, assigned_topic, default_dyn_method="方法一：物质平衡校准有效孔隙度",
    mc_mode="单点/少井先验扰动模式", n_sim=10000, dist_type="自动优选",
    phi_err_manual=0.20, c_err_manual=0.15
):
    r = {str(k).strip(): v for k, v in row_data.items() if pd.notna(v)}
    calc_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unit_id = f"UNIT-{uuid.uuid4().hex[:8].upper()}"

    zone_name = str(r.get("构造带", r.get("构造名称", r.get("计算单元", "未命名单元"))))
    res_type = str(r.get("储层类型", r.get("类型", "油气同层型"))).strip()

    # 1. 基础物理量解析
    phi_raw = r.get("多井孔隙度序列", r.get("phi_samples", ""))
    phi_sample_arr = parse_sample_values(phi_raw)
    phi_single = float(r.get("phi", r.get("ϕ", r.get("孔隙度", 0.08))))
    if phi_single > 1.0: phi_single /= 100.0
    if len(phi_sample_arr) > 0 and np.mean(phi_sample_arr) > 1.0:
        phi_sample_arr = phi_sample_arr / 100.0

    c_raw = r.get("多井品位序列", r.get("C_samples", ""))
    c_sample_arr = parse_sample_values(c_raw)
    c_single = float(r.get("C", r.get("KCl品位", r.get("品位", 0.018))))

    h_init = float(r.get("h", r.get("厚度", r.get("有效厚度", 25.0))))
    A_km2 = float(r.get("A", r.get("Aw", r.get("面积", 50.0))))
    A_m2 = A_km2 * 1e6

    has_seismic_vol = ("V" in r) or ("雕刻体积" in r) or ("储层体积" in r)
    if has_seismic_vol:
        V_raw = float(r.get("V", r.get("雕刻体积", r.get("储层体积", 0.0))))
        V_m3 = V_raw * 1e6 if V_raw < 1e5 else V_raw
        geom_method = "三维地震精细雕刻体积(V)"
    else:
        V_m3 = A_m2 * h_init
        geom_method = "平面面积乘厚度估算(A×h)"

    Sw_init = float(r.get("Sw", r.get("含水饱和度", 0.65)))
    if Sw_init > 1.0: Sw_init /= 100.0
    Bw_init = float(r.get("Bw", r.get("体积系数", 1.02)))
    S_init = float(r.get("S", r.get("弹性储水系数", 0.0005)))
    head_h = float(r.get("承压水头标高", r.get("h_head", 350.0)))
    top_H = float(r.get("储层顶面标高", r.get("H_top", -2200.0)))
    elastic_head = max(0.0, head_h - top_H)

    # 2. 蒙特卡洛抽样
    phi_draws, phi_fit = generate_parameter_draws(
        mc_mode=mc_mode, samples_arr=phi_sample_arr, base_val=phi_single,
        err_ratio=phi_err_manual, dist_type=dist_type, n_sim=n_sim, val_type="phi"
    )
    c_draws, c_fit = generate_parameter_draws(
        mc_mode=mc_mode, samples_arr=c_sample_arr, base_val=c_single,
        err_ratio=c_err_manual, dist_type=dist_type, n_sim=n_sim, val_type="C"
    )

    phi_mean = phi_fit["mean"]
    c_mean = c_fit["mean"]

    # 3. 静态容积法基准
    if "非油气" in res_type:
        calc_model = "非油气同层型"
        Q_static_raw = (phi_mean * V_m3) + (S_init * elastic_head * A_m2)
    else:
        calc_model = "油气同层型"
        Q_static_raw = (V_m3 * phi_mean * Sw_init) / Bw_init if has_seismic_vol else (A_m2 * h_init * phi_mean * Sw_init) / Bw_init
    P_static_raw = Q_static_raw * c_mean

    # 4. 动态约束校准
    dyn_choice = str(r.get("动态方法", default_dyn_method))
    dyn_method_applied = "纯静态（未启用动态数据校准）"
    dyn_process = "未录入动态试采数据，沿用静态解释基准"
    param_adjust_note = "参数未校准"

    phi_adj = phi_mean
    h_adj = h_init
    calib_phi_factor = 1.0
    calib_h_factor = 1.0

    has_m1 = ("Wp" in r or "排卤量" in r) and ("dp" in r or "压降" in r or "Δp" in r)
    has_m2 = ("qw" in r or "日产水量" in r) and ("dp_flow" in r or "流压差" in r)

    if "方法一" in dyn_choice and has_m1:
        Wp = float(r.get("Wp", r.get("排卤量", 0.0)))
        dp = float(r.get("dp", r.get("压降", 1.0)))
        ct_raw = float(r.get("ct", r.get("Ct", 8.5)))
        ct = ct_raw * 1e-4 if ct_raw > 1e-2 else ct_raw

        if dp > 0 and ct > 0 and V_m3 > 0:
            phi_inferred = Wp / (ct * dp * V_m3)
            calib_phi_factor = np.clip(phi_inferred / (phi_mean + 1e-6), 0.70, 1.30)
            phi_adj = phi_mean * calib_phi_factor
            dyn_method_applied = "方法一：物质平衡校准有效孔隙度"
            dyn_process = (
                f"由累计排卤 Wp={Wp:.0f}m³、压降 Δp={dp:.2f}MPa 反求动态孔隙；"
                f"有效孔隙校正系数={calib_phi_factor:.3f}"
            )
            param_adjust_note = f"基准孔隙度由 {phi_mean*100:.2f}% 校准为 {phi_adj*100:.2f}%"

    elif "方法二" in dyn_choice and has_m2:
        qw = float(r.get("qw", r.get("日产水量", 180.0)))
        dp_flow = float(r.get("dp_flow", r.get("流压差", 4.5)))
        k_perm = float(r.get("k", r.get("渗透率", 2.5)))
        mu = float(r.get("mu", r.get("卤水黏度", 1.15)))

        if dp_flow > 0 and k_perm > 0:
            h_inferred = (qw * mu * 3.5) / (0.00708 * k_perm * dp_flow + 1e-6)
            calib_h_factor = np.clip(h_inferred / (h_init + 1e-6), 0.70, 1.30)
            h_adj = h_init * calib_h_factor
            dyn_method_applied = "方法二：渗流压差反演校准有效厚度"
            dyn_process = (
                f"由日产水 qw={qw:.1f}m³/d、生产压差 Δp={dp_flow:.2f}MPa 反求动态厚度；"
                f"厚度校准系数={calib_h_factor:.3f}"
            )
            param_adjust_note = f"储层厚度由原始解释 {h_init:.2f}m 校准为 {h_adj:.2f}m"

    # 5. 结合校准后状态进行蒙特卡洛抽样计算
    phi_draws_adj = phi_draws * calib_phi_factor
    h_draws_adj = h_adj

    if "非油气" in res_type:
        V_final = V_m3 if has_seismic_vol else (A_m2 * h_draws_adj)
        Q_draws = (phi_draws_adj * V_final) + (S_init * elastic_head * A_m2)
        Q_final_mean = (phi_adj * V_final) + (S_init * elastic_head * A_m2)
        formula_final = (
            f"Q = ϕ_adj·V + S·(h-H)·A = {phi_adj:.4f}×{V_final:.2e} + "
            f"{S_init:.5f}×({head_h}-{top_H})×{A_m2:.2e}"
        )
    else:
        if has_seismic_vol:
            Q_draws = (V_m3 * phi_draws_adj * Sw_init) / Bw_init
            Q_final_mean = (V_m3 * phi_adj * Sw_init) / Bw_init
            formula_final = f"Q = (V·ϕ_adj·Sw)/Bw = ({V_m3:.2e}×{phi_adj:.4f}×{Sw_init:.2f})/{Bw_init:.2f}"
        else:
            Q_draws = (A_m2 * h_draws_adj * phi_draws_adj * Sw_init) / Bw_init
            Q_final_mean = (A_m2 * h_adj * phi_adj * Sw_init) / Bw_init
            formula_final = f"Q = (Aw·h_adj·ϕ_adj·Sw)/Bw = ({A_m2:.2e}×{h_adj:.2f}×{phi_adj:.4f}×{Sw_init:.2f})/{Bw_init:.2f}"

    P_draws_kcl_wan = (Q_draws * c_draws) / 1e4  # 万吨
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
        "蒙特卡洛模式": phi_fit["mode"],
        "孔隙度分布特征": phi_fit["desc"],
        "品位分布特征": c_fit["desc"],
        "动静结合方法": dyn_method_applied,
        "动态参数校准说明": param_adjust_note,
        "动静推演计算过程": dyn_process,
        "校准后容积法推演": formula_final,
        "原始静态KCl储量(万吨)": round(P_static_raw / 1e4, 2),
        "最终核定KCl储量(万吨)": round(P_final_mean / 1e4, 2),
        "校准前后储量变化率": f"{((P_final_mean - P_static_raw)/(P_static_raw+1e-6))*100:+.2f}%",
        "核定卤水体积(亿m³)": round(Q_final_mean / 1e8, 4),
        "蒙特卡洛P90(万吨)": p90_val,
        "蒙特卡洛P50(万吨)": p50_val,
        "蒙特卡洛P10(万吨)": p10_val,
        "基准孔隙度": f"{phi_mean*100:.2f}%",
        "校准后孔隙度": f"{phi_adj*100:.2f}%",
        "平均品位(t/m³)": c_mean,
        "有效厚度(m)": round(h_adj, 2),
    }

    mc_artifacts = {
        "P_draws": P_draws_kcl_wan,
        "phi_draws": phi_draws,
        "phi_fit": phi_fit,
        "c_fit": c_fit,
    }

    return record, mc_artifacts


# --------------------------------------------------
# 主界面 Tabs
# --------------------------------------------------
tab_step1, tab_archive, tab_docs = st.tabs([
    "📥 第一步：数据录入、蒙特卡洛模式选择与动静推演",
    f"📋 第二步：【{active_topic}】全要素归档详单与下载",
    "📖 第三步：双模式蒙特卡洛与动静指南",
])

# --------------------------------------------------
# TAB 1: 第一步
# --------------------------------------------------
with tab_step1:
    st.subheader(f"第一步：计算单元参数推演 —— 【{active_topic}】")
    st.markdown(
        "**自适应建模支持**：针对井数多、资料全的成熟区提供**多井概率密度统计拟合**（支持直接粘贴数值或上传独立多井测井台账文件）；针对勘探早期、井数少或仅有单点解释的区域提供**先验扰动模式**。"
    )

    with st.expander("⚙️ 蒙特卡洛建模模式与不确定性参数配置（点击展开/收起）", expanded=True):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            global_mc_mode = st.radio(
                "请选择当前计算单元的蒙特卡洛建模模式：",
                [
                    "模式一：多井实测样本统计拟合模式 (适用于资料丰富/井数≥3的成熟区)",
                    "模式二：单点/少井先验扰动模式 (适用于探井少/仅1~2口井或单点均值区)",
                ],
                index=0
            )
        with col_m2:
            global_n_sim = st.select_slider(
                "抽样模拟试验次数 (N)",
                options=[1000, 2000, 5000, 10000, 20000],
                value=10000,
            )

        if "模式一" in global_mc_mode:
            st.info("💡 **模式一启用中**：系统支持直接上传多井台账表格，自动拟合对数正态或正态概率密度分布。")
            global_dist_choice = st.selectbox(
                "概率分布拟合函数优选：",
                ["自动优选 (根据样本偏度自动判别)", "对数正态分布 (Lognormal - 推荐孔隙度)", "正态分布 (Normal)"]
            )
            global_phi_err, global_c_err = 0.20, 0.15
        else:
            st.info("💡 **模式二启用中**：系统将基于单井/代表值，结合自定义扰动区间进行蒙特卡洛模拟。")
            global_dist_choice = "三角分布"
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                global_phi_err = st.slider("基准孔隙度 ϕ 不确定性扰动幅度 (±%)", 5, 50, 20, 5) / 100.0
            with sub_c2:
                global_c_err = st.slider("KCl 品位 C 不确定性扰动幅度 (±%)", 5, 50, 15, 5) / 100.0

    st.markdown("---")

    input_channel = st.radio(
        "请选择录入通道：",
        ["方式 A：在线交互录入参数（推荐单单元核算）", "方式 B：批量上传多构造带汇总数据表 (CSV / Excel)"],
        horizontal=True
    )

    if "方式 A" in input_channel:
        st.markdown("#### 1. 构造单元基本信息与动静方法")
        ia1, ia2, ia3 = st.columns(3)
        with ia1:
            u_name = st.text_input("构造带/计算单元名称", value=f"{active_topic[:2]}某储卤构造")
        with ia2:
            u_type = st.radio("储卤层类型", ["油气同层型", "非油气同层型"], horizontal=True)
        with ia3:
            u_dyn_method = st.selectbox(
                "动-静结合参数校准方法",
                [
                    "方法一：物质平衡校准有效孔隙度 (基于排卤量 Wp 与压降 Δp)",
                    "方法二：渗流压差反演校准有效厚度 (基于日产量 qw 与流压差 Δp_wf)",
                ],
            )

        st.markdown("#### 2. 物性与几何参数录入")
        if "模式一" in global_mc_mode:
            st.caption("🔍 **模式一：多井样本获取方式**（支持直接粘贴文本或上传单井测井解释台账文件）：")
            mc_input_type = st.radio("选择多井数据来源：", ["上传本构造带多井实测台账文件 (Excel/CSV)", "手动文本框粘贴序列"], horizontal=True)

            u_phi_str = ""
            u_c_str = ""
            u_phi_single = 0.08
            u_c_single = 0.0185

            if "上传" in mc_input_type:
                col_dl_well, col_up_well = st.columns([1, 3])
                with col_dl_well:
                    # 示例文件下载
                    well_sample_df = pd.DataFrame({
                        "井号": ["Well-1", "Well-2", "Well-3", "Well-4", "Well-5", "Well-6"],
                        "测试层段": ["T3l-1", "T3l-1", "T3l-2", "T3l-1", "T3l-2", "T3l-1"],
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
                    well_file = st.file_uploader("上传多井台账文件 (包含孔隙度与品位列)", type=["csv", "xlsx", "xls"], key="well_table_up")
                    if well_file is not None:
                        f_phi, f_c, df_well_preview = extract_samples_from_file(well_file)
                        if df_well_preview is not None:
                            st.success(f"已成功识别文件！读取到 {len(f_phi)} 个孔隙度样本，{len(f_c)} 个品位样本。")
                            st.dataframe(df_well_preview.head(4), use_container_width=True)
                            u_phi_str = ", ".join([str(v) for v in f_phi])
                            u_c_str = ", ".join([str(v) for v in f_c])
            else:
                ib1, ib2 = st.columns(2)
                with ib1:
                    u_phi_str = st.text_area(
                        "多井有效孔隙度序列 ϕ (如: 0.065, 0.082, 0.071, 0.095, 0.078, 0.088)",
                        value="0.065, 0.082, 0.071, 0.095, 0.078, 0.088",
                        height=70
                    )
                with ib2:
                    u_c_str = st.text_area(
                        "多井 KCl 品位序列 C (t/m³，如: 0.0175, 0.0192, 0.0210, 0.0185)",
                        value="0.0175, 0.0192, 0.0210, 0.0185",
                        height=70
                    )
        else:
            st.caption("🔍 **模式二：单点/少井参数录入**：井数较少时，直接输入单井解释代表值：")
            ib3, ib4 = st.columns(2)
            with ib3:
                u_phi_single = st.number_input("储层平均有效孔隙度 ϕ (小数)", value=0.078, step=0.005, format="%.3f")
                u_phi_str = ""
            with ib4:
                u_c_single = st.number_input("KCl 平均品位 C (t/m³)", value=0.0190, step=0.0010, format="%.4f")
                u_c_str = ""

        ig1, ig2, ig3 = st.columns(3)
        with ig1:
            u_A = st.number_input("储层平面面积 A (km²)", value=55.0, step=1.0)
        with ig2:
            has_v = st.checkbox("已有三维地震雕刻体积 V", value=False)
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
                u_head = st.number_input("承压水头标高 h (m)", value=340.0, step=10.0)
                u_top = st.number_input("储层顶面标高 H (m)", value=-2250.0, step=50.0)
                u_sw, u_bw = None, None

        st.markdown("#### 3. 动态试采参数（可选）")
        enable_dyn_single = st.checkbox("输入动态试采参数进行动静校准", value=True)
        u_wp, u_dp, u_ct = None, None, None
        u_qw, u_dp_flow, u_k, u_mu = None, None, None, None

        if enable_dyn_single:
            if "方法一" in u_dyn_method:
                id1, id2, id3 = st.columns(3)
                with id1:
                    u_wp = st.number_input("累计排卤量 Wp (m³)", value=32000.0, step=1000.0)
                with id2:
                    u_dp = st.number_input("折算地层静态压降 Δp (MPa)", value=3.0, step=0.1)
                with id3:
                    u_ct = st.number_input("综合压缩系数 Ct (10⁻⁴ MPa⁻¹)", value=8.5, step=0.5)
            else:
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
                "多井孔隙度序列": u_phi_str,
                "多井品位序列": u_c_str,
                "phi": u_phi_single,
                "C": u_c_single,
                "A": u_A, "h": u_h, "Sw": u_sw, "Bw": u_bw,
                "S": u_S, "承压水头标高": u_head, "储层顶面标高": u_top,
                "Wp": u_wp, "dp": u_dp, "ct": u_ct,
                "qw": u_qw, "dp_flow": u_dp_flow, "k": u_k, "mu": u_mu,
            }
            if u_V is not None:
                input_dict["V"] = u_V

            single_rec, mc_artifacts = evaluate_and_archive_unit(
                input_dict,
                assigned_topic=active_topic,
                default_dyn_method=input_dict["动态方法"],
                mc_mode=global_mc_mode,
                n_sim=global_n_sim,
                dist_type=global_dist_choice,
                phi_err_manual=global_phi_err,
                c_err_manual=global_c_err,
            )
            st.session_state.temp_single_record = single_rec
            st.session_state.temp_mc_artifacts = mc_artifacts

        # 结果展示
        if st.session_state.temp_single_record is not None:
            cur_rec = st.session_state.temp_single_record
            cur_art = st.session_state.temp_mc_artifacts

            st.markdown("---")
            st.subheader(f"📊 当前推演核定结果：{cur_rec['构造带/单元名称']}")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("基准孔隙度", cur_rec['基准孔隙度'])
            r2.metric("动态校准后孔隙度", cur_rec['校准后孔隙度'])
            r3.metric("最终核定 KCl 储量", f"{cur_rec['最终核定KCl储量(万吨)']:,.2f} 万吨", delta=cur_rec['校准前后储量变化率'])
            r4.metric("核定卤水体积", f"{cur_rec['核定卤水体积(亿m³)']:.4f} 亿m³")

            st.markdown(f"**当前使用的蒙特卡洛模式**：`{cur_rec.get('蒙特卡洛模式', '自适应模式')}` | **分布特征**：`{cur_rec.get('孔隙度分布特征', '标准分布')}`")

            plot_c1, plot_c2 = st.columns(2)
            with plot_c1:
                phi_samples = cur_art["phi_fit"]["samples"]
                fig_fit = px.histogram(
                    x=cur_art["phi_draws"] * 100, nbins=50, histnorm='probability density',
                    labels={"x": "孔隙度 (%)"},
                    title="孔隙度不确定性抽样分布"
                )
                if len(phi_samples) > 0:
                    for p_s in phi_samples:
                        fig_fit.add_vline(x=p_s * 100, line_dash="dot", line_color="orange")
                st.plotly_chart(fig_fit, use_container_width=True)

            with plot_c2:
                fig_p = px.histogram(
                    x=cur_art["P_draws"], nbins=60,
                    labels={"x": "KCl 储量 (万吨)"},
                    title="最终 KCl 储量概率分布 (P10 - P50 - P90)"
                )
                fig_p.add_vline(x=cur_rec["蒙特卡洛P90(万吨)"], line_dash="dash", line_color="orange", annotation_text=f"P90: {cur_rec['蒙特卡洛P90(万吨)']}万吨")
                fig_p.add_vline(x=cur_rec["蒙特卡洛P50(万吨)"], line_dash="solid", line_color="green", annotation_text=f"P50: {cur_rec['蒙特卡洛P50(万吨)']}万吨")
                fig_p.add_vline(x=cur_rec["蒙特卡洛P10(万吨)"], line_dash="dash", line_color="red", annotation_text=f"P10: {cur_rec['蒙特卡洛P10(万吨)']}万吨")
                st.plotly_chart(fig_p, use_container_width=True)

            with st.expander("🔍 查看本单元参数校准细节与容积法推导过程", expanded=True):
                st.write(f"**动态校准过程**：{cur_rec.get('动静推演计算过程', '无')}")
                st.write(f"**校准后容积法公式**：{cur_rec.get('校准后容积法推演', '无')}")

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
        st.markdown("#### 批量上传多构造带（自适应单点与多井混合数据）")
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            batch_dyn_def = st.selectbox(
                "默认参数校准方法（未在表格指定时生效）：",
                ["方法一：物质平衡校准有效孔隙度", "方法二：渗流压差反演校准有效厚度"],
            )
        with col_t2:
            sample_template = pd.DataFrame([
                {
                    "构造带": f"{active_topic[:2]}成熟构造1号", "储层类型": "油气同层型", "动态方法": "方法一",
                    "多井孔隙度序列": "0.065, 0.082, 0.071, 0.095, 0.078",
                    "多井品位序列": "0.0175, 0.0192, 0.0210, 0.0185",
                    "A": 50.0, "h": 22.0, "Sw": 0.65, "Bw": 1.02,
                    "Wp": 40000, "dp": 3.5, "ct": 8.2,
                },
                {
                    "构造带": f"{active_topic[:2]}新探区构造2号(单井)", "储层类型": "非油气同层型", "动态方法": "方法二",
                    "phi": 0.068, "C": 0.021, "A": 75.0, "V": 1800.0, "S": 0.0004,
                    "承压水头标高": 350.0, "储层顶面标高": -2300.0,
                    "qw": 220.0, "dp_flow": 4.8, "k": 3.2, "mu": 1.1,
                },
                {
                    "构造带": f"{active_topic[:2]}早期外推圈闭3号", "储层类型": "油气同层型",
                    "phi": 0.075, "C": 0.016, "A": 28.0, "h": 16.0,
                },
            ])
            csv_buf = io.BytesIO()
            sample_template.to_csv(csv_buf, index=False, encoding="utf_8_sig")
            st.download_button(
                label="📥 下载含多井与单井混合模板 (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"{active_topic}_混合数据测试模板.csv",
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
                        row_dict = row.to_dict()
                        auto_mode = "多井实测样本拟合模式" if pd.notna(row_dict.get("多井孔隙度序列")) else "单点/少井先验扰动模式"
                        rec, _ = evaluate_and_archive_unit(
                            row_dict,
                            assigned_topic=active_topic,
                            default_dyn_method=method_code,
                            mc_mode=auto_mode,
                            n_sim=global_n_sim,
                            dist_type=global_dist_choice,
                            phi_err_manual=global_phi_err,
                            c_err_manual=global_c_err,
                        )
                        batch_records.append(rec)
                        st.session_state.topic_archives[active_topic].append(rec)

                    st.success(f"批量推演完成！已成功将 {len(batch_records)} 个单元按自适应模式写入【{active_topic}】总账！")
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
            "蒙特卡洛模式", "孔隙度分布特征", "动态参数校准说明",
            "原始静态KCl储量(万吨)", "最终核定KCl储量(万吨)", "蒙特卡洛P50(万吨)",
            "校准后容积法推演"
        ]
        
        valid_cols = [col for col in preferred_cols if col in df_arc.columns]
        if not valid_cols:
            valid_cols = list(df_arc.columns)
            
        st.dataframe(df_arc[valid_cols], use_container_width=True)

        x_col = "构造带/单元名称" if "构造带/单元名称" in df_arc.columns else df_arc.columns[0]
        y_col = "最终核定KCl储量(万吨)" if "最终核定KCl储量(万吨)" in df_arc.columns else df_arc.columns[1]
        color_col = "动静结合方法" if "动静结合方法" in df_arc.columns else None

        fig_unit_bar = px.bar(
            df_arc,
            x=x_col,
            y=y_col,
            text=y_col,
            color=color_col,
            title=f"【{active_topic}】各计算单元核定储量 (目标配额: {my_target/1e4:.0f}万吨)",
        )
        fig_unit_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_unit_bar, use_container_width=True)

        csv_archive_out = df_arc.to_csv(index=False).encode("utf_8_sig")
        st.download_button(
            label=f"📥 导出【{active_topic}】全要素归档报表 (CSV)",
            data=csv_archive_out,
            file_name=f"{active_topic}_双模式储量评价归档详单.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# TAB 3: 第三步 理论指南
# --------------------------------------------------
with tab_docs:
    st.subheader("双模式蒙特卡洛与动-静结合方法指南")

    st.markdown("### 1. 蒙特卡洛建模双模式设计准则")
    st.markdown("""
    - **模式一：多井实测样本拟合（成熟探区）**
      当同一构造带具备 3 口或以上钻井资料时，系统采用地质统计学理论，自动拟合**对数正态分布 (Lognormal)** 或 **正态分布 (Normal)**。支持上传专属的多井测井解释台账表。
    - **模式二：单点/少井先验扰动（早期新区/少井区）**
      对于仅有 1~2 口井甚至区域类比参数的单元，系统允许用户输入单点均值，并通过先验三角分布与自定义相对误差（如 \(\pm 20\%\)）实现不确定性量化。
    """)

    st.markdown("### 2. 动-静结合参数校准机理")
    st.write("#### 方法一：物质平衡有效孔隙连通校正")
    st.latex(r"\phi_{dyn} = \frac{W_p}{c_t \cdot \Delta p \cdot V}")
    st.write("以动态反求孔隙度校正静态解释孔隙度。")

    st.write("#### 方法二：渗流阻抗有效厚度校正")
    st.latex(r"h_{dyn} = \frac{q_w \cdot \mu_w \cdot \ln(R_e/r_w)}{2\pi \cdot k \cdot \Delta p_{wf}}")
    st.write("以单井稳态产流剖面校正静态电性解释有效厚度。")

    st.markdown("---")
    st.markdown("### 3. 各专题增储任务配额")
    targets_table = [
        {"专题名称": "川中专题", "新增KCl目标配额": "2000 万吨", "主要勘探层系/靶区": "震旦系灯影组、三叠系雷口坡组/嘉陵江组等"},
        {"专题名称": "川东北专题", "新增KCl目标配额": "1500 万吨", "主要勘探层系/靶区": "三叠系嘉陵江组、雷口坡组、飞仙关组等"},
        {"专题名称": "川西专题", "新增KCl目标配额": "1000 万吨", "主要勘探层系/靶区": "二叠系栖霞-茅口组、三叠系雷口坡组等"},
        {"专题名称": "川南专题", "新增KCl目标配额": "500 万吨", "主要勘探层系/靶区": "奥陶系宝塔组、三叠系雷口坡组、嘉陵江组等"},
    ]
    st.table(pd.DataFrame(targets_table))
