import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

--------------------------------------------------

页面全局配置

--------------------------------------------------

st.set_page_config(
page_title="深层卤水钾盐动-静综合资源评价系统",
page_icon="🧂",
layout="wide",
initial_sidebar_state="expanded"
)

--------------------------------------------------

会话状态初始化 (用于跨构造带协同存储)

--------------------------------------------------

if "zones_data" not in st.session_state:
st.session_state.zones_data =

全盆地增储目标常数：5000 万吨 KCl

TARGET_KCL_TONS = 50_000_000.0

--------------------------------------------------

侧边栏：协同管理与 5000 万吨目标追踪看板

--------------------------------------------------

with st.sidebar:
st.title("🧂 深层卤水钾盐评价")
st.caption("四川盆地深层钾盐资源动-静评价专用系统")
st.markdown("---")

st.subheader("📍 专题与构造单元选择")
region_choice = st.selectbox(
    "当前负责专题区块",
    ["川中专题", "川西专题", "川东北专题", "川南专题"]
)

st.markdown("---")
st.subheader("🎯 5000万吨新增增储目标")

total_added_kcl = sum([z["P_final"] for z in st.session_state.zones_data])
progress_pct = min(100.0, (total_added_kcl / TARGET_KCL_TONS) * 100) if TARGET_KCL_TONS > 0 else 0.0

st.metric(
    label="已累计核定 KCl 资源量",
    value=f"{total_added_kcl / 1e4:,.2f} 万吨",
    delta=f"距目标差额: {(TARGET_KCL_TONS - total_added_kcl)/1e4:,.2f} 万吨" if total_added_kcl < TARGET_KCL_TONS else "🎉 已圆满达成增储目标！"
)
st.progress(progress_pct / 100.0)
st.caption(f"当前整体推进进度：**{progress_pct:.2f}%** / 100.00%")

st.markdown("---")
if st.button("🗑️ 清空所有已保存数据", use_container_width=True):
    st.session_state.zones_data = []
    st.rerun()



--------------------------------------------------

主界面 Tabs：计算工作台、详细操作说明书、总账报表

--------------------------------------------------

tab_calc, tab_docs, tab_summary = st.tabs(

$$"📊 储量动态核算工作台", "📖 详细操作说明书与理论依据", "📋 四大专题总账与成果汇总"$$

)

==================================================

TAB 1: 详细操作说明书

==================================================

with tab_docs:
st.markdown("""
## 深层卤水钾盐动-静结合综合资源评价系统使用指南

本系统面向四川盆地深层富钾卤水勘查专题设计，旨在为川中、川西、川东北、川南四大专题组提供统一规范的储量计算与动-静校准平台。



一、 静态容积法数学模型

1. 油气同层型卤水（伴生卤水构造）

适用于储卤层中油/气与富钾卤水共生的层系，计算公式为：

$$Q_w = \\frac{A_w \\cdot h \\cdot \\phi \\cdot S_w}{B_w}$$

$$P_A = Q_w \\cdot C$$

$Q_w$：卤水储存量（$\text{m}^3$）

$A_w$：含卤水面积（$\text{m}^2$）

$h$：储卤层有效厚度（$\text{m}$）

$\phi$：储卤层有效孔隙度（小数）

$S_w$：含水饱和度（小数）

$B_w$：卤水地层体积系数（无因次）

$C$：氯化钾（$\text{KCl}$）平均品位（$\text{t/m}^3$）

$P_A$：卤水钾盐（$\text{KCl}$）储存量（$\text{t}$）

2. 非油气同层型卤水（纯承压含水层构造）

适用于非油气伴生的深部承压储水层系，考虑孔隙储水与高承压顶板弹性压释储水：

$$Q_{ws} = \\phi \\cdot V + S \\cdot (h - H) \\cdot A$$

$$P_A = Q_{ws} \\cdot C$$

$Q_{ws}$：卤水储存量（$\text{m}^3$）

$\phi$：储卤层岩石孔隙度（小数）

$V$：储卤层体积（$\text{m}^3$）；在资料丰富时若已由地震多属性反演精细雕刻空间几何体，可直接采用体雕刻值 $V$；若为常规资料，则通常由 $V = A \cdot h$ 计算。

$S$：卤水层的弹性储水系数（弹性给水度，无因次）

$h$：平均承压水头标高（$\text{m}$）

$H$：平均储卤层顶面标高（$\text{m}$），其中 $(h - H)$ 代表净弹性承压水头高度。

$A$：储卤层平面面积（$\text{m}^2$）

$C$：氯化钾（$\text{KCl}$）平均品位（$\text{t/m}^3$）

$P_A$：卤水钾盐储存量（$\text{t}$）

二、 资料丰富度分级评估工作流

根据地质勘探资料完备程度，系统自动提供三个等级的计算层级：

资料匮乏阶段（勘探早期/外围区）

地质特征：无三维地震精细解释，缺乏长期排卤试采数据。

推荐方案：纯静态确定性容积法，直接利用区域钻井统计参数核算基准体积；支持选配轻量蒙特卡洛模拟评估风险区间。

资料中等阶段（勘探中期/评价区）

地质特征：具备二维/三维构造图，具备多口井孔隙度与品位化验数据。

推荐方案：精细雕刻容积法 + 蒙特卡洛模拟，得到 P90（保守）、P50（适中）、P10（乐观）概率储量。

资料丰富阶段（开发准备期/试采示范区）

地质特征：具备叠前三维体雕刻体积，拥有排卤试采折算压降曲线。

推荐方案：静态精细容积法 $\rightarrow$ 蒙特卡洛模拟 $\rightarrow$ 动态压降物质平衡约束。

三、 动-静结合物质平衡约束算法

在深层封闭储卤层中，静态容积法评估的是静态孔隙总容积，而开发受制于断层遮挡、连通性及渗透性。通过单井或井组试采数据，基于弹性压释物质平衡方程：

$$Q_{dyn} = \\frac{W_p}{c_t \\cdot \\Delta p}$$

* \(W_p\)：试采累计排卤出水量（\(\\text{m}^3\)）
* \(c_t\)：储层综合压缩系数（\(\\text{MPa}^{-1}\)，含岩石基质与水体压缩系数）
* \(\\Delta p\)：折算地层静态压力降（\(\\text{MPa}\)）

根据动态控制水体与静态体积之比构建**动-静连通有效性系数** \(\\alpha = \\min\\left(1.0, \\frac{Q_{dyn}}{Q_{stat}}\\right)\)，对静态资源量进行科学折算。
""")



==================================================

TAB 2: 计算工作台

==================================================

with tab_calc:
st.subheader(f"🛠️ 构造带评价与计算 —— 【{region_choice}】")

# 顶部元数据录入
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    zone_name = st.text_input("构造带/圈闭名称", value=f"{region_choice.replace('专题','')}某构造T63层")
with m_col2:
    res_type = st.selectbox("储卤层类型", ["油气同层型", "非油气同层型"])
with m_col3:
    richness_level = st.select_slider(
        "数据资料丰富度阶段",
        options=[
            "资料匮乏（仅确定性容积法）",
            "资料中等（容积法 + 蒙特卡洛）",
            "资料丰富（精细雕刻 + 蒙特卡洛 + 动态约束）"
        ],
        value="资料丰富（精细雕刻 + 蒙特卡洛 + 动态约束）"
    )
    
st.markdown("---")
st.markdown("#### 第一阶段：静态容积法参数录入")

# 品位公共参数
c_grade_col, note_col = st.columns([1, 2])
with c_grade_col:
    grade_C = st.number_input(
        "KCl 平均品位 C (t/m³)",
        value=0.0180,
        step=0.0010,
        format="%.4f",
        help="注意：1 g/L = 1 kg/m³ = 0.001 t/m³。例如 18 g/L 应输入 0.0180 t/m³"
    )
with note_col:
    st.caption("📌 **品位换算提示**：地下深层卤水化验报告常见单位为 g/L 或 mg/L，输入时请规范转换为 \(\\text{t/m}^3\)（如 \(15\\sim 25\\text{ g/L}\) 对应 \(0.015\\sim 0.025\\text{ t/m}^3\)）。")

# 根据类型区分容积法输入
if res_type == "油气同层型":
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        area_km2 = st.number_input("含卤水面积 Aw (km²)", value=50.0, step=1.0)
        area_m2 = area_km2 * 1e6
    with c2:
        thick_h = st.number_input("储卤层有效厚度 h (m)", value=30.0, step=1.0)
    with c3:
        phi = st.number_input("有效孔隙度 ϕ (小数)", value=0.075, min_value=0.001, max_value=0.40, step=0.005, format="%.3f")
    with c4:
        sw = st.number_input("含水饱和度 Sw (小数)", value=0.60, min_value=0.01, max_value=1.00, step=0.05)
    with c5:
        bw = st.number_input("卤水体积系数 Bw", value=1.025, min_value=0.90, max_value=1.50, step=0.005, format="%.3f")
        
    # 计算静态卤水量与 KCl 储量
    Q_static = (area_m2 * thick_h * phi * sw) / bw
    P_static = Q_static * grade_C

else:
    # 非油气同层型
    c1, c2, c3, c4 = st.columns(4)
    seismic_carved = False
    
    if "精细雕刻" in richness_level:
        seismic_carved = st.checkbox("已具备三维地震精细阻抗反演雕刻储层体 (直接输入储层体积 V)", value=True)
        
    with c1:
        area_km2 = st.number_input("储层平面面积 A (km²)", value=65.0, step=1.0)
        area_m2 = area_km2 * 1e6
        
    with c2:
        if seismic_carved:
            vol_million_m3 = st.number_input("精细雕刻储层体积 V (10⁶ m³)", value=1800.0, step=50.0)
            vol_V_m3 = vol_million_m3 * 1e6
        else:
            thick_pure_h = st.number_input("储层有效厚度 h (m)", value=35.0, step=1.0)
            vol_V_m3 = area_m2 * thick_pure_h
            
    with c3:
        phi = st.number_input("岩石孔隙率 φ (小数)", value=0.068, min_value=0.001, max_value=0.40, step=0.005, format="%.3f")
    with c4:
        storage_S = st.number_input("弹性储水系数/给水度 S", value=0.00045, step=0.00005, format="%.5f")
        
    c5, c6, _ = st.columns([1, 1, 2])
    with c5:
        head_h = st.number_input("平均承压水头标高 h (m)", value=400.0, step=10.0)
    with c6:
        top_H = st.number_input("平均储卤层顶面标高 H (m)", value=-2300.0, step=50.0)
        
    head_diff = max(0.0, head_h - top_H)
    Q_static = (phi * vol_V_m3) + (storage_S * head_diff * area_m2)
    P_static = Q_static * grade_C

st.success(
    f"✅ **基础容积法确定性计算结果**：卤水总储存量 \(Q\) = **{Q_static/1e8:.4f}** 亿立方米 | "
    f"KCl 静态资源量 \(P_A\) = **{P_static/1e4:,.2f}** 万吨"
)

# --------------------------------------------------
# 第二阶段：蒙特卡洛模拟
# --------------------------------------------------
mc_executed = False
p50_kcl = P_static
p_sim_results = None

if "蒙特卡洛" in richness_level:
    st.markdown("---")
    st.markdown("#### 第二阶段：蒙特卡洛随机模拟（参数不确定性分析）")
    
    with st.expander("🎲 蒙特卡洛随机模拟控制台", expanded=True):
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            n_sim = st.select_slider("随机抽样模拟次数", options=[1000, 5000, 10000, 20000], value=10000)
        with mc2:
            phi_fluct = st.slider("孔隙度相对波动范围 (±%)", 5, 50, 20)
        with mc3:
            grade_fluct = st.slider("品位 C 相对波动范围 (±%)", 5, 50, 15)

        if st.button("▶️ 运行蒙特卡洛概率模拟", type="primary"):
            mc_executed = True
            np.random.seed(1024)
            
            # 品位三角分布抽样
            c_samples = np.random.triangular(
                grade_C * (1 - grade_fluct / 100),
                grade_C,
                grade_C * (1 + grade_fluct / 100),
                n_sim
            )
            
            # 孔隙度三角分布抽样
            phi_samples = np.random.triangular(
                phi * (1 - phi_fluct / 100),
                phi,
                phi * (1 + phi_fluct / 100),
                n_sim
            )
            
            if res_type == "油气同层型":
                sw_samples = np.random.normal(sw, sw * 0.05, n_sim)
                sw_samples = np.clip(sw_samples, 0.01, 1.0)
                q_samples = (area_m2 * thick_h * phi_samples * sw_samples) / bw
            else:
                q_samples = (phi_samples * vol_V_m3) + (storage_S * head_diff * area_m2)
                
            p_sim_results = (q_samples * c_samples) / 1e4  # 转换为万吨
            
            p90_val = np.percentile(p_sim_results, 10)  # 保守
            p50_val = np.percentile(p_sim_results, 50)  # 中值
            p10_val = np.percentile(p_sim_results, 90)  # 乐观
            p50_kcl = p50_val * 1e4
            
            col_mc_fig, col_mc_stat = st.columns([3, 1])
            with col_mc_fig:
                fig_mc = px.histogram(
                    x=p_sim_results,
                    nbins=50,
                    marginal="box",
                    labels={"x": "KCl 储量 (万吨)"},
                    title=f"{zone_name} - 蒙特卡洛资源量累积频率与不确定性分布",
                    color_discrete_sequence=["#1f77b4"]
                )
                fig_mc.add_vline(x=p90_val, line_dash="dash", line_color="#ff7f0e", annotation_text=f"P90: {p90_val:.1f}万吨")
                fig_mc.add_vline(x=p50_val, line_dash="solid", line_color="#2ca02c", annotation_text=f"P50: {p50_val:.1f}万吨")
                fig_mc.add_vline(x=p10_val, line_dash="dash", line_color="#d62728", annotation_text=f"P10: {p10_val:.1f}万吨")
                fig_mc.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350)
                st.plotly_chart(fig_mc, use_container_width=True)
                
            with col_mc_stat:
                st.markdown("##### 模拟统计特征")
                st.metric("P90 (保守型)", f"{p90_val:,.1f} 万吨")
                st.metric("P50 (期望中值)", f"{p50_val:,.1f} 万吨")
                st.metric("P10 (乐观型)", f"{p10_val:,.1f} 万吨")

# --------------------------------------------------
# 第三阶段：动态压降与物质平衡约束
# --------------------------------------------------
final_q = Q_static
final_p = p50_kcl if mc_executed else P_static
is_dynamic_applied = False

if "动态约束" in richness_level:
    st.markdown("---")
    st.markdown("#### 第三阶段：动态试采压降与物质平衡约束校准")
    
    with st.expander("📉 试水试采压降动态拟合约束", expanded=True):
        d1, d2, d3 = st.columns(3)
        with d1:
            wp = st.number_input("试采阶段累计排卤量 Wp (m³)", value=42000.0, step=2000.0)
        with d2:
            delta_p = st.number_input("储层对应折算总压降 Δp (MPa)", value=3.5, min_value=0.01, step=0.1)
        with d3:
            ct_input = st.number_input("地层综合压缩系数 Ct (10⁻⁴ MPa⁻¹)", value=8.0, step=0.5)
            ct = ct_input * 1e-4

        # 动态弹性压释体积
        Q_dyn = wp / (ct * delta_p)
        
        dc1, dc2 = st.columns([2, 2])
        with dc1:
            st.write(f"📊 **试采动态有效控水体积 (\(Q_{{dyn}}\))**：**{Q_dyn / 1e8:.4f}** 亿立方米")
        with dc2:
            apply_dyn = st.checkbox("确认应用动态有效连通系数校正静态容积法储量", value=True)
            
        if apply_dyn:
            is_dynamic_applied = True
            alpha = min(1.0, max(0.05, Q_dyn / Q_static))
            st.info(f"🔗 **动-静连通有效度系数 (α = Qdyn / Qstat)** 评估为：**{alpha:.3f}** (折算连通率约 {alpha*100:.1f}%)")
            final_q = Q_static * alpha
            final_p = final_p * alpha

# --------------------------------------------------
# 第四阶段：成果核定与汇总录入
# --------------------------------------------------
st.markdown("---")
st.markdown("#### 第四阶段：构造带成果核定与录入总账")

r1, r2, r3 = st.columns([2, 2, 1])
with r1:
    st.metric(
        label="最终核定卤水体积 (Q)",
        value=f"{final_q / 1e8:.4f} 亿 m³",
        delta="动态折减校正" if is_dynamic_applied else "静态基准"
    )
with r2:
    st.metric(
        label="最终核定 KCl 储量 (P)",
        value=f"{final_p / 1e4:,.2f} 万吨",
        delta=f"占 5000万吨总目标的 {(final_p / TARGET_KCL_TONS) * 100:.2f}%"
    )
with r3:
    if st.button("💾 确认并录入此构造带", type="primary", use_container_width=True):
        st.session_state.zones_data.append({
            "区块": region_choice,
            "构造带名称": zone_name,
            "储层类型": res_type,
            "资料完备度": richness_level,
            "KCl品位(t/m³)": grade_C,
            "卤水储存量(亿m³)": round(final_q / 1e8, 4),
            "KCl储存量(万吨)": round(final_p / 1e4, 2),
            "动态校正": "是" if is_dynamic_applied else "否",
            "P_final": final_p
        })
        st.toast(f"✅ 成功录入：{region_choice} - {zone_name}！")
        st.rerun()



==================================================

TAB 3: 构造带总账与成果汇总

==================================================

with tab_summary:
st.subheader("📋 四川盆地深层卤水钾盐储量综合核算总台账")

if len(st.session_state.zones_data) == 0:
    st.info("💡 当前总账为空。请先在【储量动态核算工作台】中完成参数测算并点击【确认并录入此构造带】。")
else:
    df_all = pd.DataFrame(st.session_state.zones_data)
    
    # 显示明细表
    st.dataframe(df_all.drop(columns=["P_final"]), use_container_width=True)
    
    st.markdown("---")
    # 统计图表展示
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        region_group = df_all.groupby("区块")["KCl储存量(万吨)"].sum().reset_index()
        fig_bar = px.bar(
            region_group,
            x="区块",
            y="KCl储存量(万吨)",
            color="区块",
            text="KCl储存量(万吨)",
            title="各专题区块 KCl 累计核定储存量 (万吨)",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with chart_col2:
        fig_pie = px.pie(
            df_all,
            names="区块",
            values="KCl储存量(万吨)",
            title="四大专题储量贡献占比",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 数据下载功能
    csv_file = df_all.drop(columns=["P_final"]).to_csv(index=False).encode("utf_8_sig")
    st.download_button(
        label="📥 导出全套成果核算报表 (CSV 格式)",
        data=csv_file,
        file_name="深层卤水钾盐动静结合储量核算总账.csv",
        mime="text/csv",
        type="primary"
    )

