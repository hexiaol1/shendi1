import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

# --------------------------------------------------
# 全局页面配置
# --------------------------------------------------
st.set_page_config(
    page_title="深层卤水钾盐动-静综合资源评价自适应系统",
    page_icon="🧂",
    layout="wide",
    initial_sidebar_state="expanded"
)

TARGET_KCL_TONS = 50_000_000.0  # 全盆地新增 5000 万吨目标

if "stored_evaluations" not in st.session_state:
    st.session_state.stored_evaluations = []

# --------------------------------------------------
# 核心自适应计算引擎
# --------------------------------------------------
def evaluate_brine_record(row_data, run_mc=True, n_sim=5000):
    res = {}
    r = {str(k).strip(): v for k, v in row_data.items() if pd.notna(v)}
    
    res["构造带"] = str(r.get("构造带", r.get("构造名称", "未命名构造")))
    res["区块"] = str(r.get("区块", r.get("专题", "川中专题")))
    res["储层类型"] = str(r.get("储层类型", r.get("类型", "油气同层型"))).strip()
    
    # 品位 C (t/m3)
    C = float(r.get("C", r.get("KCl品位", r.get("品位", 0.018))))
    res["KCl品位(t/m³)"] = C
    
    # 面积 A (km2 -> m2)
    A_km2 = float(r.get("A", r.get("Aw", r.get("面积", 0.0))))
    A_m2 = A_km2 * 1e6
    
    # 厚度 h (m)
    h = float(r.get("h", r.get("厚度", r.get("有效厚度", 0.0))))
    
    # 孔隙度 phi (支持百分数或小数输入)
    phi = float(r.get("phi", r.get("ϕ", r.get("孔隙度", r.get("孔隙率", 0.08)))))
    if phi > 1.0:
        phi = phi / 100.0
        
    # 判断是否具备地震精细雕刻体积 V (10^6 m3)
    has_seismic_vol = ("V" in r) or ("雕刻体积" in r) or ("储层体积" in r)
    if has_seismic_vol:
        V_raw = float(r.get("V", r.get("雕刻体积", r.get("储层体积", 0.0))))
        V_m3 = V_raw * 1e6 if V_raw < 1e5 else V_raw
        res["几何建模方式"] = "地震精细三维雕刻体积 V"
    else:
        V_m3 = A_m2 * h
        res["几何建模方式"] = "面积×厚度估算 (A×h)"

    # 1. 容积法基准计算
    if "非油气" in res["储层类型"]:
        res["计算模型"] = "非油气同层型"
        S = float(r.get("S", r.get("弹性储水系数", 0.0005)))
        head_h = float(r.get("承压水头标高", r.get("h_head", 350.0)))
        top_H = float(r.get("储层顶面标高", r.get("H_top", -2200.0)))
        elastic_head = max(0.0, head_h - top_H)
        
        # 公式: Q = phi * V + S * (h - H) * A
        Q_static = (phi * V_m3) + (S * elastic_head * A_m2)
    else:
        res["计算模型"] = "油气同层型"
        Sw = float(r.get("Sw", r.get("含水饱和度", 0.65)))
        if Sw > 1.0: Sw /= 100.0
        Bw = float(r.get("Bw", r.get("体积系数", 1.02)))
        
        # 公式: Q = (A * h * phi * Sw) / Bw
        if has_seismic_vol:
            Q_static = (V_m3 * phi * Sw) / Bw
        else:
            Q_static = (A_m2 * h * phi * Sw) / Bw
            
    P_static = Q_static * C
    res["静态卤水体积(亿m³)"] = round(Q_static / 1e8, 4)
    res["静态KCl储量(万吨)"] = round(P_static / 1e4, 2)
    
    # 2. 动态参数感知与校准计算
    has_dynamic = ("Wp" in r or "排卤量" in r or "累计产水量" in r) and ("dp" in r or "压降" in r or "Δp" in r)
    if has_dynamic:
        Wp = float(r.get("Wp", r.get("排卤量", r.get("累计产水量", 0.0))))
        dp = float(r.get("dp", r.get("压降", r.get("Δp", 1.0))))
        ct_raw = float(r.get("ct", r.get("Ct", r.get("压缩系数", 8.5))))
        ct = ct_raw * 1e-4 if ct_raw > 1e-2 else ct_raw
        
        if dp > 0 and ct > 0:
            Q_dynamic = Wp / (ct * dp)
            alpha = min(1.0, max(0.05, Q_dynamic / Q_static)) if Q_static > 0 else 1.0
            res["动态约束状态"] = f"已约束 (动静连通比 α={alpha:.3f})"
            Q_constrained = Q_static * alpha
            P_constrained = P_static * alpha
        else:
            res["动态约束状态"] = "数据异常未启用"
            Q_constrained = Q_static
            P_constrained = P_static
            alpha = 1.0
    else:
        res["动态约束状态"] = "无动态数据(使用纯静态)"
        Q_constrained = Q_static
        P_constrained = P_static
        alpha = 1.0
        
    res["最终核定卤水体积(亿m³)"] = round(Q_constrained / 1e8, 4)
    res["最终核定KCl储量(万吨)"] = round(P_constrained / 1e4, 2)
    res["P_final_raw"] = P_constrained

    # 3. 蒙特卡洛模拟
    mc_results = {}
    if run_mc:
        np.random.seed(42)
        phi_err = float(r.get("phi_err", r.get("孔隙度相对误差", 0.20)))
        c_err = float(r.get("c_err", r.get("品位相对误差", 0.15)))
        
        phi_samples = np.random.triangular(phi * (1 - phi_err), phi, phi * (1 + phi_err), n_sim)
        c_samples = np.random.triangular(C * (1 - c_err), C, C * (1 + c_err), n_sim)
        
        if "非油气" in res["储层类型"]:
            q_samples = (phi_samples * V_m3) + (S * elastic_head * A_m2)
        else:
            if has_seismic_vol:
                q_samples = (V_m3 * phi_samples * Sw) / Bw
            else:
                q_samples = (A_m2 * h * phi_samples * Sw) / Bw
                
        p_samples = (q_samples * c_samples * alpha) / 1e4
        mc_results = {
            "P90(万吨)": round(float(np.percentile(p_samples, 10)), 2),
            "P50(万吨)": round(float(np.percentile(p_samples, 50)), 2),
            "P10(万吨)": round(float(np.percentile(p_samples, 90)), 2),
            "samples": p_samples
        }
        res["P90保守储量(万吨)"] = mc_results["P90(万吨)"]
        res["P50中值储量(万吨)"] = mc_results["P50(万吨)"]
        res["P10乐观储量(万吨)"] = mc_results["P10(万吨)"]
        
    return res, mc_results

# --------------------------------------------------
# 页面布局与侧边栏
# --------------------------------------------------
with st.sidebar:
    st.title("🧂 增储总目标监控台")
    st.caption("四川盆地深层富钾卤水重点研发专项")
    
    current_total_kcl = sum([item["P_final_raw"] for item in st.session_state.stored_evaluations])
    progress = min(1.0, current_total_kcl / TARGET_KCL_TONS)
    
    st.metric(
        label="已核算 KCl 总储存量",
        value=f"{current_total_kcl / 1e4:,.2f} 万吨",
        delta=f"距离5000万吨还差: {(TARGET_KCL_TONS - current_total_kcl)/1e4:,.2f} 万吨" if current_total_kcl < TARGET_KCL_TONS else "🎉 已圆满达成增储目标！"
    )
    st.progress(progress)
    st.caption(f"当前整体目标达成度：{progress * 100:.2f}%")
    
    st.markdown("---")
    st.write("### 🧭 自适应计算机制说明")
    st.info(
        "系统具备自动降级与升级计算能力：\n"
        "- 传入基础参数：自动运行容积法；\n"
        "- 包含地震雕刻 V：自动替代平面 A × h；\n"
        "- 包含试采压降数据：自动执行动静结合修正；\n"
        "- 任意数据均可自适应生成蒙特卡洛 P10/P50/P90 区间。"
    )
    if st.button("🗑️ 清空所有已存成果", use_container_width=True):
        st.session_state.stored_evaluations = []
        st.rerun()

# --------------------------------------------------
# 主界面 Tabs
# --------------------------------------------------
tab_upload, tab_single, tab_manual, tab_summary = st.tabs([
    "📁 任意数据批量上传自适应计算",
    "✍️ 单构造带灵活交互测算",
    "📖 参数规范与操作说明书",
    "📊 四大专题总账与成果对比"
])

# --------------------------------------------------
# TAB 1: 批量数据上传计算
# --------------------------------------------------
with tab_upload:
    st.subheader("批量上传任意格式数据文件 (CSV / Excel)")
    st.markdown(
        "各专题（川中、川西、川东北、川南）可直接将已有报表上传。表头无需完全一致，系统内置多别名模糊匹配。缺少某些参数时自动进行保底计算。"
    )
    
    sample_df = pd.DataFrame([
        {"构造带": "广安东翼T63", "区块": "川中专题", "储层类型": "油气同层型", "A": 45, "h": 25, "phi": 0.08, "Sw": 0.65, "Bw": 1.02, "C": 0.018, "Wp": 35000, "dp": 3.2, "ct": 8.5},
        {"构造带": "川西深层断褶带", "区块": "川西专题", "储层类型": "非油气同层型", "A": 60, "V": 1500, "phi": 0.065, "S": 0.0004, "承压水头标高": 300, "储层顶面标高": -2500, "C": 0.021},
        {"构造带": "川东北平落坝", "区块": "川东北专题", "储层类型": "油气同层型", "A": 30, "h": 18, "phi": 0.07, "C": 0.015}
    ])
    csv_buffer = io.BytesIO()
    sample_df.to_csv(csv_buffer, index=False, encoding="utf_8_sig")
    st.download_button(
        label="📥 下载多类型通用测试模板 (CSV)",
        data=csv_buffer.getvalue(),
        file_name="深层卤水储量计算任意数据模板.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("选择要计算的数据文件", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
                
            st.write("📋 **已成功解析上传数据预览：**")
            st.dataframe(df_input.head(5), use_container_width=True)
            
            if st.button("⚡ 一键自适应智能核算全表", type="primary"):
                batch_results = []
                for idx, row in df_input.iterrows():
                    calc_res, _ = evaluate_brine_record(row.to_dict(), run_mc=True, n_sim=2000)
                    batch_results.append(calc_res)
                    st.session_state.stored_evaluations.append(calc_res)
                    
                st.success(f"已完成全表 {len(batch_results)} 个构造带的自适应核算，并已合并至全盆地总账！")
                df_batch_show = pd.DataFrame(batch_results).drop(columns=["P_final_raw"])
                st.dataframe(df_batch_show, use_container_width=True)
        except Exception as e:
            st.error(f"文件解析或计算时发生错误: {str(e)}")

# --------------------------------------------------
# TAB 2: 单构造带手动输入测算
# --------------------------------------------------
with tab_single:
    st.subheader("单构造带参数灵活输入（未填项系统自动平滑过渡）")
    
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        s_zone = st.text_input("构造带名称", value="广安构造某富钾储卤层")
    with c_m2:
        s_region = st.selectbox("归属专题区块", ["川中专题", "川西专题", "川东北专题", "川南专题"])
    with c_m3:
        s_type = st.radio("储层赋存类型", ["油气同层型", "非油气同层型"], horizontal=True)

    st.markdown("##### 1. 必填/通用几何与物性参数")
    cg1, cg2, cg3, cg4 = st.columns(4)
    with cg1:
        s_C = st.number_input("KCl 平均品位 C (t/m³)", value=0.0190, step=0.0010, format="%.4f")
    with cg2:
        s_A = st.number_input("含水面积 A (km²)", value=50.0, step=1.0)
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
        st.markdown("##### 2. 油气同层专用参数")
        co1, co2 = st.columns(2)
        with co1:
            s_sw = st.number_input("含水饱和度 Sw (小数)", value=0.60, step=0.05)
        with co2:
            s_bw = st.number_input("卤水体积系数 Bw (无因次)", value=1.02, step=0.01)
        s_S, s_head, s_top = None, None, None
    else:
        st.markdown("##### 2. 非油气同层承压水参数")
        cn1, cn2, cn3 = st.columns(3)
        with cn1:
            s_S = st.number_input("弹性储水系数 S", value=0.0005, step=0.0001, format="%.5f")
        with cn2:
            s_head = st.number_input("平均承压水头标高 h (m)", value=320.0, step=10.0)
        with cn3:
            s_top = st.number_input("平均储层顶面标高 H (m)", value=-2100.0, step=50.0)
        s_sw, s_bw = None, None

    st.markdown("##### 3. 动态约束参数（可选，留空则仅计算静态）")
    has_dyn_input = st.checkbox("输入试采压降数据进行动静结合约束", value=False)
    if has_dyn_input:
        cd1, cd2, cd3 = st.columns(3)
        with cd1:
            s_wp = st.number_input("试采累计排卤量 Wp (m³)", value=28000.0, step=1000.0)
        with cd2:
            s_dp = st.number_input("生产折算压降 Δp (MPa)", value=2.8, step=0.1)
        with cd3:
            s_ct = st.number_input("综合压缩系数 Ct (10⁻⁴ MPa⁻¹)", value=8.0, step=0.5)
    else:
        s_wp, s_dp, s_ct = None, None, None

    if st.button("🚀 计算本构造带并生成不确定性分析", type="primary"):
        input_dict = {
            "构造带": s_zone, "区块": s_region, "储层类型": s_type,
            "C": s_C, "A": s_A, "phi": s_phi, "h": s_h,
            "Sw": s_sw, "Bw": s_bw, "S": s_S, "承压水头标高": s_head, "储层顶面标高": s_top,
            "Wp": s_wp, "dp": s_dp, "ct": s_ct
        }
        if s_V is not None:
            input_dict["V"] = s_V
            
        res_calc, mc_res = evaluate_brine_record(input_dict, run_mc=True, n_sim=5000)
        
        r_c1, r_c2, r_c3, r_c4 = st.columns(4)
        r_c1.metric("建模模式", res_calc["几何建模方式"])
        r_c2.metric("动态约束", res_calc["动态约束状态"])
        r_c3.metric("核定卤水量", f"{res_calc['最终核定卤水体积(亿m³)']:.4f} 亿m³")
        r_c4.metric("核定 KCl 储量", f"{res_calc['最终核定KCl储量(万吨)']:,.2f} 万吨")
        
        if mc_res:
            fig_hist = px.histogram(
                x=mc_res["samples"], nbins=50,
                labels={"x": "KCl 储量 (万吨)"},
                title="蒙特卡洛不确定性模拟概率分布 (P10 - P50 - P90)"
            )
            fig_hist.add_vline(x=mc_res["P90(万吨)"], line_dash="dash", line_color="orange", annotation_text=f"P90: {mc_res['P90(万吨)']}万吨")
            fig_hist.add_vline(x=mc_res["P50(万吨)"], line_dash="solid", line_color="green", annotation_text=f"P50: {mc_res['P50(万吨)']}万吨")
            fig_hist.add_vline(x=mc_res["P10(万吨)"], line_dash="dash", line_color="red", annotation_text=f"P10: {mc_res['P10(万吨)']}万吨")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        if st.button("💾 确认录入该结果到成果总账"):
            st.session_state.stored_evaluations.append(res_calc)
            st.success(f"已录入 {s_zone}，请前往“四大专题总账”查看！")

# --------------------------------------------------
# TAB 3: 详细操作说明书与规范 (使用原始字符串 r""" 避免转义报错)
# --------------------------------------------------
with tab_manual:
    st.markdown(r"""
## 深层卤水钾盐资源动-静综合评价操作手册与自适应规则

### 1. 计算公式标准

#### (1) 油气同层型卤水
