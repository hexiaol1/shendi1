import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# 页面基础配置
# --------------------------------------------------
st.set_page_config(
    page_title="深层卤水钾盐动-静综合资源评价系统",
    page_icon="🧂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 会话状态管理 (多构造带存储)
# --------------------------------------------------
if "zones_data" not in st.session_state:
    st.session_state.zones_data = []

TARGET_KCL_TONS = 50_000_000.0  # 目标新增 5000 万吨 KCl

# --------------------------------------------------
# 侧边栏：全局设置与目标进度看板
# --------------------------------------------------
with st.sidebar:
    st.title("⚙️ 专题评估控制台")
    region_choice = st.selectbox(
        "选择当前负责专题区块",
        ["川中专题", "川西专题", "川东北专题", "川南专题", "综合汇总"]
    )
    
    st.markdown("---")
    st.subheader("🎯 全盆地增储目标进度追踪")
    
    # 汇总已核算构造带
    total_added_kcl = sum([z["P_final"] for z in st.session_state.zones_data])
    progress_pct = min(100.0, (total_added_kcl / TARGET_KCL_TONS) * 100)
    
    st.metric(
        label="已评价总储存量 (KCl)",
        value=f"{total_added_kcl / 1e4:,.2f} 万吨",
        delta=f"距离目标差额: {(TARGET_KCL_TONS - total_added_kcl)/1e4:,.2f} 万吨" if total_added_kcl < TARGET_KCL_TONS else "已达成目标！"
    )
    st.progress(progress_pct / 100.0)
    st.caption(f"目标：5000.00 万吨 | 当前完成度：{progress_pct:.2f}%")
    
    if st.button("🗑️ 清空所有已存构造带数据", use_container_width=True):
        st.session_state.zones_data = []
        st.rerun()

# --------------------------------------------------
# 页面顶部选项卡：评估工作台 vs 详细操作说明书
# --------------------------------------------------
tab_calc, tab_docs, tab_summary = st.tabs(["📊 综合储量计算工作台", "📖 详细操作说明书与理论指南", "📋 构造带总账与区块成果对比"])

with tab_docs:
    st.markdown("""
    ## 深层卤水钾盐动-静结合综合评价操作手册

    本工具面向四川盆地深层卤水钾盐资源评价专题（川中、川西、川东北、川南）设计，用于应对不同勘探程度下资料丰富度不均的问题。
