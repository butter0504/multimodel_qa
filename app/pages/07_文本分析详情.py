import streamlit as st
import pandas as pd
import json
import numpy as np


def convert_to_serializable(obj):
    """递归转换 numpy 类型为 Python 原生类型，确保 JSON 可序列化"""
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k: 
                convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

st.markdown("""
<style>
[data-testid="stSidebarNav"] {
    display: none;
}

:root {
    --primary-color: #1E3A8A;
    --primary-light: #4F46E5;
    --success-color: #10B981;
    --warning-color: #F59E0B;
    --error-color: #EF4444;
    --text-primary: #1F2937;
    --text-secondary: #6B7280;
    --bg-light: #F3F4F6;
    --bg-white: #FFFFFF;
    --border-color: #E5E7EB;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: var(--text-primary);
    line-height: 1.6;
    background-color: var(--bg-light);
}

.page-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1.5rem;
}

.card {
    background: var(--bg-white);
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

.stButton > button {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(30, 58, 138, 0.3);
}

.metric-card {
    background: var(--bg-light);
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    border-left: 4px solid var(--primary-color);
}

.metric-card.success {
    border-left-color: var(--success-color);
}

.metric-card.warning {
    border-left-color: var(--warning-color);
}

.metric-card.error {
    border-left-color: var(--error-color);
}

.status-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.875rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}

.status-tag.success {
    background: #d1fae5;
    color: #065f46;
}

.status-tag.warning {
    background: #fef3c7;
    color: #92400e;
}

.status-tag.error {
    background: #fee2e2;
    color: #991b1b;
}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "文本分析详情"

# 侧边栏 - 只保留返回首页按钮
with st.sidebar:
    if st.button("返回首页", width='stretch'):
        st.session_state.page = "首页"
        st.session_state.current_project = None
        st.switch_page("main.py")

# 检查是否有分析结果
if "analysis_result" not in st.session_state:
    st.error("没有分析结果，请先进行文本分析")
    if st.button("返回首页"):
        st.switch_page("main.py")
    st.stop()

result = st.session_state.analysis_result
uploaded_file = st.session_state.get("uploaded_file")

# 页面标题
st.markdown('<h1 class="page-title">文本分析详情</h1>', unsafe_allow_html=True)

# 标签页组织
tab1, tab2, tab3 = st.tabs(["概览", "问题列表", "原始JSON"])

with tab1:
    # 数据概览
    st.markdown("<h2>数据概览</h2>", unsafe_allow_html=True)
    with st.container():
        metrics = result.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("总文本数", metrics.get("total_texts", 0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
            st.metric("空文本率", f"{metrics.get('empty_text_rate', 0):.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
            st.metric("重复文本率", f"{metrics.get('duplicate_text_rate', 0):.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("平均长度", f"{metrics.get('average_length', 0):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # 问题列表
    st.markdown("<h2>问题检测</h2>", unsafe_allow_html=True)
    issues = result.get("issues", [])
    if issues:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            issues_df = pd.DataFrame(issues)
            st.dataframe(issues_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.success("未检测到任何问题，数据质量良好！")

with tab3:
    # 原始JSON数据
    st.markdown("<h2>原始分析结果</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.json(convert_to_serializable(result))
        st.markdown('</div>', unsafe_allow_html=True)

# 下载报告按钮
st.markdown("<h2 style='margin-top: 30px;'>导出报告</h2>", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    report_data = {
        "file_name": uploaded_file.name if uploaded_file else "unknown.txt",
        "metrics": result.get("metrics", {}),
        "issues": result.get("issues", [])
    }
    
    report_data = convert_to_serializable(report_data)
    report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
    
    st.download_button(
        label="下载详细报告",
        data=report_json,
        file_name=f"text_analysis_report_{uploaded_file.name.split('.')[0]}.json" if uploaded_file else "text_analysis_report.json",
        mime="application/json"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("""
<div style="margin-top: 50px; text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)
