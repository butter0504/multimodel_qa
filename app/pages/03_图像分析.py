import streamlit as st
import pandas as pd
from modules.image_detector import ImageDetector
from PIL import Image
import io
import requests
import time

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

.upload-area {
    border: 2px dashed var(--border-color);
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    transition: all 0.3s ease;
    background: var(--bg-white);
}

.upload-area:hover {
    border-color: var(--primary-light);
    background: rgba(79, 70, 229, 0.02);
}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "图像分析"

if "recent_analyses" not in st.session_state:
    st.session_state.recent_analyses = []

# 侧边栏 - 只保留返回首页按钮
with st.sidebar:
    if st.button("返回首页", width='stretch'):
        st.session_state.page = "首页"
        st.session_state.current_project = None
        st.switch_page("main.py")

# 页面标题
st.markdown('<h1 class="page-title">图像数据质量分析</h1>', unsafe_allow_html=True)

# 文件上传区域
st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "拖拽文件到此处或点击上传图像文件",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="支持 JPG、JPEG、PNG 图像格式"
)
st.markdown('</div>', unsafe_allow_html=True)

# 项目名称
project_name = st.text_input("项目名称", placeholder="输入项目名称")

# 开始分析按钮
if uploaded_files and project_name:
    if st.button("开始分析", use_container_width=True):
        with st.spinner("正在分析图像数据..."):
            images = []
            for file in uploaded_files:
                images.append(file.read())
            
            detector = ImageDetector()
            result = detector.detect(images)
            
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            analysis_record = {
                "name": project_name,
                "type": "图像",
                "file": f"{len(uploaded_files)}张图像",
                "time": current_time,
                "results": result,
                "uploaded_files": uploaded_files
            }
            
            if "analysis_history" not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append(analysis_record)
            
            if "recent_analyses" not in st.session_state:
                st.session_state.recent_analyses = []
            st.session_state.recent_analyses.append(analysis_record)
            
            if len(st.session_state.recent_analyses) > 5:
                st.session_state.recent_analyses = st.session_state.recent_analyses[-5:]
            
            st.session_state.analysis_result = result
            st.session_state.uploaded_file = uploaded_files[0]
            
            st.switch_page("pages/06_图像分析详情.py")

# 页脚
st.markdown("""
<div style="margin-top: 50px; text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)
