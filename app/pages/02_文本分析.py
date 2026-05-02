import streamlit as st
import pandas as pd
import requests
import time
from modules.text_detector import TextDetector

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
    st.session_state.page = "文本分析"

if "recent_analyses" not in st.session_state:
    st.session_state.recent_analyses = []

with st.sidebar:
    if st.button("返回首页", width='stretch'):
        st.session_state.page = "首页"
        st.session_state.current_project = None
        st.switch_page("main.py")

st.markdown('<h1 class="page-title">文本数据质量分析</h1>', unsafe_allow_html=True)

st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "拖拽文件到此处或点击上传文本文件",
    type=["txt", "csv"],
    accept_multiple_files=False,
    help="支持 TXT 和 CSV 格式"
)
st.markdown('</div>', unsafe_allow_html=True)

project_name = st.text_input("项目名称", placeholder="输入项目名称")

st.markdown("### 检测模块选择")
st.markdown("选择需要运行的检测模块：")

col_a, col_b = st.columns(2)
with col_a:
    mod_text_length = st.checkbox("📏 文本长度检测", value=True, key="mod_text_length")
    mod_duplicate = st.checkbox("🔄 重复检测", value=True, key="mod_duplicate")
    mod_char_anomaly = st.checkbox("⚠️ 字符异常检测", value=True, key="mod_char_anomaly")
    mod_language = st.checkbox("🌐 语言检测", value=False, key="mod_language")
with col_b:
    mod_label_error = st.checkbox("🏷️ 标签错误检测", value=False, key="mod_label_error",
                                   help="需要上传标签文件")
    mod_perplexity = st.checkbox("📊 困惑度检测", value=False, key="mod_perplexity")
    mod_sentiment = st.checkbox("💬 情感一致性检测", value=False, key="mod_sentiment",
                                 help="需要上传标签文件")

needs_labels = mod_label_error or mod_sentiment

labels = None
if needs_labels:
    st.markdown("### 标签文件（可选）")
    st.markdown("上传标签文件以启用标签错误检测和情感一致性检测。格式：每行一个标签，或 CSV 文件（label 列）。")
    label_file = st.file_uploader(
        "上传标签文件",
        type=["txt", "csv"],
        accept_multiple_files=False,
        key="label_file_uploader"
    )
    if label_file:
        try:
            if label_file.name.endswith('.csv'):
                label_df = pd.read_csv(label_file)
                if 'label' in label_df.columns:
                    labels = label_df['label'].tolist()
                elif 'Label' in label_df.columns:
                    labels = label_df['Label'].tolist()
                else:
                    labels = label_df.iloc[:, 0].tolist()
            else:
                label_content = label_file.read().decode("utf-8")
                labels = [line.strip() for line in label_content.splitlines() if line.strip()]
            st.success(f"已加载 {len(labels)} 个标签")
        except Exception as e:
            st.error(f"标签文件解析失败: {str(e)}")
            labels = None

if uploaded_file and project_name:
    if st.button("开始分析", width='stretch'):
        with st.spinner("正在分析文本数据..."):
            contents = uploaded_file.read().decode("utf-8")
            texts = contents.splitlines()
            texts = [t for t in texts if t.strip() != ""]

            if labels and len(labels) != len(texts):
                st.warning(f"标签数量（{len(labels)}）与文本数量（{len(texts)}）不匹配，将忽略标签")
                labels = None

            modules = []
            if mod_text_length:
                modules.append('text_length')
            if mod_duplicate:
                modules.append('duplicate')
            if mod_label_error and labels:
                modules.append('label_error')
            if mod_char_anomaly:
                modules.append('character_anomaly')
            if mod_language:
                modules.append('language')
            if mod_perplexity:
                modules.append('perplexity')
            if mod_sentiment and labels:
                modules.append('sentiment_consistency')

            if not modules:
                modules = ['text_length', 'duplicate', 'character_anomaly']

            detector = TextDetector()
            result = detector.detect(texts, labels=labels, modules=modules)

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            analysis_record = {
                "name": project_name,
                "type": "文本",
                "file": uploaded_file.name,
                "time": current_time,
                "results": result,
                "file_object": uploaded_file
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
            st.session_state.uploaded_file = uploaded_file
            st.session_state.texts_data = texts
            st.session_state.text_labels = labels if labels else []

            st.switch_page("pages/07_文本分析详情.py")

st.markdown("""
<div style="margin-top: 50px; text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)
