import streamlit as st
from app.utils.common import (
    inject_css, render_sidebar, run_detection,
    get_module_options, get_default_modules, get_format_options,
)

st.set_page_config(page_title="文本分析", layout="wide")
inject_css()
render_sidebar()

st.markdown("## 📝 文本数据质量分析")

uploaded_file = st.file_uploader("上传文本文件", type=["txt", "csv"])
label_file = st.file_uploader("标签文件（可选，选择标签错误或情感一致性检测时需要）", type=["txt", "csv"])
project_name = st.text_input("项目名称", value="文本分析项目")

format_options = get_format_options()
format_label = st.selectbox("数据格式", list(format_options.keys()), key="text_format")
format_hint = format_options[format_label]

module_opts = get_module_options("text")
selected_modules = []
st.markdown("### 检测模块")
cols = st.columns(2)
for i, (key, desc) in enumerate(module_opts.items()):
    with cols[i % 2]:
        default = key in get_default_modules("text")
        if st.checkbox(desc, value=default, key=f"txt_{key}"):
            selected_modules.append(key)

if st.button("🚀 开始分析", use_container_width=True, type="primary"):
    if uploaded_file is None:
        st.error("请先上传文件")
    else:
        with st.spinner("正在分析..."):
            file_bytes = uploaded_file.read()
            labels = None
            if label_file:
                label_content = label_file.read().decode("utf-8")
                labels = [l.strip() for l in label_content.splitlines() if l.strip()]

            response = run_detection(
                source=file_bytes,
                modality="text",
                modules=selected_modules if selected_modules else None,
                format_hint=format_hint,
                project_name=project_name,
                filename=uploaded_file.name,
                extra={"filename": uploaded_file.name, "labels": labels},
            )
            st.session_state.analysis_result = response.detection_result
            st.session_state.analysis_modality = "text"
            st.session_state.uploaded_file_name = uploaded_file.name

            if response.success:
                st.session_state.data_object_info = response.data_object_info
                if response.warnings:
                    for w in response.warnings:
                        st.warning(w)
                st.success("分析完成！")
                st.switch_page("pages/07_文本分析详情.py")
            else:
                for e in response.errors:
                    st.error(e)
