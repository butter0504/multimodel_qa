import streamlit as st
from app.utils.common import (
    inject_css, render_sidebar, run_detection,
    get_module_options, get_default_modules, get_format_options,
)

st.set_page_config(page_title="表格分析", layout="wide")
inject_css()
render_sidebar()

st.markdown("## 📊 表格数据质量分析")

uploaded_file = st.file_uploader("上传表格文件", type=["csv", "xlsx", "tsv"])
label_col = st.text_input("标签列名称（可选）", value="")
text_col = st.text_input("文本列名称（可选）", value="")
project_name = st.text_input("项目名称", value="表格分析项目")

format_options = get_format_options()
format_label = st.selectbox("数据格式", list(format_options.keys()), key="table_format")
format_hint = format_options[format_label]

module_opts = get_module_options("table")
selected_modules = []
st.markdown("### 检测模块")
cols = st.columns(2)
for i, (key, desc) in enumerate(module_opts.items()):
    with cols[i % 2]:
        default = key in get_default_modules("table")
        if st.checkbox(desc, value=default, key=f"tbl_{key}"):
            selected_modules.append(key)

if st.button("🚀 开始分析", use_container_width=True, type="primary"):
    if uploaded_file is None:
        st.error("请先上传文件")
    else:
        with st.spinner("正在分析..."):
            file_bytes = uploaded_file.read()
            response = run_detection(
                source=file_bytes,
                modality="table",
                modules=selected_modules if selected_modules else None,
                format_hint=format_hint or "csv",
                label_column=label_col if label_col else None,
                text_column=text_col if text_col else None,
                project_name=project_name,
                filename=uploaded_file.name,
                extra={"filename": uploaded_file.name},
            )
            st.session_state.analysis_result = response.detection_result
            st.session_state.analysis_modality = "table"
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.label_col = label_col

            if response.success:
                st.session_state.data_object_info = response.data_object_info
                if response.warnings:
                    for w in response.warnings:
                        st.warning(w)
                st.success("分析完成！")
                st.switch_page("pages/05_表格分析详情.py")
            else:
                for e in response.errors:
                    st.error(e)
