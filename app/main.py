import streamlit as st
from app.utils.common import (
    inject_css, render_sidebar, render_metric_cards,
    run_detection, get_modality_options, get_module_options,
    get_default_modules, get_format_options, convert_to_serializable,
    get_storage,
)

st.set_page_config(page_title="多模态数据集质量检测系统", layout="wide")
inject_css()
render_sidebar()

st.markdown("""
<h1 style='text-align:center; background: linear-gradient(135deg, #4F46E5, #818CF8, #06B6D4);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem;'>
     基于置信学习的多模态数据质量检测系统
</h1>
<p style='text-align:center; color: #94A3B8; font-size: 0.95rem;'>
     六层架构 · 多格式适配 · LLM辅助解析 · 置信学习核心
</p>
""", unsafe_allow_html=True)

if st.button("📋 查看历史报告", use_container_width=True):
    st.switch_page("pages/04_历史报告.py")

st.markdown("---")

modality_labels = get_modality_options()
modality_choice = st.selectbox("选择数据模态", list(modality_labels.keys()))
modality = modality_labels[modality_choice]

format_options = get_format_options()
format_label = st.selectbox("数据格式", list(format_options.keys()))
format_hint = format_options[format_label]

st.markdown("### 📤 数据上传")

project_name = st.text_input("项目名称", value=f"{modality_choice}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M')}")

if modality == "table":
    uploaded_file = st.file_uploader("上传表格文件", type=["csv", "xlsx", "tsv"])
    label_col = st.text_input("标签列名称（可选）", value="")
    text_col = st.text_input("文本列名称（可选）", value="")

    module_opts = get_module_options("table")
    selected_modules = []
    st.markdown("#### 检测模块选择")
    cols = st.columns(2)
    for i, (key, desc) in enumerate(module_opts.items()):
        with cols[i % 2]:
            default = key in get_default_modules("table")
            if st.checkbox(desc, value=default, key=f"table_mod_{key}"):
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
                    modules=selected_modules,
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

elif modality == "text":
    uploaded_file = st.file_uploader("上传文本文件", type=["txt", "csv"])
    label_file = st.file_uploader("标签文件（可选）", type=["txt", "csv"])

    module_opts = get_module_options("text")
    selected_modules = []
    st.markdown("#### 检测模块选择")
    cols = st.columns(2)
    for i, (key, desc) in enumerate(module_opts.items()):
        with cols[i % 2]:
            default = key in get_default_modules("text")
            if st.checkbox(desc, value=default, key=f"text_mod_{key}"):
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
                    modules=selected_modules,
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

elif modality == "image":
    upload_mode = st.radio("上传方式", ["多张图像", "标注文件（COCO/VOC/YOLO）", "ZIP压缩包"], horizontal=True)

    selected_modules = []
    module_opts = get_module_options("image")
    st.markdown("#### 检测模块选择")
    cols = st.columns(2)
    for i, (key, desc) in enumerate(module_opts.items()):
        with cols[i % 2]:
            default = key in get_default_modules("image")
            if st.checkbox(desc, value=default, key=f"image_mod_{key}"):
                selected_modules.append(key)

    if upload_mode == "多张图像":
        uploaded_files = st.file_uploader("上传图像", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                          accept_multiple_files=True)
        label_file = st.file_uploader("标签文件（可选，CSV含filename和label列）", type=["csv"])

        if st.button("🚀 开始分析", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.error("请先上传图像")
            else:
                with st.spinner("正在分析..."):
                    images = [f.read() for f in uploaded_files]
                    labels = None
                    if label_file:
                        import pandas as pd
                        import io
                        df = pd.read_csv(io.BytesIO(label_file.read()))
                        if "filename" in df.columns and "label" in df.columns:
                            name_to_label = dict(zip(df["filename"], df["label"]))
                            labels = [name_to_label.get(f.name, None) for f in uploaded_files]

                    response = run_detection(
                        source=images,
                        modality="image",
                        modules=selected_modules,
                        project_name=project_name,
                        filename=f"{len(uploaded_files)}_images",
                        extra={"filenames": [f.name for f in uploaded_files], "labels": labels},
                    )
                    st.session_state.analysis_result = response.detection_result
                    st.session_state.analysis_modality = "image"
                    st.session_state.image_filenames = [f.name for f in uploaded_files]

                    if response.success:
                        st.session_state.data_object_info = response.data_object_info
                        if response.warnings:
                            for w in response.warnings:
                                st.warning(w)
                        st.success("分析完成！")
                        st.switch_page("pages/06_图像分析详情.py")
                    else:
                        for e in response.errors:
                            st.error(e)

    elif upload_mode == "标注文件（COCO/VOC/YOLO）":
        annotation_file = st.file_uploader("上传标注文件", type=["json", "xml", "txt", "zip"])
        image_dir = st.text_input("图像目录路径（可选）", value="")
        class_file = st.text_input("类别文件路径（YOLO格式可选）", value="")

        if st.button("🚀 开始分析", use_container_width=True, type="primary"):
            if annotation_file is None:
                st.error("请先上传标注文件")
            else:
                with st.spinner("正在分析..."):
                    file_bytes = annotation_file.read()
                    response = run_detection(
                        source=file_bytes,
                        modality="image",
                        modules=selected_modules,
                        format_hint=format_hint,
                        project_name=project_name,
                        filename=annotation_file.name,
                        extra={
                            "filename": annotation_file.name,
                            "image_dir": image_dir if image_dir else None,
                            "class_file": class_file if class_file else None,
                        },
                    )
                    st.session_state.analysis_result = response.detection_result
                    st.session_state.analysis_modality = "image"

                    if response.success:
                        st.session_state.data_object_info = response.data_object_info
                        st.success("分析完成！")
                        st.switch_page("pages/06_图像分析详情.py")
                    else:
                        for e in response.errors:
                            st.error(e)

    elif upload_mode == "ZIP压缩包":
        zip_file = st.file_uploader("上传ZIP文件", type=["zip"])
        if st.button("🚀 开始分析", use_container_width=True, type="primary"):
            if zip_file is None:
                st.error("请先上传ZIP文件")
            else:
                with st.spinner("正在解压和分析..."):
                    import zipfile
                    import io
                    import tempfile
                    import os

                    tmp_dir = tempfile.mkdtemp()
                    try:
                        with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
                            zf.extractall(tmp_dir)

                        response = run_detection(
                            source=tmp_dir,
                            modality="image",
                            modules=selected_modules,
                            format_hint=format_hint,
                            project_name=project_name,
                            filename=zip_file.name,
                        )
                        st.session_state.analysis_result = response.detection_result
                        st.session_state.analysis_modality = "image"

                        if response.success:
                            st.session_state.data_object_info = response.data_object_info
                            st.success("分析完成！")
                            st.switch_page("pages/06_图像分析详情.py")
                        else:
                            for e in response.errors:
                                st.error(e)
                    finally:
                        import shutil
                        shutil.rmtree(tmp_dir, ignore_errors=True)

st.markdown("---")
st.markdown("### 📊 最近分析记录")
try:
    storage = get_storage()
    projects = storage.get_projects(limit=5)
    if projects:
        for p in projects:
            atype = p.get("analysis_type", "unknown")
            name = p.get("name", "未命名")
            ftime = p.get("upload_time", "")
            st.markdown(f"""
            <div style="background:#1E293B; border:1px solid #334155; border-radius:8px; padding:0.8rem 1rem; margin:0.3rem 0;">
                <span style="color:#818CF8; font-weight:600;">{name}</span>
                <span class="status-tag info" style="margin-left:0.5rem;">{atype}</span>
                <span style="color:#94A3B8; font-size:0.8rem; float:right;">{ftime[:19] if ftime else ''}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("暂无分析记录")
except Exception:
    st.info("暂无分析记录")
