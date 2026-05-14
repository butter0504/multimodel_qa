import streamlit as st
from app.utils.common import (
    inject_css, render_sidebar, run_detection,
    get_module_options, get_default_modules, get_format_options,
)

st.set_page_config(page_title="图像分析", layout="wide")
inject_css()
render_sidebar()

st.markdown("## 🖼️ 图像数据质量分析")

upload_mode = st.radio("上传方式", ["多张图像", "标注文件（COCO/VOC/YOLO）", "ZIP压缩包"], horizontal=True)

format_options = get_format_options()
format_label = st.selectbox("数据格式", list(format_options.keys()), key="img_format")
format_hint = format_options[format_label]

project_name = st.text_input("项目名称", value="图像分析项目")

module_opts = get_module_options("image")
selected_modules = []
st.markdown("### 检测模块")
cols = st.columns(2)
for i, (key, desc) in enumerate(module_opts.items()):
    with cols[i % 2]:
        default = key in get_default_modules("image")
        if st.checkbox(desc, value=default, key=f"img_{key}"):
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
                    modules=selected_modules if selected_modules else None,
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
    image_dir = st.text_input("图像目录路径（可选，本地路径）", value="")
    class_file = st.text_input("类别文件路径（YOLO格式可选，本地路径）", value="")

    if st.button("🚀 开始分析", use_container_width=True, type="primary"):
        if annotation_file is None:
            st.error("请先上传标注文件")
        else:
            with st.spinner("正在分析..."):
                file_bytes = annotation_file.read()
                response = run_detection(
                    source=file_bytes,
                    modality="image",
                    modules=selected_modules if selected_modules else None,
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
                    if response.warnings:
                        for w in response.warnings:
                            st.warning(w)
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
                import shutil

                tmp_dir = tempfile.mkdtemp()
                try:
                    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
                        zf.extractall(tmp_dir)

                    response = run_detection(
                        source=tmp_dir,
                        modality="image",
                        modules=selected_modules if selected_modules else None,
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
                    shutil.rmtree(tmp_dir, ignore_errors=True)
