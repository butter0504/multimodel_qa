import streamlit as st
import json
from app.utils.common import inject_css, render_sidebar, get_storage, convert_to_serializable

st.set_page_config(page_title="历史报告", layout="wide")
inject_css()
render_sidebar()

st.markdown("## 📋 历史分析报告")

storage = get_storage()

filter_type = st.selectbox("按类型筛选", ["全部", "table", "text", "image"])
sort_order = st.selectbox("排序方式", ["时间最新", "时间最早"])

projects = storage.get_projects(limit=100)

if filter_type != "全部":
    projects = [p for p in projects if p.get("analysis_type") == filter_type]

if sort_order == "时间最早":
    projects = list(reversed(projects))

if not projects:
    st.info("暂无分析记录")
    st.stop()

page_size = 10
total_pages = (len(projects) + page_size - 1) // page_size
current_page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)
start_idx = (current_page - 1) * page_size
end_idx = start_idx + page_size
page_projects = projects[start_idx:end_idx]

for p in page_projects:
    pid = p.get("id", 0)
    name = p.get("name", "未命名")
    atype = p.get("analysis_type", "unknown")
    ftime = p.get("upload_time", "")
    fname = p.get("file_name", "")
    basic_info = p.get("basic_info", {})

    type_labels = {"table": "表格分析", "text": "文本分析", "image": "图像分析"}
    type_label = type_labels.get(atype, atype)

    with st.expander(f"📁 {name} — {type_label} — {ftime[:19] if ftime else ''}"):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**文件**: {fname}")
            if basic_info:
                sample_count = basic_info.get("sample_count", "N/A")
                num_classes = basic_info.get("num_classes", "N/A")
                task_type = basic_info.get("task_type", "N/A")
                source_format = basic_info.get("source_format", "N/A")
                st.markdown(f"**样本数**: {sample_count} | **类别数**: {num_classes} | **任务类型**: {task_type} | **数据格式**: {source_format}")

        with col2:
            if st.button("查看详情", key=f"view_{pid}"):
                result = storage.get_analysis_result(pid)
                if result:
                    st.session_state.analysis_result = result
                    st.session_state.analysis_modality = atype
                    st.session_state.current_project_id = pid
                    page_map = {
                        "table": "pages/05_表格分析详情.py",
                        "text": "pages/07_文本分析详情.py",
                        "image": "pages/06_图像分析详情.py",
                    }
                    st.switch_page(page_map.get(atype, "app.py"))
                else:
                    st.warning("未找到分析结果")

            if st.button("导出JSON", key=f"export_{pid}"):
                result = storage.get_analysis_result(pid)
                if result:
                    st.download_button(
                        label="下载",
                        data=json.dumps(convert_to_serializable(result), ensure_ascii=False, indent=2),
                        file_name=f"report_{pid}.json",
                        mime="application/json",
                    )

st.markdown(f"**共 {len(projects)} 条记录，第 {current_page}/{total_pages} 页**")
