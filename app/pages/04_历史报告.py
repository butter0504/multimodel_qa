import streamlit as st
import json
from datetime import datetime

# 尝试导入数据库模块
try:
    from app.utils.database import db
    print("数据库模块导入成功")
except Exception as e:
    print(f"数据库模块导入失败: {str(e)}")
    # 创建一个简单的数据库模拟对象
    class MockDatabase:
        def get_projects(self, limit=10):
            import streamlit as st
            if "projects" in st.session_state:
                return st.session_state.projects[-limit:]
            return []
        def get_analysis_result(self, project_id):
            import streamlit as st
            if "projects" in st.session_state:
                for project in st.session_state.projects:
                    if project.get('db_id') == project_id and 'analysis_result' in project:
                        return project['analysis_result']
            return None
    db = MockDatabase()
    print("使用模拟数据库对象")

# 设置页面配置
st.set_page_config(
    page_title="历史报告 - 多模态数据质量检测系统",
    page_icon="📊",
    layout="wide"
)

# 禁用Streamlit自动生成的导航
st.markdown("""
<style>
/* 隐藏Streamlit自动生成的导航 */
[data-testid="stSidebarNav"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "recent_analyses" not in st.session_state:
    st.session_state.recent_analyses = []

# 侧边栏 - 只保留返回首页按钮
with st.sidebar:
    if st.button("返回首页", width='stretch'):
        st.switch_page("main.py")

# 页面标题
st.markdown('<h1 class="main-title">历史分析报告</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">查看和管理所有历史分析记录</p>', unsafe_allow_html=True)

# 从数据库获取项目列表
projects = db.get_projects(limit=100)

# 同时从 analysis_history 获取分析记录
analysis_history = st.session_state.get("analysis_history", [])

# 合并数据源：将 analysis_history 中不在 projects 中的记录添加进来
existing_names = {p.get('name', '') for p in projects}
for record in analysis_history:
    if record.get('name', '') not in existing_names:
        projects.append({
            "id": len(projects) + 1,
            "name": record.get("name", ""),
            "file_name": record.get("file", ""),
            "analysis_type": record.get("type", "") + "分析" if not record.get("type", "").endswith("分析") else record.get("type", ""),
            "upload_time": record.get("time", ""),
            "results": record.get("results", {})
        })

# 历史报告列表
if projects:
    # 筛选选项
    col1, col2 = st.columns(2)
    with col1:
        analysis_type_filter = st.selectbox(
            "按类型筛选",
            ["全部", "表格分析", "文本分析", "图像分析"]
        )
    with col2:
        sort_by = st.selectbox(
            "排序方式",
            ["时间（最新）", "时间（最早）"]
        )
    
    # 应用筛选和排序
    filtered_projects = projects.copy()
    
    if analysis_type_filter != "全部":
        filtered_projects = [item for item in filtered_projects if item["analysis_type"] == analysis_type_filter]
    
    if sort_by == "时间（最新）":
        filtered_projects.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
    else:
        filtered_projects.sort(key=lambda x: x.get("upload_time", ""))
    
    # 分页显示
    items_per_page = 10
    total_pages = (len(filtered_projects) + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        page = st.number_input(
            "页码",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        paginated_projects = filtered_projects[start_idx:end_idx]
    else:
        paginated_projects = filtered_projects
    
    # 显示历史记录
    for i, project in enumerate(paginated_projects):
        # 获取分析结果 - 先从数据库获取，再从项目的 results 字段获取
        result = db.get_analysis_result(project.get('id', project.get('db_id')))
        if not result and 'results' in project:
            result = project['results']
        
        # 确定分析类型
        analysis_type = project.get('analysis_type', '')
        if not analysis_type.endswith('分析'):
            analysis_type = analysis_type + '分析'
        
        with st.expander(f"{analysis_type}: {project.get('name', '未知')} - {project.get('upload_time', '未知时间')}"):
            # 基本信息
            st.markdown("**基本信息**")
            st.write(f"项目名称: {project.get('name', '未知')}")
            st.write(f"文件: {project.get('file_name', '未知')}")
            st.write(f"类型: {analysis_type}")
            st.write(f"上传时间: {project.get('upload_time', '未知')}")
            
            # 分析结果预览
            if result:
                st.markdown("**分析结果预览**")
                if analysis_type == "表格分析":
                    if isinstance(result, dict):
                        if "basic_info" in result:
                            st.write(f"行数: {result['basic_info'].get('rows', 'N/A')}")
                            st.write(f"列数: {result['basic_info'].get('columns', 'N/A')}")
                        if "missing_stats" in result:
                            st.write(f"缺失值: {result['missing_stats'].get('total', 0)}")
                        if "label_issues" in result:
                            st.write(f"标签错误: {result['label_issues'].get('error_count', 0)}")
                elif analysis_type == "文本分析":
                    if isinstance(result, dict):
                        if "metrics" in result:
                            st.write(f"总文本数: {result['metrics'].get('total_texts', 'N/A')}")
                            st.write(f"重复率: {result['metrics'].get('duplicate_text_rate', 0):.2f}%")
                elif analysis_type == "图像分析":
                    if isinstance(result, dict):
                        if "metrics" in result:
                            st.write(f"总图像数: {result['metrics'].get('total_images', 'N/A')}")
                            st.write(f"问题图像: {result['metrics'].get('issue_count', 0)}")
            
            # 操作按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("查看详情", key=f"view_{i}"):
                    st.session_state.current_project = project
                    if result:
                        st.session_state.analysis_result = result
                    else:
                        fc = project.get('file_content')
                        if fc is not None:
                            from pathlib import Path
                            import sys
                            sys.path.insert(0, str(Path(__file__).parent.parent))
                            with st.spinner("正在重新执行检测..."):
                                if analysis_type == "图像分析":
                                    from modules.image_detector import ImageDetector
                                    from modules.config import Config
                                    cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                                    detector = ImageDetector(cfg)
                                    st.session_state.analysis_result = detector.detect(fc, modules=['basic_quality'])
                                elif analysis_type == "表格分析":
                                    from modules.table_detector import TableDetector
                                    from modules.config import Config
                                    cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                                    detector = TableDetector(cfg)
                                    lc = project.get('label_col')
                                    st.session_state.analysis_result = detector.detect(fc, label_col=lc)
                                elif analysis_type == "文本分析":
                                    from modules.text_detector import TextDetector
                                    from modules.config import Config
                                    cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                                    detector = TextDetector(cfg)
                                    st.session_state.analysis_result = detector.detect(fc)
                    if analysis_type == "表格分析":
                        st.switch_page("pages/05_表格分析详情.py")
                    elif analysis_type == "文本分析":
                        st.switch_page("pages/07_文本分析详情.py")
                    elif analysis_type == "图像分析":
                        st.switch_page("pages/06_图像分析详情.py")
            with col2:
                is_latest = len(st.session_state.get("projects", [])) > 0 and project is st.session_state["projects"][-1]
                if is_latest:
                    if st.button("重新分析", key=f"reanalyze_{i}"):
                        st.session_state.current_project = project
                        st.session_state.reanalyze_pending = True
                        st.switch_page("main.py")
                else:
                    st.button("重新分析", key=f"reanalyze_{i}", disabled=True, help="仅最近一次分析支持重新分析")
else:
    st.info("暂无历史分析记录")

# 导出所有历史记录
if projects:
    st.markdown("""<hr style='margin: 2rem 0;'>""", unsafe_allow_html=True)
    st.markdown("**导出历史记录**")
    
    # 准备导出数据
    export_data = []
    for project in projects:
        # 获取分析结果
        result = db.get_analysis_result(project['id'])
        export_item = {
            "name": project["name"],
            "file": project["file_name"],
            "type": project["analysis_type"],
            "time": project.get("upload_time", ""),
            "results": result or {}
        }
        export_data.append(export_item)
    
    # 导出为JSON
    json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
    st.download_button(
        label="导出为JSON",
        data=json_str,
        file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# 页脚
st.markdown("""
<div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #E5E7EB; text-align: center; color: #6B7280;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)