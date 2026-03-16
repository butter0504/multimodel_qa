import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from modules.table_detector import TableDetector

# 禁用Streamlit自动生成的导航
st.markdown("""
<style>
/* 隐藏Streamlit自动生成的导航 */
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

/* 全局样式 */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: var(--text-primary);
    line-height: 1.6;
    background-color: var(--bg-light);
}

/* 标题样式 */
.page-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1.5rem;
}

/* 卡片样式 */
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

/* 按钮样式 */
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

/* 指标卡片 */
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

/* 上传区域样式 */
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

/* 标签样式 */
.tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}

.tag.success {
    background: #d1fae5;
    color: #065f46;
}

.tag.warning {
    background: #fef3c7;
    color: #92400e;
}

.tag.error {
    background: #fee2e2;
    color: #991b1b;
}

/* 侧边栏样式 */
.sidebar-section {
    margin-bottom: 2rem;
}

.sidebar-section-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* 导航样式 */
.nav-radio .stRadio > div {
    flex-direction: column;
}

.nav-radio .stRadio label {
    padding: 10px 16px;
    border-radius: 8px;
    transition: all 0.3s ease;
    margin-bottom: 4px;
}

.nav-radio .stRadio label:hover {
    background: rgba(30, 58, 138, 0.05);
}

.nav-radio .stRadio input[type="radio"]:checked + label {
    background: rgba(30, 58, 138, 0.1);
    font-weight: 600;
    color: var(--primary-color);
    border-left: 4px solid var(--primary-color);
}

/* 状态标签 */
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

# 缓存函数
@st.cache_data(ttl=60)
def check_api_status():
    """检查API状态并缓存60秒"""
    try:
        import time
        start_time = time.time()
        response = requests.get("http://localhost:8000/health", timeout=2)
        response_time = (time.time() - start_time) * 1000
        return {
            "status": response.status_code == 200,
            "response_time": round(response_time, 2)
        }
    except Exception as e:
        return {
            "status": False,
            "response_time": None,
            "error": str(e)
        }

# 初始化会话状态
if "page" not in st.session_state:
    st.session_state.page = "表格分析"

if "recent_analyses" not in st.session_state:
    st.session_state.recent_analyses = []

# 侧边栏
with st.sidebar:
    # 主要导航
    st.markdown('<div class="sidebar-section nav-radio">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">导航</div>', unsafe_allow_html=True)
    selected_page = st.radio(
        "选择功能",
        ["首页", "表格分析", "文本分析", "图像分析", "API测试"],
        key="nav_radio",
        label_visibility="collapsed",
        index=1  # 默认选中表格分析
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 系统状态
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">系统状态</div>', unsafe_allow_html=True)
    
    with st.expander("查看状态", expanded=False):
        # 刷新按钮
        if st.button("刷新状态", use_container_width=True):
            # 清除缓存
            check_api_status.clear()
            st.rerun()
        
        # API状态
        api_status = check_api_status()
        if api_status["status"]:
            st.markdown(f'<span class="status-tag success">API: 正常</span>', unsafe_allow_html=True)
            if api_status["response_time"]:
                st.markdown(f'<span class="status-tag success">响应: {api_status["response_time"]}ms</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-tag error">API: 未运行</span>', unsafe_allow_html=True)
        
        # 环境信息
        st.markdown('<div style="margin-top: 10px;">', unsafe_allow_html=True)
        st.write("环境信息:")
        st.info("Python 3.9+")
        st.info("Cleanlab 2.7.1")
        st.info("Streamlit 1.50.0")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 最近分析记录
    if st.session_state.recent_analyses:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">最近分析</div>', unsafe_allow_html=True)
        for analysis in st.session_state.recent_analyses[-5:]:  # 显示最近5条
            st.write(f"• {analysis['type']}: {analysis['file']}")
        st.markdown('</div>', unsafe_allow_html=True)

# 页面导航逻辑
if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    # 对于非当前页面，使用Streamlit的页面导航
    if selected_page == "首页":
        st.switch_page("main.py")
    elif selected_page == "文本分析":
        st.switch_page("pages/02_文本分析.py")
    elif selected_page == "图像分析":
        st.switch_page("pages/03_图像分析.py")
    elif selected_page == "API测试":
        st.switch_page("pages/04_API测试.py")

# 页面标题
st.markdown('<h1 class="page-title">表格数据质量分析</h1>', unsafe_allow_html=True)

# 文件上传区域
st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "拖拽文件到此处或点击上传 CSV/Excel 文件",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
    help="支持 CSV 和 Excel 文件格式"
)
st.markdown('</div>', unsafe_allow_html=True)

# 标签列选择
label_col = st.text_input("标签列名称（可选）", placeholder="例如：label")

if uploaded_file is not None:
    # 读取数据
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # 检测数据质量
    detector = TableDetector()
    result = detector.detect(df, label_col=label_col if label_col else None)
    
    # 显示数据预览
    st.markdown("<h2 style='margin-top: 30px;'>数据预览</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 用 tabs 组织结果
    tab1, tab2, tab3 = st.tabs(["📊 质量指标", "⚠️ 问题检测", "🔍 详细分析"])
    
    with tab1:
        st.markdown("<h3>数据质量指标</h3>", unsafe_allow_html=True)
        
        # 指标卡片
        metrics = result["metrics"]
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("行数", metrics["rows"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("列数", metrics["columns"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
            st.metric("缺失值率", f"{metrics['missing_value_rate']:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
            st.metric("重复行率", f"{metrics['duplicate_row_rate']:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-card error">', unsafe_allow_html=True)
            st.metric("异常值率", f"{metrics['outlier_rate']:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h3>检测到的问题</h3>", unsafe_allow_html=True)
        
        issues = result["issues"]
        if issues:
            # 问题分类统计
            issue_types = {}
            for issue in issues:
                issue_type = issue.get("type", "unknown")
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            # 显示问题类型分布
            st.markdown("<h4>问题类型分布</h4>", unsafe_allow_html=True)
            for issue_type, count in issue_types.items():
                if issue_type == "missing_value":
                    st.markdown(f'<span class="tag warning">{issue_type}: {count}</span>', unsafe_allow_html=True)
                elif issue_type == "outlier":
                    st.markdown(f'<span class="tag error">{issue_type}: {count}</span>', unsafe_allow_html=True)
                elif issue_type == "duplicate_rows":
                    st.markdown(f'<span class="tag warning">{issue_type}: {count}</span>', unsafe_allow_html=True)
                elif issue_type == "label_error":
                    st.markdown(f'<span class="tag error">{issue_type}: {count}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="tag">{issue_type}: {count}</span>', unsafe_allow_html=True)
            
            # 显示详细问题列表
            st.markdown("<h4>详细问题列表</h4>", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                issues_df = pd.DataFrame(issues)
                st.dataframe(issues_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.success("🎉 未检测到任何问题，数据质量良好！")
    
    with tab3:
        st.markdown("<h3>详细分析</h3>", unsafe_allow_html=True)
        
        # 缺失值分析
        st.markdown("<h4>缺失值分析</h4>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            missing_stats = result["missing_stats"]
            st.write(f"总缺失值: {missing_stats['total']} ({missing_stats['percentage']:.2f}%)")
            
            # 显示每列缺失值
            if missing_stats['per_column']:
                missing_df = pd.DataFrame.from_dict(missing_stats['per_column'], orient='index')
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("没有缺失值")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 异常值分析
        st.markdown("<h4>异常值分析</h4>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            outlier_stats = result["outlier_stats"]
            if outlier_stats:
                for col, info in outlier_stats.items():
                    st.write(f"**{col}**: {info['count']} 个异常值 ({info['percentage']:.2f}%)")
                    st.write(f"  边界: [{info['bounds']['lower']:.2f}, {info['bounds']['upper']:.2f}]")
            else:
                st.success("没有异常值")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 标签错误分析
        if label_col and result.get("label_issues"):
            st.markdown("<h4>标签错误分析</h4>", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                label_issues = result["label_issues"]
                st.write(f"检测到 {label_issues.get('error_count', 0)} 个标签错误 ({label_issues.get('error_rate', 0):.2f}%)")
                
                if label_issues.get('error_indices'):
                    st.write("错误标签索引:", label_issues['error_indices'])
                
                # 显示质量分数统计
                if label_issues.get('debug_info'):
                    debug_info = label_issues['debug_info']
                    st.write(f"平均质量分数: {debug_info.get('mean_quality_score', 0):.2f}")
                    st.write(f"最低质量分数: {debug_info.get('min_quality_score', 0):.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # 下载报告按钮
    st.markdown("<h2 style='margin-top: 30px;'>导出报告</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # 准备报告数据
        report_data = {
            "file_name": uploaded_file.name,
            "basic_info": result["basic_info"],
            "metrics": result["metrics"],
            "issues": result["issues"],
            "missing_stats": result["missing_stats"],
            "outlier_stats": result["outlier_stats"],
            "duplicate_stats": result["duplicate_stats"],
            "label_issues": result.get("label_issues", {})
        }
        
        # 转换为 JSON 字符串
        report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
        
        # 下载按钮
        st.download_button(
            label="📥 下载 JSON 报告",
            data=report_json,
            file_name=f"table_analysis_report_{uploaded_file.name.split('.')[0]}.json",
            mime="application/json"
        )
        
        # 添加到最近分析记录
        st.session_state.recent_analyses.append({
            "type": "表格",
            "file": uploaded_file.name,
            "time": st.session_state.get("current_time", "")
        })
        # 只保留最近5条记录
        if len(st.session_state.recent_analyses) > 5:
            st.session_state.recent_analyses = st.session_state.recent_analyses[-5:]
        
        st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("""
<div style="margin-top: 50px; text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)