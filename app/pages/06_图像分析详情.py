import streamlit as st
import pandas as pd
import plotly.express as px
import json
from PIL import Image
import io

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

# 初始化会话状态
if "page" not in st.session_state:
    st.session_state.page = "图像分析详情"

# 侧边栏 - 只保留返回首页按钮
with st.sidebar:
    if st.button("返回首页", width='stretch'):
        # 重置会话状态，返回到初始页面位置
        st.session_state.page = "首页"
        st.session_state.current_project = None
        st.switch_page("main.py")

# 检查是否有分析结果
if "analysis_result" not in st.session_state:
    # 尝试从查询参数获取
    query_params = st.query_params
    if "page" in query_params and query_params.page == "图像分析详情":
        st.error("分析结果丢失，请重新进行图像分析")
        st.stop()
    else:
        st.error("没有分析结果，请先进行图像分析")
        st.stop()

result = st.session_state.analysis_result
uploaded_file = st.session_state.get("uploaded_file")

# 页面标题
st.markdown('<h1 class="page-title">图像分析详情</h1>', unsafe_allow_html=True)

# 标签页组织
tab1, tab2, tab3, tab4 = st.tabs(["概览", "问题列表", "可视化", "原始JSON"])

with tab1:
    # 图像预览
    st.markdown("<h2>图像预览</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if uploaded_file:
            img = Image.open(io.BytesIO(uploaded_file.getvalue()))
            st.image(img, caption=uploaded_file.name, use_column_width=True)
            st.write(f"尺寸: {img.width}x{img.height}")
            st.write(f"格式: {img.format}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 基本统计卡片
    st.markdown("<h2>数据概览</h2>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        # 总图像数
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("总图像数", result["metrics"]["total_images"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 问题率
        with col2:
            st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
            st.metric("问题率", f"{result['metrics']['issue_rate']:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 平均质量分数
        with col3:
            st.markdown('<div class="metric-card success">', unsafe_allow_html=True)
            st.metric("平均质量分数", f"{result['metrics'].get('average_quality_score', 0):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # 问题样本列表
    st.markdown("<h2>问题检测</h2>", unsafe_allow_html=True)
    issues = result["issues"]
    if issues:
        # 问题分类统计
        issue_types = {}
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        # 显示问题类型分布
        st.markdown("<h3>问题类型分布</h3>", unsafe_allow_html=True)
        issue_colors = {
            "corrupted_image": "error",
            "unsupported_format": "warning",
            "small_image": "warning",
            "blurry_image": "warning",
            "bad_exposure": "warning",
            "low_contrast": "warning",
            "low_color_contrast": "warning"
        }
        
        # 根据问题类型数量动态创建列
        num_issues = len(issue_types)
        if num_issues > 0:
            cols = st.columns(num_issues)
            for i, (issue_type, count) in enumerate(issue_types.items()):
                color = issue_colors.get(issue_type, "")
                with cols[i]:
                    st.markdown(f'<div class="metric-card {color}">', unsafe_allow_html=True)
                    st.metric(issue_type.replace("_", " ").title(), count)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # 显示详细问题列表（带颜色标记）
        st.markdown("<h3>问题样本列表</h3>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # 构建问题数据框
            issues_data = []
            for issue in issues:
                issue_data = {
                    "类型": issue.get("type", "unknown").replace("_", " ").title(),
                    "索引": issue.get("index", "N/A"),
                    "错误信息": issue.get("error", "N/A")
                }
                if "sharpness" in issue:
                    issue_data["清晰度"] = f"{issue.get('sharpness', 0):.2f}"
                if "dark_ratio" in issue:
                    issue_data["暗像素比例"] = f"{issue.get('dark_ratio', 0):.2f}"
                if "bright_ratio" in issue:
                    issue_data["亮像素比例"] = f"{issue.get('bright_ratio', 0):.2f}"
                if "rms_contrast" in issue:
                    issue_data["对比度"] = f"{issue.get('rms_contrast', 0):.2f}"
                if "color_std" in issue:
                    issue_data["颜色标准差"] = f"{issue.get('color_std', 0):.2f}"
                issues_data.append(issue_data)
            
            issues_df = pd.DataFrame(issues_data)
            
            # 筛选功能
            issue_type_filter = st.selectbox("按问题类型筛选", ["全部"] + list(issue_types.keys()))
            if issue_type_filter != "全部":
                issues_df = issues_df[issues_df["类型"] == issue_type_filter.replace("_", " ").title()]
            
            st.dataframe(issues_df, width='stretch')
            
            # 导出问题列表
            if not issues_df.empty:
                csv = issues_df.to_csv(index=False)
                st.download_button(
                    label="导出问题列表",
                    data=csv,
                    file_name=f"image_issues_{uploaded_file.name.split('.')[0]}.csv" if uploaded_file else "image_issues.csv",
                    mime="text/csv"
                )
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.success("🎉 未检测到任何问题，数据质量良好！")

with tab3:
    # 数据质量分析图表
    st.markdown("<h2>数据质量分析</h2>", unsafe_allow_html=True)
    
    # 质量分数分布
    if "quality_scores" in result["metrics"]:
        st.markdown("<h3>质量分数分布</h3>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            import plotly.express as px
            import pandas as pd
            score_df = pd.DataFrame({
                "质量分数": result["metrics"]["quality_scores"]
            })
            fig = px.histogram(score_df, x="质量分数", bins=20, title="图像质量分数分布")
            fig.update_layout(
                xaxis_title="质量分数",
                yaxis_title="图像数量",
                height=400
            )
            st.plotly_chart(fig, config={'responsive': True})
            st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    # 原始JSON数据
    st.markdown("<h2>原始分析结果</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.json(result)
        st.markdown('</div>', unsafe_allow_html=True)

# 下载报告按钮
st.markdown("<h2 style='margin-top: 30px;'>导出报告</h2>", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # 准备报告数据
    report_data = {
        "file_name": uploaded_file.name if uploaded_file else "unknown.jpg",
        "metrics": result["metrics"],
        "issues": result["issues"]
    }
    
    # 转换为 JSON 字符串
    report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
    
    # 下载按钮
    st.download_button(
        label="📥 下载详细报告",
        data=report_json,
        file_name=f"image_analysis_report_{uploaded_file.name.split('.')[0]}.json" if uploaded_file else "image_analysis_report.json",
        mime="application/json"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("""
<div style="margin-top: 50px; text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)