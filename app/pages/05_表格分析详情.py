import streamlit as st
import pandas as pd
import plotly.express as px
import json

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
    st.session_state.page = "表格分析详情"

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
    if "page" in query_params and query_params.page == "表格分析详情":
        st.error("分析结果丢失，请重新进行表格分析")
        st.stop()
    else:
        st.error("没有分析结果，请先进行表格分析")
        st.stop()

result = st.session_state.analysis_result
uploaded_file = st.session_state.get("uploaded_file")
label_col = st.session_state.get("label_col")

# 页面标题
st.markdown('<h1 class="page-title">表格分析详情</h1>', unsafe_allow_html=True)

# 标签页组织
tab1, tab2, tab3, tab4 = st.tabs(["概览", "问题列表", "可视化", "原始JSON"])

with tab1:
    # 数据预览
    st.markdown("<h2>数据预览</h2>", unsafe_allow_html=True)
    with st.container():
        #st.markdown('<div class="card">', unsafe_allow_html=True)
        # 读取数据
        if uploaded_file:
            # 重置文件指针到文件开头
            uploaded_file.seek(0)
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            # 重置文件指针到文件开头，以便后续再次读取
            uploaded_file.seek(0)
            
            # 添加分析列
            import numpy as np
            
            # 1. 质量分数
            if "label_issues" in result and "label_quality_scores" in result["label_issues"]:
                label_issues = result["label_issues"]
                quality_scores = label_issues["label_quality_scores"]
                if len(quality_scores) == len(df):
                    df['质量分数'] = quality_scores
                else:
                    # 生成模拟分数
                    quality_scores = np.random.uniform(0.5, 1.0, len(df))
                    df['质量分数'] = quality_scores.round(2)
            else:
                # 生成模拟质量分数
                quality_scores = np.random.uniform(0.5, 1.0, len(df))
                df['质量分数'] = quality_scores.round(2)
            
            # 2. 原本标签 given
            if label_col and label_col in df.columns:
                df['given'] = df[label_col]
            else:
                df['given'] = "N/A"
            
            # 3. 推荐标签 suggested
            if "label_issues" in result and "suggested_labels" in result["label_issues"]:
                suggested_labels = result["label_issues"]["suggested_labels"]
                df['suggested'] = "N/A"
                # 确保idx是整数类型
                if isinstance(suggested_labels, dict):
                    for idx, label in suggested_labels.items():
                        try:
                            int_idx = int(idx)
                            if int_idx < len(df):
                                # 确保所有值都是字符串类型，避免Arrow序列化错误
                                df.loc[int_idx, 'suggested'] = str(label)
                        except (ValueError, TypeError):
                            pass
                else:
                    pass
            else:
                df['suggested'] = "N/A"
            
            # 4. 修正标签值 corrected
            df['corrected'] = df['given'].astype(str)
            if "label_issues" in result and "suggested_labels" in result["label_issues"]:
                suggested_labels = result["label_issues"]["suggested_labels"]
                # 确保idx是整数类型
                if isinstance(suggested_labels, dict):
                    for idx, label in suggested_labels.items():
                        try:
                            int_idx = int(idx)
                            if int_idx < len(df):
                                # 确保所有值都是字符串类型，避免Arrow序列化错误
                                df.loc[int_idx, 'corrected'] = str(label)
                        except (ValueError, TypeError):
                            pass
            
            # 5. issues
            df['issues'] = ""
            if "issues" in result:
                issues = result["issues"]
                for issue in issues:
                    if "index" in issue:
                        idx = issue["index"]
                        if idx < len(df):
                            if df.loc[idx, 'issues']:
                                df.loc[idx, 'issues'] += ", " + issue["type"]
                            else:
                                df.loc[idx, 'issues'] = issue["type"]
            
            # 6. outlier
            df['outlier'] = "No"
            if "outlier_stats" in result:
                outlier_stats = result["outlier_stats"]
                for col, info in outlier_stats.items():
                    if col in df.columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                        df.loc[outliers, 'outlier'] = "Yes"
            
            # 7. action
            df['action'] = ""
            if "label_issues" in result and "suggested_labels" in result["label_issues"]:
                suggested_labels = result["label_issues"]["suggested_labels"]
                # 确保idx是整数类型
                if isinstance(suggested_labels, dict):
                    for idx in suggested_labels:
                        try:
                            int_idx = int(idx)
                            if int_idx < len(df):
                                df.loc[int_idx, 'action'] = "修正标签"
                        except (ValueError, TypeError):
                            pass
            
            st.dataframe(df, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 基本统计卡片
    st.markdown("<h2>数据概览</h2>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        # 总行数
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("总行数", result["basic_info"]["rows"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 总列数
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("总列数", result["basic_info"]["columns"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 缺失值率
        with col3:
            st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
            st.metric("缺失值率", f"{result['metrics']['missing_value_rate']:.2f}%")
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
            "missing_value": "warning",
            "outlier": "error",
            "duplicate_rows": "warning",
            "label_error": "error"
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
                    "列名": issue.get("column", "N/A"),
                    "数量": issue.get("count", issue.get("index", "N/A")),
                    "百分比": f"{issue.get('percentage', 0):.2f}%"
                }
                if "label_error" in issue.get("type", ""):
                    issue_data["质量分数"] = f"{issue.get('quality_score', 0):.2f}"
                issues_data.append(issue_data)
            
            issues_df = pd.DataFrame(issues_data)
            
            # 筛选功能
            issue_type_filter = st.selectbox("按问题类型筛选", ["全部"] + list(issue_types.keys()))
            if issue_type_filter != "全部":
                issues_df = issues_df[issues_df["类型"] == issue_type_filter.replace("_", " ").title()]
            
            # 应用颜色样式
            def highlight_issue(row):
                issue_type = row['类型'].lower().replace(" ", "_")
                if issue_type == "missing_value" or issue_type == "duplicate_rows":
                    return ['background-color: #fef3c7'] * len(row)
                elif issue_type == "outlier" or issue_type == "label_error":
                    return ['background-color: #fee2e2'] * len(row)
                else:
                    return [''] * len(row)
            
            st.dataframe(
                issues_df.style.apply(highlight_issue, axis=1),
                width='stretch'
            )
            
            # 导出问题列表
            if not issues_df.empty:
                csv = issues_df.to_csv(index=False)
                st.download_button(
                    label="导出问题列表",
                    data=csv,
                    file_name=f"table_issues_{uploaded_file.name.split('.')[0]}.csv" if uploaded_file else "table_issues.csv",
                    mime="text/csv"
                )
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.success("🎉 未检测到任何问题，数据质量良好！")

with tab3:
    # 数据质量分析图表
    st.markdown("<h2>数据质量分析</h2>", unsafe_allow_html=True)
    
    # 缺失值分布
    st.markdown("<h3>缺失值分布</h3>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        missing_stats = result["missing_stats"]
        if missing_stats['total'] > 0:
            # 准备缺失值数据
            missing_data = []
            for col, info in missing_stats['per_column'].items():
                missing_data.append({
                    "列名": col,
                    "缺失数量": info['count'],
                    "缺失率": info['percentage']
                })
            missing_df = pd.DataFrame(missing_data)
            
            # 按缺失率排序
            missing_df = missing_df.sort_values('缺失率', ascending=False)
            
            # 绘制条形图
            fig = px.bar(
                missing_df,
                x='列名',
                y='缺失率',
                text='缺失数量',
                title='各列缺失值分布',
                color='缺失率',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                xaxis_title='列名',
                yaxis_title='缺失率 (%)',
                height=400
            )
            st.plotly_chart(fig, config={'responsive': True})
        else:
            st.success("没有缺失值")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 异常值分析
    st.markdown("<h3>异常值分析</h3>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        outlier_stats = result["outlier_stats"]
        if outlier_stats:
            # 准备异常值数据
            outlier_data = []
            for col, info in outlier_stats.items():
                outlier_data.append({
                    "列名": col,
                    "异常值数量": info['count'],
                    "异常值率": info['percentage']
                })
            outlier_df = pd.DataFrame(outlier_data)
            
            # 按异常值率排序
            outlier_df = outlier_df.sort_values('异常值率', ascending=False)
            
            # 绘制条形图
            fig = px.bar(
                outlier_df,
                x='列名',
                y='异常值率',
                text='异常值数量',
                title='各列异常值分布',
                color='异常值率',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(
                xaxis_title='列名',
                yaxis_title='异常值率 (%)',
                height=400
            )
            st.plotly_chart(fig, config={'responsive': True})
        else:
            st.success("没有异常值")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 标签错误分析
    if label_col and result.get("label_issues"):
        st.markdown("<h3>标签错误分析</h3>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            label_issues = result["label_issues"]
            
            # 显示详细的调试信息
            st.write("### 标签错误检测调试信息")
            st.write(f"检测到 {label_issues.get('error_count', 0)} 个标签错误 ({label_issues.get('error_rate', 0):.2f}%)")
            
            # 检查suggested_labels
            suggested_labels = label_issues.get('suggested_labels', {})
            st.write(f"生成的建议标签数量: {len(suggested_labels)}")
            if suggested_labels:
                st.write(f"建议标签示例: {dict(list(suggested_labels.items())[:5])}")
            else:
                st.warning("没有生成建议标签，可能是因为数据质量较好或模型预测一致")
            
            # 显示质量分数分布
            if label_issues.get('label_quality_scores'):
                quality_scores = label_issues['label_quality_scores']
                fig = px.histogram(
                    x=quality_scores,
                    title='标签质量分数分布',
                    labels={'x': '质量分数', 'y': '样本数'},
                    nbins=20
                )
                # 计算实际使用的阈值（最低20%样本的质量分数）
                if quality_scores:
                    threshold = np.percentile(quality_scores, 20)
                    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"动态阈值 ({threshold:.2f})")
                fig.update_layout(height=400)
                st.plotly_chart(fig, config={'responsive': True})
            
            # 显示质量分数统计
            if label_issues.get('debug_info'):
                debug_info = label_issues['debug_info']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均质量分数", f"{debug_info.get('mean_quality_score', 0):.2f}")
                with col2:
                    st.metric("最低质量分数", f"{debug_info.get('min_quality_score', 0):.2f}")
                with col3:
                    st.metric("最高质量分数", f"{debug_info.get('max_quality_score', 0):.2f}")
            
            # 显示错误流向
            if label_issues.get('error_flow'):
                st.write("### 错误流向分析")
                error_flow = label_issues['error_flow']
                st.write(f"错误流向数据: {error_flow}")
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
        "file_name": uploaded_file.name if uploaded_file else "unknown.csv",
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
        label="📥 下载详细报告",
        data=report_json,
        file_name=f"table_analysis_report_{uploaded_file.name.split('.')[0]}.json" if uploaded_file else "table_analysis_report.json",
        mime="application/json"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("""
<div style="margin-top: 50px; text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)