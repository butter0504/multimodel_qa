import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np
from app.utils.common import inject_css, render_sidebar, render_metric_cards, convert_to_serializable, format_number

st.set_page_config(page_title="表格分析详情", layout="wide")
inject_css()
render_sidebar()

st.markdown("## 📊 表格分析详情")

result = st.session_state.get("analysis_result")
data_info = st.session_state.get("data_object_info", {})
label_col = st.session_state.get("label_col", "")

if not result:
    st.error("未找到分析结果，请先进行分析")
    if st.button("返回首页"):
        st.switch_page("app.py")
    st.stop()

if data_info:
    st.markdown("### 数据概览")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="value">{data_info.get("sample_count", "N/A")}</div><div class="label">样本数</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="value">{data_info.get("num_classes", "N/A")}</div><div class="label">类别数</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="value">{data_info.get("task_type", "N/A")}</div><div class="label">任务类型</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="value">{data_info.get("source_format", "N/A")}</div><div class="label">数据格式</div></div>', unsafe_allow_html=True)

tab_names = ["概览", "问题列表", "可视化", "原始JSON"]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.markdown("### 分析概览")

    basic_info = result.get("basic_info", {})
    missing_stats = result.get("missing_stats", {})
    outlier_stats = result.get("outlier_stats", {})
    duplicate_stats = result.get("duplicate_stats", {})
    label_issues = result.get("label_issues", {})

    total_rows = basic_info.get("total_rows", 0)
    total_cols = basic_info.get("total_columns", 0)
    missing_total = missing_stats.get("total_missing", 0)
    missing_pct = missing_stats.get("missing_percentage", 0)
    dup_count = duplicate_stats.get("duplicate_count", 0)
    label_error_count = label_issues.get("label_error_count", 0)

    render_metric_cards([
        {"value": total_rows, "label": "总行数"},
        {"value": total_cols, "label": "总列数"},
        {"value": format_number(missing_pct, 1) + "%", "label": "缺失率", "color": "var(--warning)" if missing_pct > 10 else "var(--success)"},
        {"value": dup_count, "label": "重复行", "color": "var(--danger)" if dup_count > 0 else "var(--success)"},
    ])

    if label_issues:
        st.markdown("#### 标签质量")
        le_count = label_issues.get("label_error_count", 0)
        noise_rate = label_issues.get("noise_rate", 0)
        render_metric_cards([
            {"value": le_count, "label": "标签错误数", "color": "var(--danger)" if le_count > 0 else "var(--success)"},
            {"value": format_number(noise_rate, 4), "label": "噪声率"},
        ])

with tabs[1]:
    st.markdown("### 问题列表")

    issues = result.get("issues", [])
    if issues:
        issues_df = pd.DataFrame(issues)
        st.dataframe(issues_df, use_container_width=True)
        csv_data = issues_df.to_csv(index=False).encode("utf-8")
        st.download_button("导出问题列表 CSV", data=csv_data, file_name="table_issues.csv", mime="text/csv")
    else:
        st.success("未检测到问题")

with tabs[2]:
    st.markdown("### 可视化")

    if missing_stats and missing_stats.get("column_stats"):
        st.markdown("#### 缺失值分布")
        col_stats = missing_stats["column_stats"]
        if isinstance(col_stats, dict):
            fig = px.bar(x=list(col_stats.keys()), y=list(col_stats.values()),
                        labels={"x": "列名", "y": "缺失数量"},
                        title="各列缺失值数量")
            fig.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
            st.plotly_chart(fig, use_container_width=True)

    if outlier_stats and outlier_stats.get("column_stats"):
        st.markdown("#### 异常值分布")
        col_stats = outlier_stats["column_stats"]
        if isinstance(col_stats, dict):
            fig = px.bar(x=list(col_stats.keys()), y=list(col_stats.values()),
                        labels={"x": "列名", "y": "异常值数量"},
                        title="各列异常值数量")
            fig.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
            st.plotly_chart(fig, use_container_width=True)

    if label_issues and label_issues.get("label_quality_scores"):
        st.markdown("#### 标签质量分数分布")
        scores = label_issues["label_quality_scores"]
        fig = px.histogram(x=scores, nbins=30, title="标签质量分数分布",
                          labels={"x": "质量分数", "y": "样本数"})
        fig.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.markdown("### 原始JSON")
    st.json(convert_to_serializable(result))

if st.button("📥 导出完整报告", use_container_width=True):
    report_json = json.dumps(convert_to_serializable(result), ensure_ascii=False, indent=2)
    st.download_button(
        label="下载JSON报告",
        data=report_json,
        file_name="table_analysis_report.json",
        mime="application/json",
    )
