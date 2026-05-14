import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import base64
import io
import numpy as np
from PIL import Image
from app.utils.common import inject_css, render_sidebar, render_metric_cards, convert_to_serializable, format_number

st.set_page_config(page_title="图像分析详情", layout="wide")
inject_css()
render_sidebar()

st.markdown("## 🖼️ 图像分析详情")

result = st.session_state.get("analysis_result")
data_info = st.session_state.get("data_object_info", {})
filenames = st.session_state.get("image_filenames", [])

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

modules_run = result.get("modules_run", [])
issues = result.get("issues", [])
metrics = result.get("metrics", {})

tab_list = ["概览"]
if "basic_quality" in modules_run:
    tab_list.append("基础质量")
if "label_error" in modules_run:
    tab_list.append("标签错误")
if "distribution_shift" in modules_run:
    tab_list.append("分布偏移")
if "uncertainty" in modules_run:
    tab_list.append("不确定性")
tab_list.extend(["问题列表", "原始JSON"])
tabs = st.tabs(tab_list)

with tabs[0]:
    st.markdown("### 分析概览")

    total_images = metrics.get("total_images", 0)
    issue_count = len(issues)
    issue_rate = issue_count / max(total_images, 1) * 100
    avg_quality = metrics.get("average_quality_score", "N/A")

    render_metric_cards([
        {"value": total_images, "label": "总图像数"},
        {"value": format_number(issue_rate, 1) + "%", "label": "问题率", "color": "var(--warning)" if issue_rate > 20 else "var(--success)"},
        {"value": format_number(avg_quality) if isinstance(avg_quality, (int, float)) else avg_quality, "label": "平均质量"},
        {"value": len(modules_run), "label": "检测模块数"},
    ])

    if modules_run:
        st.markdown("**已运行模块**: " + ", ".join(modules_run))

    quality_scores = metrics.get("quality_scores", [])
    if quality_scores:
        fig = px.histogram(x=quality_scores, nbins=30, title="质量分数分布",
                          labels={"x": "质量分数", "y": "图像数"})
        fig.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
        st.plotly_chart(fig, use_container_width=True)

tab_idx = 1
if "basic_quality" in modules_run:
    with tabs[tab_idx]:
        st.markdown("### 基础质量检测")

        bq_metrics = metrics.get("basic_quality", {})
        blur_count = bq_metrics.get("blur_count", 0)
        exposure_count = bq_metrics.get("exposure_count", 0)
        noise_count = bq_metrics.get("noise_count", 0)
        resolution_count = bq_metrics.get("low_resolution_count", 0)
        contrast_count = bq_metrics.get("low_contrast_count", 0)

        render_metric_cards([
            {"value": blur_count, "label": "模糊", "color": "var(--danger)" if blur_count > 0 else "var(--success)"},
            {"value": exposure_count, "label": "曝光异常", "color": "var(--warning)" if exposure_count > 0 else "var(--success)"},
            {"value": noise_count, "label": "噪声", "color": "var(--warning)" if noise_count > 0 else "var(--success)"},
            {"value": resolution_count, "label": "低分辨率", "color": "var(--danger)" if resolution_count > 0 else "var(--success)"},
        ])

        bq_issues = [i for i in issues if i.get("type", "").startswith(("blur", "exposure", "noise", "resolution", "contrast", "corrupt"))]
        if bq_issues:
            type_filter = st.selectbox("按类型筛选", ["全部"] + list(set(i["type"] for i in bq_issues)), key="bq_filter")
            filtered = bq_issues if type_filter == "全部" else [i for i in bq_issues if i["type"] == type_filter]
            for issue in filtered[:20]:
                with st.expander(f"🔍 {issue['type']} — 图像 #{issue.get('index', '?')}"):
                    st.markdown(f"**类型**: {issue['type']}")
                    if "suggestion" in issue:
                        st.markdown(f'<div class="suggestion-box">💡 {issue["suggestion"]}</div>', unsafe_allow_html=True)
                    if "details" in issue:
                        st.json(convert_to_serializable(issue["details"]))
    tab_idx += 1

if "label_error" in modules_run:
    with tabs[tab_idx]:
        st.markdown("### 标签错误检测")

        le_metrics = metrics.get("label_error", {})
        le_count = le_metrics.get("label_error_count", 0)
        le_rate = le_metrics.get("error_rate", 0)
        feature_method = le_metrics.get("feature_method", "unknown")

        render_metric_cards([
            {"value": le_count, "label": "标签错误数", "color": "var(--danger)" if le_count > 0 else "var(--success)"},
            {"value": format_number(le_rate, 4), "label": "错误率"},
            {"value": feature_method, "label": "特征提取方式"},
        ])

        quality_scores = le_metrics.get("label_quality_scores", [])
        if quality_scores:
            fig = px.histogram(x=quality_scores, nbins=30, title="标签质量分数分布",
                              labels={"x": "质量分数", "y": "样本数"})
            fig.add_vline(x=0.15, line_dash="dash", line_color="red", annotation_text="阈值 0.15")
            fig.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
            st.plotly_chart(fig, use_container_width=True)

        le_issues = [i for i in issues if i.get("type") == "label_error"]
        if le_issues:
            top_df = pd.DataFrame(le_issues[:20])
            st.dataframe(top_df, use_container_width=True)
    tab_idx += 1

if "distribution_shift" in modules_run:
    with tabs[tab_idx]:
        st.markdown("### 分布偏移检测")

        ds_metrics = metrics.get("distribution_shift", {})
        shifted_dims = ds_metrics.get("shifted_dimensions", 0)
        brightness_js = ds_metrics.get("brightness_js_divergence", "N/A")
        semantic_shift = ds_metrics.get("semantic_shift_detected", False)

        render_metric_cards([
            {"value": shifted_dims, "label": "偏移维度数", "color": "var(--warning)" if shifted_dims > 0 else "var(--success)"},
            {"value": format_number(brightness_js) if isinstance(brightness_js, (int, float)) else brightness_js, "label": "亮度JS散度"},
            {"value": "是" if semantic_shift else "否", "label": "语义偏移", "color": "var(--danger)" if semantic_shift else "var(--success)"},
        ])

        if ds_metrics.get("dimension_details"):
            st.json(convert_to_serializable(ds_metrics["dimension_details"]))
    tab_idx += 1

if "uncertainty" in modules_run:
    with tabs[tab_idx]:
        st.markdown("### 不确定性估计")

        u_metrics = metrics.get("uncertainty", {})
        high_uncertainty = u_metrics.get("high_uncertainty_count", 0)
        avg_uncertainty = u_metrics.get("average_uncertainty", "N/A")
        method = u_metrics.get("method", "unknown")

        render_metric_cards([
            {"value": high_uncertainty, "label": "高不确定性样本", "color": "var(--warning)" if high_uncertainty > 0 else "var(--success)"},
            {"value": format_number(avg_uncertainty) if isinstance(avg_uncertainty, (int, float)) else avg_uncertainty, "label": "平均不确定性"},
            {"value": method, "label": "检测方法"},
        ])

        uncertainty_scores = u_metrics.get("uncertainty_scores", [])
        if uncertainty_scores:
            fig = px.scatter(y=uncertainty_scores, title="不确定性分数",
                           labels={"y": "不确定性分数", "x": "样本索引"})
            fig.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
            st.plotly_chart(fig, use_container_width=True)
    tab_idx += 1

with tabs[tab_idx]:
    st.markdown("### 问题列表")
    if issues:
        issues_df = pd.DataFrame(issues)
        st.dataframe(issues_df, use_container_width=True)
        csv_data = issues_df.to_csv(index=False).encode("utf-8")
        st.download_button("导出问题列表 CSV", data=csv_data, file_name="image_issues.csv", mime="text/csv")
    else:
        st.success("未检测到问题")
    tab_idx += 1

with tabs[tab_idx]:
    st.markdown("### 原始JSON")
    st.json(convert_to_serializable(result))

if st.button("📥 导出完整报告", use_container_width=True):
    report_json = json.dumps(convert_to_serializable(result), ensure_ascii=False, indent=2)
    st.download_button(
        label="下载JSON报告",
        data=report_json,
        file_name="image_analysis_report.json",
        mime="application/json",
    )
