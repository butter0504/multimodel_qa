import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from app.utils.common import inject_css, render_sidebar, render_metric_cards, convert_to_serializable, format_number

st.set_page_config(page_title="文本分析详情", layout="wide")
inject_css()
render_sidebar()

st.markdown("## 📝 文本分析详情")

result = st.session_state.get("analysis_result")
data_info = st.session_state.get("data_object_info", {})

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
module_tab_map = {}
for m in modules_run:
    label_map = {
        "text_length": "文本长度",
        "duplicate": "重复检测",
        "label_error": "标签错误",
        "character_anomaly": "字符异常",
        "language": "语言检测",
        "perplexity": "困惑度",
        "sentiment_consistency": "情感一致性",
    }
    tab_name = label_map.get(m, m)
    tab_list.append(tab_name)
    module_tab_map[m] = len(tab_list) - 1
tab_list.extend(["问题列表", "原始JSON"])
tabs = st.tabs(tab_list)

with tabs[0]:
    st.markdown("### 分析概览")

    total_texts = metrics.get("total_texts", 0)
    issue_count = len(issues)
    issue_rate = issue_count / max(total_texts, 1) * 100
    avg_length = metrics.get("average_length", "N/A")

    render_metric_cards([
        {"value": total_texts, "label": "总文本数"},
        {"value": format_number(issue_rate, 1) + "%", "label": "问题率", "color": "var(--warning)" if issue_rate > 20 else "var(--success)"},
        {"value": format_number(avg_length, 0) if isinstance(avg_length, (int, float)) else avg_length, "label": "平均长度"},
        {"value": len(modules_run), "label": "检测模块数"},
    ])

    if modules_run:
        st.markdown("**已运行模块**: " + ", ".join(modules_run))

for m, tab_idx in module_tab_map.items():
    with tabs[tab_idx]:
        m_metrics = metrics.get(m, {})
        m_issues = [i for i in issues if i.get("module") == m or i.get("type", "").startswith(m)]

        if m == "text_length":
            st.markdown("### 文本长度检测")
            short_count = m_metrics.get("short_count", 0)
            long_count = m_metrics.get("long_count", 0)
            outlier_count = m_metrics.get("outlier_count", 0)
            render_metric_cards([
                {"value": short_count, "label": "过短文本", "color": "var(--warning)" if short_count > 0 else "var(--success)"},
                {"value": long_count, "label": "过长文本", "color": "var(--warning)" if long_count > 0 else "var(--success)"},
                {"value": outlier_count, "label": "长度异常", "color": "var(--danger)" if outlier_count > 0 else "var(--success)"},
            ])
            length_stats = m_metrics.get("length_stats", {})
            if length_stats:
                st.markdown("#### 长度统计")
                stats_df = pd.DataFrame([length_stats])
                st.dataframe(stats_df, use_container_width=True)
            lengths = m_metrics.get("lengths", [])
            if lengths:
                fig = px.histogram(x=lengths, nbins=30, title="文本长度分布",
                                  labels={"x": "字符数", "y": "文本数"})
                fig.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
                st.plotly_chart(fig, use_container_width=True)

        elif m == "duplicate":
            st.markdown("### 重复检测")
            exact_dup = m_metrics.get("exact_duplicate_count", 0)
            near_dup = m_metrics.get("near_duplicate_count", 0)
            unique = m_metrics.get("unique_count", 0)
            render_metric_cards([
                {"value": exact_dup, "label": "完全重复", "color": "var(--danger)" if exact_dup > 0 else "var(--success)"},
                {"value": near_dup, "label": "近似重复", "color": "var(--warning)" if near_dup > 0 else "var(--success)"},
                {"value": unique, "label": "唯一文本", "color": "var(--success)"},
            ])
            dup_groups = m_metrics.get("duplicate_groups", [])
            if dup_groups:
                st.markdown(f"**重复组数**: {len(dup_groups)}")

        elif m == "label_error":
            st.markdown("### 标签错误检测")
            le_count = m_metrics.get("label_error_count", 0)
            noise_rate = m_metrics.get("noise_rate", 0)
            feature_method = m_metrics.get("feature_method", "unknown")
            render_metric_cards([
                {"value": le_count, "label": "标签错误数", "color": "var(--danger)" if le_count > 0 else "var(--success)"},
                {"value": format_number(noise_rate, 4), "label": "噪声率"},
                {"value": feature_method, "label": "特征方法"},
            ])
            quality_scores = m_metrics.get("label_quality_scores", [])
            if quality_scores:
                fig = px.histogram(x=quality_scores, nbins=30, title="标签质量分数分布",
                                  labels={"x": "质量分数", "y": "样本数"})
                fig.add_vline(x=0.15, line_dash="dash", line_color="red", annotation_text="阈值 0.15")
                fig.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
                st.plotly_chart(fig, use_container_width=True)

        elif m == "character_anomaly":
            st.markdown("### 字符异常检测")
            garbled = m_metrics.get("garbled_count", 0)
            special = m_metrics.get("special_char_count", 0)
            whitespace = m_metrics.get("whitespace_count", 0)
            render_metric_cards([
                {"value": garbled, "label": "乱码文本", "color": "var(--danger)" if garbled > 0 else "var(--success)"},
                {"value": special, "label": "特殊字符过多", "color": "var(--warning)" if special > 0 else "var(--success)"},
                {"value": whitespace, "label": "首尾空白", "color": "var(--warning)" if whitespace > 0 else "var(--success)"},
            ])

        elif m == "language":
            st.markdown("### 语言检测")
            mixed = m_metrics.get("mixed_language_count", 0)
            non_target = m_metrics.get("non_target_language_count", 0)
            render_metric_cards([
                {"value": mixed, "label": "混合语言文本", "color": "var(--warning)" if mixed > 0 else "var(--success)"},
                {"value": non_target, "label": "非目标语言", "color": "var(--warning)" if non_target > 0 else "var(--success)"},
            ])
            lang_dist = m_metrics.get("language_distribution", {})
            if lang_dist:
                fig = px.pie(names=list(lang_dist.keys()), values=list(lang_dist.values()),
                           title="语言分布")
                fig.update_layout(paper_bgcolor="#1E293B", font_color="#F1F5F9")
                st.plotly_chart(fig, use_container_width=True)

        elif m == "perplexity":
            st.markdown("### 困惑度检测")
            method = m_metrics.get("method", "unknown")
            high_ppl = m_metrics.get("high_perplexity_count", 0)
            avg_ppl = m_metrics.get("average_perplexity", "N/A")
            threshold = m_metrics.get("threshold", "N/A")
            render_metric_cards([
                {"value": high_ppl, "label": "高困惑度样本", "color": "var(--warning)" if high_ppl > 0 else "var(--success)"},
                {"value": format_number(avg_ppl) if isinstance(avg_ppl, (int, float)) else avg_ppl, "label": "平均困惑度"},
                {"value": format_number(threshold) if isinstance(threshold, (int, float)) else threshold, "label": "阈值"},
            ])
            ppl_scores = m_metrics.get("perplexity_scores", [])
            if ppl_scores:
                fig = px.scatter(y=ppl_scores, title="困惑度分数",
                               labels={"y": "困惑度", "x": "样本索引"})
                fig.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
                st.plotly_chart(fig, use_container_width=True)

        elif m == "sentiment_consistency":
            st.markdown("### 情感一致性检测")
            inconsistent = m_metrics.get("inconsistent_count", 0)
            inconsistency_rate = m_metrics.get("inconsistency_rate", 0)
            render_metric_cards([
                {"value": inconsistent, "label": "不一致样本", "color": "var(--danger)" if inconsistent > 0 else "var(--success)"},
                {"value": format_number(inconsistency_rate, 4), "label": "不一致率"},
            ])

        if m_issues:
            with st.expander(f"问题详情 ({len(m_issues)} 条)"):
                for issue in m_issues[:20]:
                    idx = issue.get("index", "?")
                    itype = issue.get("type", "unknown")
                    st.markdown(f"**#{idx}** — {itype}")
                    if "suggestion" in issue:
                        st.markdown(f'<div class="suggestion-box">💡 {issue["suggestion"]}</div>', unsafe_allow_html=True)
                    if "details" in issue:
                        st.json(convert_to_serializable(issue["details"]))

with tabs[-2]:
    st.markdown("### 问题列表")
    if issues:
        issues_df = pd.DataFrame(issues)
        st.dataframe(issues_df, use_container_width=True)
        csv_data = issues_df.to_csv(index=False).encode("utf-8")
        st.download_button("导出问题列表 CSV", data=csv_data, file_name="text_issues.csv", mime="text/csv")
    else:
        st.success("未检测到问题")

with tabs[-1]:
    st.markdown("### 原始JSON")
    st.json(convert_to_serializable(result))

if st.button("📥 导出完整报告", use_container_width=True):
    report_json = json.dumps(convert_to_serializable(result), ensure_ascii=False, indent=2)
    st.download_button(
        label="下载JSON报告",
        data=report_json,
        file_name="text_analysis_report.json",
        mime="application/json",
    )
