import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k:
                convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none; }
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
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: var(--text-primary); line-height: 1.6; background-color: var(--bg-light); }
.page-title { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, var(--primary-color), var(--primary-light)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.5rem; }
.card { display: none; }
.metric-card { background: var(--bg-light); border-radius: 8px; padding: 15px; text-align: center; border-left: 4px solid var(--primary-color); }
.metric-card.success { border-left-color: var(--success-color); }
.metric-card.warning { border-left-color: var(--warning-color); }
.metric-card.error { border-left-color: var(--error-color); }
.suggestion-box { background: #EFF6FF; border-left: 4px solid #3B82F6; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; font-size: 0.9rem; }
.warning-box { background: #FEF3C7; border-left: 4px solid #F59E0B; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "文本分析详情"

with st.sidebar:
    if st.button("返回首页", width='stretch'):
        st.session_state.page = "首页"
        st.session_state.current_project = None
        st.switch_page("main.py")

if "analysis_result" not in st.session_state:
    st.error("没有分析结果，请先进行文本分析")
    if st.button("返回首页"):
        st.switch_page("main.py")
    st.stop()

result = st.session_state.analysis_result
uploaded_file = st.session_state.get("uploaded_file")

st.markdown('<h1 class="page-title">文本分析详情</h1>', unsafe_allow_html=True)

modules_run = result.get("modules_run", ["text_length", "duplicate", "character_anomaly"])

MODULE_LABELS = {
    "text_length": "📏 文本长度",
    "duplicate": "🔄 重复检测",
    "label_error": "🏷️ 标签错误",
    "character_anomaly": "⚠️ 字符异常",
    "language": "🌐 语言检测",
    "perplexity": "📊 困惑度",
    "sentiment_consistency": "💬 情感一致性",
}

tab_names = ["概览"]
for m in modules_run:
    tab_names.append(MODULE_LABELS.get(m, m))
tab_names.append("问题列表")
tab_names.append("原始JSON")

tabs = st.tabs(tab_names)
tab_idx = 0

with tabs[tab_idx]:
    tab_idx += 1
    st.markdown("<h2>数据概览</h2>", unsafe_allow_html=True)

    metrics = result.get("metrics", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("总文本数", metrics.get("total_texts", 0))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
        st.metric("问题率", f"{metrics.get('issue_rate', 0):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("平均长度", f"{metrics.get('average_length', 0):.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("检测模块", len(modules_run))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h3>已运行检测模块</h3>", unsafe_allow_html=True)
    for m in modules_run:
        st.markdown(f"- {MODULE_LABELS.get(m, m)}")

if "text_length" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>文本长度检测</h2>", unsafe_allow_html=True)

        tl = result.get("text_length", {})
        if tl:
            tl_metrics = tl.get("metrics", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("过短文本", tl_metrics.get("short_text_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("过长文本", tl_metrics.get("long_text_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("长度异常", tl_metrics.get("abnormal_length_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)

            ld = tl.get("length_distribution", {})
            stats = ld.get("statistics", {})
            if stats:
                st.markdown("<h3>长度统计</h3>", unsafe_allow_html=True)
                stat_cols = st.columns(7)
                stat_items = [
                    ("最小值", stats.get("min", 0)),
                    ("Q1", stats.get("q1", 0)),
                    ("中位数", stats.get("median", 0)),
                    ("均值", stats.get("mean", 0)),
                    ("Q3", stats.get("q3", 0)),
                    ("最大值", stats.get("max", 0)),
                    ("标准差", stats.get("std", 0)),
                ]
                for i, (label, val) in enumerate(stat_items):
                    with stat_cols[i]:
                        st.metric(label, f"{val:.1f}")

            histogram = ld.get("histogram", {})
            if histogram.get("bins"):
                st.markdown("<h3>长度分布直方图</h3>", unsafe_allow_html=True)
                fig = go.Figure(data=[go.Bar(
                    x=histogram["bins"],
                    y=histogram["counts"],
                    marker_color='#4F46E5',
                )])
                fig.update_layout(
                    title="文本长度分布",
                    xaxis_title="长度区间",
                    yaxis_title="文本数量",
                    height=400,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, width='stretch')

            tl_issues = [iss for iss in tl.get("issues", []) if iss.get("type") in ("short_text", "long_text")]
            if tl_issues:
                st.markdown("<h3>问题详情与修复建议</h3>", unsafe_allow_html=True)
                for issue in tl_issues[:20]:
                    itype = issue.get("type", "")
                    idx = issue.get("index", "?")
                    length = issue.get("length", 0)
                    suggestion = issue.get("suggestion", "")
                    type_names = {"short_text": "过短文本", "long_text": "过长文本"}
                    with st.expander(f"文本 #{idx} - {type_names.get(itype, itype)} (长度: {length})"):
                        if suggestion:
                            st.markdown(f'<div class="suggestion-box">💡 {suggestion}</div>', unsafe_allow_html=True)
                        texts_data = st.session_state.get("texts_data", [])
                        if idx < len(texts_data):
                            preview = texts_data[idx][:300]
                            st.text(f"内容预览: {preview}{'...' if len(texts_data[idx]) > 300 else ''}")
        else:
            st.info("未运行文本长度检测模块")

if "duplicate" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>文本重复检测</h2>", unsafe_allow_html=True)

        dup = result.get("duplicate", {})
        if dup:
            dup_metrics = dup.get("metrics", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("完全重复", dup_metrics.get("exact_duplicate_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("近似重复", dup_metrics.get("approximate_duplicate_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card success">', unsafe_allow_html=True)
                st.metric("唯一文本", dup_metrics.get("unique_text_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)

            dup_groups = dup.get("exact_duplicate_groups", {})
            if dup_groups:
                st.markdown("<h3>重复文本分组</h3>", unsafe_allow_html=True)
                group_data = []
                for key, indices in dup_groups.items():
                    group_data.append({
                        "文本摘要": key[:80] + "..." if len(key) > 80 else key,
                        "重复次数": len(indices),
                        "索引列表": str(indices[:10]),
                    })
                if group_data:
                    st.dataframe(pd.DataFrame(group_data), width='stretch')

            dup_issues = dup.get("issues", [])
            if dup_issues:
                st.markdown("<h3>重复文本详情</h3>", unsafe_allow_html=True)
                type_names = {"exact_duplicate": "完全重复", "approximate_duplicate": "近似重复"}
                for issue in dup_issues[:20]:
                    itype = issue.get("type", "")
                    idx = issue.get("index", "?")
                    original_idx = issue.get("original_index", "?")
                    similarity = issue.get("similarity", "")
                    label = type_names.get(itype, itype)
                    extra = f" (相似度: {similarity})" if similarity else ""
                    with st.expander(f"文本 #{idx} - {label}{extra}"):
                        st.write(f"原始文本索引: #{original_idx}")
                        suggestion = issue.get("suggestion", "")
                        if suggestion:
                            st.markdown(f'<div class="suggestion-box">💡 {suggestion}</div>', unsafe_allow_html=True)
        else:
            st.info("未运行重复检测模块")

if "label_error" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>标签错误检测（置信学习）</h2>", unsafe_allow_html=True)

        le = result.get("label_error", {})
        if le and "error" not in le:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card error">', unsafe_allow_html=True)
                st.metric("标签错误数", le.get("error_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("错误率", f"{le.get('error_rate', 0):.1%}")
                st.markdown('</div>', unsafe_allow_html=True)

            if le.get("debug_info"):
                di = le["debug_info"]
                st.markdown(f"**类别数**: {di.get('n_classes', '?')} | "
                           f"**交叉验证折数**: {di.get('cv_folds_used', '?')} | "
                           f"**特征方法**: {di.get('feature_method', 'tfidf')} | "
                           f"**特征维度**: {di.get('feature_dim', '?')} | "
                           f"**平均质量分数**: {di.get('mean_quality_score', 0):.3f}")

            if le.get("label_quality_scores"):
                st.markdown("<h3>标签质量分数分布</h3>", unsafe_allow_html=True)
                lq_df = pd.DataFrame({"质量分数": le["label_quality_scores"]})
                fig = px.histogram(lq_df, x="质量分数", nbins=20, title="标签质量分数分布")
                fig.add_vline(x=0.15, line_dash="dash", line_color="red", annotation_text="阈值")
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')

            if le.get("error_indices"):
                st.markdown("<h3>Top 可疑样本</h3>", unsafe_allow_html=True)
                scores = le.get("label_quality_scores", [])
                suggested_labels = le.get("suggested_labels", {})
                class_names = le.get("class_names", [])
                texts_data = st.session_state.get("texts_data", [])
                text_labels = st.session_state.get("text_labels", [])

                error_data = []
                for idx in le["error_indices"]:
                    score = scores[idx] if idx < len(scores) else 0
                    original_label = text_labels[idx] if idx < len(text_labels) else None
                    suggested_label_idx = suggested_labels.get(str(idx))

                    if class_names and original_label is not None:
                        try:
                            orig_name = class_names[int(original_label)] if int(original_label) < len(class_names) else str(original_label)
                        except (ValueError, TypeError):
                            orig_name = str(original_label)
                    else:
                        orig_name = str(original_label) if original_label is not None else "?"

                    if class_names and suggested_label_idx is not None:
                        try:
                            sugg_name = class_names[int(suggested_label_idx)] if int(suggested_label_idx) < len(class_names) else str(suggested_label_idx)
                        except (ValueError, TypeError):
                            sugg_name = str(suggested_label_idx)
                    else:
                        sugg_name = str(suggested_label_idx) if suggested_label_idx is not None else "?"

                    text_preview = texts_data[idx][:60] if idx < len(texts_data) else ""
                    error_data.append({
                        "索引": idx,
                        "原标签": orig_name,
                        "建议标签": sugg_name,
                        "质量分数": f"{score:.3f}",
                        "文本预览": text_preview,
                    })

                error_data.sort(key=lambda x: float(x["质量分数"]))
                st.dataframe(pd.DataFrame(error_data[:30]), width='stretch')
        elif le and "error" in le:
            st.warning(f"标签错误检测未运行: {le['error']}")
        else:
            st.info("未运行标签错误检测模块（需要提供标签）")

if "character_anomaly" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>字符异常检测</h2>", unsafe_allow_html=True)

        ca = result.get("character_anomaly", {})
        if ca:
            ca_metrics = ca.get("metrics", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card error">', unsafe_allow_html=True)
                st.metric("乱码文本", ca_metrics.get("garbled_text_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("特殊字符过多", ca_metrics.get("high_special_char_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("首尾空白", ca_metrics.get("whitespace_padding_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)

            ca_issues = ca.get("issues", [])
            if ca_issues:
                st.markdown("<h3>问题详情</h3>", unsafe_allow_html=True)
                type_names = {
                    "garbled_chars": "乱码字符",
                    "high_special_char_ratio": "特殊字符过多",
                    "whitespace_padding": "首尾空白",
                }
                for issue in ca_issues[:20]:
                    itype = issue.get("type", "")
                    idx = issue.get("index", "?")
                    label = type_names.get(itype, itype)
                    with st.expander(f"文本 #{idx} - {label}"):
                        details = {k: v for k, v in issue.items() if k not in ("type", "index", "suggestion")}
                        if details:
                            st.json(details)
                        suggestion = issue.get("suggestion", "")
                        if suggestion:
                            st.markdown(f'<div class="suggestion-box">💡 {suggestion}</div>', unsafe_allow_html=True)
                        texts_data = st.session_state.get("texts_data", [])
                        if idx < len(texts_data):
                            preview = texts_data[idx][:200]
                            st.text(f"内容预览: {preview}")
            else:
                st.success("🎉 字符异常检测未发现任何问题！")
        else:
            st.info("未运行字符异常检测模块")

if "language" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>文本语言检测</h2>", unsafe_allow_html=True)

        lang = result.get("language", {})
        if lang:
            lang_metrics = lang.get("metrics", {})
            lang_dist = lang_metrics.get("language_distribution", {})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("混合语言文本", lang_metrics.get("mixed_language_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("非目标语言文本", lang_metrics.get("non_target_language_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)

            if lang_dist:
                st.markdown("<h3>语言分布</h3>", unsafe_allow_html=True)
                lang_df = pd.DataFrame({
                    "语言": list(lang_dist.keys()),
                    "文本数": list(lang_dist.values()),
                })
                fig = px.pie(lang_df, values="文本数", names="语言", title="语言分布")
                st.plotly_chart(fig, width='stretch')

            lang_issues = lang.get("issues", [])
            if lang_issues:
                st.markdown("<h3>问题详情</h3>", unsafe_allow_html=True)
                type_names = {"mixed_language": "混合语言", "non_target_language": "非目标语言"}
                for issue in lang_issues[:20]:
                    itype = issue.get("type", "")
                    idx = issue.get("index", "?")
                    label = type_names.get(itype, itype)
                    with st.expander(f"文本 #{idx} - {label}"):
                        details = {k: v for k, v in issue.items() if k not in ("type", "index", "suggestion")}
                        if details:
                            st.json(details)
                        suggestion = issue.get("suggestion", "")
                        if suggestion:
                            st.markdown(f'<div class="suggestion-box">💡 {suggestion}</div>', unsafe_allow_html=True)
        else:
            st.info("未运行语言检测模块")

if "perplexity" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>文本困惑度检测</h2>", unsafe_allow_html=True)

        ppl = result.get("perplexity", {})
        if ppl:
            ppl_metrics = ppl.get("metrics", {})
            method = ppl.get("method", "unknown")

            st.caption(f"检测方法: {method}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("高困惑度样本", ppl_metrics.get("high_perplexity_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("平均困惑度", f"{ppl_metrics.get('mean_perplexity', 0):.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("困惑度阈值", f"{ppl_metrics.get('perplexity_threshold', 0):.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            ppl_scores = ppl.get("perplexity_scores", [])
            if ppl_scores:
                st.markdown("<h3>困惑度分数分布</h3>", unsafe_allow_html=True)
                ppl_df = pd.DataFrame({"样本序号": range(len(ppl_scores)), "困惑度": ppl_scores})
                threshold = ppl_metrics.get("perplexity_threshold", 0)
                fig = px.scatter(ppl_df, x="样本序号", y="困惑度", title="困惑度分数分布")
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="阈值")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')

            ppl_issues = ppl.get("issues", [])
            if ppl_issues:
                st.markdown("<h3>高困惑度样本</h3>", unsafe_allow_html=True)
                for issue in ppl_issues[:10]:
                    idx = issue.get("index", "?")
                    perplexity = issue.get("perplexity", 0)
                    with st.expander(f"文本 #{idx} - 困惑度: {perplexity:.2f}"):
                        suggestion = issue.get("suggestion", "")
                        if suggestion:
                            st.markdown(f'<div class="suggestion-box">💡 {suggestion}</div>', unsafe_allow_html=True)
                        texts_data = st.session_state.get("texts_data", [])
                        if idx < len(texts_data):
                            preview = texts_data[idx][:300]
                            st.text(f"内容预览: {preview}")
        else:
            st.info("未运行困惑度检测模块")

if "sentiment_consistency" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>情感一致性检测</h2>", unsafe_allow_html=True)

        sent = result.get("sentiment_consistency", {})
        if sent:
            sent_metrics = sent.get("metrics", {})
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card error">', unsafe_allow_html=True)
                st.metric("不一致样本", sent_metrics.get("inconsistency_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("不一致率", f"{sent_metrics.get('inconsistency_rate', 0):.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            sent_issues = sent.get("issues", [])
            if sent_issues:
                st.markdown("<h3>情感不一致样本</h3>", unsafe_allow_html=True)
                for issue in sent_issues[:20]:
                    idx = issue.get("index", "?")
                    text_sent = issue.get("text_sentiment", "?")
                    label_sent = issue.get("label_sentiment", "?")
                    score = issue.get("inconsistency_score", 0)
                    with st.expander(f"文本 #{idx} - 文本情感: {text_sent}, 标签情感: {label_sent} (不一致度: {score:.2f})"):
                        suggestion = issue.get("suggestion", "")
                        if suggestion:
                            st.markdown(f'<div class="suggestion-box">💡 {suggestion}</div>', unsafe_allow_html=True)
                        texts_data = st.session_state.get("texts_data", [])
                        if idx < len(texts_data):
                            preview = texts_data[idx][:300]
                            st.text(f"内容预览: {preview}")
            else:
                st.success("🎉 情感一致性检测未发现不一致！")
        else:
            st.info("未运行情感一致性检测模块（需要提供标签）")

if tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>全部问题列表</h2>", unsafe_allow_html=True)
        issues = result.get("issues", [])
        if issues:
            type_names = {
                "short_text": "过短文本", "long_text": "过长文本",
                "abnormal_length": "长度异常", "exact_duplicate": "完全重复",
                "approximate_duplicate": "近似重复", "label_error": "标签错误",
                "garbled_chars": "乱码字符", "high_special_char_ratio": "特殊字符过多",
                "whitespace_padding": "首尾空白", "mixed_language": "混合语言",
                "non_target_language": "非目标语言", "high_perplexity": "高困惑度",
                "sentiment_inconsistency": "情感不一致",
            }
            rows = []
            for issue in issues:
                row = {
                    "类型": type_names.get(issue.get("type", ""), issue.get("type", "")),
                    "索引": issue.get("index", "N/A"),
                    "修复建议": issue.get("suggestion", ""),
                }
                for k, v in issue.items():
                    if k not in ("type", "index", "suggestion"):
                        row[k] = v
                rows.append(row)
            issues_df = pd.DataFrame(rows)
            st.dataframe(issues_df, width='stretch')

            csv = issues_df.to_csv(index=False)
            st.download_button("📥 导出问题列表 (CSV)", data=csv,
                              file_name="text_issues.csv", mime="text/csv")
        else:
            st.success("🎉 未检测到任何问题！")

if tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>原始分析结果</h2>", unsafe_allow_html=True)
        st.json(convert_to_serializable(result))

st.markdown("**导出报告**")

report_data = {k: v for k, v in result.items() if k != "diagnosis_report"}
report_data = convert_to_serializable(report_data)
report_json = json.dumps(report_data, indent=2, ensure_ascii=False)

st.download_button(
    label="📥 下载 JSON 报告",
    data=report_json,
    file_name=f"text_report_{uploaded_file.name.split('.')[0] if uploaded_file else 'unknown'}.json",
    mime="application/json"
)

if result.get("issues"):
    issues_df_export = pd.DataFrame(result["issues"])
    csv_export = issues_df_export.to_csv(index=False)
    st.download_button(
        label="📥 下载问题清单 (CSV)",
        data=csv_export,
        file_name=f"text_issues_{uploaded_file.name.split('.')[0] if uploaded_file else 'unknown'}.csv",
        mime="text/csv"
    )

st.markdown("""
<div style="margin-top: 50px; text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)
