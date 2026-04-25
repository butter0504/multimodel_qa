import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import base64
import io
import numpy as np
from PIL import Image


def convert_to_serializable(obj):
    """递归转换 numpy 类型为 Python 原生类型，确保 JSON 可序列化"""
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
.card { background: var(--bg-white); border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 20px; margin-bottom: 20px; border: 1px solid var(--border-color); transition: all 0.3s ease; }
.card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
.metric-card { background: var(--bg-light); border-radius: 8px; padding: 15px; text-align: center; border-left: 4px solid var(--primary-color); }
.metric-card.success { border-left-color: var(--success-color); }
.metric-card.warning { border-left-color: var(--warning-color); }
.metric-card.error { border-left-color: var(--error-color); }
.suggestion-box { background: #EFF6FF; border-left: 4px solid #3B82F6; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; font-size: 0.9rem; }
.warning-box { background: #FEF3C7; border-left: 4px solid #F59E0B; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "图像分析详情"

with st.sidebar:
    if st.button("返回首页", width='stretch'):
        st.session_state.page = "首页"
        st.session_state.current_project = None
        st.switch_page("main.py")

if "analysis_result" not in st.session_state:
    st.error("没有分析结果，请先进行图像分析")
    st.stop()

result = st.session_state.analysis_result
project = st.session_state.get("current_project", {})
file_content = project.get("file_content", [])
filenames = st.session_state.get("image_filenames", [])

st.markdown('<h1 class="page-title">图像分析详情</h1>', unsafe_allow_html=True)

modules_run = result.get("modules_run", ["basic_quality"])

tab_names = ["概览"]
if "basic_quality" in modules_run:
    tab_names.append("基础质量")
if "label_error" in modules_run:
    tab_names.append("标签错误")
if "distribution_shift" in modules_run:
    tab_names.append("分布偏移")
if "uncertainty" in modules_run:
    tab_names.append("不确定性")
tab_names.append("问题列表")
tab_names.append("原始JSON")

tabs = st.tabs(tab_names)
tab_idx = 0

with tabs[tab_idx]:
    tab_idx += 1
    st.markdown("<h2>数据概览</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("总图像数", result["metrics"].get("total_images", 0))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
        st.metric("问题率", f"{result['metrics'].get('issue_rate', 0):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card success">', unsafe_allow_html=True)
        st.metric("平均质量", f"{result['metrics'].get('average_quality_score', 0):.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("检测模块", len(modules_run))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h3>已运行检测模块</h3>", unsafe_allow_html=True)
    module_labels = {
        "basic_quality": "✅ 基础质量检测",
        "label_error": "🏷️ 标签错误检测",
        "distribution_shift": "📊 分布偏移检测",
        "uncertainty": "❓ 不确定性估计",
    }
    for m in modules_run:
        st.markdown(f"- {module_labels.get(m, m)}")

    if "basic_quality" in modules_run and result.get("basic_quality"):
        bq = result["basic_quality"]
        if bq.get("image_metadata"):
            st.markdown("<h3>图像元数据</h3>", unsafe_allow_html=True)
            meta_rows = []
            for meta in bq["image_metadata"][:20]:
                row = {"索引": meta.get("index", 0)}
                if "height" in meta:
                    row["分辨率"] = f"{meta['height']}×{meta['width']}"
                    row["通道数"] = meta.get("channels", "?")
                    row["模式"] = meta.get("mode", "?")
                meta_rows.append(row)
            if meta_rows:
                st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)
                if len(bq["image_metadata"]) > 20:
                    st.info(f"仅显示前 20 张，共 {len(bq['image_metadata'])} 张图像")

        if bq.get("quality_scores"):
            st.markdown("<h3>质量分数分布</h3>", unsafe_allow_html=True)
            score_df = pd.DataFrame({"质量分数": bq["quality_scores"]})
            fig = px.histogram(score_df, x="质量分数", nbins=20, title="图像质量分数分布")
            fig.update_layout(xaxis_title="质量分数", yaxis_title="图像数量", height=350)
            st.plotly_chart(fig, use_container_width=True)

if "basic_quality" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>基础质量检测</h2>", unsafe_allow_html=True)

        bq = result.get("basic_quality", {})
        if bq and bq.get("issues"):
            issue_types = {}
            for issue in bq["issues"]:
                t = issue.get("type", "unknown")
                issue_types[t] = issue_types.get(t, 0) + 1

            cols = st.columns(min(len(issue_types), 4))
            type_names = {
                "blurry_image": "模糊", "under_exposed": "欠曝", "over_exposed": "过曝",
                "noisy_image": "噪声", "small_image": "低分辨率", "low_contrast": "低对比度",
                "corrupted_image": "损坏",
            }
            for i, (itype, count) in enumerate(issue_types.items()):
                with cols[i % len(cols)]:
                    st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                    st.metric(type_names.get(itype, itype), count)
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<h3>问题详情与修复建议</h3>", unsafe_allow_html=True)
            filter_type = st.selectbox("按类型筛选", ["全部"] + list(issue_types.keys()),
                                        format_func=lambda x: type_names.get(x, x) if x != "全部" else "全部")
            for issue in bq["issues"]:
                if filter_type != "全部" and issue.get("type") != filter_type:
                    continue
                idx = issue.get("index", "?")
                itype = issue.get("type", "unknown")
                suggestion = issue.get("suggestion", "")
                with st.expander(f"图像 #{idx} - {type_names.get(itype, itype)}"):
                    details = {k: v for k, v in issue.items() if k not in ("type", "index", "suggestion")}
                    if details:
                        st.json(details)
                    if suggestion:
                        st.markdown(f'<div class="suggestion-box">💡 {suggestion}</div>', unsafe_allow_html=True)
                    if idx < len(file_content):
                        try:
                            img = Image.open(io.BytesIO(file_content[idx]))
                            st.image(img, caption=f"图像 #{idx}", width=300)
                        except Exception:
                            pass
        else:
            st.success("🎉 基础质量检测未发现任何问题！")

if "label_error" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>标签错误检测（置信学习）</h2>", unsafe_allow_html=True)

        le = result.get("label_error", {})
        label_names = st.session_state.get("label_names", [])
        
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
                           f"**平均质量分数**: {di.get('mean_quality_score', 0):.3f}")

            if le.get("label_quality_scores"):
                st.markdown("<h3>标签质量分数分布</h3>", unsafe_allow_html=True)
                lq_df = pd.DataFrame({"质量分数": le["label_quality_scores"]})
                fig = px.histogram(lq_df, x="质量分数", nbins=20, title="标签质量分数分布")
                fig.add_vline(x=0.15, line_dash="dash", line_color="red", annotation_text="阈值")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            if le.get("error_indices"):
                st.markdown("<h3>Top 可疑样本</h3>", unsafe_allow_html=True)
                
                image_labels = st.session_state.get("image_labels", [])
                scores = le.get("label_quality_scores", [])
                suggested_labels = le.get("suggested_labels", {})
                
                error_data = []
                for idx in le["error_indices"]:
                    score = scores[idx] if idx < len(scores) else 0
                    
                    original_label_idx = image_labels[idx] if idx < len(image_labels) else None
                    suggested_label_idx = suggested_labels.get(str(idx))
                    
                    if label_names and original_label_idx is not None:
                        original_label_name = label_names[original_label_idx] if original_label_idx < len(label_names) else str(original_label_idx)
                    else:
                        original_label_name = str(original_label_idx) if original_label_idx is not None else "?"
                    
                    if label_names and suggested_label_idx is not None:
                        suggested_label_name = label_names[suggested_label_idx] if suggested_label_idx < len(label_names) else str(suggested_label_idx)
                    else:
                        suggested_label_name = str(suggested_label_idx) if suggested_label_idx is not None else "?"
                    
                    error_data.append({
                        "索引": idx,
                        "原标签": original_label_name,
                        "建议标签": suggested_label_name,
                        "质量分数": f"{score:.3f}",
                    })
                
                error_data.sort(key=lambda x: float(x["质量分数"]))
                st.dataframe(pd.DataFrame(error_data[:20]), use_container_width=True)

                st.markdown("<h3>可疑样本缩略图</h3>", unsafe_allow_html=True)
                cols = st.columns(5)
                for i, idx in enumerate(le["error_indices"][:10]):
                    with cols[i % 5]:
                        if idx < len(file_content):
                            try:
                                img = Image.open(io.BytesIO(file_content[idx]))
                                
                                original_label_idx = image_labels[idx] if idx < len(image_labels) else None
                                suggested_label_idx = suggested_labels.get(str(idx))
                                
                                if label_names and original_label_idx is not None:
                                    orig_name = label_names[original_label_idx] if original_label_idx < len(label_names) else str(original_label_idx)
                                else:
                                    orig_name = "?"
                                
                                if label_names and suggested_label_idx is not None:
                                    sugg_name = label_names[suggested_label_idx] if suggested_label_idx < len(label_names) else str(suggested_label_idx)
                                else:
                                    sugg_name = "?"
                                
                                caption = f"#{idx}\n{orig_name} -> {sugg_name}"
                                st.image(img, caption=caption, width=120)
                            except Exception:
                                st.warning(f"#{idx} 无法显示")
        elif le and "error" in le:
            st.warning(f"标签错误检测未运行: {le['error']}")
        else:
            st.info("未运行标签错误检测模块")

if "distribution_shift" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>分布偏移检测</h2>", unsafe_allow_html=True)

        ds = result.get("distribution_shift", {})
        if ds:
            cs = ds.get("covariate_shift", {})
            ss = ds.get("subgroup_shift", {})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("偏移维度数", cs.get("shifted_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                js_val = ss.get("brightness_js_divergence", 0)
                color = "error" if ss.get("is_shifted") else "success"
                st.markdown(f'<div class="metric-card {color}">', unsafe_allow_html=True)
                st.metric("亮度JS散度", f"{js_val:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)

            if cs.get("shifted_dimensions"):
                st.markdown("<h3>偏移维度详情</h3>", unsafe_allow_html=True)
                shift_df = pd.DataFrame(cs["shifted_dimensions"])
                st.dataframe(shift_df, use_container_width=True)

                dims = [d["dimension"] for d in cs["shifted_dimensions"]]
                stats = [d["ks_statistic"] for d in cs["shifted_dimensions"]]
                fig = go.Figure(data=[go.Bar(x=dims, y=stats)])
                fig.update_layout(title="各维度 KS 统计量", xaxis_title="维度", yaxis_title="KS 统计量", height=350)
                st.plotly_chart(fig, use_container_width=True)

            warnings = ds.get("warnings", [])
            for w in warnings:
                st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)
        else:
            st.info("未运行分布偏移检测模块")

if "uncertainty" in modules_run and tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>不确定性估计</h2>", unsafe_allow_html=True)

        ue = result.get("uncertainty", {})
        if ue and "error" not in ue:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
                st.metric("高不确定性样本", ue.get("high_uncertainty_count", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                stats = ue.get("statistics", {})
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("平均距离", f"{stats.get('mean_distance', 0):.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            if ue.get("uncertainty_scores"):
                st.markdown("<h3>不确定性分数散点图</h3>", unsafe_allow_html=True)
                unc_df = pd.DataFrame({"样本序号": range(len(ue["uncertainty_scores"])),
                                       "不确定性": ue["uncertainty_scores"]})
                threshold = ue.get("threshold", 0)
                fig = px.scatter(unc_df, x="样本序号", y="不确定性", title="不确定性分数分布")
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="阈值")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            if ue.get("high_uncertainty_indices"):
                st.markdown("<h3>高不确定性样本缩略图</h3>", unsafe_allow_html=True)
                cols = st.columns(5)
                for i, idx in enumerate(ue["high_uncertainty_indices"][:10]):
                    with cols[i % 5]:
                        if idx < len(file_content):
                            try:
                                img = Image.open(io.BytesIO(file_content[idx]))
                                st.image(img, caption=f"#{idx}", width=120)
                            except Exception:
                                st.warning(f"#{idx} 无法显示")
        elif ue and "error" in ue:
            st.warning(f"不确定性估计未运行: {ue['error']}")
        else:
            st.info("未运行不确定性估计模块")

if tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>全部问题列表</h2>", unsafe_allow_html=True)
        issues = result.get("issues", [])
        if issues:
            type_names = {
                "blurry_image": "模糊", "under_exposed": "欠曝", "over_exposed": "过曝",
                "noisy_image": "噪声", "small_image": "低分辨率", "low_contrast": "低对比度",
                "corrupted_image": "损坏", "label_error": "标签错误",
                "distribution_shift": "分布偏移", "high_uncertainty": "高不确定性",
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
            st.dataframe(issues_df, use_container_width=True)

            csv = issues_df.to_csv(index=False)
            st.download_button("📥 导出问题列表 (CSV)", data=csv,
                              file_name="image_issues.csv", mime="text/csv")
        else:
            st.success("🎉 未检测到任何问题！")

if tab_idx < len(tabs):
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("<h2>原始分析结果</h2>", unsafe_allow_html=True)
        st.json(convert_to_serializable(result))

st.markdown("<h2 style='margin-top: 30px;'>导出报告</h2>", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    report_data = {k: v for k, v in result.items() if k != "diagnosis_report"}
    report_data = convert_to_serializable(report_data)
    report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 下载 JSON 报告",
        data=report_json,
        file_name=f"image_report_{project.get('file_name', 'unknown')}.json",
        mime="application/json"
    )

    if result.get("issues"):
        issues_df_export = pd.DataFrame(result["issues"])
        csv_export = issues_df_export.to_csv(index=False)
        st.download_button(
            label="📥 下载问题清单 (CSV)",
            data=csv_export,
            file_name=f"image_issues_{project.get('file_name', 'unknown')}.csv",
            mime="text/csv"
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-top: 50px; text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)
