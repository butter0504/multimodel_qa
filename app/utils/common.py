from __future__ import annotations

import json
import streamlit as st
from typing import Any, Dict, List, Optional

from modules.interface import Interface, DetectionRequest, DetectionResponse
from modules.storage import StorageLayer
from modules.config import Config


def get_interface() -> Interface:
    if "interface" not in st.session_state:
        cfg = Config.fromfile("config.yaml") if __file__ else Config()
        st.session_state.interface = Interface(cfg)
    return st.session_state.interface


def get_storage() -> StorageLayer:
    if "storage" not in st.session_state:
        cfg = Config.fromfile("config.yaml") if __file__ else Config()
        st.session_state.storage = StorageLayer(cfg)
    return st.session_state.storage


def get_config() -> Config:
    if "config" not in st.session_state:
        try:
            st.session_state.config = Config.fromfile("config.yaml")
        except Exception:
            st.session_state.config = Config()
    return st.session_state.config


def inject_css():
    st.markdown("""<style>
:root {
    --primary: #4F46E5;
    --primary-light: #818CF8;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --info: #3B82F6;
    --bg: #0F172A;
    --card-bg: #1E293B;
    --text: #F1F5F9;
    --text-muted: #94A3B8;
    --border: #334155;
}
.stApp { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    border-right: 1px solid var(--border);
}
.stButton>button {
    background: linear-gradient(135deg, var(--primary), var(--primary-light));
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(79,70,229,0.4);
}
.metric-card {
    background: var(--card-bg); border-radius: 12px;
    padding: 1.2rem; border: 1px solid var(--border);
    text-align: center; transition: all 0.3s ease;
}
.metric-card:hover { border-color: var(--primary); transform: translateY(-2px); }
.metric-card .value { font-size: 2rem; font-weight: 700; color: var(--primary-light); }
.metric-card .label { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.3rem; }
.upload-zone {
    border: 2px dashed var(--border); border-radius: 12px;
    padding: 2rem; text-align: center; transition: all 0.3s ease;
}
.upload-zone:hover { border-color: var(--primary); background: rgba(79,70,229,0.05); }
.status-tag {
    display: inline-block; padding: 0.2rem 0.8rem;
    border-radius: 9999px; font-size: 0.75rem; font-weight: 600;
}
.status-tag.success { background: rgba(16,185,129,0.15); color: var(--success); }
.status-tag.warning { background: rgba(245,158,11,0.15); color: var(--warning); }
.status-tag.danger  { background: rgba(239,68,68,0.15);  color: var(--danger); }
.status-tag.info    { background: rgba(59,130,246,0.15);  color: var(--info); }
.suggestion-box {
    background: rgba(16,185,129,0.08); border-left: 3px solid var(--success);
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
}
.warning-box {
    background: rgba(245,158,11,0.08); border-left: 3px solid var(--warning);
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
}
.danger-box {
    background: rgba(239,68,68,0.08); border-left: 3px solid var(--danger);
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
}
.tab-content { padding: 1rem 0; }
div[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 8px; }
.stDataFrame { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
</style>""", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        if st.button("🏠 返回首页", use_container_width=True):
            st.switch_page("app.py")


def render_metric_cards(metrics: List[Dict[str, Any]], cols: int = 4):
    columns = st.columns(min(cols, len(metrics)))
    for i, m in enumerate(metrics):
        with columns[i % cols]:
            color = m.get("color", "var(--primary-light)")
            st.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:{color}">{m['value']}</div>
                <div class="label">{m['label']}</div>
            </div>""", unsafe_allow_html=True)


def run_detection(source, modality: str, modules: Optional[List[str]] = None,
                  format_hint: Optional[str] = None,
                  label_column: Optional[str] = None,
                  text_column: Optional[str] = None,
                  filename_column: Optional[str] = None,
                  extra: Optional[Dict[str, Any]] = None,
                  project_name: str = "未命名项目",
                  filename: str = "") -> DetectionResponse:
    iface = get_interface()

    request = DetectionRequest(
        source=source,
        source_type="auto",
        modality=modality,
        modules=modules,
        format_hint=format_hint,
        label_column=label_column,
        text_column=text_column,
        filename_column=filename_column,
        extra=extra or {},
    )

    response = iface.process_request(request)

    if response.success:
        storage = get_storage()
        pid = storage.add_project({
            "name": project_name,
            "file_name": filename,
            "analysis_type": modality,
            "basic_info": response.data_object_info,
        })
        if pid and response.detection_result:
            storage.add_analysis_result(pid, response.detection_result)
        st.session_state.current_project_id = pid

    return response


def convert_to_serializable(obj: Any) -> Any:
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, '__float__'):
        return float(obj)
    return str(obj)


def format_number(val: Any, decimals: int = 2) -> str:
    try:
        if val is None:
            return "N/A"
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def get_modality_options() -> Dict[str, str]:
    return {
        "表格分析": "table",
        "文本分析": "text",
        "图像分析": "image",
    }


def get_module_options(modality: str) -> Dict[str, str]:
    options = {
        "image": {
            "basic_quality": "基础质量检测（模糊/曝光/噪声/分辨率/对比度）",
            "label_error": "标签错误检测（需标签）",
            "distribution_shift": "分布偏移检测（需基准集）",
            "uncertainty": "不确定性估计（需标签）",
        },
        "text": {
            "text_length": "文本长度检测",
            "duplicate": "重复检测",
            "character_anomaly": "字符异常检测",
            "language": "语言检测",
            "label_error": "标签错误检测（需标签）",
            "perplexity": "困惑度检测",
            "sentiment_consistency": "情感一致性检测（需标签）",
        },
        "table": {
            "missing_values": "缺失值检测",
            "outliers": "异常值检测",
            "duplicates": "重复行检测",
            "label_error": "标签错误检测（需标签列）",
        },
    }
    return options.get(modality, {})


def get_default_modules(modality: str) -> List[str]:
    defaults = {
        "image": ["basic_quality"],
        "text": ["text_length", "duplicate", "character_anomaly"],
        "table": ["missing_values", "outliers", "duplicates"],
    }
    return defaults.get(modality, [])


def get_format_options() -> Dict[str, str]:
    return {
        "自动检测": None,
        "COCO JSON": "coco",
        "VOC XML": "voc",
        "YOLO TXT": "yolo",
        "CSV/TSV": "csv",
        "LLM 辅助解析": "llm",
    }
