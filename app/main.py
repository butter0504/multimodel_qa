import streamlit as st
import requests
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 加入项目根目录

# 设置页面配置
st.set_page_config(
    page_title="多模态数据质量检测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
.main-title {
    font-size: clamp(2rem, 6vw, 3.5rem);
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary-color), var(--primary-light), var(--success-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1.5rem;
    text-align: center;
    animation: fadeIn 0.8s ease-out;
}

.subtitle {
    font-size: 1.2rem;
    color: var(--text-secondary);
    text-align: center;
    margin-bottom: 2rem;
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
    padding: 10px 20px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3);
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4);
}

/* 次要按钮 */
.secondary-button {
    background: white;
    color: var(--primary-color);
    border: 2px solid var(--primary-color);
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.secondary-button:hover {
    background: var(--primary-color);
    color: white;
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

/* 上传区域 */
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

/* 动画效果 */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in {
    animation: fadeIn 0.6s ease-out;
}

/* 分割线 */
.divider {
    height: 1px;
    background: var(--border-color);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# 缓存函数
@st.cache_data(ttl=60)
def check_api_status():
    """检查API状态并缓存60秒"""
    try:
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
    st.session_state.page = "首页"

if "recent_analyses" not in st.session_state:
    st.session_state.recent_analyses = []

if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False

# 侧边栏
with st.sidebar:
    # 主要导航
    st.markdown('<div class="sidebar-section nav-radio">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">导航</div>', unsafe_allow_html=True)
    selected_page = st.radio(
        "选择功能",
        ["首页", "表格分析", "文本分析", "图像分析", "API测试"],
        key="nav_radio",
        label_visibility="collapsed"
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
    # 对于非首页，使用Streamlit的页面导航
    if selected_page == "表格分析":
        st.switch_page("pages/01_表格分析.py")
    elif selected_page == "文本分析":
        st.switch_page("pages/02_文本分析.py")
    elif selected_page == "图像分析":
        st.switch_page("pages/03_图像分析.py")
    elif selected_page == "API测试":
        st.switch_page("pages/04_API测试.py")

# 主页面 (仅当当前页面是首页时显示)
if st.session_state.page == "首页":
    # 欢迎语
    st.markdown('<h1 class="main-title">基于置信学习的多模态数据质量检测系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">快速检测表格、文本、图像数据的质量问题</p>', unsafe_allow_html=True)
    
    # 快捷入口按钮
    st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
    st.markdown("<h3>快捷入口</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("表格分析", use_container_width=True):
            st.switch_page("pages/01_表格分析.py")
    
    with col2:
        if st.button("文本分析", use_container_width=True):
            st.switch_page("pages/02_文本分析.py")
    
    with col3:
        if st.button("图像分析", use_container_width=True):
            st.switch_page("pages/03_图像分析.py")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 快速上传区域
    st.markdown("<h3 style='margin-top: 2rem;'>快速体验</h3>", unsafe_allow_html=True)
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "拖拽文件到此处或点击上传",
        type=["csv", "xlsx", "txt", "jpg", "png"],
        accept_multiple_files=False,
        help="支持 CSV、Excel、TXT、JPG、PNG 文件"
    )
    
    if uploaded_file:
        # 根据文件类型自动跳转
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext in ['csv', 'xlsx']:
            # 存储文件到会话状态
            st.session_state.uploaded_file = uploaded_file
            st.switch_page("pages/01_表格分析.py")
        elif file_ext == 'txt':
            st.session_state.uploaded_file = uploaded_file
            st.switch_page("pages/02_文本分析.py")
        elif file_ext in ['jpg', 'png']:
            st.session_state.uploaded_file = uploaded_file
            st.switch_page("pages/03_图像分析.py")
    
    # 加载示例数据
    st.markdown("<h4 style='margin-top: 1.5rem;'>加载示例数据</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("加载表格示例", use_container_width=True):
            st.session_state.load_sample = "table"
            st.switch_page("pages/01_表格分析.py")
    
    with col2:
        if st.button("加载文本示例", use_container_width=True):
            st.session_state.load_sample = "text"
            st.switch_page("pages/02_文本分析.py")
    
    with col3:
        if st.button("加载图像示例", use_container_width=True):
            st.session_state.load_sample = "image"
            st.switch_page("pages/03_图像分析.py")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("""
<div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #E5E7EB; text-align: center; color: #6B7280;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)