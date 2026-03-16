import streamlit as st
import pandas as pd
from modules.image_detector import ImageDetector
from PIL import Image
import io
import requests

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
    st.session_state.page = "图像分析"

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
        index=3  # 默认选中图像分析
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
    elif selected_page == "表格分析":
        st.switch_page("pages/01_表格分析.py")
    elif selected_page == "文本分析":
        st.switch_page("pages/02_文本分析.py")
    elif selected_page == "API测试":
        st.switch_page("pages/04_API测试.py")

# 页面标题
st.markdown('<h1 class="page-title">图像数据质量分析</h1>', unsafe_allow_html=True)

# 文件上传区域
st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "拖拽文件到此处或点击上传图像文件",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="支持 JPG、JPEG、PNG 图像格式"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    # 读取图像数据
    images = []
    for file in uploaded_files:
        images.append(file.read())
    
    # 显示数据预览
    st.markdown("<h2 style='margin-top: 30px;'>数据预览</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"总图像数量: {len(uploaded_files)}")
        
        # 显示前3张图像
        st.write("前3张图像:")
        cols = st.columns(3)
        for i, (file, img_bytes) in enumerate(zip(uploaded_files[:3], images[:3])):
            img = Image.open(io.BytesIO(img_bytes))
            cols[i].image(img, caption=file.name, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 检测数据质量
    detector = ImageDetector()
    result = detector.detect(images)
    
    # 显示检测结果
    st.markdown("<h2 style='margin-top: 30px;'>检测结果</h2>", unsafe_allow_html=True)
    
    # 显示指标
    st.markdown("<h3>数据质量指标</h3>", unsafe_allow_html=True)
    metrics = result["metrics"]
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("总图像数", metrics["total_images"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
        st.metric("问题率", f"{metrics['issue_rate']:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 显示问题
    st.markdown("<h3>检测到的问题</h3>", unsafe_allow_html=True)
    issues = result["issues"]
    if issues:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            issues_df = pd.DataFrame(issues)
            st.dataframe(issues_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.success("🎉 未检测到任何问题，数据质量良好！")

# 页脚
st.markdown("""
<div style="margin-top: 50px; text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)
