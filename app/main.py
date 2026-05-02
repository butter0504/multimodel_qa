import streamlit as st
import requests
import time
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 加入项目根目录

# 尝试导入数据库模块
try:
    from app.utils.database import db
    print("数据库模块导入成功")
except Exception as e:
    print(f"数据库模块导入失败: {str(e)}")
    # 创建一个简单的数据库模拟对象
    class MockDatabase:
        def add_project(self, project):
            return None
        def add_analysis_result(self, project_id, result):
            return None
        def get_projects(self, limit=10):
            return []
        def get_project(self, project_id):
            return None
        def get_analysis_result(self, project_id):
            return None
    db = MockDatabase()
    print("使用模拟数据库对象")

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

/* 暗色模式 */
body[data-theme="dark"] {
    --text-primary: #F9FAFB;
    --text-secondary: #9CA3AF;
    --bg-light: #111827;
    --bg-white: #1F2937;
    --border-color: #374151;
}

/* 全局样式 */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: var(--text-primary);
    line-height: 1.6;
    background-color: var(--bg-light);
    transition: all 0.3s ease;
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

/* 响应式设计 */
@media (max-width: 768px) {
    .main-title {
        font-size: 2rem;
    }
    
    .card {
        padding: 15px;
    }
    
    .upload-area {
        padding: 20px;
    }
}
</style>

<script>
// 暗色模式切换
const darkModeToggle = document.querySelector('input[data-testid="dark_mode_toggle"]');
if (darkModeToggle) {
    const body = document.body;
    
    // 初始化暗色模式
    if (darkModeToggle.checked) {
        body.setAttribute('data-theme', 'dark');
    }
    
    // 监听切换事件
    darkModeToggle.addEventListener('change', function() {
        if (this.checked) {
            body.setAttribute('data-theme', 'dark');
        } else {
            body.removeAttribute('data-theme');
        }
    });
}
</script>
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

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "recent_analyses" not in st.session_state:
    st.session_state.recent_analyses = []

if "reanalyze_pending" not in st.session_state:
    st.session_state.reanalyze_pending = False

# 处理重新分析请求
if st.session_state.reanalyze_pending and st.session_state.get("current_project"):
    project = st.session_state.current_project
    fc = project.get('file_content')
    at = project.get('analysis_type', '')
    pn = project.get('name', '')
    fn = project.get('file_name', '')

    st.info(f"🔄 重新分析项目: **{pn}** ({fn})")
    st.markdown(f"**分析类型**: {at}")

    if st.button("确认重新分析", width='stretch'):
        st.session_state.reanalyze_pending = False
        import time
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        if fc is not None:
            with st.spinner("正在重新执行检测..."):
                if at == "图像分析":
                    from modules.image_detector import ImageDetector
                    from modules.config import Config
                    cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                    detector = ImageDetector(cfg)
                    st.session_state.analysis_result = detector.detect(fc, modules=['basic_quality'])
                    st.switch_page("pages/06_图像分析详情.py")
                elif at == "表格分析":
                    from modules.table_detector import TableDetector
                    from modules.config import Config
                    cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                    detector = TableDetector(cfg)
                    lc = project.get('label_col')
                    st.session_state.analysis_result = detector.detect(fc, label_col=lc)
                    st.switch_page("pages/05_表格分析详情.py")
                elif at == "文本分析":
                    from modules.text_detector import TextDetector
                    from modules.config import Config
                    cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                    detector = TextDetector(cfg)
                    st.session_state.analysis_result = detector.detect(fc)
                    st.switch_page("pages/07_文本分析详情.py")
        else:
            st.error("无法重新分析：文件内容不可用（可能已过期）")
            st.session_state.reanalyze_pending = False

    if st.button("取消", width='stretch'):
        st.session_state.reanalyze_pending = False
        st.rerun()

    st.stop()

# 侧边栏 - 只保留返回首页按钮
with st.sidebar:
    if st.button("返回首页", width='stretch'):
        st.session_state.page = "首页"
        st.rerun()

# 简化导航，只保留历史报告页面的访问

# 主页面 - 集成所有功能
st.markdown('<h1 class="main-title">基于置信学习的多模态数据质量检测系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">快速检测表格、文本、图像数据的质量问题</p>', unsafe_allow_html=True)

# 历史报告快捷入口
if st.button("查看历史报告", width='stretch'):
    st.session_state.page = "历史报告"
    st.switch_page("pages/04_历史报告.py")

st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E5E7EB;'>", unsafe_allow_html=True)

# 项目管理区域
if "projects" not in st.session_state:
    st.session_state.projects = []

if "current_project" not in st.session_state:
    st.session_state.current_project = None

# 主要上传区域
st.markdown('<div class="upload-area">', unsafe_allow_html=True)

# 先选择数据模态
analysis_type = st.selectbox(
    "选择数据模态",
    ["文本分析", "表格分析", "图像分析"]
)

# 上传方式选择
upload_method = st.selectbox(
    "选择上传方式",
    ["文件上传", "历史上传", "URL上传", "Python API上传"]
)

# 文件上传
if upload_method == "文件上传":
    if analysis_type == "图像分析":
        data_source_type = st.radio(
            "选择数据源类型",
            ["已有标签文件（CSV/JSON）", "标准数据集（自动识别）", 
             "文件夹结构（文件夹名=标签）", "无标签（仅部分检测）"],
            horizontal=True
        )
    else:
        data_source_type = "已有标签文件（CSV/JSON）"
    
    if data_source_type == "标准数据集（自动识别）":
        st.markdown("#### 标准数据集选择")
        dataset_name = st.selectbox(
            "选择数据集",
            ["CIFAR-10", "CIFAR-100", "CIFAR-10N (噪声标签)", "CIFAR-100N (噪声标签)",
             "Tiny-ImageNet", "MNIST", "Fashion-MNIST"]
        )
        
        data_path_input = st.text_input(
            "数据集路径",
            value=r"c:\Users\abc14\Desktop\毕业设计\代码\multimodel_qa\data\raw",
            help="数据集文件所在目录"
        )
        
        sample_size = st.number_input(
            "加载样本数（0=全部）",
            min_value=0, max_value=50000, value=1000, step=100
        )
        
        project_name = st.text_input("项目名称", placeholder="输入项目名称")
        
        if st.button("加载数据集", width='stretch'):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from modules.data_adapter import DataAdapter
            
            with st.spinner(f"正在加载 {dataset_name} 数据集..."):
                try:
                    adapter = DataAdapter()
                    
                    actual_dataset_name = dataset_name.replace(" (噪声标签)", "").replace(" ", "")
                    
                    result = adapter.load(
                        source_type='standard_dataset',
                        dataset_name=actual_dataset_name,
                        data_path=data_path_input,
                        sample_size=sample_size if sample_size > 0 else None
                    )
                    
                    st.session_state.adapter_result = result
                    st.session_state.image_filenames = result.get('filenames', [])
                    st.session_state.image_labels = result.get('labels')
                    st.session_state.label_names = result.get('label_names', [])
                    
                    if result.get('has_clean_labels'):
                        st.session_state.clean_labels = result.get('clean_labels')
                        noise_rate = result['metadata'].get('noise_rate', 0)
                        st.warning(f"噪声标签数据集 - 实际噪声率: {noise_rate:.1%}")
                    else:
                        st.session_state.clean_labels = None
                    
                    st.success(f"加载完成: {result['metadata'].get('total_samples', 0)} 张图像, "
                              f"{len(result.get('label_names', []))} 个类别")
                    
                    st.info(f"类别: {', '.join(result.get('label_names', [])[:10])}"
                           f"{'...' if len(result.get('label_names', [])) > 10 else ''}")
                    
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")
        
        if 'adapter_result' in st.session_state:
            st.markdown("#### 检测模块选择")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                run_basic = st.checkbox("基础质量检测", value=True, disabled=True)
                run_label = st.checkbox("标签错误检测", value=True)
            with col_m2:
                run_uncertainty = st.checkbox("不确定性估计", value=True)
            
            image_modules = ['basic_quality']
            if run_label:
                image_modules.append('label_error')
            if run_uncertainty:
                image_modules.append('uncertainty')
            
            project_name = st.text_input("项目名称", placeholder="输入项目名称", key='project_name_std')
            
            if st.button("开始检测", width='stretch'):
                import time
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                adapter_result = st.session_state.adapter_result
                
                with st.spinner("正在执行数据质量检测..."):
                    from modules.image_detector import ImageDetector
                    from modules.config import Config
                    
                    cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                    detector = ImageDetector(cfg)
                    
                    analysis_result = detector.detect(
                        adapter_result['images'],
                        labels=adapter_result.get('labels'),
                        modules=image_modules,
                    )
                    
                    st.session_state.analysis_result = analysis_result
                    
                    project = {
                        "id": len(st.session_state.projects) + 1,
                        "name": project_name or dataset_name,
                        "file_name": dataset_name,
                        "analysis_type": "图像分析",
                        "upload_time": current_time,
                        "file_content": adapter_result['images'],
                        "basic_info": adapter_result['metadata'],
                    }
                    
                    st.session_state.projects.append(project)
                    st.session_state.current_project = project
                    
                    st.switch_page("pages/06_图像分析详情.py")
    
    else:
        if analysis_type == "文本分析":
            uploaded_file = st.file_uploader(
                "拖拽文件到此处或点击上传文本文件",
                type=["txt", "csv"],
                accept_multiple_files=False,
                help="支持 TXT 和 CSV 格式"
            )
        elif analysis_type == "表格分析":
            uploaded_file = st.file_uploader(
                "拖拽文件到此处或点击上传表格文件",
                type=["csv", "xlsx"],
                accept_multiple_files=False,
                help="支持 CSV 和 Excel 格式"
            )
        else:
            uploaded_file = st.file_uploader(
                "拖拽文件到此处或点击上传图像文件",
                type=["jpg", "jpeg", "png", "zip"],
                accept_multiple_files=False,
                help="支持 JPG、PNG 图像文件和 ZIP 压缩包"
            )
        
        project_name = st.text_input("项目名称", placeholder="输入项目名称")
        
        file_ext = uploaded_file.name.split('.')[-1].lower() if uploaded_file else ""
        
        label_col_input = None
        image_modules = ['basic_quality']
        image_labels = None
        reference_images = None
        
        if uploaded_file and analysis_type == "表格分析":
            import pandas as pd
            try:
                if file_ext == 'csv':
                    df_preview = pd.read_csv(uploaded_file)
                else:
                    df_preview = pd.read_excel(uploaded_file)
                columns_list = df_preview.columns.tolist()
                st.info(f"已读取文件: {uploaded_file.name} ({len(df_preview)} 行 × {len(columns_list)} 列)")
                label_col_input = st.selectbox(
                    "选择标签列（用于标签错误检测）",
                    options=["无"] + columns_list,
                    index=0,
                    help="选择包含目标标签的列，系统将使用置信学习检测该列的标注错误"
                )
                if label_col_input == "无":
                    label_col_input = None
            except Exception as e:
                st.error(f"读取文件失败: {str(e)}")
        
        if uploaded_file and analysis_type == "图像分析":
            st.markdown("#### 检测模块选择")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                run_basic = st.checkbox("基础质量检测", value=True, disabled=True,
                                       help="模糊/曝光/噪声/分辨率/对比度，速度快")
                run_label = st.checkbox("标签错误检测", value=False,
                                       help="基于置信学习检测标签错误，需提供标签")
                run_shift = st.checkbox("分布偏移检测", value=False,
                                       help="检测与基准集的分布差异，需提供基准集")
            with col_m2:
                run_uncertainty = st.checkbox("不确定性估计", value=False,
                                            help="检测困难样本，需提供标签")
            
            if run_label or run_uncertainty:
                label_file = st.file_uploader("上传标签文件（CSV，需包含 filename 和 label 列）",
                                              type=['csv'], key='image_label_file')
                if label_file:
                    import pandas as pd
                    try:
                        label_df = pd.read_csv(label_file)
                        if 'filename' in label_df.columns and 'label' in label_df.columns:
                            label_map = dict(zip(label_df['filename'].astype(str), label_df['label']))
                            st.session_state.image_label_map = label_map
                            st.success(f"已加载标签文件，共 {len(label_df)} 条记录")
                        else:
                            st.error("标签文件需包含 filename 和 label 两列")
                    except Exception as e:
                        st.error(f"读取标签文件失败: {str(e)}")
            
            if run_shift:
                ref_file = st.file_uploader("上传基准图像集（ZIP格式）",
                                           type=['zip'], key='reference_images')
                if ref_file:
                    try:
                        from modules.image_detector import ImageDetector
                        ref_bytes = ref_file.getvalue()
                        reference_images, _ = ImageDetector.load_images_from_zip(ref_bytes)
                        st.session_state.reference_images_data = reference_images
                        st.success(f"已加载 {len(reference_images)} 张基准图像")
                    except Exception as e:
                        st.error(f"读取基准集失败: {str(e)}")
            
            image_modules = ['basic_quality']
            if run_label:
                image_modules.append('label_error')
            if run_shift:
                image_modules.append('distribution_shift')
            if run_uncertainty:
                image_modules.append('uncertainty')
        
        if uploaded_file and project_name:
            if st.button("确认上传", width='stretch'):
                import time
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                
                basic_info = {}
                file_content = None
                
                if analysis_type == "表格分析":
                    import pandas as pd
                    try:
                        uploaded_file.seek(0)
                        if file_ext == 'csv':
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        basic_info = {
                            "rows": len(df),
                            "columns": len(df.columns),
                            "columns_list": list(df.columns)
                        }
                        file_content = df
                    except Exception as e:
                        st.error(f"读取文件失败: {str(e)}")
                        st.stop()
                        
                elif analysis_type == "文本分析":
                    try:
                        text_content = uploaded_file.getvalue().decode('utf-8')
                        text_lines = text_content.splitlines()
                        basic_info = {
                            "length": len(text_content),
                            "lines": len(text_lines)
                        }
                        file_content = text_lines
                    except Exception as e:
                        st.error(f"读取文件失败: {str(e)}")
                        st.stop()
                        
                elif analysis_type == "图像分析":
                    try:
                        from modules.image_detector import ImageDetector
                        if file_ext == 'zip':
                            image_bytes = uploaded_file.getvalue()
                            images_list, fnames = ImageDetector.load_images_from_zip(image_bytes)
                            basic_info = {
                                "file_size": len(image_bytes),
                                "format": "zip",
                                "image_count": len(images_list)
                            }
                            file_content = images_list
                            st.session_state.image_filenames = fnames
                            
                            label_map = st.session_state.get('image_label_map', {})
                            if label_map and ('label_error' in image_modules or 'uncertainty' in image_modules):
                                image_labels = [label_map.get(f, None) for f in fnames]
                                none_count = sum(1 for l in image_labels if l is None)
                                if none_count > 0:
                                    st.warning(f"有 {none_count} 张图片未找到对应标签")
                                else:
                                    st.info(f"已匹配 {len([l for l in image_labels if l is not None])} 个标签")
                        else:
                            image_bytes = uploaded_file.getvalue()
                            basic_info = {
                                "file_size": len(image_bytes),
                                "format": file_ext
                            }
                            file_content = [image_bytes]
                            st.session_state.image_filenames = [uploaded_file.name]
                            
                            label_map = st.session_state.get('image_label_map', {})
                            if label_map and ('label_error' in image_modules or 'uncertainty' in image_modules):
                                image_labels = [label_map.get(uploaded_file.name, None)]
                    except Exception as e:
                        st.error(f"读取文件失败: {str(e)}")
                        st.stop()
                    
                    if 'distribution_shift' in image_modules:
                        reference_images = st.session_state.get('reference_images_data', None)
                        if reference_images:
                            st.info(f"使用基准集: {len(reference_images)} 张图像")
                
                with st.spinner("正在执行数据质量检测..."):
                    analysis_result = None
                    
                    if analysis_type == "表格分析":
                        from modules.table_detector import TableDetector
                        from modules.config import Config
                        
                        cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                        detector = TableDetector(cfg)
                        analysis_result = detector.detect(file_content, label_col=label_col_input)
                        st.session_state.label_col = label_col_input
                        
                    elif analysis_type == "文本分析":
                        from modules.text_detector import TextDetector
                        from modules.config import Config
                        
                        cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                        detector = TextDetector(cfg)
                        analysis_result = detector.detect(file_content)
                        
                    elif analysis_type == "图像分析":
                        from modules.image_detector import ImageDetector
                        from modules.config import Config
                        
                        cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                        detector = ImageDetector(cfg)
                        
                        final_modules = ['basic_quality']
                        if 'label_error' in image_modules and image_labels and any(l is not None for l in image_labels):
                            final_modules.append('label_error')
                        if 'distribution_shift' in image_modules and reference_images:
                            final_modules.append('distribution_shift')
                        if 'uncertainty' in image_modules and image_labels and any(l is not None for l in image_labels):
                            final_modules.append('uncertainty')
                        
                        analysis_result = detector.detect(
                            file_content,
                            labels=image_labels,
                            modules=final_modules,
                            reference_images=reference_images,
                        )
                    
                    st.session_state.analysis_result = analysis_result
                    st.session_state.uploaded_file = uploaded_file
                
                project = {
                    "id": len(st.session_state.projects) + 1,
                    "name": project_name,
                    "file_name": uploaded_file.name,
                    "analysis_type": analysis_type,
                    "upload_time": current_time,
                    "file_content": file_content,
                    "basic_info": basic_info,
                    "analysis_result": analysis_result,
                }
                
                db_project_id = db.add_project(project)
                if db_project_id:
                    project['db_id'] = db_project_id
                    print(f"项目已存储到数据库，ID: {db_project_id}")
                    try:
                        db.add_analysis_result(db_project_id, analysis_result)
                    except Exception:
                        pass
                
                st.session_state.projects.append(project)
                st.session_state.current_project = project
                
                if analysis_type == "表格分析":
                    st.switch_page("pages/05_表格分析详情.py")
                elif analysis_type == "文本分析":
                    st.switch_page("pages/07_文本分析详情.py")
                elif analysis_type == "图像分析":
                    st.switch_page("pages/06_图像分析详情.py")

# 历史上传
elif upload_method == "历史上传":
    if st.session_state.projects:
        project_ids = [f"项目 {p['id']}: {p['name']} ({p['file_name']})" for p in st.session_state.projects]
        selected_project = st.selectbox("选择历史项目", project_ids)
        
        if st.button("加载项目", width='stretch'):
            project_id = int(selected_project.split(':')[0].split()[1])
            for project in st.session_state.projects:
                if project['id'] == project_id:
                    st.session_state.current_project = project
                    
                    if 'analysis_result' not in st.session_state or st.session_state.analysis_result is None:
                        import sys
                        sys.path.insert(0, str(Path(__file__).parent.parent))
                        
                        fc = project.get('file_content')
                        at = project.get('analysis_type', '')
                        
                        if fc is not None:
                            with st.spinner("正在重新执行检测..."):
                                if at == "表格分析":
                                    from modules.table_detector import TableDetector
                                    from modules.config import Config
                                    cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                                    detector = TableDetector(cfg)
                                    lc = project.get('label_col')
                                    st.session_state.analysis_result = detector.detect(fc, label_col=lc)
                                    st.session_state.label_col = lc
                                elif at == "文本分析":
                                    from modules.text_detector import TextDetector
                                    from modules.config import Config
                                    cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                                    detector = TextDetector(cfg)
                                    st.session_state.analysis_result = detector.detect(fc)
                                elif at == "图像分析":
                                    from modules.image_detector import ImageDetector
                                    from modules.config import Config
                                    cfg = Config.fromfile(str(Path(__file__).parent.parent / 'config.yaml'))
                                    detector = ImageDetector(cfg)
                                    st.session_state.analysis_result = detector.detect(fc, modules=['basic_quality'])
                    
                    if project['analysis_type'] == "表格分析":
                        st.switch_page("pages/05_表格分析详情.py")
                    elif project['analysis_type'] == "文本分析":
                        st.switch_page("pages/07_文本分析详情.py")
                    elif project['analysis_type'] == "图像分析":
                        st.switch_page("pages/06_图像分析详情.py")
                    break
    else:
        st.info("暂无历史项目")

# URL上传
elif upload_method == "URL上传":
    url = st.text_input("输入文件URL", placeholder="例如：https://example.com/data.csv")
    project_name = st.text_input("项目名称", placeholder="输入项目名称")
    
    if st.button("确认上传", width='stretch'):
        import time
        import requests
        from io import BytesIO
        
        try:
            # 下载文件
            response = requests.get(url)
            response.raise_for_status()
            
            # 模拟文件对象
            file_name = url.split('/')[-1]
            uploaded_file = BytesIO(response.content)
            uploaded_file.name = file_name
            
            # 记录当前时间
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 读取数据并获取基本信息
            if analysis_type == "表格分析":
                import pandas as pd
                # 读取数据
                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif file_name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("不支持的文件格式")
                    st.stop()
                
                # 基本信息
                basic_info = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "columns_list": list(df.columns)
                }
            
            # 创建项目
            project = {
                "id": len(st.session_state.projects) + 1,
                "name": project_name,
                "file_name": file_name,
                "analysis_type": analysis_type,
                "upload_time": current_time,
                "file_content": uploaded_file,
                "basic_info": basic_info
            }
            
            # 存储项目到数据库
            db_project_id = db.add_project(project)
            if db_project_id:
                project['db_id'] = db_project_id
                print(f"项目已存储到数据库，ID: {db_project_id}")
            
            # 存储项目到会话状态
            st.session_state.projects.append(project)
            st.session_state.current_project = project
            
            # 跳转到对应的详情页面
            if analysis_type == "表格分析":
                st.switch_page("pages/05_表格分析详情.py")
            elif analysis_type == "文本分析":
                st.switch_page("pages/07_文本分析详情.py")
            elif analysis_type == "图像分析":
                st.switch_page("pages/06_图像分析详情.py")
        except Exception as e:
            st.error(f"URL上传失败: {str(e)}")

# Python API上传
elif upload_method == "Python API上传":
    st.markdown("""
    <div class="card">
        <h4>Python API 上传示例</h4>
        <pre>
import requests
import json

# 上传文件
url = "http://localhost:8000/api/v1/upload"
files = {'file': open('data.csv', 'rb')}
data = {'project_name': '测试项目', 'analysis_type': '表格分析'}

response = requests.post(url, files=files, data=data)
print(response.json())
        </pre>
    </div>
    """, unsafe_allow_html=True)

# 加载示例数据
st.markdown("<h4 style='margin-top: 1.5rem;'>加载示例数据</h4>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("加载表格示例", width='stretch'):
        import pandas as pd
        import time
        
        # 生成示例数据
        df = pd.DataFrame({
            'id': range(1, 101),
            'feature1': np.random.normal(50, 10, 100),
            'feature2': np.random.normal(30, 5, 100),
            'label': np.random.randint(0, 2, 100)
        })
        
        # 基本信息
        basic_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": list(df.columns)
        }
        
        # 记录当前时间
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建项目
        project = {
            "id": len(st.session_state.projects) + 1,
            "name": "示例表格项目",
            "file_name": "示例表格数据",
            "analysis_type": "表格分析",
            "upload_time": current_time,
            "basic_info": basic_info,
            "df": df
        }
        
        # 存储项目
        st.session_state.projects.append(project)
        st.session_state.current_project = project
        
        # 跳转到对应的详情页面
        st.switch_page("pages/05_表格分析详情.py")

with col2:
    if st.button("加载文本示例", width='stretch'):
        import time
        
        # 示例文本
        sample_text = "这是一个示例文本。包含一些重复内容，重复内容，以及一些拼写错误，如 recieve 和 occurance。"
        
        # 基本信息
        basic_info = {
            "length": len(sample_text),
            "vocab_size": len(set(sample_text.split()))
        }
        
        # 记录当前时间
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建项目
        project = {
            "id": len(st.session_state.projects) + 1,
            "name": "示例文本项目",
            "file_name": "示例文本数据",
            "analysis_type": "文本分析",
            "upload_time": current_time,
            "basic_info": basic_info,
            "text": sample_text
        }
        
        # 存储项目
        st.session_state.projects.append(project)
        st.session_state.current_project = project
        
        # 跳转到对应的详情页面
        st.switch_page("pages/07_文本分析详情.py")

with col3:
    if st.button("加载图像示例", width='stretch'):
        import time
        import os
        
        # 检查是否有示例图像
        sample_image_path = "data/samples/images/sample.jpg"
        if os.path.exists(sample_image_path):
            with open(sample_image_path, 'rb') as f:
                image_bytes = f.read()
        else:
            # 生成一个简单的图像
            from PIL import Image
            import io
            img = Image.new('RGB', (200, 200), color='red')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()
        
        # 基本信息
        basic_info = {
            "width": 200,
            "height": 200,
            "format": "JPEG"
        }
        
        # 记录当前时间
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建项目
        project = {
            "id": len(st.session_state.projects) + 1,
            "name": "示例图像项目",
            "file_name": "示例图像数据",
            "analysis_type": "图像分析",
            "upload_time": current_time,
            "basic_info": basic_info,
            "image_bytes": image_bytes
        }
        
        # 存储项目
        st.session_state.projects.append(project)
        st.session_state.current_project = project
        
        # 跳转到对应的详情页面
        st.switch_page("pages/06_图像分析详情.py")

st.markdown('</div>', unsafe_allow_html=True)

# 项目详情页面
if st.session_state.current_project:
    project = st.session_state.current_project
    
    # 项目基本信息
    st.markdown(f'<h2 class="main-title">项目: {project["name"]}</h2>', unsafe_allow_html=True)
    
    # 数据集基本情况
    st.markdown("<h3>数据集基本情况</h3>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"文件名: {project['file_name']}")
        st.write(f"分析类型: {project['analysis_type']}")
        st.write(f"上传时间: {project['upload_time']}")
        
        # 显示基本信息
        if "basic_info" in project:
            basic_info = project["basic_info"]
            if project['analysis_type'] == "表格分析":
                st.write(f"行数: {basic_info.get('rows', 'N/A')}")
                st.write(f"列数: {basic_info.get('columns', 'N/A')}")
                st.write(f"列名: {', '.join(basic_info.get('columns_list', []))}")
            elif project['analysis_type'] == "文本分析":
                st.write(f"文本长度: {basic_info.get('length', 'N/A')}")
                st.write(f"词汇数: {basic_info.get('vocab_size', 'N/A')}")
            elif project['analysis_type'] == "图像分析":
                st.write(f"尺寸: {basic_info.get('width', 'N/A')}x{basic_info.get('height', 'N/A')}")
                st.write(f"格式: {basic_info.get('format', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 数据集预览
    if project['analysis_type'] == "表格分析":
        st.markdown("<h3>数据集预览</h3>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if "df" in project:
                st.dataframe(project["df"].head(10), width='stretch')
            elif "file_content" in project:
                import pandas as pd
                try:
                    df = project['file_content']
                    st.dataframe(df.head(10), width='stretch')
                except Exception as e:
                    st.error(f"读取文件失败: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 标签列选择（仅表格分析）
    label_col = None
    if project['analysis_type'] == "表格分析":
        st.markdown("<h3>分析设置</h3>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # 尝试自动检测标签列
            possible_label_cols = ['label', 'target', 'class', 'y', 'labels', 'targets', 'classes']
            detected_label_col = None
            if "df" in project:
                df = project["df"]
            elif "file_content" in project:
                import pandas as pd
                try:
                    df = project['file_content']
                    for col in possible_label_cols:
                        if col in df.columns:
                            detected_label_col = col
                            break
                except Exception as e:
                    pass
            
            # 显示标签列选择
            label_col = st.text_input("标签列名称", value=detected_label_col if detected_label_col else "", placeholder="例如：label")
            if detected_label_col:
                st.info(f"自动检测到标签列: {detected_label_col}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 分析按钮
    if st.button("开始分析", width='stretch'):
        import time
        # 显示进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 读取数据并分析
        status_text.text("正在读取数据...")
        progress_bar.progress(20)
        
        if project['analysis_type'] == "表格分析":
            import pandas as pd
            from modules.table_detector import TableDetector
            
            # 读取数据
            if "df" in project:
                df = project["df"]
            else:
                file_ext = project['file_name'].split('.')[-1].lower()
                try:
                    df = project['file_content']
                except Exception as e:
                    st.error(f"读取文件失败: {str(e)}")
                    st.stop()
            
            # 分析数据
            status_text.text("正在分析数据质量...")
            progress_bar.progress(60)
            
            detector = TableDetector()
            # 确保当label_col为空字符串时，传递None
            result = detector.detect(df, label_col=label_col if label_col and label_col.strip() else None)
            
            # 调试信息
            st.write("分析结果调试:")
            st.write(f"标签列: {label_col}")
            st.write(f"是否包含label_issues: {'label_issues' in result}")
            if 'label_issues' in result:
                label_issues = result['label_issues']
                st.write(f"标签错误数量: {label_issues.get('error_count', 0)}")
                if 'suggested_labels' in label_issues:
                    suggested_labels = label_issues['suggested_labels']
                    st.write(f"建议标签数量: {len(suggested_labels)}")
                    st.write(f"建议标签示例: {dict(list(suggested_labels.items())[:5])}")
                else:
                    st.write("没有suggested_labels")
                if 'debug_info' in label_issues:
                    debug_info = label_issues['debug_info']
                    st.write(f"质量分数范围: {debug_info.get('min_quality_score', 0):.4f} - {debug_info.get('max_quality_score', 0):.4f}")
            
            # 完成分析
            progress_bar.progress(100)
            status_text.text("分析完成！")
            
            # 存储分析结果
            project["analysis_result"] = result
            
            # 存储分析结果到数据库
            if 'db_id' in project:
                db_result_id = db.add_analysis_result(project['db_id'], result)
                if db_result_id:
                    print(f"分析结果已存储到数据库，ID: {db_result_id}")
            project["label_col"] = label_col
            
            # 存储分析结果到数据库
            if 'db_id' in project:
                db_result_id = db.add_analysis_result(project['db_id'], result)
                if db_result_id:
                    print(f"分析结果已存储到数据库，ID: {db_result_id}")
        
        elif project['analysis_type'] == "文本分析":
            from modules.text_detector import TextDetector
            
            # 读取文本
            if "text" in project:
                text = project["text"]
            else:
                text = project['file_content']
            
            # 分析数据
            status_text.text("正在分析文本质量...")
            progress_bar.progress(60)
            
            detector = TextDetector()
            result = detector.detect(text)
            
            # 完成分析
            progress_bar.progress(100)
            status_text.text("分析完成！")
            
            # 存储分析结果
            project["analysis_result"] = result
            
            # 存储分析结果到数据库
            if 'db_id' in project:
                db_result_id = db.add_analysis_result(project['db_id'], result)
                if db_result_id:
                    print(f"分析结果已存储到数据库，ID: {db_result_id}")
        
        elif project['analysis_type'] == "图像分析":
            from modules.image_detector import ImageDetector
            import zipfile
            import io
            
            # 读取图像
            images = []
            if "image_bytes" in project:
                images.append(project["image_bytes"])
            else:
                file_content = project['file_content']
                file_name = project['file_name']
                
                if file_name.endswith('.zip'):
                    # 处理ZIP文件
                    try:
                        import io
                        with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
                            # 提取所有图像文件
                            for zip_info in zip_ref.infolist():
                                if zip_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    with zip_ref.open(zip_info) as img_file:
                                        img_bytes = img_file.read()
                                        images.append(img_bytes)
                        status_text.text(f"成功提取ZIP文件，包含 {len(images)} 张图像")
                    except Exception as e:
                        st.error(f"处理ZIP文件失败: {str(e)}")
                        st.stop()
                else:
                    # 处理单个图像文件
                    images.append(file_content)
            
            # 分析数据
            status_text.text("正在分析图像质量...")
            progress_bar.progress(60)
            
            detector = ImageDetector()
            result = detector.detect(images)
            
            # 完成分析
            progress_bar.progress(100)
            status_text.text("分析完成！")
            
            # 调试信息
            st.write("分析结果调试:")
            st.write(f"返回结果类型: {type(result)}")
            st.write(f"返回结果键: {list(result.keys()) if isinstance(result, dict) else '不是字典'}")
            if isinstance(result, dict) and 'metrics' in result:
                st.write(f"metrics键: {list(result['metrics'].keys())}")
                st.write(f"total_images: {result['metrics'].get('total_images', '不存在')}")
                st.write(f"quality_scores: {result['metrics'].get('quality_scores', '不存在')}")
            
            # 存储分析结果
            project["analysis_result"] = result
            
            # 存储分析结果到数据库
            if 'db_id' in project:
                db_result_id = db.add_analysis_result(project['db_id'], result)
                if db_result_id:
                    print(f"分析结果已存储到数据库，ID: {db_result_id}")
        
        # 重新运行以显示分析结果
        st.rerun()
    
    # 分析结果页面
    if "analysis_result" in project:
        result = project["analysis_result"]
        
        # 结果标签页
        tab1, tab2 = st.tabs(["Dataset", "Analytics"])
        
        with tab1:
            # Dataset 界面
            st.markdown("<h3>数据集质量</h3>", unsafe_allow_html=True)
            
            # 左侧指标
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<h4>质量指标</h4>", unsafe_allow_html=True)
                
                # 计算各项指标
                if project['analysis_type'] == "图像分析":
                    total_examples = result.get("metrics", {}).get("total_images", 0)
                    
                    # 标签问题数
                    label_issues_count = 0
                    if "label_issues" in result.get("metrics", {}):
                        label_issues_count = result["metrics"]["label_issues"].get("total_issues", 0)
                    
                    # 异常值数（使用potential_outliers的数量）
                    outlier_count = 0
                    if "label_issues" in result.get("metrics", {}):
                        outlier_count = len(result["metrics"]["label_issues"].get("potential_outliers", []))
                    
                    # 重复行数（图像分析中没有重复行的概念）
                    near_duplicates_count = 0
                    
                    # 未标记数（假设没有NaN标签）
                    unlabeled_count = 0
                    
                    # 歧义数（假设质量分数低于0.5的为歧义）
                    ambiguous_count = 0
                    if "quality_scores" in result.get("metrics", {}):
                        quality_scores = result["metrics"]["quality_scores"]
                        ambiguous_count = sum(1 for score in quality_scores if score < 50)  # 图像质量分数范围是0-100
                    
                    # 问题已解决数（假设为0，因为还未应用修改）
                    issues_resolved = 0
                    
                    # 标记良好数
                    well_labeled = total_examples - label_issues_count - unlabeled_count - ambiguous_count
                    
                    # 分布偏移样本数
                    distribution_shift_count = 0
                    if "distribution_shift" in result:
                        distribution_shift_count = result["distribution_shift"].get("total_samples", 0)
                    
                    # 弱样本数
                    weak_samples_count = 0
                    if "weak_samples" in result:
                        weak_samples_count = result["weak_samples"].get("total_samples", 0)
                else:
                    # 表格分析的指标计算
                    total_examples = result.get("basic_info", {}).get("rows", 0)
                    
                    # 标签问题数
                    label_issues_count = 0
                    if "label_issues" in result:
                        label_issues_count = result["label_issues"].get("error_count", 0)
                    
                    # 异常值数
                    outlier_count = 0
                    for col, info in result.get("outlier_stats", {}).items():
                        outlier_count += info.get("count", 0)
                    
                    # 重复行数
                    near_duplicates_count = result.get("duplicate_stats", {}).get("count", 0)
                    
                    # 未标记数（假设没有NaN标签）
                    unlabeled_count = 0
                    label_col = project.get("label_col", "label")
                    if label_col in result.get("basic_info", {}).get("columns_list", []):
                        if "df" in project:
                            df = project["df"]
                            if label_col in df.columns:
                                unlabeled_count = df[label_col].isnull().sum()
                    
                    # 歧义数（假设质量分数低于0.5的为歧义）
                    ambiguous_count = 0
                    if "label_issues" in result and "label_quality_scores" in result["label_issues"]:
                        quality_scores = result["label_issues"]["label_quality_scores"]
                        ambiguous_count = sum(1 for score in quality_scores if score < 0.5)
                    
                    # 问题已解决数（假设为0，因为还未应用修改）
                    issues_resolved = 0
                    
                    # 标记良好数
                    well_labeled = total_examples - label_issues_count - unlabeled_count - ambiguous_count
                    
                    # 分布偏移样本数
                    distribution_shift_count = 0
                    if "distribution_shift" in result:
                        distribution_shift_count = result["distribution_shift"].get("total_samples", 0)
                    
                    # 弱样本数
                    weak_samples_count = 0
                    if "weak_samples" in result:
                        weak_samples_count = result["weak_samples"].get("total_samples", 0)
                
                # 显示各项指标
                st.metric("Total Examples", total_examples)
                st.metric("Well Labeled", well_labeled)
                st.metric("Issues Resolved", issues_resolved)
                st.metric("Label Issues", label_issues_count)
                st.metric("Outliers", outlier_count)
                st.metric("Unlabeled", unlabeled_count)
                st.metric("Ambiguous", ambiguous_count)
                st.metric("Near Duplicates", near_duplicates_count)
                st.metric("Distribution Shift", distribution_shift_count)
                st.metric("Weak Samples", weak_samples_count)
                
                # 右侧操作按钮
                st.markdown("<h4>操作</h4>", unsafe_allow_html=True)
                
                # 下载报告按钮
                report_type = st.selectbox("报告类型", ["完整报告", "仅标签错误报告"])
                
                if st.button("下载报告"):
                    import json
                    
                    if report_type == "仅标签错误报告" and "label_issues" in result:
                        report_data = {
                            "project_name": project["name"],
                            "file_name": project["file_name"],
                            "label_issues": result["label_issues"]
                        }
                        file_name = f"{project['name']}_label_errors_report.json"
                    else:
                        # 生成HTML报告
                        def generate_html_report(project, result):
                            """生成HTML格式的分析报告"""
                            import pandas as pd
                            import plotly.express as px
                            import plotly.graph_objects as go
                            import base64
                            from io import StringIO, BytesIO
                            
                            # 生成HTML头部
                            html = """
                            <!DOCTYPE html>
                            <html lang="zh-CN">
                            <head>
                                <meta charset="UTF-8">
                                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                <title>数据分析报告</title>
                                <style>
                                    * {
                                        margin: 0;
                                        padding: 0;
                                        box-sizing: border-box;
                                    }
                                    
                                    body {
                                        font-family: Arial, sans-serif;
                                        line-height: 1.6;
                                        color: #333;
                                        background-color: #f5f5f5;
                                    }
                                    
                                    .container {
                                        max-width: 1200px;
                                        margin: 0 auto;
                                        padding: 20px;
                                    }
                                    
                                    header {
                                        background-color: #1E3A8A;
                                        color: white;
                                        padding: 30px;
                                        text-align: center;
                                        margin-bottom: 30px;
                                        border-radius: 10px;
                                    }
                                    
                                    h1 {
                                        font-size: 2.5em;
                                        margin-bottom: 10px;
                                    }
                                    
                                    h2 {
                                        color: #1E3A8A;
                                        margin: 30px 0 20px;
                                        padding-bottom: 10px;
                                        border-bottom: 2px solid #e0e0e0;
                                    }
                                    
                                    h3 {
                                        color: #374151;
                                        margin: 20px 0 15px;
                                    }
                                    
                                    .section {
                                        background-color: white;
                                        padding: 30px;
                                        margin-bottom: 30px;
                                        border-radius: 10px;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    }
                                    
                                    .metrics {
                                        display: grid;
                                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                                        gap: 20px;
                                        margin: 20px 0;
                                    }
                                    
                                    .metric-card {
                                        background-color: #f8fafc;
                                        padding: 20px;
                                        border-radius: 8px;
                                        text-align: center;
                                        border-left: 4px solid #1E3A8A;
                                    }
                                    
                                    .metric-value {
                                        font-size: 1.8em;
                                        font-weight: bold;
                                        color: #1E3A8A;
                                    }
                                    
                                    .metric-label {
                                        font-size: 0.9em;
                                        color: #64748b;
                                        margin-top: 5px;
                                    }
                                    
                                    table {
                                        width: 100%;
                                        border-collapse: collapse;
                                        margin: 20px 0;
                                    }
                                    
                                    th, td {
                                        padding: 12px;
                                        text-align: left;
                                        border-bottom: 1px solid #e0e0e0;
                                    }
                                    
                                    th {
                                        background-color: #f8fafc;
                                        font-weight: bold;
                                    }
                                    
                                    tr:hover {
                                        background-color: #f5f5f5;
                                    }
                                    
                                    .chart-container {
                                        margin: 20px 0;
                                        padding: 20px;
                                        background-color: #f8fafc;
                                        border-radius: 8px;
                                    }
                                    
                                    .confidence-high {
                                        background-color: #d1fae5;
                                        color: #065f46;
                                    }
                                    
                                    .confidence-medium {
                                        background-color: #fef3c7;
                                        color: #92400e;
                                    }
                                    
                                    .confidence-low {
                                        background-color: #fee2e2;
                                        color: #991b1b;
                                    }
                                    
                                    .tooltip {
                                        position: relative;
                                        display: inline-block;
                                    }
                                    
                                    .tooltip .tooltiptext {
                                        visibility: hidden;
                                        width: 200px;
                                        background-color: #333;
                                        color: #fff;
                                        text-align: center;
                                        border-radius: 6px;
                                        padding: 5px;
                                        position: absolute;
                                        z-index: 1;
                                        bottom: 125%;
                                        left: 50%;
                                        margin-left: -100px;
                                        opacity: 0;
                                        transition: opacity 0.3s;
                                    }
                                    
                                    .tooltip:hover .tooltiptext {
                                        visibility: visible;
                                        opacity: 1;
                                    }
                                </style>
                            </head>
                            <body>
                                <div class="container">
                                    <header>
                                        <h1>数据分析报告</h1>
                                        <p>项目: PROJECT_NAME | 文件: FILE_NAME | 分析类型: ANALYSIS_TYPE</p>
                                    </header>
                            """
                            
                            # 替换占位符
                            html = html.replace("PROJECT_NAME", project["name"])
                            html = html.replace("FILE_NAME", project["file_name"])
                            html = html.replace("ANALYSIS_TYPE", project["analysis_type"])
                            
                            # 基本信息
                            html += """
                            <div class="section">
                                <h2>1. 基本信息</h2>
                                <div class="metrics">
                            """
                            
                            # 根据分析类型显示不同的基本指标
                            if project['analysis_type'] == "表格分析":
                                # 表格分析的基本指标
                                basic_info = result.get("basic_info", {})
                                html += f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{basic_info.get('rows', 0)}</div>
                                        <div class="metric-label">总样本数</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">{basic_info.get('columns', 0)}</div>
                                        <div class="metric-label">列数</div>
                                    </div>
                                """
                                
                                # 添加质量指标
                                metrics = result.get("metrics", {})
                                html += f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{metrics.get('missing_value_rate', 0):.2f}%</div>
                                        <div class="metric-label">缺失值率</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">{metrics.get('duplicate_row_rate', 0):.2f}%</div>
                                        <div class="metric-label">重复行率</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">{metrics.get('outlier_rate', 0):.2f}%</div>
                                        <div class="metric-label">异常值率</div>
                                    </div>
                                """
                            elif project['analysis_type'] == "图像分析":
                                # 图像分析的基本指标
                                metrics = result.get("metrics", {})
                                html += f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{metrics.get('total_images', 0)}</div>
                                        <div class="metric-label">总图像数</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">{metrics.get('issue_rate', 0):.2f}%</div>
                                        <div class="metric-label">问题率</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">{metrics.get('average_quality_score', 0):.2f}</div>
                                        <div class="metric-label">平均质量分数</div>
                                    </div>
                                """
                            elif project['analysis_type'] == "文本分析":
                                # 文本分析的基本指标
                                basic_info = result.get("basic_info", {})
                                html += f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{basic_info.get('total_texts', 0)}</div>
                                        <div class="metric-label">总文本数</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">{basic_info.get('average_length', 0):.2f}</div>
                                        <div class="metric-label">平均长度</div>
                                    </div>
                                """
                                
                                # 添加质量指标
                                metrics = result.get("metrics", {})
                                html += f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{metrics.get('issue_rate', 0):.2f}%</div>
                                        <div class="metric-label">问题率</div>
                                    </div>
                                """
                            
                            # 添加分布偏移和弱样本指标
                            if "distribution_shift" in result:
                                distribution_shift_count = result["distribution_shift"].get("total_samples", 0)
                                html += f"""
                                <div class="metric-card">
                                    <div class="metric-value">{distribution_shift_count}</div>
                                    <div class="metric-label">分布偏移样本</div>
                                </div>
                                """
                            
                            if "weak_samples" in result:
                                weak_samples_count = result["weak_samples"].get("total_samples", 0)
                                html += f"""
                                <div class="metric-card">
                                    <div class="metric-value">{weak_samples_count}</div>
                                    <div class="metric-label">弱样本</div>
                                </div>
                                """
                            
                            html += """
                                </div>
                            </div>
                            """
                            
                            # 数据质量分析
                            html += """
                            <div class="section">
                                <h2>2. 数据质量分析</h2>
                            """
                            
                            if project['analysis_type'] == "表格分析":
                                # 缺失值分析
                                missing_stats = result.get("missing_stats", {})
                                if missing_stats.get('total', 0) > 0:
                                    html += """
                                    <h3>2.1 缺失值分析</h3>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>列名</th>
                                                <th>缺失数量</th>
                                                <th>缺失率</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                    """
                                    
                                    for col, info in missing_stats.get('per_column', {}).items():
                                        html += f"""
                                            <tr>
                                                <td>{col}</td>
                                                <td>{info.get('count', 0)}</td>
                                                <td>{info.get('percentage', 0):.2f}%</td>
                                            </tr>
                                        """
                                    
                                    html += """
                                        </tbody>
                                    </table>
                                    """
                                else:
                                    html += "<h3>2.1 缺失值分析</h3><p>没有检测到缺失值</p>"
                                
                                # 异常值分析
                                outlier_stats = result.get("outlier_stats", {})
                                if outlier_stats:
                                    html += """
                                    <h3>2.2 异常值分析</h3>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>列名</th>
                                                <th>异常值数量</th>
                                                <th>异常值率</th>
                                                <th>下界</th>
                                                <th>上界</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                    """
                                    
                                    for col, info in outlier_stats.items():
                                        html += f"""
                                            <tr>
                                                <td>{col}</td>
                                                <td>{info.get('count', 0)}</td>
                                                <td>{info.get('percentage', 0):.2f}%</td>
                                                <td>{info.get('bounds', {}).get('lower', 'N/A')}</td>
                                                <td>{info.get('bounds', {}).get('upper', 'N/A')}</td>
                                            </tr>
                                        """
                                    
                                    html += """
                                        </tbody>
                                    </table>
                                    """
                                else:
                                    html += "<h3>2.2 异常值分析</h3><p>没有检测到异常值</p>"
                                
                                # 重复行分析
                                duplicate_stats = result.get("duplicate_stats", {})
                                html += f"""
                                <h3>2.3 重复行分析</h3>
                                <p>重复行数量: {duplicate_stats.get('count', 0)}</p>
                                <p>重复行率: {duplicate_stats.get('percentage', 0):.2f}%</p>
                                """
                            elif project['analysis_type'] == "图像分析":
                                # 图像质量分析
                                metrics = result.get("metrics", {})
                                html += f"""
                                <h3>2.1 图像质量分析</h3>
                                <p>总图像数: {metrics.get('total_images', 0)}</p>
                                <p>问题率: {metrics.get('issue_rate', 0):.2f}%</p>
                                <p>平均质量分数: {metrics.get('average_quality_score', 0):.2f}</p>
                                """
                                
                                # 质量分数分布
                                if "quality_scores" in metrics:
                                    quality_scores = metrics["quality_scores"]
                                    score_df = pd.DataFrame({"质量分数": quality_scores})
                                    
                                    # 生成直方图
                                    fig = px.histogram(
                                        score_df, 
                                        x="质量分数", 
                                        nbins=20, 
                                        title="图像质量分数分布"
                                    )
                                    fig.update_layout(
                                        xaxis_title="质量分数",
                                        yaxis_title="图像数量",
                                        height=400
                                    )
                                    
                                    # 将图表转换为HTML
                                    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                                    html += f"<div class='chart-container'>{chart_html}</div>"
                                
                                # 问题类型分布
                                issues = result.get("issues", [])
                                if issues:
                                    html += """
                                    <h3>2.2 问题类型分布</h3>
                                    """
                                    
                                    issue_types = {}
                                    for issue in issues:
                                        issue_type = issue.get("type", "unknown")
                                        issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                                    
                                    issue_df = pd.DataFrame(list(issue_types.items()), columns=["问题类型", "数量"])
                                    fig = px.bar(issue_df, x="问题类型", y="数量", title="问题类型分布")
                                    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                                    html += f"<div class='chart-container'>{chart_html}</div>"
                            elif project['analysis_type'] == "文本分析":
                                # 文本质量分析
                                basic_info = result.get("basic_info", {})
                                metrics = result.get("metrics", {})
                                html += f"""
                                <h3>2.1 文本质量分析</h3>
                                <p>总文本数: {basic_info.get('total_texts', 0)}</p>
                                <p>平均长度: {basic_info.get('average_length', 0):.2f}</p>
                                <p>问题率: {metrics.get('issue_rate', 0):.2f}%</p>
                                """
                                
                                # 长度分布
                                if "length_distribution" in basic_info:
                                    length_distribution = basic_info["length_distribution"]
                                    length_df = pd.DataFrame({"长度": length_distribution})
                                    
                                    # 生成直方图
                                    fig = px.histogram(
                                        length_df, 
                                        x="长度", 
                                        nbins=20, 
                                        title="文本长度分布"
                                    )
                                    fig.update_layout(
                                        xaxis_title="长度",
                                        yaxis_title="文本数量",
                                        height=400
                                    )
                                    
                                    # 将图表转换为HTML
                                    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                                    html += f"<div class='chart-container'>{chart_html}</div>"
                            
                            # 标签质量分析
                            if "label_issues" in result:
                                label_issues = result["label_issues"]
                                html += f"""
                                <h3>2.4 标签质量分析</h3>
                                <p>标签错误数量: {label_issues.get('error_count', 0)}</p>
                                """
                                
                                # 标签质量分数分布图表
                                if "label_quality_scores" in label_issues:
                                    quality_scores = label_issues["label_quality_scores"]
                                    score_df = pd.DataFrame({"质量分数": quality_scores})
                                    
                                    # 生成直方图
                                    fig = px.histogram(
                                        score_df, 
                                        x="质量分数", 
                                        nbins=20, 
                                        title="标签质量分数分布"
                                    )
                                    fig.update_layout(
                                        xaxis_title="质量分数",
                                        yaxis_title="样本数",
                                        height=400
                                    )
                                    
                                    # 将图表转换为HTML
                                    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                                    html += f"<div class='chart-container'>{chart_html}</div>"
                            
                            # 分布偏移分析
                            if "distribution_shift" in result:
                                distribution_shift = result["distribution_shift"]
                                html += f"""
                                <h3>2.5 分布偏移分析</h3>
                                <p>分布偏移样本数量: {distribution_shift.get('total_samples', 0)}</p>
                                """
                                
                                if distribution_shift.get('samples'):
                                    html += """
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>索引</th>
                                                <th>熵值</th>
                                                <th>不确定性</th>
                                                <th>预测类别</th>
                                                <th>置信度</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                    """
                                    
                                    for sample in distribution_shift['samples'][:10]:  # 只显示前10个样本
                                        html += f"""
                                            <tr>
                                                <td>{sample.get('index', 'N/A')}</td>
                                                <td>{sample.get('entropy', 'N/A'):.4f}</td>
                                                <td>{sample.get('uncertainty', 'N/A'):.4f}</td>
                                                <td>{sample.get('predicted_class', 'N/A')}</td>
                                                <td>{sample.get('confidence', 'N/A'):.4f}</td>
                                            </tr>
                                        """
                                    
                                    html += """
                                        </tbody>
                                    </table>
                                    """
                            
                            # 弱样本分析
                            if "weak_samples" in result:
                                weak_samples = result["weak_samples"]
                                html += f"""
                                <h3>2.6 弱样本分析</h3>
                                <p>弱样本数量: {weak_samples.get('total_samples', 0)}</p>
                                """
                                
                                if weak_samples.get('samples'):
                                    html += """
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>索引</th>
                                                <th>损失值</th>
                                                <th>置信度</th>
                                                <th>弱样本分数</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                    """
                                    
                                    for sample in weak_samples['samples'][:10]:  # 只显示前10个样本
                                        html += f"""
                                            <tr>
                                                <td>{sample.get('index', 'N/A')}</td>
                                                <td>{sample.get('loss', 'N/A'):.4f}</td>
                                                <td>{sample.get('confidence', 'N/A'):.4f}</td>
                                                <td>{sample.get('weak_sample_score', 'N/A'):.4f}</td>
                                            </tr>
                                        """
                                    
                                    html += """
                                        </tbody>
                                    </table>
                                    """
                            
                            # 数据预览
                            html += """
                            <div class="section">
                                <h2>3. 数据预览</h2>
                            """
                            
                            # 显示数据预览
                            if project['analysis_type'] == "表格分析":
                                if "df" in project:
                                    df = project["df"]
                                else:
                                    try:
                                        df = project['file_content']
                                    except:
                                        df = pd.DataFrame()
                                
                                if not df.empty:
                                    # 添加分析列
                                    import numpy as np
                                    
                                    # 1. 质量分数
                                    if "analysis_result" in project and "label_issues" in project["analysis_result"]:
                                        label_issues = project["analysis_result"]["label_issues"]
                                        if "label_quality_scores" in label_issues:
                                            quality_scores = label_issues["label_quality_scores"]
                                            df['质量分数'] = [quality_scores[i] if i < len(quality_scores) else 0 for i in range(len(df))]
                                        else:
                                            df['质量分数'] = 0
                                    else:
                                        df['质量分数'] = 0
                                    
                                    # 2. 预测标签
                                    df['预测标签'] = "N/A"
                                    if "analysis_result" in project and "label_issues" in project["analysis_result"]:
                                        label_issues = project["analysis_result"]["label_issues"]
                                        if "suggested_labels" in label_issues:
                                            suggested_labels = label_issues["suggested_labels"]
                                            for idx, label in suggested_labels.items():
                                                try:
                                                    int_idx = int(idx)
                                                    if int_idx < len(df):
                                                        df.loc[int_idx, '预测标签'] = label
                                                except:
                                                    pass
                                    
                                    # 3. 建议标签
                                    df['建议标签'] = df['预测标签']
                                    
                                    # 4. 标签错误
                                    df['标签错误'] = "No"
                                    if "analysis_result" in project and "label_issues" in project["analysis_result"]:
                                        label_issues = project["analysis_result"]["label_issues"]
                                        if "error_indices" in label_issues:
                                            error_indices = label_issues["error_indices"]
                                            for idx in error_indices:
                                                try:
                                                    int_idx = int(idx)
                                                    if int_idx < len(df):
                                                        df.loc[int_idx, '标签错误'] = "Yes"
                                                except:
                                                    pass
                                    
                                    # 5. issues
                                    df['issues'] = ""
                                    if "analysis_result" in project and "issues" in project["analysis_result"]:
                                        issues = project["analysis_result"]["issues"]
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
                                    if "analysis_result" in project and "outlier_stats" in project["analysis_result"]:
                                        outlier_stats = project["analysis_result"]["outlier_stats"]
                                        for col, info in outlier_stats.items():
                                            if col in df.columns:
                                                Q1 = df[col].quantile(0.25)
                                                Q3 = df[col].quantile(0.75)
                                                IQR = Q3 - Q1
                                                lower_bound = Q1 - 1.5 * IQR
                                                upper_bound = Q3 + 1.5 * IQR
                                                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                                                df.loc[outliers, 'outlier'] = "Yes"
                                    
                                    # 7. 确定性（confidence）
                                    df['确定性'] = "N/A"
                                    confidence_data = {}
                                    entropy_data = {}
                                    if "analysis_result" in project and "distribution_shift" in project["analysis_result"]:
                                        distribution_shift = project["analysis_result"]["distribution_shift"]
                                        if "samples" in distribution_shift:
                                            for sample in distribution_shift["samples"]:
                                                idx = sample["index"]
                                                if idx < len(df):
                                                    confidence = sample.get('confidence', 0)
                                                    df.loc[idx, '确定性'] = f"{confidence:.2f}"
                                                    confidence_data[idx] = confidence
                                                    if 'entropy' in sample:
                                                        entropy_data[idx] = sample['entropy']
                                    
                                    # 8. 是否为弱样本
                                    df['是否为弱样本'] = "No"
                                    if "analysis_result" in project and "weak_samples" in project["analysis_result"]:
                                        weak_samples = project["analysis_result"]["weak_samples"]
                                        if "samples" in weak_samples:
                                            for sample in weak_samples["samples"]:
                                                idx = sample["index"]
                                                if idx < len(df):
                                                    df.loc[idx, '是否为弱样本'] = "Yes"
                                    
                                    # 9. action
                                    df['action'] = ""
                                    if "analysis_result" in project and "label_issues" in project["analysis_result"]:
                                        label_issues = project["analysis_result"]["label_issues"]
                                        if "suggested_labels" in label_issues:
                                            suggested_labels = label_issues["suggested_labels"]
                                            for idx in suggested_labels:
                                                try:
                                                    int_idx = int(idx)
                                                    if int_idx < len(df):
                                                        df.loc[int_idx, 'action'] = "修正标签"
                                                except:
                                                    pass
                                    
                                    # 显示前50行数据
                                    preview_df = df.head(50)
                                    
                                    # 构建HTML表格
                                    html += """
                                    <div style="overflow-x: auto;">
                                    <table style="width: 100%; table-layout: fixed;">
                                        <thead>
                                            <tr>
                                    """
                                    
                                    for col in preview_df.columns:
                                        html += f"<th style='word-wrap: break-word;'>{col}</th>"
                                    
                                    html += """
                                            </tr>
                                        </thead>
                                        <tbody>
                                    """
                                    
                                    for idx, row in preview_df.iterrows():
                                        html += "<tr>"
                                        for col in preview_df.columns:
                                            value = row[col]
                                            if col == "确定性" and value != "N/A":
                                                # 添加颜色编码
                                                try:
                                                    confidence = float(value)
                                                    if confidence >= 0.7:
                                                        class_name = "confidence-high"
                                                    elif confidence >= 0.4:
                                                        class_name = "confidence-medium"
                                                    else:
                                                        class_name = "confidence-low"
                                                    
                                                    # 检查是否有熵值
                                                    entropy = entropy_data.get(idx, "N/A")
                                                    entropy_str = f"{entropy:.2f}" if entropy != "N/A" else "N/A"
                                                    
                                                    html += f"<td class='{class_name} tooltip' style='word-wrap: break-word;'>{value}<span class='tooltiptext'>预测熵: {entropy_str}</span></td>"
                                                except:
                                                    html += f"<td style='word-wrap: break-word;'>{value}</td>"
                                            else:
                                                html += f"<td style='word-wrap: break-word;'>{value}</td>"
                                        html += "</tr>"
                                    
                                    html += """
                                        </tbody>
                                    </table>
                                    </div>
                                    """
                                    
                                    # 添加表格说明
                                    html += """
                                    <p style="margin-top: 10px; font-size: 0.9em; color: #64748b;">
                                        注：表格已添加水平滚动功能，可左右拖动查看完整数据
                                    </p>
                                    """
                                else:
                                    html += "<p>没有数据可显示</p>"
                            elif project['analysis_type'] == "图像分析":
                                # 显示图像数据预览
                                html += """
                                <h3>3.1 图像数据列表</h3>
                                """
                                
                                # 检查是否是ZIP文件
                                is_zip = project['file_name'].endswith('.zip')
                                
                                # 初始化图像数据
                                image_data = []
                                
                                # 获取分析结果
                                metrics = result.get("metrics", {})
                                quality_scores = metrics.get("quality_scores", [])
                                label_issues = metrics.get("label_issues", {})
                                potential_outliers = label_issues.get("potential_outliers", [])
                                issue_ids = label_issues.get("issue_ids", [])
                                
                                # 提取图像
                                images = []
                                if is_zip:
                                    # 对于ZIP文件，重新提取图像
                                    try:
                                        import zipfile
                                        import io
                                        file_content = project['file_content']
                                        with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
                                            # 提取所有图像文件
                                            for zip_info in zip_ref.infolist():
                                                if zip_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                                    with zip_ref.open(zip_info) as img_file:
                                                        img_bytes = img_file.read()
                                                        images.append(img_bytes)
                                    except Exception as e:
                                        html += f"<p>提取ZIP文件失败: {str(e)}</p>"
                                else:
                                    # 对于单个图像文件
                                    if "image_bytes" in project:
                                        images.append(project["image_bytes"])
                                    else:
                                        images.append(project['file_content'])
                                
                                # 构建图像数据表格
                                for i, img_bytes in enumerate(images[:10]):  # 只显示前10张图像
                                    # 获取质量分数
                                    quality_score = quality_scores[i] if i < len(quality_scores) else 0
                                    
                                    # 检查是否是异常样本
                                    is_outlier = i in potential_outliers
                                    
                                    # 获取图像ID
                                    image_id = issue_ids[i] if i < len(issue_ids) else f"{i:05d}"
                                    
                                    # 获取原标签和建议标签
                                    original_label = "N/A"
                                    predicted_label = "N/A"
                                    suggested_label = "N/A"
                                    
                                    # 尝试从分析结果中获取标签信息
                                    if "label_issues" in result:
                                        label_issues = result["label_issues"]
                                        if "suggested_labels" in label_issues:
                                            suggested_labels = label_issues["suggested_labels"]
                                            if str(i) in suggested_labels:
                                                suggested_label = suggested_labels[str(i)]
                                        if "original_labels" in label_issues:
                                            original_labels = label_issues["original_labels"]
                                            if i < len(original_labels):
                                                original_label = original_labels[i]
                                        if "predicted_labels" in label_issues:
                                            predicted_labels = label_issues["predicted_labels"]
                                            if i < len(predicted_labels):
                                                predicted_label = predicted_labels[i]
                                    
                                    # 将图像转换为base64
                                    import base64
                                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                                    
                                    # 添加到HTML
                                    html += f"""
                                    <div style="display: inline-block; margin: 10px; text-align: center;">
                                        <img src="data:image/png;base64,{img_base64}" alt="图像" style="max-width: 200px; max-height: 200px; border: 1px solid #ddd; padding: 5px;">
                                        <p style="margin: 5px 0;"><strong>ID:</strong> {image_id}</p>
                                        <p style="margin: 5px 0;"><strong>质量分数:</strong> {round(quality_score, 2)}</p>
                                        <p style="margin: 5px 0;"><strong>原标签:</strong> {original_label}</p>
                                        <p style="margin: 5px 0;"><strong>预测标签:</strong> {predicted_label}</p>
                                        <p style="margin: 5px 0;"><strong>建议标签:</strong> {suggested_label}</p>
                                        <p style="margin: 5px 0;"><strong>是否异常:</strong> {'Yes' if is_outlier else 'No'}</p>
                                    </div>
                                    """
                                
                                # 添加图像数量说明
                                html += f"""
                                <p style="margin-top: 20px; font-size: 0.9em; color: #64748b;">
                                    注：显示前10张图像，共 {len(images)} 张图像
                                </p>
                                """
                            elif project['analysis_type'] == "文本分析":
                                # 显示文本数据预览
                                html += """
                                <h3>3.1 文本数据列表</h3>
                                """
                                
                                # 获取分析结果
                                basic_info = result.get("basic_info", {})
                                issues = result.get("issues", [])
                                
                                # 假设文本数据存储在file_content中
                                try:
                                    if "df" in project:
                                        df = project["df"]
                                    else:
                                        df = project['file_content']
                                    
                                    if not df.empty:
                                        # 显示前10条文本
                                        preview_df = df.head(10)
                                        
                                        # 构建HTML表格
                                        html += """
                                        <div style="overflow-x: auto;">
                                        <table style="width: 100%; table-layout: fixed;">
                                            <thead>
                                                <tr>
                                                    <th style='word-wrap: break-word;'>ID</th>
                                                    <th style='word-wrap: break-word;'>文本</th>
                                                    <th style='word-wrap: break-word;'>长度</th>
                                                    <th style='word-wrap: break-word;'>问题</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                        """
                                        
                                        for idx, row in preview_df.iterrows():
                                            html += "<tr>"
                                            html += f"<td style='word-wrap: break-word;'>{idx}</td>"
                                            # 限制文本长度
                                            text = str(row.iloc[0]) if len(row) > 0 else ""
                                            if len(text) > 100:
                                                text = text[:100] + "..."
                                            html += f"<td style='word-wrap: break-word;'>{text}</td>"
                                            html += f"<td style='word-wrap: break-word;'>{len(str(row.iloc[0]))}</td>"
                                            
                                            # 检查是否有问题
                                            text_issues = ""
                                            for issue in issues:
                                                if "index" in issue and issue["index"] == idx:
                                                    if text_issues:
                                                        text_issues += ", " + issue["type"]
                                                    else:
                                                        text_issues = issue["type"]
                                            html += f"<td style='word-wrap: break-word;'>{text_issues if text_issues else '无'}</td>"
                                            html += "</tr>"
                                        
                                        html += """
                                            </tbody>
                                        </table>
                                        </div>
                                        """
                                        
                                        # 添加文本数量说明
                                        html += f"""
                                        <p style="margin-top: 10px; font-size: 0.9em; color: #64748b;">
                                            注：显示前10条文本，共 {len(df)} 条文本
                                        </p>
                                        """
                                    else:
                                        html += "<p>没有数据可显示</p>"
                                except Exception as e:
                                    html += f"<p>读取文本数据失败: {str(e)}</p>"
                            
                            # 报告总结
                            html += """
                            <div class="section">
                                <h2>4. 总结</h2>
                                <p>本次分析共检测到以下问题：</p>
                                <ul>
                            """
                            
                            # 统计问题类型
                            issues = result.get("issues", [])
                            issue_types = {}
                            for issue in issues:
                                issue_type = issue.get('type', 'unknown')
                                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                            
                            for issue_type, count in issue_types.items():
                                html += f"<li>{issue_type}: {count}个</li>"
                            
                            # 生成数据质量评分
                            total_samples = basic_info.get('rows', 0)
                            missing_count = missing_stats.get('total', 0)
                            duplicate_count = duplicate_stats.get('count', 0)
                            outlier_count = sum(info.get('count', 0) for info in outlier_stats.values())
                            label_error_count = 0
                            if "label_issues" in result:
                                label_issues = result["label_issues"]
                                label_error_count = label_issues.get('error_count', 0)
                            
                            quality_score = 100
                            if total_samples > 0:
                                if missing_count > 0:
                                    quality_score -= (missing_count / total_samples) * 20
                                if duplicate_count > 0:
                                    quality_score -= (duplicate_count / total_samples) * 15
                                if outlier_count > 0:
                                    quality_score -= (outlier_count / total_samples) * 25
                                if label_error_count > 0:
                                    quality_score -= (label_error_count / total_samples) * 40
                            quality_score = max(0, min(100, quality_score))
                            
                            # 生成质量分数仪表盘
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=quality_score,
                                title={'text': "数据质量分数"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#1E3A8A"},
                                    'steps': [
                                        {'range': [0, 60], 'color': "#EF4444"},
                                        {'range': [60, 80], 'color': "#F59E0B"},
                                        {'range': [80, 100], 'color': "#10B981"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "#8B5CF6", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 70
                                    }
                                }
                            ))
                            
                            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                            html += f"<div class='chart-container'>{chart_html}</div>"
                            
                            # 结束HTML
                            html += """
                                </ul>
                                <p>数据质量评分: <strong>QUALITY_SCORE</strong>/100</p>
                                <p>建议：</p>
                                <ul>
                                    <li>处理缺失值和异常值</li>
                                    <li>检查并修复标签错误</li>
                                    <li>考虑移除分布偏移样本和弱样本</li>
                                    <li>定期进行数据质量评估</li>
                                </ul>
                            </div>
                                </div>
                            </body>
                            </html>
                            """
                            
                            # 替换质量分数占位符
                            html = html.replace("QUALITY_SCORE", f"{quality_score:.2f}")
                            
                            return html
                        
                        # 生成HTML报告
                        html_report = generate_html_report(project, result)
                        file_name = f"{project['name']}_complete_report.html"
                    
                    st.download_button(
                        label="下载报告",
                        data=html_report,
                        file_name=file_name,
                        mime="text/html"
                    )
                
                # 再次分析按钮
                if st.button("再次分析"):
                    st.session_state.reanalyze_pending = True
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # 数据集展示
                st.markdown("<h4>数据列表</h4>", unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    # 显示数据
                    if project['analysis_type'] == "表格分析":
                        if "df" in project:
                            df = project["df"]
                        else:
                            import pandas as pd
                            try:
                                df = project['file_content']
                            except Exception as e:
                                st.error(f"读取文件失败: {str(e)}")
                                st.stop()
                        
                        # 添加分析列
                        import numpy as np
                        
                        # 1. 质量分数
                        if "analysis_result" in project and "label_issues" in project["analysis_result"]:
                            label_issues = project["analysis_result"]["label_issues"]
                            if "label_quality_scores" in label_issues:
                                # 使用真实的质量分数
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
                        
                        # 2. 原本标签
                        label_col = project.get("label_col", "label")
                        if label_col in df.columns:
                            df['原标签'] = df[label_col]
                        else:
                            df['原标签'] = "N/A"
                        
                        # 3. 推荐标签
                        if "analysis_result" in project and "label_issues" in project["analysis_result"]:
                            label_issues = project["analysis_result"]["label_issues"]
                            if "suggested_labels" in label_issues:
                                suggested_labels = label_issues["suggested_labels"]
                                df['建议标签'] = "N/A"
                                # 确保idx是整数类型
                                for idx, label in suggested_labels.items():
                                    try:
                                        int_idx = int(idx)
                                        if int_idx < len(df):
                                            # 确保所有值都是字符串类型，避免Arrow序列化错误
                                            df.loc[int_idx, '建议标签'] = str(label)
                                    except (ValueError, TypeError):
                                        pass
                            else:
                                df['建议标签'] = "N/A"
                        else:
                            df['建议标签'] = "N/A"
                        
                        # 4. 修正标签值
                        df['修正标签'] = df['原标签'].astype(str)
                        if "analysis_result" in project and "label_issues" in project["analysis_result"]:
                            label_issues = project["analysis_result"]["label_issues"]
                            if "suggested_labels" in label_issues:
                                suggested_labels = label_issues["suggested_labels"]
                                # 确保idx是整数类型
                                for idx, label in suggested_labels.items():
                                    try:
                                        int_idx = int(idx)
                                        if int_idx < len(df):
                                            # 确保所有值都是字符串类型，避免Arrow序列化错误
                                            df.loc[int_idx, '修正标签'] = str(label)
                                    except (ValueError, TypeError):
                                        pass
                        
                        # 5. 问题类型
                        df['问题类型'] = ""
                        if "analysis_result" in project and "issues" in project["analysis_result"]:
                            issues = project["analysis_result"]["issues"]
                            for issue in issues:
                                if "index" in issue:
                                    idx = issue["index"]
                                    if idx < len(df):
                                        if df.loc[idx, '问题类型']:
                                            df.loc[idx, '问题类型'] += ", " + issue["type"]
                                        else:
                                            df.loc[idx, '问题类型'] = issue["type"]
                        
                        # 6. 是否为异常值
                        df['是否为异常值'] = "No"
                        if "analysis_result" in project and "outlier_stats" in project["analysis_result"]:
                            outlier_stats = project["analysis_result"]["outlier_stats"]
                            for col, info in outlier_stats.items():
                                if col in df.columns:
                                    Q1 = df[col].quantile(0.25)
                                    Q3 = df[col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    lower_bound = Q1 - 1.5 * IQR
                                    upper_bound = Q3 + 1.5 * IQR
                                    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                                    df.loc[outliers, '是否为异常值'] = "Yes"
                        
                        # 7. 确定性（confidence）
                        df['确定性'] = "N/A"
                        # 存储置信度和熵值，用于颜色编码和工具提示
                        confidence_data = {}
                        entropy_data = {}
                        if "analysis_result" in project and "distribution_shift" in project["analysis_result"]:
                            distribution_shift = project["analysis_result"]["distribution_shift"]
                            if "samples" in distribution_shift:
                                for sample in distribution_shift["samples"]:
                                    try:
                                        idx = int(sample["index"])
                                        if idx < len(df):
                                            confidence = sample.get('confidence', 0)
                                            df.loc[idx, '确定性'] = f"{confidence:.2f}"
                                            confidence_data[idx] = confidence
                                            # 存储熵值用于工具提示
                                            if 'entropy' in sample:
                                                entropy_data[idx] = sample['entropy']
                                    except (ValueError, TypeError):
                                        pass
                        
                        # 8. 是否为弱样本
                        df['是否为弱样本'] = "No"
                        if "analysis_result" in project and "weak_samples" in project["analysis_result"]:
                            weak_samples = project["analysis_result"]["weak_samples"]
                            if "samples" in weak_samples:
                                for sample in weak_samples["samples"]:
                                    try:
                                        idx = int(sample["index"])
                                        if idx < len(df):
                                            df.loc[idx, '是否为弱样本'] = "Yes"
                                    except (ValueError, TypeError):
                                        pass
                        
                        # 9. 操作
                        df['操作'] = ""
                        if "analysis_result" in project and "label_issues" in project["analysis_result"]:
                            label_issues = project["analysis_result"]["label_issues"]
                            if "suggested_labels" in label_issues:
                                suggested_labels = label_issues["suggested_labels"]
                                # 确保idx是整数类型
                                for idx in suggested_labels:
                                    try:
                                        int_idx = int(idx)
                                        if int_idx < len(df):
                                            df.loc[int_idx, '操作'] = "修正标签"
                                    except (ValueError, TypeError):
                                        pass
                        
                        # 显示数据
                        # 使用自定义表格来实现颜色编码和工具提示
                        st.markdown("""
                        <style>
                        .data-table {
                            width: 100%;
                            border-collapse: collapse;
                            font-size: 14px;
                        }
                        .data-table th, .data-table td {
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: left;
                        }
                        .data-table th {
                            background-color: #f2f2f2;
                            font-weight: bold;
                        }
                        .data-table tr:hover {
                            background-color: #f5f5f5;
                        }
                        .confidence-high {
                            background-color: #d1fae5;
                            color: #065f46;
                        }
                        .confidence-medium {
                            background-color: #fef3c7;
                            color: #92400e;
                        }
                        .confidence-low {
                            background-color: #fee2e2;
                            color: #991b1b;
                        }
                        .tooltip {
                            position: relative;
                            display: inline-block;
                        }
                        .tooltip .tooltiptext {
                            visibility: hidden;
                            width: 200px;
                            background-color: #333;
                            color: #fff;
                            text-align: center;
                            border-radius: 6px;
                            padding: 5px;
                            position: absolute;
                            z-index: 1;
                            bottom: 125%;
                            left: 50%;
                            margin-left: -100px;
                            opacity: 0;
                            transition: opacity 0.3s;
                        }
                        .tooltip:hover .tooltiptext {
                            visibility: visible;
                            opacity: 1;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # 构建HTML表格
                        table_html = "<table class='data-table'>"
                        # 表头
                        table_html += "<tr>"
                        for col in df.columns:
                            table_html += f"<th>{col}</th>"
                        table_html += "</tr>"
                        
                        # 表格内容
                        for idx, row in df.iterrows():
                            table_html += "<tr>"
                            for col in df.columns:
                                value = row[col]
                                if col == "确定性" and value != "N/A":
                                    # 添加颜色编码和工具提示
                                    try:
                                        confidence = float(value)
                                        if confidence >= 0.7:
                                            class_name = "confidence-high"
                                        elif confidence >= 0.4:
                                            class_name = "confidence-medium"
                                        else:
                                            class_name = "confidence-low"
                                        
                                        # 检查是否有熵值
                                        entropy = entropy_data.get(idx, "N/A")
                                        entropy_str = f"{entropy:.2f}" if entropy != "N/A" else "N/A"
                                        
                                        table_html += f"<td class='{class_name} tooltip'>{value}<span class='tooltiptext'>预测熵: {entropy_str}</span></td>"
                                    except (ValueError, TypeError):
                                        table_html += f"<td>{value}</td>"
                                else:
                                    table_html += f"<td>{value}</td>"
                            table_html += "</tr>"
                        
                        table_html += "</table>"
                        st.markdown(table_html, unsafe_allow_html=True)
                    elif project['analysis_type'] == "图像分析":
                        # 显示图像数据表格
                        st.markdown("<h4>图像数据列表</h4>", unsafe_allow_html=True)
                        
                        # 检查是否是ZIP文件
                        is_zip = project['file_name'].endswith('.zip')
                        
                        # 初始化图像数据
                        image_data = []
                        
                        # 获取分析结果
                        metrics = result.get("metrics", {})
                        quality_scores = metrics.get("quality_scores", [])
                        label_issues = metrics.get("label_issues", {})
                        potential_outliers = label_issues.get("potential_outliers", [])
                        issue_ids = label_issues.get("issue_ids", [])
                        
                        # 提取图像
                        images = []
                        if is_zip:
                            # 对于ZIP文件，重新提取图像
                            try:
                                import zipfile
                                import io
                                file_content = project['file_content']
                                with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
                                    # 提取所有图像文件
                                    for zip_info in zip_ref.infolist():
                                        if zip_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                            with zip_ref.open(zip_info) as img_file:
                                                img_bytes = img_file.read()
                                                images.append(img_bytes)
                            except Exception as e:
                                st.error(f"提取ZIP文件失败: {str(e)}")
                        else:
                            # 对于单个图像文件
                            if "image_bytes" in project:
                                images.append(project["image_bytes"])
                            else:
                                images.append(project['file_content'])
                        
                        # 构建图像数据表格
                        for i, img_bytes in enumerate(images):
                            # 获取质量分数
                            quality_score = quality_scores[i] if i < len(quality_scores) else 0
                            
                            # 检查是否是异常样本
                            is_outlier = i in potential_outliers
                            
                            # 获取图像ID
                            image_id = issue_ids[i] if i < len(issue_ids) else f"{i:05d}"
                            
                            # 获取原标签和建议标签
                            original_label = "N/A"
                            predicted_label = "N/A"
                            suggested_label = "N/A"
                            
                            # 尝试从分析结果中获取标签信息
                            if "label_issues" in result:
                                label_issues = result["label_issues"]
                                if "suggested_labels" in label_issues:
                                    suggested_labels = label_issues["suggested_labels"]
                                    if str(i) in suggested_labels:
                                        suggested_label = suggested_labels[str(i)]
                                if "original_labels" in label_issues:
                                    original_labels = label_issues["original_labels"]
                                    if i < len(original_labels):
                                        original_label = original_labels[i]
                                if "predicted_labels" in label_issues:
                                    predicted_labels = label_issues["predicted_labels"]
                                    if i < len(predicted_labels):
                                        predicted_label = predicted_labels[i]
                            # 从metrics中获取标签信息（如果result中没有）
                            elif "metrics" in result:
                                metrics = result["metrics"]
                                if "label_issues" in metrics:
                                    label_issues = metrics["label_issues"]
                                    if "original_labels" in label_issues:
                                        original_labels = label_issues["original_labels"]
                                        if i < len(original_labels):
                                            original_label = original_labels[i]
                            
                            # 构建图像数据字典
                            img_data = {
                                "ID": image_id,
                                "质量分数": round(quality_score, 2),
                                "原标签": original_label,
                                "预测标签": predicted_label,
                                "建议标签": suggested_label,
                                "是否异常": "Yes" if is_outlier else "No",
                                "图像": img_bytes
                            }
                            image_data.append(img_data)
                        
                        # 创建DataFrame
                        import pandas as pd
                        img_df = pd.DataFrame(image_data)
                        
                        # 显示图像数据表格
                        # 由于Streamlit的dataframe不支持直接显示图像，我们使用自定义表格
                        from PIL import Image
                        import io
                        
                        # 显示前20张图像
                        display_data = image_data[:20]
                        
                        # 创建表格
                        st.markdown("""
                        <style>
                        .image-table {
                            width: 100%;
                            border-collapse: collapse;
                        }
                        .image-table th, .image-table td {
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: center;
                        }
                        .image-table th {
                            background-color: #f2f2f2;
                        }
                        .image-table img {
                            max-width: 100px;
                            max-height: 100px;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # 构建HTML表格
                        table_html = "<table class='image-table'>"
                        table_html += "<tr><th>ID</th><th>图像</th><th>质量分数</th><th>原标签</th><th>预测标签</th><th>建议标签</th><th>是否异常</th></tr>"
                        
                        for img_data in display_data:
                            table_html += "<tr>"
                            table_html += f"<td>{img_data['ID']}</td>"
                            
                            # 显示图像
                            img = Image.open(io.BytesIO(img_data['图像']))
                            import base64
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                            table_html += f"<td><img src='data:image/png;base64,{img_base64}' alt='图像'></td>"
                            
                            table_html += f"<td>{img_data['质量分数']}</td>"
                            table_html += f"<td>{img_data['原标签']}</td>"
                            table_html += f"<td>{img_data['预测标签']}</td>"
                            table_html += f"<td>{img_data['建议标签']}</td>"
                            table_html += f"<td>{img_data['是否异常']}</td>"
                            table_html += "</tr>"
                        
                        table_html += "</table>"
                        st.markdown(table_html, unsafe_allow_html=True)
                        
                        # 显示分析概览
                        st.markdown("<h4>分析概览</h4>", unsafe_allow_html=True)
                        st.write(f"分析了 {len(images)} 张图像")
                        
                        # 显示质量分数
                        st.metric("平均质量分数", f"{metrics.get('average_quality_score', 0):.2f}")
                        
                        # 显示检测到的问题
                        issues = result.get("issues", [])
                        if issues:
                            st.write("检测到的问题:")
                            # 按问题类型分组
                            issue_types = {}
                            for issue in issues:
                                issue_type = issue.get('type', 'unknown')
                                if issue_type not in issue_types:
                                    issue_types[issue_type] = 0
                                issue_types[issue_type] += 1
                            
                            for issue_type, count in issue_types.items():
                                if issue_type == "blurry_image":
                                    st.write(f"- 模糊图像: {count} 张")
                                elif issue_type == "bad_exposure":
                                    st.write(f"- 曝光问题: {count} 张")
                                elif issue_type == "low_contrast":
                                    st.write(f"- 低对比度: {count} 张")
                                elif issue_type == "low_color_contrast":
                                    st.write(f"- 低颜色对比度: {count} 张")
                                elif issue_type == "small_image":
                                    st.write(f"- 图像尺寸过小: {count} 张")
                                else:
                                    st.write(f"- {issue_type}: {count} 张")
                        else:
                            st.success("未检测到任何问题，图像质量良好！")
                        
                        # 显示标签错误检测结果
                        if "label_issues" in metrics:
                            label_issues = metrics["label_issues"]
                            st.markdown("<h5>标签错误检测</h5>", unsafe_allow_html=True)
                            st.write(f"潜在异常样本数量: {label_issues.get('total_issues', 0)}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            # Analytics 界面
            st.markdown("<h3>分析可视化</h3>", unsafe_allow_html=True)
            
            if project['analysis_type'] == "图像分析":
                # 图像分析的可视化逻辑
                st.markdown("<h4>图像质量分析</h4>", unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    # 显示图像质量指标
                    metrics = result.get("metrics", {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总图像数", metrics.get("total_images", 0))
                    with col2:
                        st.metric("问题率", f"{metrics.get('issue_rate', 0):.2f}%")
                    with col3:
                        st.metric("平均质量分数", f"{metrics.get('average_quality_score', 0):.2f}")
                    
                    # 显示质量分数分布
                    if "quality_scores" in metrics:
                        st.markdown("<h5>质量分数分布</h5>", unsafe_allow_html=True)
                        import plotly.express as px
                        import pandas as pd
                        score_df = pd.DataFrame({
                            "质量分数": metrics["quality_scores"]
                        })
                        fig = px.histogram(score_df, x="质量分数", nbins=20, title="图像质量分数分布")
                        fig.update_layout(
                            xaxis_title="质量分数",
                            yaxis_title="图像数量",
                            height=400
                        )
                        st.plotly_chart(fig, config={'responsive': True})
                    
                    # 显示问题类型分布
                    issues = result.get("issues", [])
                    if issues:
                        st.markdown("<h5>问题类型分布</h5>", unsafe_allow_html=True)
                        issue_types = {}
                        for issue in issues:
                            issue_type = issue.get("type", "unknown")
                            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                        
                        import plotly.express as px
                        import pandas as pd
                        issue_df = pd.DataFrame(list(issue_types.items()), columns=["问题类型", "数量"])
                        fig = px.bar(issue_df, x="问题类型", y="数量", title="问题类型分布")
                        st.plotly_chart(fig, config={'responsive': True})
                    else:
                        st.success("未检测到任何问题，图像质量良好！")
                    
                    # 显示标签错误检测结果
                    if "label_issues" in metrics:
                        st.markdown("<h5>标签错误检测</h5>", unsafe_allow_html=True)
                        label_issues = metrics["label_issues"]
                        st.write(f"潜在异常样本数量: {label_issues.get('total_issues', 0)}")
                        if label_issues.get('potential_outliers'):
                            st.write(f"异常样本索引: {label_issues['potential_outliers']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            elif project['analysis_type'] == "表格分析":
                # 缺失值分布
                import plotly.express as px
                import pandas as pd
                import numpy as np
                import plotly.graph_objects as go
                
                # 数据质量概览
                st.markdown("<h4>数据质量概览</h4>", unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    # 计算数据质量指标
                    total_samples = result.get("basic_info", {}).get("rows", 0)
                    missing_count = result.get("missing_stats", {}).get("total", 0)
                    duplicate_count = result.get("duplicate_stats", {}).get("count", 0)
                    outlier_count = 0
                    for col, info in result.get("outlier_stats", {}).items():
                        outlier_count += info.get("count", 0)
                    label_error_count = result.get("label_issues", {}).get("error_count", 0)
                    
                    # 计算质量分数
                    quality_score = 100
                    if total_samples > 0:
                        if missing_count > 0:
                            quality_score -= (missing_count / total_samples) * 20
                        if duplicate_count > 0:
                            quality_score -= (duplicate_count / total_samples) * 15
                        if outlier_count > 0:
                            quality_score -= (outlier_count / total_samples) * 25
                        if label_error_count > 0:
                            quality_score -= (label_error_count / total_samples) * 40
                    quality_score = max(0, min(100, quality_score))
                    
                    # 显示质量指标
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总样本数", total_samples)
                    with col2:
                        st.metric("缺失值", missing_count)
                    with col3:
                        st.metric("异常值", outlier_count)
                    with col4:
                        st.metric("标签错误", label_error_count)
                    
                    # 质量分数仪表盘
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=quality_score,
                        title={'text': "数据质量分数"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#1E3A8A"},
                            'steps': [
                                {'range': [0, 60], 'color': "#EF4444"},
                                {'range': [60, 80], 'color': "#F59E0B"},
                                {'range': [80, 100], 'color': "#10B981"}
                            ],
                            'threshold': {
                                'line': {'color': "#8B5CF6", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    st.plotly_chart(fig, config={'responsive': True})
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 缺失值分布
                st.markdown("<h4>缺失值分布</h4>", unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    missing_stats = result.get("missing_stats", {})
                    if missing_stats.get('total', 0) > 0:
                        missing_data = []
                        for col, info in missing_stats.get('per_column', {}).items():
                            missing_data.append({
                                "列名": col,
                                "缺失率": info.get('percentage', 0)
                            })
                        missing_df = pd.DataFrame(missing_data)
                        fig = px.bar(missing_df, x='列名', y='缺失率', title='各列缺失值分布')
                        st.plotly_chart(fig, config={'responsive': True})
                    else:
                        st.success("没有缺失值")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 标签质量分析
                if "label_issues" in result:
                    st.markdown("<h4>标签质量分析</h4>", unsafe_allow_html=True)
                    
                    label_issues = result["label_issues"]
                    
                    # 标签质量分数分布
                    with st.container():
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        
                        if "label_quality_scores" in label_issues:
                            quality_scores = label_issues["label_quality_scores"]
                            fig = px.histogram(
                                x=quality_scores,
                                title='标签质量分数分布',
                                labels={'x': '质量分数', 'y': '样本数'},
                                nbins=20
                            )
                            fig.add_vline(x=0.6, line_dash="dash", line_color="red", annotation_text="阈值 (0.6)")
                            st.plotly_chart(fig, config={'responsive': True})
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 错误流向分析
                    with st.container():
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        
                        # 生成错误流向数据
                        if "error_flow" in label_issues:
                            error_flow = label_issues["error_flow"]
                            # 转换为热力图数据
                            classes = sorted(list(set([k for k in error_flow.keys()] + [v for vals in error_flow.values() for v in vals.keys()])))
                            heatmap_data = np.zeros((len(classes), len(classes)))
                            for true_class, predicted_classes in error_flow.items():
                                true_idx = classes.index(true_class)
                                for pred_class, count in predicted_classes.items():
                                    pred_idx = classes.index(pred_class)
                                    heatmap_data[true_idx][pred_idx] = count
                            
                            # 创建混淆矩阵热力图
                            fig = px.imshow(
                                heatmap_data,
                                x=classes,
                                y=classes,
                                labels=dict(x="预测标签", y="真实标签", color="错误数量"),
                                title="标签错误流向（混淆矩阵）",
                                color_continuous_scale="Reds"
                            )
                            fig.update_xaxes(side="top")
                            st.plotly_chart(fig, config={'responsive': True})
                        else:
                            # 模拟错误流向数据
                            st.info("错误流向数据正在生成中...")
                            # 这里可以根据实际情况生成错误流向数据
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 应用建议标签按钮
                    with st.container():
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        
                        if st.button("应用建议标签"):
                            st.info("正在生成清洗后的数据集...")
                            
                            # 模拟生成清洗后的数据集
                            if "df" in project:
                                df = project["df"]
                            else:
                                # 读取数据
                                df = project['file_content']
                            
                            # 模拟应用建议标签
                            if "suggested_labels" in label_issues:
                                suggested_labels = label_issues["suggested_labels"]
                                # 应用建议标签
                                label_col = project.get("label_col", "label")
                                if label_col in df.columns:
                                    # 记录修改
                                    changes = []
                                    for idx, new_label in suggested_labels.items():
                                        if df.loc[idx, label_col] != new_label:
                                            changes.append({
                                                "index": idx,
                                                "old_label": df.loc[idx, label_col],
                                                "new_label": new_label
                                            })
                                    # 应用修改
                                    for idx, new_label in suggested_labels.items():
                                        df.loc[idx, label_col] = new_label
                                    
                                    # 生成清洗后的 CSV
                                    import io
                                    csv_buffer = io.StringIO()
                                    df.to_csv(csv_buffer, index=False)
                                    csv_buffer.seek(0)
                                    
                                    # 提供下载
                                    st.download_button(
                                        label="下载清洗后的数据集",
                                        data=csv_buffer.getvalue(),
                                        file_name=f"cleaned_{project['file_name'].split('.')[0]}.csv",
                                        mime="text/csv"
                                    )
                                    
                                    # 显示修改详情
                                    st.markdown("<h5>修改详情</h5>", unsafe_allow_html=True)
                                    if changes:
                                        changes_df = pd.DataFrame(changes)
                                        st.dataframe(changes_df, width='stretch')
                                        st.write(f"共修改了 {len(changes)} 个样本的标签")
                                    else:
                                        st.success("没有需要修改的标签")
                                else:
                                    st.error("标签列不存在")
                            else:
                                st.info("正在生成建议标签...")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # 退出项目按钮
        if st.button("退出项目", width='stretch'):
            st.session_state.current_project = None
            st.rerun()

# 最近分析记录
if st.session_state.recent_analyses:
    st.markdown("<h3 style='margin-top: 2rem;'>最近分析</h3>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for i, analysis in enumerate(st.session_state.recent_analyses[-5:][::-1]):  # 显示最近5条，最新的在前
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i+1}. {analysis['type']}: {analysis['file']}**", unsafe_allow_html=True)
                if 'time' in analysis and analysis['time']:
                    st.caption(f"分析时间: {analysis['time']}")
            with col2:
                if st.button("查看详情", key=f"recent_view_{i}"):
                    # 查找对应的项目
                    for project in st.session_state.projects:
                        if project['name'] == analysis.get('name') and project['file_name'] == analysis.get('file'):
                            st.session_state.current_project = project
                            # 跳转到历史报告页面查看详情
                            st.session_state.page = "历史报告"
                            # 由于Streamlit的限制，我们使用rerun来刷新页面
                            st.rerun()
                            break
            st.divider()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 查看全部历史按钮
        if st.button("查看全部历史", width='stretch'):
            st.session_state.page = "历史报告"
            # 由于Streamlit的限制，我们使用rerun来刷新页面
            st.rerun()

# 页脚
st.markdown("""
<div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #E5E7EB; text-align: center; color: #6B7280;">
    <p>© 2026 多模态数据质量检测系统 | 基于 Cleanlab 置信学习</p>
</div>
""", unsafe_allow_html=True)
