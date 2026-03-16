# 多模态数据质量检测系统

基于 Cleanlab 置信学习的多模态数据质量检测系统，支持表格、文本、图像数据的质量分析。

## 功能特性

- **表格数据质量检测**：缺失值、异常值、重复行、标签错误检测
- **文本数据质量检测**：空文本、重复文本、长度异常检测
- **图像数据质量检测**：格式错误、大小异常检测
- **置信学习**：基于Cleanlab的标签错误检测算法
- **可视化分析**：直观的数据质量指标和问题展示
- **API接口**：完整的FastAPI后端接口
- **用户友好**：Streamlit前端界面，操作简单直观

## 系统架构

- **前端**：Streamlit
- **后端**：FastAPI
- **核心算法**：Cleanlab、scikit-learn
- **数据处理**：Pandas、NumPy
- **可视化**：Plotly

## 项目结构

```
multimodal_qa/
├── app/              # Streamlit前端
│   ├── main.py       # 首页
│   └── pages/        # 功能页面
│       ├── 01_表格分析.py
│       ├── 02_文本分析.py
│       ├── 03_图像分析.py
│       └── 04_API测试.py
├── api/              # FastAPI后端
│   ├── main.py       # API入口
│   ├── routes/       # 路由
│   ├── models/       # Pydantic模型
│   └── core/         # 核心配置
├── modules/          # 检测器模块
│   ├── base_detector.py    # 基类
│   ├── table_detector.py   # 表格检测器
│   ├── text_detector.py    # 文本检测器
│   ├── image_detector.py   # 图像检测器
│   └── cleanlab_wrapper.py # Cleanlab封装
├── data/             # 数据目录
│   ├── samples/      # 示例数据
│   └── uploads/      # 上传文件
├── tests/            # 测试文件
├── requirements.txt  # 依赖文件
├── Dockerfile        # Docker配置
├── docker-compose.yml # Docker Compose配置
├── .env.example      # 环境变量示例
├── .gitignore        # Git忽略文件
└── README.md         # 项目说明
```

## 安装指南

### 环境要求

- Python 3.9+
- Conda（推荐）或虚拟环境

### 步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd multimodal_qa
   ```

2. **创建并激活虚拟环境**
   ```bash
   # 使用Conda
   conda create -n multimodal_qa python=3.9
   conda activate multimodal_qa
   
   # 或使用venv
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 快速开始

### 启动后端API

```bash
uvicorn api.main:app --reload
```

API将在 `http://localhost:8000` 运行

### 启动前端界面

```bash
streamlit run app/main.py
```

前端将在 `http://localhost:8501` 运行

## API文档

启动后端后，可以访问以下地址查看API文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 主要API端点

- `GET /health` - 健康检查
- `POST /api/v1/table/detect` - 表格数据质量检测
- `POST /api/v1/text/detect` - 文本数据质量检测
- `POST /api/v1/image/detect` - 图像数据质量检测

## 配置信息

### 环境变量

复制 `.env.example` 文件为 `.env` 并根据需要修改：

```bash
cp .env.example .env
```

### 主要配置项

- `API_HOST` - API服务主机（默认：0.0.0.0）
- `API_PORT` - API服务端口（默认：8000）
- `STREAMLIT_PORT` - Streamlit端口（默认：8501）

## 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 前端 | Streamlit | 1.50.0+ |
| 后端 | FastAPI | 0.100.0+ |
| 核心算法 | Cleanlab | 2.7.1+ |
| 机器学习 | scikit-learn | 1.3.0+ |
| 数据处理 | Pandas, NumPy | 2.0.0+, 1.25.0+ |
| 可视化 | Plotly | 5.15.0+ |
| API文档 | Swagger UI | 内置 |

## 使用指南

### 表格分析
1. 上传CSV或Excel文件
2. 可选：指定标签列名称
3. 点击分析按钮查看结果
4. 查看质量指标和问题详情
5. 下载分析报告

### 文本分析
1. 上传文本文件（.txt）
2. 系统自动分析文本质量
3. 查看空文本、重复文本等问题

### 图像分析
1. 上传图像文件（JPG、PNG）
2. 系统自动分析图像质量
3. 查看格式和大小异常

### API测试
1. 在前端界面测试各API端点
2. 上传测试文件并查看响应结果

## 开发指南

### 代码风格
- 遵循PEP 8规范
- 使用类型提示
- 编写清晰的文档字符串

### 测试
运行测试：
```bash
pytest tests/
```

## 部署

### Docker部署

1. **构建镜像**
   ```bash
docker-compose build
```

2. **启动服务**
   ```bash
docker-compose up -d
```

3. **访问服务**
   - 前端：`http://localhost:8501`
   - 后端：`http://localhost:8000`

### 生产部署

- 使用Gunicorn作为WSGI服务器
- 配置Nginx反向代理
- 设置适当的安全措施

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目维护者：[butter0504]
- 邮箱：[butter0504@outlook.com]
- GitHub：[butter0504]

---

# Git工作流程

# 只在main分支上工作
# 每天结束前：
git status                 # 看看改了啥
git add .                  # 添加所有（确保.gitignore配置好）
git commit -m "日期：完成了X功能"
git push                   # 推送到GitHub/Gitee备份

# 启动命令
uvicorn api.main:app --reload

streamlit run app/main.py

