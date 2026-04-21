# MySQL 安装与配置说明

## 1. MySQL 安装步骤

### Windows 系统

1. **下载 MySQL 安装包**
   - 访问 MySQL 官方网站：https://dev.mysql.com/downloads/installer/
   - 下载 MySQL Installer for Windows
   - 选择适合您系统的版本（32位或64位）

2. **运行安装程序**
   - 双击下载的安装包
   - 选择 "Custom" 安装类型
   - 选择 MySQL Server 8.0 或更高版本
   - 按照向导完成安装

3. **配置 MySQL**
   - 选择 "Standalone MySQL Server/Classic MySQL Replication"
   - 设置 root 密码（建议使用：password）
   - 选择默认端口 3306
   - 选择字符集为 utf8mb4
   - 完成配置向导

### Linux 系统

1. **使用包管理器安装**
   - Ubuntu/Debian:
     ```bash
     sudo apt update
     sudo apt install mysql-server
     ```
   - CentOS/RHEL:
     ```bash
     sudo yum update
     sudo yum install mysql-server
     ```

2. **启动 MySQL 服务**
   - Ubuntu/Debian:
     ```bash
     sudo systemctl start mysql
     sudo systemctl enable mysql
     ```
   - CentOS/RHEL:
     ```bash
     sudo systemctl start mysqld
     sudo systemctl enable mysqld
     ```

3. **设置 root 密码**
   - 运行安全脚本：
     ```bash
     sudo mysql_secure_installation
     ```
   - 按照提示设置 root 密码（建议使用：password）

### macOS 系统

1. **使用 Homebrew 安装**
   ```bash
   brew install mysql
   ```

2. **启动 MySQL 服务**
   ```bash
   brew services start mysql
   ```

3. **设置 root 密码**
   ```bash
   mysql_secure_installation
   ```

## 2. 数据库配置

### 1. 登录 MySQL

```bash
mysql -u root -p
```

### 2. 创建数据库（如果不存在）

```sql
CREATE DATABASE IF NOT EXISTS multimodal_qa CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 3. 验证数据库创建

```sql
SHOW DATABASES;
```

## 3. 环境变量配置

在项目根目录创建 `.env` 文件，添加以下配置：

```
# 数据库配置
DB_USER=root
DB_PASSWORD=password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=multimodal_qa

# API 配置
API_V1_STR=/api/v1
PROJECT_NAME=多模态数据质量检测系统

# 数据目录配置
DATA_DIR=data
UPLOAD_DIR=data/uploads
SAMPLE_DIR=data/samples

# 环境配置
CONDA_ENV=multimodal_qa
```

## 4. 安装 Python 依赖

```bash
pip install mysql-connector-python
```

## 5. 启动 API 服务

```bash
cd api
uvicorn main:app --reload
```

API 服务启动时会自动：
- 创建数据库（如果不存在）
- 创建必要的表结构
- 初始化数据库连接池

## 6. 验证数据库连接

访问 API 健康检查接口：
- URL: http://localhost:8000/health
- 响应应包含数据库连接状态

## 7. 故障排除

### 连接失败
- 检查 MySQL 服务是否正在运行
- 验证数据库用户名和密码是否正确
- 确保 MySQL 允许远程连接（如果需要）

### 字符集问题
- 确保数据库和表使用 utf8mb4 字符集
- 检查连接字符串中是否指定了正确的字符集

### 端口问题
- 确保 MySQL 监听在 3306 端口
- 检查防火墙是否允许访问 3306 端口

## 8. 最佳实践

- 使用强密码保护数据库
- 定期备份数据库
- 限制数据库用户的权限
- 监控数据库性能

## 9. 联系方式

如果遇到问题，请联系系统管理员或查看 MySQL 官方文档。