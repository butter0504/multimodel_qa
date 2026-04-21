import json
from datetime import datetime

# 尝试导入MySQL模块
try:
    import mysql.connector
    from mysql.connector import pooling
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    print("MySQL模块不可用，将使用会话状态作为后备存储")

# 尝试导入配置
try:
    from api.core.config import settings
except ImportError:
    # 如果无法导入API配置，使用默认值
    class Settings:
        DB_USER = "root"
        DB_PASSWORD = "Mxy2003031611"
        DB_HOST = "localhost"
        DB_PORT = "3306"
        DB_NAME = "multimodel_qa"
    settings = Settings()

class Database:
    def __init__(self):
        self.pool = None
        self.use_mysql = MYSQL_AVAILABLE
        self.local_storage = "data/local_storage.json"
        if self.use_mysql:
            self.init_pool()
        else:
            print("使用本地文件存储作为后备存储")
            self._ensure_local_storage()
    
    def _ensure_local_storage(self):
        """确保本地存储文件存在"""
        import os
        import json
        os.makedirs(os.path.dirname(self.local_storage), exist_ok=True)
        if not os.path.exists(self.local_storage):
            with open(self.local_storage, 'w') as f:
                json.dump({"projects": [], "analysis_results": []}, f)
    
    def init_pool(self):
        """初始化数据库连接池，添加重试机制"""
        max_retries = 3
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                print(f"尝试连接数据库 (尝试 {attempt + 1}/{max_retries})...")
                # 先创建数据库连接（不指定数据库）
                conn = mysql.connector.connect(
                    host=settings.DB_HOST,
                    port=settings.DB_PORT,
                    user=settings.DB_USER,
                    password=settings.DB_PASSWORD,
                    connect_timeout=10
                )
                cursor = conn.cursor()
                
                # 创建数据库（如果不存在）
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {settings.DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                conn.commit()
                
                # 关闭连接
                cursor.close()
                conn.close()
                
                # 创建连接池
                self.pool = pooling.MySQLConnectionPool(
                    pool_name="multimodal_qa_pool",
                    pool_size=5,
                    pool_reset_session=True,
                    host=settings.DB_HOST,
                    port=settings.DB_PORT,
                    user=settings.DB_USER,
                    password=settings.DB_PASSWORD,
                    database=settings.DB_NAME,
                    charset="utf8mb4",
                    collation="utf8mb4_unicode_ci"
                )
                
                # 创建表
                self.create_tables()
                print("数据库连接池初始化成功")
                return
            except mysql.connector.Error as e:
                print(f"数据库连接池初始化失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    print(f"{retry_delay}秒后重试...")
                    import time
                    time.sleep(retry_delay)
                else:
                    print("达到最大重试次数，数据库连接池初始化失败")
                    self.use_mysql = False
                    print("使用本地文件存储作为后备存储")
                    self._ensure_local_storage()
            except Exception as e:
                print(f"未知错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    print(f"{retry_delay}秒后重试...")
                    import time
                    time.sleep(retry_delay)
                else:
                    print("达到最大重试次数，数据库连接池初始化失败")
                    self.use_mysql = False
                    print("使用本地文件存储作为后备存储")
                    self._ensure_local_storage()
    
    def get_connection(self):
        """获取数据库连接"""
        if not self.use_mysql or not self.pool:
            return None
        try:
            return self.pool.get_connection()
        except Exception as e:
            print(f"获取数据库连接失败: {str(e)}")
            # 尝试重新初始化连接池
            self.init_pool()
            try:
                return self.pool.get_connection()
            except:
                return None
    
    def create_tables(self):
        """创建必要的表"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            # 创建项目表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    file_name VARCHAR(255) NOT NULL,
                    analysis_type VARCHAR(50) NOT NULL,
                    upload_time DATETIME NOT NULL,
                    basic_info TEXT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # 创建分析结果表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    project_id INT NOT NULL,
                    result LONGTEXT NOT NULL,
                    analysis_time DATETIME NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            conn.commit()
            print("表创建成功")
        except Exception as e:
            print(f"表创建失败: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            conn.close()
    
    def add_project(self, project):
        """添加项目到数据库"""
        import json
        if self.use_mysql:
            conn = self.get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    sql = """
                        INSERT INTO projects (name, file_name, analysis_type, upload_time, basic_info)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    values = (
                        project['name'],
                        project['file_name'],
                        project['analysis_type'],
                        datetime.now(),
                        json.dumps(project.get('basic_info', {}))
                    )
                    cursor.execute(sql, values)
                    conn.commit()
                    project_id = cursor.lastrowid
                    if 'cursor' in locals():
                        cursor.close()
                    conn.close()
                    return project_id
                except Exception as e:
                    print(f"添加项目失败: {str(e)}")
                    if 'cursor' in locals():
                        cursor.close()
                    conn.close()
        
        # 使用本地文件存储作为后备
        try:
            with open(self.local_storage, 'r') as f:
                data = json.load(f)
            
            # 生成项目ID
            project_id = len(data['projects']) + 1
            
            # 添加项目
            project_data = {
                'id': project_id,
                'name': project['name'],
                'file_name': project['file_name'],
                'analysis_type': project['analysis_type'],
                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'basic_info': project.get('basic_info', {})
            }
            data['projects'].append(project_data)
            
            # 保存到本地文件
            with open(self.local_storage, 'w') as f:
                json.dump(data, f, indent=2)
            
            return project_id
        except Exception as e:
            print(f"本地文件存储添加项目失败: {str(e)}")
            return None
    
    def add_analysis_result(self, project_id, result):
        """添加分析结果到数据库"""
        import json
        if self.use_mysql:
            conn = self.get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    sql = """
                        INSERT INTO analysis_results (project_id, result, analysis_time)
                        VALUES (%s, %s, %s)
                    """
                    values = (
                        project_id,
                        json.dumps(result),
                        datetime.now()
                    )
                    cursor.execute(sql, values)
                    conn.commit()
                    result_id = cursor.lastrowid
                    if 'cursor' in locals():
                        cursor.close()
                    conn.close()
                    return result_id
                except Exception as e:
                    print(f"添加分析结果失败: {str(e)}")
                    if 'cursor' in locals():
                        cursor.close()
                    conn.close()
        
        # 使用本地文件存储作为后备
        try:
            with open(self.local_storage, 'r') as f:
                data = json.load(f)
            
            # 生成结果ID
            result_id = len(data['analysis_results']) + 1
            
            # 添加分析结果
            result_data = {
                'id': result_id,
                'project_id': project_id,
                'result': result,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            data['analysis_results'].append(result_data)
            
            # 保存到本地文件
            with open(self.local_storage, 'w') as f:
                json.dump(data, f, indent=2)
            
            return result_id
        except Exception as e:
            print(f"本地文件存储添加分析结果失败: {str(e)}")
            return None
    
    def get_projects(self, limit=10):
        """获取项目列表"""
        import json
        if self.use_mysql:
            conn = self.get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    sql = "SELECT * FROM projects ORDER BY upload_time DESC LIMIT %s"
                    cursor.execute(sql, (limit,))
                    projects = []
                    for row in cursor.fetchall():
                        project = {
                            'id': row[0],
                            'name': row[1],
                            'file_name': row[2],
                            'analysis_type': row[3],
                            'upload_time': row[4].strftime('%Y-%m-%d %H:%M:%S'),
                            'basic_info': json.loads(row[5]) if row[5] else {}
                        }
                        projects.append(project)
                    if 'cursor' in locals():
                        cursor.close()
                    conn.close()
                    return projects
                except Exception as e:
                    print(f"获取项目列表失败: {str(e)}")
                    if 'cursor' in locals():
                        cursor.close()
                    conn.close()
        
        # 使用本地文件存储作为后备
        try:
            with open(self.local_storage, 'r') as f:
                data = json.load(f)
            
            # 按上传时间降序排序
            projects = sorted(data['projects'], key=lambda x: x['upload_time'], reverse=True)
            # 限制返回数量
            return projects[:limit]
        except Exception as e:
            print(f"本地文件存储获取项目列表失败: {str(e)}")
            # 最后使用会话状态作为后备
            import streamlit as st
            if "projects" in st.session_state:
                return st.session_state.projects[-limit:]
            return []
    
    def get_project(self, project_id):
        """获取单个项目"""
        import json
        if self.use_mysql:
            conn = self.get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    sql = "SELECT * FROM projects WHERE id = %s"
                    cursor.execute(sql, (project_id,))
                    row = cursor.fetchone()
                    if row:
                        project = {
                            'id': row[0],
                            'name': row[1],
                            'file_name': row[2],
                            'analysis_type': row[3],
                            'upload_time': row[4].strftime('%Y-%m-%d %H:%M:%S'),
                            'basic_info': json.loads(row[5]) if row[5] else {}
                        }
                        if 'cursor' in locals():
                            cursor.close()
                        conn.close()
                        return project
                    if 'cursor' in locals():
                        cursor.close()
                    conn.close()
                except Exception as e:
                    print(f"获取项目失败: {str(e)}")
                    if 'cursor' in locals():
                        cursor.close()
                    conn.close()
        
        # 使用本地文件存储作为后备
        try:
            with open(self.local_storage, 'r') as f:
                data = json.load(f)
            
            # 查找项目
            for project in data['projects']:
                if project['id'] == project_id:
                    return project
        except Exception as e:
            print(f"本地文件存储获取项目失败: {str(e)}")
        
        # 最后使用会话状态作为后备
        import streamlit as st
        if "projects" in st.session_state:
            for project in st.session_state.projects:
                if project.get('db_id') == project_id or project.get('id') == project_id:
                    return project
        return None
    
    def get_analysis_result(self, project_id):
        """获取项目的分析结果"""
        import json
        if self.use_mysql:
            conn = self.get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    sql = "SELECT result FROM analysis_results WHERE project_id = %s ORDER BY analysis_time DESC LIMIT 1"
                    cursor.execute(sql, (project_id,))
                    row = cursor.fetchone()
                    if row:
                        result = json.loads(row[0])
                        if 'cursor' in locals():
                            cursor.close()
                        conn.close()
                        return result
                    if 'cursor' in locals():
                        cursor.close()
                    conn.close()
                except Exception as e:
                    print(f"获取分析结果失败: {str(e)}")
                    if 'cursor' in locals():
                        cursor.close()
                    conn.close()
        
        # 使用本地文件存储作为后备
        try:
            with open(self.local_storage, 'r') as f:
                data = json.load(f)
            
            # 查找最新的分析结果
            project_results = []
            for result in data['analysis_results']:
                if result['project_id'] == project_id:
                    project_results.append(result)
            
            # 按分析时间降序排序
            if project_results:
                project_results.sort(key=lambda x: x['analysis_time'], reverse=True)
                return project_results[0]['result']
        except Exception as e:
            print(f"本地文件存储获取分析结果失败: {str(e)}")
        
        # 最后使用会话状态作为后备
        import streamlit as st
        if "projects" in st.session_state:
            for project in st.session_state.projects:
                if project.get('db_id') == project_id or project.get('id') == project_id:
                    if 'analysis_result' in project:
                        return project['analysis_result']
        return None
    
    def close(self):
        """关闭数据库连接池"""
        # 连接池会自动管理连接，不需要手动关闭

# 创建数据库实例
db = Database()
