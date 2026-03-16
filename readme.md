# 只在main分支上工作
# 每天结束前：
git status                 # 看看改了啥
git add .                  # 添加所有（确保.gitignore配置好）
git commit -m "日期：完成了X功能"
git push                   # 推送到GitHub/Gitee备份

uvicorn api.main:app --reload

streamlit run app/main.py

