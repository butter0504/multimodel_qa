from modules.table_detector import TableDetector
import pandas as pd
import numpy as np

# 创建测试数据
df = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'label': np.random.randint(0, 2, 100)
})

# 初始化检测器
detector = TableDetector()

# 执行检测
print("开始检测...")
result = detector.detect(df, 'label')

# 分析结果
print("检测完成")
print(f"结果大小: {len(str(result))} 字符")
print(f"总问题数: {len(result['issues'])}")

# 统计不同类型的问题
issue_types = {}
for issue in result['issues']:
    issue_type = issue['type']
    issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

print("问题类型分布:")
for issue_type, count in issue_types.items():
    print(f"  {issue_type}: {count}")

# 统计标签错误问题
label_errors = [issue for issue in result['issues'] if issue['type'] == 'label_error']
print(f"标签错误样本数: {len(label_errors)}")

# 检查质量分数分级
if label_errors:
    scores = [issue['quality_score'] for issue in label_errors]
    print(f"标签错误样本质量分数范围: {min(scores):.4f} - {max(scores):.4f}")

print("测试完成")
