import pandas as pd
import numpy as np

# 简单测试脚本
print("开始生成测试数据...")

# 测试参数
n_samples = 100
n_classes = 5

# 生成简单数据
true_labels = np.random.choice(n_classes, n_samples)
feature_1 = np.random.normal(10, 2, n_samples)

# 创建DataFrame
df = pd.DataFrame({
    'id': range(1, n_samples+1),
    'feature_1': feature_1.round(2),
    'true_label': [f'Class_{l}' for l in true_labels],
    'label': [f'Class_{l}' for l in true_labels]
})

# 保存文件
df.to_csv('data/test_sample.csv', index=False)
print(f"测试数据生成完成，保存到 data/test_sample.csv")
print(f"数据形状: {df.shape}")