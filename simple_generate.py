import pandas as pd
import numpy as np
import os

print("开始生成简单测试数据...")

# 基本参数
n_samples = 100
n_classes = 5

# 生成数据
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
file_path = os.path.join(os.getcwd(), 'data', 'samples', 'test.csv')
df.to_csv(file_path, index=False)
print(f"文件保存成功: {file_path}")
print(f"数据形状: {df.shape}")
print(f"前5行数据:\n{df.head()}")