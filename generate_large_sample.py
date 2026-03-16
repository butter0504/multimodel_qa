import pandas as pd
import numpy as np
import os

# 设置随机种子
np.random.seed(42)

# 参数
n_samples = 2000
n_classes = 5

# 生成真实标签
true_labels = np.random.choice(n_classes, n_samples)

# 生成特征
feature_means = [10, 20, 30, 40, 50]
feature_1 = np.array([np.random.normal(feature_means[l], 2) for l in true_labels])
feature_2 = np.array([np.random.normal(feature_means[l] + 5, 3) for l in true_labels])
feature_3 = np.array([np.random.normal(feature_means[l] - 3, 2.5) for l in true_labels])
feature_4 = np.array([np.random.normal(feature_means[l] * 0.8, 5) for l in true_labels])
feature_5 = np.array([np.random.normal(feature_means[l] * 1.2, 6) for l in true_labels])
feature_6 = np.array([np.random.normal(feature_means[l] * 0.5, 8) for l in true_labels])
feature_7 = np.random.normal(50, 20, n_samples)
feature_8 = np.random.uniform(0, 100, n_samples)
feature_9 = np.random.exponential(30, n_samples)

# 生成分类特征
cat1_probs = np.array([[0.4,0.3,0.2,0.05,0.05],[0.05,0.4,0.3,0.15,0.1],[0.1,0.05,0.4,0.3,0.15],[0.15,0.1,0.05,0.4,0.3],[0.3,0.15,0.1,0.05,0.4]])
cat_1 = [np.random.choice(['A','B','C','D','E'], p=cat1_probs[l]) for l in true_labels]
cat2_probs = np.array([[0.5,0.3,0.2],[0.3,0.5,0.2],[0.2,0.3,0.5],[0.4,0.4,0.2],[0.2,0.4,0.4]])
cat_2 = [np.random.choice(['X','Y','Z'], p=cat2_probs[l]) for l in true_labels]
cat_3 = np.random.choice(['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10'], n_samples)

# 添加标签错误
noisy_labels = true_labels.copy()
error_indices = np.random.choice(n_samples, 500, replace=False)

# 随机错误
for idx in error_indices[:200]:
    possible = [c for c in range(n_classes) if c != true_labels[idx]]
    noisy_labels[idx] = np.random.choice(possible)

# 邻近错误
adjacent_map = {0:[1], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3]}
for idx in error_indices[200:375]:
    noisy_labels[idx] = np.random.choice(adjacent_map[true_labels[idx]])

# 混淆错误
confusion_map = {0:[2], 1:[3], 2:[0,4], 3:[1], 4:[2]}
for idx in error_indices[375:]:
    noisy_labels[idx] = np.random.choice(confusion_map[true_labels[idx]])

# 创建DataFrame
df = pd.DataFrame({
    'id': range(1, n_samples+1),
    'feature_1': feature_1.round(2),
    'feature_2': feature_2.round(2),
    'feature_3': feature_3.round(2),
    'feature_4': feature_4.round(2),
    'feature_5': feature_5.round(2),
    'feature_6': feature_6.round(2),
    'feature_7': feature_7.round(2),
    'feature_8': feature_8.round(2),
    'feature_9': feature_9.round(2),
    'cat_1': cat_1,
    'cat_2': cat_2,
    'cat_3': cat_3,
    'true_label': [f'Class_{l}' for l in true_labels],
    'label': [f'Class_{l}' for l in noisy_labels]
})

# 保存文件
df.to_csv('large_error_sample.csv', index=False)
print("数据集生成完成！")
print(f"总样本数: {len(df)}")
print(f"错误率: 25%")