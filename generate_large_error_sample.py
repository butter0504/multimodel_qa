import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 设置随机种子保证可复现
np.random.seed(42)

# 参数设置
n_samples = 2000
n_classes = 5
error_rate = 0.25
n_errors = int(n_samples * error_rate)

# 1. 生成真实标签（均匀分布）
true_labels = np.random.choice(n_classes, n_samples)

# 2. 生成数值特征
def generate_features(true_labels):
    # 强相关特征
    feature_means = [10, 20, 30, 40, 50]
    feature_1 = np.array([np.random.normal(feature_means[l], 2) for l in true_labels])
    feature_2 = np.array([np.random.normal(feature_means[l] + 5, 3) for l in true_labels])
    feature_3 = np.array([np.random.normal(feature_means[l] - 3, 2.5) for l in true_labels])
    
    # 中等相关特征
    feature_4 = np.array([np.random.normal(feature_means[l] * 0.8, 5) for l in true_labels])
    feature_5 = np.array([np.random.normal(feature_means[l] * 1.2, 6) for l in true_labels])
    feature_6 = np.array([np.random.normal(feature_means[l] * 0.5, 8) for l in true_labels])
    
    # 随机噪声特征
    feature_7 = np.random.normal(50, 20, n_samples)
    feature_8 = np.random.uniform(0, 100, n_samples)
    feature_9 = np.random.exponential(30, n_samples)
    
    return np.column_stack([feature_1, feature_2, feature_3, feature_4, 
                           feature_5, feature_6, feature_7, feature_8, feature_9])

X_numeric = generate_features(true_labels)

# 3. 生成分类特征
# cat_1：与标签相关
cat1_probs = np.array([
    [0.4, 0.3, 0.2, 0.05, 0.05],  # Class_0
    [0.05, 0.4, 0.3, 0.15, 0.1],  # Class_1
    [0.1, 0.05, 0.4, 0.3, 0.15],  # Class_2
    [0.15, 0.1, 0.05, 0.4, 0.3],  # Class_3
    [0.3, 0.15, 0.1, 0.05, 0.4]   # Class_4
])
cat_1 = [np.random.choice(['A','B','C','D','E'], p=cat1_probs[l]) for l in true_labels]

# cat_2：中等相关
cat2_probs = np.array([
    [0.5, 0.3, 0.2],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5],
    [0.4, 0.4, 0.2],
    [0.2, 0.4, 0.4]
])
cat_2 = [np.random.choice(['X','Y','Z'], p=cat2_probs[l]) for l in true_labels]

# cat_3：完全随机
cat_3 = np.random.choice(['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10'], n_samples)

# 4. 添加标签错误
noisy_labels = true_labels.copy()
error_indices = np.random.choice(n_samples, n_errors, replace=False)

error_types = {
    'random': error_indices[:int(n_errors*0.4)],
    'adjacent': error_indices[int(n_errors*0.4):int(n_errors*0.75)],
    'confusion': error_indices[int(n_errors*0.75):]
}

# 随机错误
for idx in error_types['random']:
    possible = [c for c in range(n_classes) if c != true_labels[idx]]
    noisy_labels[idx] = np.random.choice(possible)

# 邻近错误
adjacent_map = {0:[1], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3]}
for idx in error_types['adjacent']:
    noisy_labels[idx] = np.random.choice(adjacent_map[true_labels[idx]])

# 混淆错误
confusion_map = {0:[2], 1:[3], 2:[0,4], 3:[1], 4:[2]}
for idx in error_types['confusion']:
    noisy_labels[idx] = np.random.choice(confusion_map[true_labels[idx]])

# 5. 创建DataFrame
df = pd.DataFrame({
    'id': range(1, n_samples+1),
    'feature_1': X_numeric[:, 0].round(2),
    'feature_2': X_numeric[:, 1].round(2),
    'feature_3': X_numeric[:, 2].round(2),
    'feature_4': X_numeric[:, 3].round(2),
    'feature_5': X_numeric[:, 4].round(2),
    'feature_6': X_numeric[:, 5].round(2),
    'feature_7': X_numeric[:, 6].round(2),
    'feature_8': X_numeric[:, 7].round(2),
    'feature_9': X_numeric[:, 8].round(2),
    'cat_1': cat_1,
    'cat_2': cat_2,
    'cat_3': cat_3,
    'true_label': [f'Class_{l}' for l in true_labels],
    'label': [f'Class_{l}' for l in noisy_labels]
})

# 6. 保存文件
df.to_csv('data/large_error_sample.csv', index=False)

# 7. 打印统计信息
print("="*50)
print("数据集生成完成！")
print("="*50)
print(f"总样本数: {len(df)}")
print(f"特征维度: 9个数值 + 3个分类")
print(f"标签类别: 5个 (Class_0 ~ Class_4)")
print(f"真实错误率: {n_errors/n_samples:.1%}")
print(f"错误样本分布:")
for i in range(n_classes):
    class_errors = sum((true_labels == i) & (noisy_labels != i))
    print(f"  Class_{i}: {class_errors} 个错误")
print("="*50)
print("保存路径: data/large_error_sample.csv")