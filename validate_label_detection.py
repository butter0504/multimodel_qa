import pandas as pd
import numpy as np
from modules.table_detector import TableDetector

# 生成带有已知标签错误的数据集
def generate_dataset_with_label_errors():
    """生成一个带有已知标签错误的数据集"""
    # 生成基本特征
    np.random.seed(42)
    n_samples = 100
    
    # 生成两个特征
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    
    # 生成真实标签
    y_true = np.zeros(n_samples)
    y_true[X1 + X2 > 0] = 1
    
    # 人为添加标签错误（约10%的错误率）
    y_noisy = y_true.copy()
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y_noisy[error_indices] = 1 - y_noisy[error_indices]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'feature1': X1,
        'feature2': X2,
        'label': y_noisy
    })
    
    return df, error_indices

# 验证标签错误检测
def validate_label_detection():
    """验证标签错误检测功能"""
    print("生成带有标签错误的数据集...")
    df, true_error_indices = generate_dataset_with_label_errors()
    
    print(f"数据集形状: {df.shape}")
    print(f"真实标签错误索引: {true_error_indices}")
    print(f"真实标签错误数量: {len(true_error_indices)}")
    
    print("\n使用表格分析器检测标签错误...")
    detector = TableDetector()
    result = detector.detect(df, label_col='label')
    
    print("\n检测结果:")
    print(f"检测到的标签错误数量: {result['label_issues'].get('error_count', 0)}")
    
    if 'label_issues' in result:
        label_issues = result['label_issues']
        print(f"建议标签: {label_issues.get('suggested_labels', {})}")
        print(f"错误索引: {label_issues.get('error_indices', [])}")
        
        # 计算检测准确率
        detected_error_indices = set(label_issues.get('error_indices', []))
        true_error_indices_set = set(true_error_indices)
        
        # 计算 true positive
        true_positive = len(detected_error_indices.intersection(true_error_indices_set))
        # 计算 false positive
        false_positive = len(detected_error_indices - true_error_indices_set)
        # 计算 false negative
        false_negative = len(true_error_indices_set - detected_error_indices)
        
        print(f"\n检测性能:")
        print(f"True Positive: {true_positive}")
        print(f"False Positive: {false_positive}")
        print(f"False Negative: {false_negative}")
        
        if len(detected_error_indices) > 0:
            precision = true_positive / len(detected_error_indices)
            print(f"Precision: {precision:.2f}")
        
        if len(true_error_indices_set) > 0:
            recall = true_positive / len(true_error_indices_set)
            print(f"Recall: {recall:.2f}")
    else:
        print("没有检测到标签问题")

if __name__ == "__main__":
    validate_label_detection()
