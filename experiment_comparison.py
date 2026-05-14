"""
标签错误检测算法对比实验
======================
在 CIFAR-10N 数据集上对比四种标签错误检测方法的性能：
    1. Random: 随机检测方法，为每个样本随机生成质量分数
    2. Confidence: 基于置信度的方法，使用模型预测的最大概率作为质量分数
    3. Loss-based: 基于损失的方法，使用交叉熵损失作为检测依据
    4. Ours (CL): 本系统采用的置信学习方法（Confidence Learning）

评估指标：
    - Precision: 精确率
    - Recall: 召回率
    - F1: F1分数
    - AUC: ROC曲线下面积
    - 时间: 检测耗时

使用方式：
    python experiment_comparison.py
    python experiment_comparison.py --sample_size 5000
    python experiment_comparison.py --data_path ./data/raw
"""

import sys
import os
import time
import argparse
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config import Config
from modules.image_detector import LabelErrorDetector

CIFAR10_CONFUSION_PAIRS = {
    0: [2, 3],
    1: [8, 7],
    2: [0, 3],
    3: [0, 2, 5],
    4: [6, 7],
    5: [3, 7],
    6: [4, 2],
    7: [5, 1],
    8: [1, 0],
    9: [1, 7],
}


def inject_class_conditional_noise(clean_labels, n_classes, noise_rate=0.2, seed=42):
    rng = np.random.RandomState(seed)
    noisy_labels = clean_labels.copy()
    n_samples = len(clean_labels)

    for i in range(n_samples):
        if rng.rand() < noise_rate:
            original = noisy_labels[i]
            confusion_targets = CIFAR10_CONFUSION_PAIRS.get(original, [])
            valid_targets = [t for t in confusion_targets if t != original]
            if valid_targets:
                noisy_labels[i] = rng.choice(valid_targets)
            else:
                candidates = [c for c in range(n_classes) if c != original]
                noisy_labels[i] = rng.choice(candidates)

    return noisy_labels


def load_cifar10n_data(uploads_dir, sample_size=None):
    """
    从 cifar10n_images.zip + cifar10n_labels.csv 加载 CIFAR-10N 数据集。

    Parameters
    ----------
    uploads_dir : str
        data/uploads 目录路径
    sample_size : int or None
        采样数量

    Returns
    -------
    dict : 包含 images, labels, clean_labels, label_names 的字典
    """
    import pandas as pd
    import zipfile
    from io import BytesIO
    from PIL import Image

    csv_path = os.path.join(uploads_dir, 'cifar10n_labels.csv')
    zip_path = os.path.join(uploads_dir, 'cifar10n_images.zip')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到标签文件: {csv_path}")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"未找到图像文件: {zip_path}")

    print("[1/2] 加载 CIFAR-10N 标签文件...")
    df = pd.read_csv(csv_path)

    if sample_size and sample_size < len(df):
        df = df.head(sample_size)

    noisy_labels = df['label'].values
    clean_labels = df['clean_label'].values
    filenames = df['filename'].values

    label_names = sorted(df['label_name'].unique().tolist(),
                         key=lambda x: df.loc[df['label_name'] == x, 'label'].iloc[0])
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"  标签数: {len(df)}")

    print("[2/2] 加载 CIFAR-10N 图像文件...")
    images = []
    filename_set = set(filenames)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        name_list = sorted([n for n in zf.namelist()
                           if not n.startswith('__MACOSX')
                           and n.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for name in name_list:
            basename = os.path.basename(name)
            if basename in filename_set:
                img_bytes = zf.read(name)
                images.append(img_bytes)

    if len(images) != len(df):
        print(f"  警告: 图像数({len(images)})与标签数({len(df)})不匹配，按文件名对齐...")
        img_map = {}
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in name_list:
                basename = os.path.basename(name)
                if basename in filename_set:
                    img_map[basename] = zf.read(name)

        images = []
        for fn in filenames:
            if fn in img_map:
                images.append(img_map[fn])
            else:
                img = Image.new('RGB', (32, 32), (0, 0, 0))
                buf = BytesIO()
                img.save(buf, format='PNG')
                images.append(buf.getvalue())

    n_errors = int(np.sum(noisy_labels != clean_labels))
    noise_rate = n_errors / len(noisy_labels)

    print(f"  成功加载 CIFAR-10N")
    print(f"  样本数: {len(images)}")
    print(f"  实际错误标签数: {n_errors}")
    print(f"  实际噪声率: {noise_rate:.2%}")

    return {
        'images': images,
        'labels': noisy_labels.tolist(),
        'clean_labels': clean_labels.tolist(),
        'label_names': label_names,
        'filenames': filenames.tolist(),
        'metadata': {
            'source': 'CIFAR-10N',
            'total_samples': len(images),
            'noise_rate': noise_rate,
            'actual_errors': n_errors,
        },
    }


def extract_features(images, cfg):
    from sklearn.decomposition import PCA
    from PIL import Image
    import io

    pixel_vectors = []
    for img_data in images:
        if isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data))
        else:
            img = img_data
        img_resized = img.resize((32, 32))
        if img_resized.mode != 'RGB':
            img_resized = img_resized.convert('RGB')
        arr = np.array(img_resized, dtype=np.float32).flatten()
        arr /= 255.0
        pixel_vectors.append(arr)

    X = np.array(pixel_vectors)
    n_components = min(128, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    led = LabelErrorDetector(cfg)
    color_features = led._extract_features(images)

    features = np.hstack([X_pca, color_features])
    return features


def compute_pred_probs(features, y, cfg):
    cv_folds = min(cfg.get('detection.cross_validation_folds', 5),
                   int(np.min(np.bincount(y))))
    if cv_folds < 2:
        cv_folds = 2

    model = RandomForestClassifier(
        n_estimators=cfg.get('detection.n_estimators', 200),
        random_state=cfg.get('detection.random_state', 42),
        n_jobs=-1,
    )
    pred_probs = cross_val_predict(
        model, features, y, cv=cv_folds, method='predict_proba'
    )
    return pred_probs


class RandomDetector:
    """
    随机检测方法。

    为每个样本随机生成 (0, 1) 区间的质量分数，
    分数越低表示越可疑。使用 0.5 作为判定阈值，
    即标记约 50% 的样本为可疑。
    """

    def __init__(self, seed=42):
        self.seed = seed

    def get_quality_scores(self, y):
        rng = np.random.RandomState(self.seed)
        return rng.rand(len(y))

    def detect(self, y):
        scores = self.get_quality_scores(y)
        return scores < 0.5


class ConfidenceDetector:
    """
    基于置信度的检测方法。

    使用模型预测的最大概率（max probability）作为质量分数，
    最大预测概率越低，说明模型越不确定，标签越可能有问题。

    判定规则：标记 max_prob 最低的 top_k_frac 比例样本为可疑。

    注意：max_prob 衡量的是模型对其最可能类别的信心程度，
    而非对给定标签的认同程度。当模型高置信度地预测一个
    与给定标签不同的类别时，max_prob 仍然很高，因此
    该方法无法有效识别此类标签错误。

    参考：
        Hendrycks & Gimpel (2017). A Baseline for Detecting Misclassified
        and Out-of-Distribution Examples in Neural Networks. ICLR.
    """

    def __init__(self, top_k_frac=0.2):
        self.top_k_frac = top_k_frac

    def get_quality_scores(self, pred_probs):
        return np.max(pred_probs, axis=1)

    def detect(self, pred_probs, y):
        scores = self.get_quality_scores(pred_probs)
        threshold = np.quantile(scores, self.top_k_frac)
        return scores < threshold


class LossBasedDetector:
    """
    基于损失的方法。

    使用交叉熵损失作为检测依据，损失值越高表示越可疑。
    判定规则：损失值超过均值 + k * 标准差。

    参考：
        MentorNet (Jiang et al., 2018) 使用样本损失作为
        课程学习的数据加权依据。
    """

    def __init__(self, k=0.5):
        self.k = k

    def get_quality_scores(self, pred_probs, y):
        losses = np.zeros(len(y))
        for i in range(len(y)):
            p = pred_probs[i, y[i]]
            losses[i] = -np.log(np.clip(p, 1e-10, 1.0))
        max_loss = np.max(losses) if np.max(losses) > 0 else 1.0
        return 1.0 - losses / max_loss

    def detect(self, pred_probs, y):
        losses = np.zeros(len(y))
        for i in range(len(y)):
            p = pred_probs[i, y[i]]
            losses[i] = -np.log(np.clip(p, 1e-10, 1.0))
        threshold = np.mean(losses) + self.k * np.std(losses)
        return losses > threshold


class CLDetector:
    """
    本系统方法：置信学习（Confidence Learning）。

    使用 cleanlab 的 CleanLearning 完整管线检测标签错误。
    CleanLearning 先通过交叉验证获取预测概率，再构建
    置信联合矩阵（Confident Joint）估计噪声转移概率，
    然后清洗训练数据中的问题样本，在清洗后的数据上
    重新训练模型，最终使用每类自适应阈值判定标签错误。

    相比基线方法的全局阈值，CL 的逐类阈值能更准确
    地识别标签问题，尤其在类条件噪声下优势明显。

    质量分数使用 normalized_margin 方法计算：
    score_i = P(y_i | x_i) - max_{j != y_i} P(j | x_i)

    参考：
        Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021).
        Confident Learning: Estimating Uncertainty in Dataset Labels.
        JAIR, 70, 1373-1411.
    """

    def get_quality_scores(self, pred_probs, y):
        from cleanlab.rank import get_label_quality_scores
        return get_label_quality_scores(
            y, pred_probs, method="normalized_margin"
        )

    def detect(self, features, y, cfg):
        from cleanlab.classification import CleanLearning

        cv_folds = min(cfg.get('detection.cross_validation_folds', 5),
                       int(np.min(np.bincount(y))))
        if cv_folds < 2:
            cv_folds = 2

        model = RandomForestClassifier(
            n_estimators=cfg.get('detection.n_estimators', 200),
            random_state=cfg.get('detection.random_state', 42),
            n_jobs=-1,
        )

        cl = CleanLearning(clf=model, cv_n_folds=cv_folds)
        cl.fit(features, y)

        label_issues_df = cl.get_label_issues()
        issues_mask = np.zeros(len(y), dtype=bool)
        if label_issues_df is not None and len(label_issues_df) > 0:
            issues_mask[label_issues_df.index] = label_issues_df['is_label_issue'].values

        pred_probs = cl.predict_proba(features)
        self._pred_probs = pred_probs

        return issues_mask


def evaluate(y_true, y_pred, quality_scores=None):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    auc = 0.5
    if quality_scores is not None:
        suspiciousness = 1.0 - quality_scores
        try:
            auc = roc_auc_score(y_true, suspiciousness)
        except ValueError:
            auc = 0.5

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }


def run_experiment(uploads_dir, sample_size=None):
    print("=" * 50)
    print(" 标签错误检测算法对比实验")

    print("\n[Step 1] 加载数据集...")
    data = load_cifar10n_data(uploads_dir, sample_size)
    images = data['images']
    noisy_labels = np.array(data['labels'])
    clean_labels = np.array(data['clean_labels'])

    n_samples = len(images)
    n_errors = int(np.sum(noisy_labels != clean_labels))
    noise_rate = n_errors / n_samples
    y_true = (noisy_labels != clean_labels).astype(int)

    print(f"  数据集: CIFAR-10N ({n_samples}样本)")
    print(f"  实际错误标签数: {n_errors}")
    print(f"  实际噪声率: {noise_rate:.2%}")

    cfg = Config()
    cfg.set('detection.threshold', 0.3)

    print("\n[Step 2] 提取图像特征...")
    t0 = time.time()
    features = extract_features(images, cfg)
    feat_time = time.time() - t0
    print(f"  特征维度: {features.shape}")
    print(f"  特征提取耗时: {feat_time:.1f}s")

    print("\n[Step 3] 交叉验证获取预测概率（基线方法共享）...")
    t0 = time.time()
    pred_probs = compute_pred_probs(features, noisy_labels, cfg)
    cv_time = time.time() - t0
    print(f"  预测概率矩阵: {pred_probs.shape}")
    print(f"  交叉验证耗时: {cv_time:.1f}s")

    results = {}

    # ---- Random ----
    print("\n[Step 4] 运行 Random 方法...")
    detector_random = RandomDetector()
    t0 = time.time()
    y_pred_random = detector_random.detect(noisy_labels)
    scores_random = detector_random.get_quality_scores(noisy_labels)
    time_random = time.time() - t0
    eval_random = evaluate(y_true, y_pred_random, scores_random)
    results['Random'] = {**eval_random, 'time': time_random}
    print(f"  P:{eval_random['precision']:.3f} R:{eval_random['recall']:.3f} "
          f"F1:{eval_random['f1']:.3f} AUC:{eval_random['auc']:.3f} "
          f"时间:{time_random:.1f}s")

    # ---- Confidence ----
    print("\n[Step 5] 运行 Confidence 方法...")
    detector_conf = ConfidenceDetector(top_k_frac=0.2)
    t0 = time.time()
    y_pred_conf = detector_conf.detect(pred_probs, noisy_labels)
    scores_conf = detector_conf.get_quality_scores(pred_probs)
    time_conf = time.time() - t0 + cv_time
    eval_conf = evaluate(y_true, y_pred_conf, scores_conf)
    results['Confidence'] = {**eval_conf, 'time': time_conf}
    print(f"  P:{eval_conf['precision']:.3f} R:{eval_conf['recall']:.3f} "
          f"F1:{eval_conf['f1']:.3f} AUC:{eval_conf['auc']:.3f} "
          f"时间:{time_conf:.1f}s")

    # ---- Loss-based ----
    print("\n[Step 6] 运行 Loss-based 方法...")
    detector_loss = LossBasedDetector(k=0.5)
    t0 = time.time()
    y_pred_loss = detector_loss.detect(pred_probs, noisy_labels)
    scores_loss = detector_loss.get_quality_scores(pred_probs, noisy_labels)
    time_loss = time.time() - t0 + cv_time
    eval_loss = evaluate(y_true, y_pred_loss, scores_loss)
    results['Loss-based'] = {**eval_loss, 'time': time_loss}
    print(f"  P:{eval_loss['precision']:.3f} R:{eval_loss['recall']:.3f} "
          f"F1:{eval_loss['f1']:.3f} AUC:{eval_loss['auc']:.3f} "
          f"时间:{time_loss:.1f}s")

    # ---- Ours (CL) ----
    print("\n[Step 7] 运行 Ours (CL) 方法（完整 CleanLearning 管线）...")
    detector_cl = CLDetector()
    t0 = time.time()
    y_pred_cl = detector_cl.detect(features, noisy_labels, cfg)
    cl_pred_probs = detector_cl._pred_probs
    scores_cl = detector_cl.get_quality_scores(cl_pred_probs, noisy_labels)
    time_cl = time.time() - t0
    eval_cl = evaluate(y_true, y_pred_cl, scores_cl)
    results['Ours (CL)'] = {**eval_cl, 'time': time_cl}
    print(f"  P:{eval_cl['precision']:.3f} R:{eval_cl['recall']:.3f} "
          f"F1:{eval_cl['f1']:.3f} AUC:{eval_cl['auc']:.3f} "
          f"时间:{time_cl:.1f}s")

    # ---- 打印最终对比结果 ----
    print("\n" + "=" * 50)
    print(" 最终对比结果")
    print("=" * 50)
    header = f"{'方法':<16} {'Precision':<11} {'Recall':<11} {'F1':<11} {'AUC':<11} {'时间(秒)':<10}"
    print(header)
    print("-" * 70)
    for method_name, r in results.items():
        row = (f"{method_name:<16} "
               f"{r['precision']:<11.3f} "
               f"{r['recall']:<11.3f} "
               f"{r['f1']:<11.3f} "
               f"{r['auc']:<11.3f} "
               f"{r['time']:<10.1f}")
        print(row)

    # ---- 保存结果 ----
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'comparison_results.txt')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write(" 标签错误检测算法对比实验\n")
        f.write(f" 数据集: CIFAR-10N ({n_samples}样本)\n")
        f.write("=" * 50 + "\n\n")

        for method_name, r in results.items():
            f.write(f"{method_name} - "
                    f"P:{r['precision']:.3f} "
                    f"R:{r['recall']:.3f} "
                    f"F1:{r['f1']:.3f} "
                    f"AUC:{r['auc']:.3f} "
                    f"时间:{r['time']:.1f}s\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write(" 最终对比结果\n")
        f.write("=" * 50 + "\n")
        f.write(header + "\n")
        f.write("-" * 70 + "\n")
        for method_name, r in results.items():
            row = (f"{method_name:<16} "
                   f"{r['precision']:<11.3f} "
                   f"{r['recall']:<11.3f} "
                   f"{r['f1']:<11.3f} "
                   f"{r['auc']:<11.3f} "
                   f"{r['time']:<10.1f}")
            f.write(row + "\n")

    print(f"\n结果已保存到: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='标签错误检测算法对比实验')
    parser.add_argument('--uploads_dir', type=str,
                        default=r'c:\Users\abc14\Desktop\毕业设计\代码\multimodel_qa\data\uploads',
                        help='data/uploads 目录路径')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='采样数量（0=全部，默认使用全部数据）')

    args = parser.parse_args()
    sample_size = None if args.sample_size == 0 else args.sample_size

    run_experiment(args.uploads_dir, sample_size)


if __name__ == '__main__':
    main()
