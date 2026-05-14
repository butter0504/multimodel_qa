import sys
import os
import io
import glob
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from modules.image_detector import ImageDetector, DistributionShiftDetector
from modules.config import Config


def load_cifar10_from_zip(zip_path, max_samples=None):
    images, fnames = ImageDetector.load_images_from_zip(open(zip_path, 'rb').read())
    if max_samples and len(images) > max_samples:
        images = images[:max_samples]
        fnames = fnames[:max_samples]
    print(f"Loaded CIFAR-10: {len(images)} images")
    return images, fnames


def load_tiny_imagenet_from_dir(image_dir, max_samples=None):
    images = []
    fnames = []
    exts = ('*.JPEG', '*.jpg', '*.png')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    files.sort()

    if max_samples:
        files = files[:max_samples]

    for fpath in files:
        try:
            with open(fpath, 'rb') as f:
                img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode == 'L':
                img = img.convert('RGB')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            images.append(buf.getvalue())
            fnames.append(os.path.basename(fpath))
        except Exception as e:
            pass

    print(f"Loaded Tiny-ImageNet: {len(images)} images")
    return images, fnames


def run_distribution_shift_experiment(ref_images, target_images):
    cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    detector = DistributionShiftDetector(cfg)

    print("\n" + "=" * 60)
    print("Running Distribution Shift Detection...")
    print("=" * 60)

    result = detector.detect(target_images, ref_images)

    print("\n--- Covariate Shift (KS Test) ---")
    cs = result.get("covariate_shift", {})
    shifted = cs.get("shifted_dimensions", [])
    total = cs.get("total_dimensions", 0)
    print(f"Shifted dimensions: {len(shifted)} / {total}")
    for dim in shifted:
        print(f"  {dim['dimension']}: KS={dim['ks_statistic']:.4f}, p={dim['p_value']:.2e}")

    print("\n--- Subgroup Shift (JS Divergence) ---")
    ss = result.get("subgroup_shift", {})
    js_val = ss.get("brightness_js_divergence", 0)
    js_threshold = ss.get("js_threshold", 0.1)
    is_shifted = ss.get("is_shifted", False)
    print(f"Brightness JS divergence: {js_val:.4f}")
    print(f"JS threshold: {js_threshold}")
    print(f"Is shifted: {is_shifted}")

    print("\n--- Warnings ---")
    for w in result.get("warnings", []):
        print(f"  {w}")

    return result


def run_threshold_validation_experiment(ref_images):
    print("\n" + "=" * 60)
    print("Running Threshold Validation Experiment...")
    print("=" * 60)

    cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    detector = DistributionShiftDetector(cfg)

    ref_features = detector._extract_distribution_features(ref_images)

    scenarios = {
        "no_shift": ref_images,
    }

    np.random.seed(42)
    sample_size = min(2000, len(ref_images))
    sample_indices = np.random.choice(len(ref_images), sample_size, replace=False)
    sample_images = [ref_images[i] for i in sample_indices]

    weak_shift_images = []
    for img_bytes in sample_images:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            img_array = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img_array).save(buf, format='PNG')
            weak_shift_images.append(buf.getvalue())
        except Exception:
            weak_shift_images.append(img_bytes)
    scenarios["weak_brightness_shift"] = weak_shift_images

    strong_shift_images = []
    for img_bytes in sample_images:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            img_array = np.clip(img_array * 2.0, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img_array).save(buf, format='PNG')
            strong_shift_images.append(buf.getvalue())
        except Exception:
            strong_shift_images.append(img_bytes)
    scenarios["strong_brightness_shift"] = strong_shift_images

    mixed_shift_images = []
    for img_bytes in sample_images:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            img_array = np.clip(img_array * 1.5, 0, 255).astype(np.uint8)
            gray = np.mean(img_array, axis=2, keepdims=True)
            contrast_factor = 1.3
            img_array = np.clip(
                (img_array - gray) * contrast_factor + gray * 1.2,
                0, 255
            ).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img_array).save(buf, format='PNG')
            mixed_shift_images.append(buf.getvalue())
        except Exception:
            mixed_shift_images.append(img_bytes)
    scenarios["mixed_shift"] = mixed_shift_images

    moderate_shift_images = []
    for img_bytes in sample_images:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 40, 0, 255)
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] - 20, 0, 255)
            img_array = img_array.astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img_array).save(buf, format='PNG')
            moderate_shift_images.append(buf.getvalue())
        except Exception:
            moderate_shift_images.append(img_bytes)
    scenarios["moderate_color_shift"] = moderate_shift_images

    dark_shift_images = []
    for img_bytes in sample_images:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            img_array = np.clip(img_array * 0.5, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img_array).save(buf, format='PNG')
            dark_shift_images.append(buf.getvalue())
        except Exception:
            dark_shift_images.append(img_bytes)
    scenarios["dark_shift"] = dark_shift_images

    low_contrast_images = []
    for img_bytes in sample_images:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            img_array = np.clip((img_array - 128) * 0.5 + 128, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img_array).save(buf, format='PNG')
            low_contrast_images.append(buf.getvalue())
        except Exception:
            low_contrast_images.append(img_bytes)
    scenarios["low_contrast_shift"] = low_contrast_images

    slight_brightness_images = []
    for img_bytes in sample_images:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            img_array = np.clip(img_array * 1.1, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img_array).save(buf, format='PNG')
            slight_brightness_images.append(buf.getvalue())
        except Exception:
            slight_brightness_images.append(img_bytes)
    scenarios["slight_brightness_shift"] = slight_brightness_images

    results = {}
    for name, target_imgs in scenarios.items():
        target_features = detector._extract_distribution_features(target_imgs)
        js_div = detector._js_divergence(target_features[:, 0], ref_features[:, 0])
        is_detected = js_div > 0.1
        results[name] = {
            "js_divergence": float(js_div),
            "is_detected": is_detected,
        }
        print(f"\n  {name}:")
        print(f"    JS divergence = {js_div:.4f}")
        print(f"    Detected (threshold=0.1): {is_detected}")

    correct = 0
    total = len(results)
    expected = {
        "no_shift": False,
        "weak_brightness_shift": False,
        "strong_brightness_shift": True,
        "mixed_shift": True,
        "moderate_color_shift": True,
        "dark_shift": True,
        "low_contrast_shift": True,
        "slight_brightness_shift": False,
    }
    for name, res in results.items():
        if res["is_detected"] == expected[name]:
            correct += 1
    accuracy = correct / total * 100
    print(f"\nDetection accuracy: {correct}/{total} = {accuracy:.1f}%")

    return results, accuracy


def compute_detailed_stats(ref_images, target_images):
    print("\n" + "=" * 60)
    print("Computing Detailed Statistics...")
    print("=" * 60)

    cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    detector = DistributionShiftDetector(cfg)

    ref_features = detector._extract_distribution_features(ref_images)
    target_features = detector._extract_distribution_features(target_images)

    feature_names = [
        "brightness_mean", "brightness_std", "saturation_mean",
        "hue_mean", "contrast", "sharpness",
        "r_mean", "g_mean", "b_mean",
    ]

    from scipy.stats import ks_2samp

    print(f"\n{'Dimension':<20} {'Ref Mean':>10} {'Tgt Mean':>10} {'KS Stat':>10} {'p-value':>12} {'Significant':>12}")
    print("-" * 80)

    shifted_count = 0
    for dim_idx in range(min(ref_features.shape[1], target_features.shape[1])):
        ref_dim = ref_features[:, dim_idx]
        tgt_dim = target_features[:, dim_idx]
        stat, p_value = ks_2samp(ref_dim, tgt_dim)
        name = feature_names[dim_idx] if dim_idx < len(feature_names) else f"dim_{dim_idx}"
        sig = p_value < 0.05
        if sig:
            shifted_count += 1
        print(f"{name:<20} {np.mean(ref_dim):>10.4f} {np.mean(tgt_dim):>10.4f} {stat:>10.4f} {p_value:>12.2e} {'Yes' if sig else 'No':>12}")

    print(f"\nShifted dimensions: {shifted_count} / {len(feature_names)}")

    ref_dark_ratio = float(np.mean(ref_features[:, 0] < 0.3))
    tgt_dark_ratio = float(np.mean(target_features[:, 0] < 0.3))
    ref_bright_ratio = float(np.mean(ref_features[:, 0] > 0.8))
    tgt_bright_ratio = float(np.mean(target_features[:, 0] > 0.8))

    print(f"\nCIFAR-10 dark ratio (<0.3): {ref_dark_ratio:.3f}")
    print(f"Tiny-ImageNet dark ratio (<0.3): {tgt_dark_ratio:.3f}")
    print(f"CIFAR-10 bright ratio (>0.8): {ref_bright_ratio:.3f}")
    print(f"Tiny-ImageNet bright ratio (>0.8): {tgt_bright_ratio:.3f}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)

    cifar10_zip = os.path.join(base_dir, "data", "uploads", "cifar10_images.zip")
    tiny_imagenet_dir = os.path.join(base_dir, "data", "raw", "tiny-imagenet-200", "test", "images")

    print("Loading CIFAR-10 (reference set)...")
    ref_images, _ = load_cifar10_from_zip(cifar10_zip, max_samples=5000)

    print("Loading Tiny-ImageNet (target set)...")
    target_images, _ = load_tiny_imagenet_from_dir(tiny_imagenet_dir, max_samples=5000)

    result = run_distribution_shift_experiment(ref_images, target_images)

    compute_detailed_stats(ref_images, target_images)

    threshold_results, accuracy = run_threshold_validation_experiment(ref_images)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
