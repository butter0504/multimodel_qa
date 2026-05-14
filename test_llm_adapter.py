"""
LLM 适配器数据接入测试
=====================
测试 DeepSeek LLM 辅助通道的完整流程：
  1. 规则回退解析（无需 API 调用）
  2. Schema 校验
  3. DataObject 构建
  4. DeepSeek API 调用（需要 .env 配置）
  5. JSON 提取与容错
  6. 标记与人工介入机制
  7. AdapterManager 自动路由 + LLM 回退
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from modules.adapters.llm_adapter import LLMAdapter
from modules.adapters.manager import AdapterManager
from modules.data_object import DataObject


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_rule_based_fallback():
    print_section("测试 1: 规则回退解析（无需 API）")

    adapter = LLMAdapter()

    # 1a: YOLO 风格数据
    yolo_data = "0 0.5 0.3 0.2 0.4\n1 0.3 0.6 0.15 0.3"
    result = adapter._rule_based_fallback(yolo_data)
    print(f"  [YOLO 风格] samples={len(result['samples'])}, labels={result['label_names']}, task={result['task_type']}")
    for s in result["samples"]:
        print(f"    {s}")
    assert result["task_type"] == "detection"
    assert len(result["samples"]) == 2

    # 1b: 文本标注 + bbox 数据
    text_bbox_data = "img001 cat 120 80 300 250\nimg002 dog 50 30 200 180\nimg003 cat"
    result = adapter._rule_based_fallback(text_bbox_data)
    print(f"\n  [文本+bbox] samples={len(result['samples'])}, labels={result['label_names']}, task={result['task_type']}")
    for s in result["samples"]:
        print(f"    {s}")
    assert result["task_type"] == "detection"
    assert len(result["samples"]) == 3

    # 1c: 纯分类标签
    classify_data = "sample_01 positive\nsample_02 negative\nsample_03 positive"
    result = adapter._rule_based_fallback(classify_data)
    print(f"\n  [纯分类] samples={len(result['samples'])}, labels={result['label_names']}, task={result['task_type']}")
    for s in result["samples"]:
        print(f"    {s}")
    assert result["task_type"] == "classification"
    assert len(result["samples"]) == 3

    # 1d: 空数据
    result = adapter._rule_based_fallback("")
    print(f"\n  [空数据] result={result}")
    assert result is None

    print("\n  ✅ 规则回退解析全部通过")


def test_schema_validation():
    print_section("测试 2: Schema 校验")

    adapter = LLMAdapter()

    # 2a: 合法结构
    valid = {
        "samples": [
            {"sample_id": "001", "label": "cat", "bbox": [10, 20, 100, 200]},
            {"sample_id": "002", "label": "dog"},
        ],
        "label_names": ["cat", "dog"],
        "task_type": "detection",
    }
    assert adapter._validate_schema(valid) is True
    print("  [合法结构] ✅ 通过")

    # 2b: 缺少 samples
    invalid1 = {"label_names": ["cat"], "task_type": "classification"}
    assert adapter._validate_schema(invalid1) is False
    print("  [缺少 samples] ✅ 正确拒绝")

    # 2c: samples 不是列表
    invalid2 = {"samples": "not a list"}
    assert adapter._validate_schema(invalid2) is False
    print("  [samples 非列表] ✅ 正确拒绝")

    # 2d: sample 缺少 sample_id
    invalid3 = {"samples": [{"label": "cat"}]}
    assert adapter._validate_schema(invalid3) is False
    print("  [缺少 sample_id] ✅ 正确拒绝")

    # 2e: bbox 格式错误
    invalid4 = {"samples": [{"sample_id": "1", "bbox": [1, 2, 3]}]}
    assert adapter._validate_schema(invalid4) is False
    print("  [bbox 长度不足] ✅ 正确拒绝")

    invalid5 = {"samples": [{"sample_id": "1", "bbox": [1, 2, 3, "a"]}]}
    assert adapter._validate_schema(invalid5) is False
    print("  [bbox 含非数字] ✅ 正确拒绝")

    # 2f: 合法但无 bbox
    valid_no_bbox = {"samples": [{"sample_id": "1", "label": "cat"}]}
    assert adapter._validate_schema(valid_no_bbox) is True
    print("  [无 bbox 合法] ✅ 通过")

    print("\n  ✅ Schema 校验全部通过")


def test_build_data_object():
    print_section("测试 3: DataObject 构建")

    adapter = LLMAdapter()

    parsed = {
        "samples": [
            {"sample_id": "img001", "label": "cat", "bbox": [10, 20, 100, 200]},
            {"sample_id": "img002", "label": "dog"},
            {"sample_id": "img003", "label": "cat", "bbox": [50, 60, 150, 200], "extra": {"difficult": 1}},
        ],
        "label_names": ["cat", "dog"],
        "task_type": "detection",
    }

    data_obj = adapter._build_data_object(parsed)

    print(f"  sample_count={data_obj.sample_count}")
    print(f"  num_classes={data_obj.num_classes}")
    print(f"  has_labels={data_obj.has_labels}")
    print(f"  has_bboxes={data_obj.has_bboxes}")
    print(f"  labels={data_obj.get_labels()}")
    print(f"  label_indices={data_obj.get_label_indices()}")
    print(f"  source_format={data_obj.source_format}")
    print(f"  metadata={data_obj.metadata}")

    assert data_obj.sample_count == 3
    assert data_obj.num_classes == 2
    assert data_obj.has_labels is True
    assert data_obj.has_bboxes is True
    assert data_obj.source_format == "llm"
    assert data_obj.metadata["parse_status"] == "success"

    # 检查 bbox 构建
    first_ann = data_obj.samples[0].annotations[0]
    assert first_ann.bbox is not None
    assert first_ann.bbox.x_min == 10
    assert first_ann.bbox.x_max == 100

    print("\n  ✅ DataObject 构建全部通过")


def test_json_extraction():
    print_section("测试 4: JSON 提取与容错")

    adapter = LLMAdapter()

    # 4a: 标准带 markdown 代码块的输出
    md_output = '```json\n{"samples": [{"sample_id": "1", "label": "cat"}], "label_names": ["cat"], "task_type": "classification"}\n```'
    result = adapter._extract_json(md_output)
    assert result is not None
    assert result["samples"][0]["label"] == "cat"
    print("  [markdown 代码块] ✅ 正确提取")

    # 4b: 纯 JSON 输出
    pure_json = '{"samples": [{"sample_id": "1", "label": "dog"}], "label_names": ["dog"]}'
    result = adapter._extract_json(pure_json)
    assert result is not None
    assert result["samples"][0]["label"] == "dog"
    print("  [纯 JSON] ✅ 正确提取")

    # 4c: 带前后说明文字的输出
    mixed_output = '好的，以下是解析结果：\n{"samples": [{"sample_id": "1", "label": "bird"}], "label_names": ["bird"]}\n以上是结果。'
    result = adapter._extract_json(mixed_output)
    assert result is not None
    assert result["samples"][0]["label"] == "bird"
    print("  [带前后文字] ✅ 正确提取")

    # 4d: 无代码块的 markdown
    no_lang_output = '```\n{"samples": [{"sample_id": "1", "label": "fish"}], "label_names": ["fish"]}\n```'
    result = adapter._extract_json(no_lang_output)
    assert result is not None
    print("  [无语言标记代码块] ✅ 正确提取")

    # 4e: 无法解析的输出
    garbage = "This is not JSON at all, just plain text."
    result = adapter._extract_json(garbage)
    assert result is None
    print("  [无效输出] ✅ 正确返回 None")

    print("\n  ✅ JSON 提取与容错全部通过")


def test_flag_and_human_review():
    print_section("测试 5: 标记与人工介入机制")

    adapter = LLMAdapter()

    # 5a: 解析失败 → 标记需人工介入
    data_obj = adapter.load("", raw_data="")
    assert data_obj.metadata.get("error") == "数据为空" or data_obj.metadata.get("parse_status") == "failed"
    print(f"  [空数据] metadata={data_obj.metadata}")

    # 5b: 模拟 Schema 校验失败
    original_validate = adapter._validate_schema
    adapter._validate_schema = lambda x: False
    data_obj = adapter.load("", raw_data="some data here")
    assert data_obj.metadata.get("needs_human_review") is True
    assert "schema_validation_failed" in data_obj.metadata.get("parse_status", "")
    print(f"  [Schema 失败] metadata={data_obj.metadata}")
    adapter._validate_schema = original_validate

    # 5c: 成功解析 → 不需要人工介入
    data_obj = adapter.load("", raw_data="img001 cat 120 80 300 250")
    if data_obj.metadata.get("parse_status") == "success":
        assert data_obj.metadata.get("needs_human_review") is False
        print(f"  [成功解析] metadata={data_obj.metadata}")
    else:
        print(f"  [回退解析] metadata={data_obj.metadata}")

    print("\n  ✅ 标记与人工介入机制全部通过")


def test_deepseek_api():
    print_section("测试 6: DeepSeek API 调用（需要 .env 配置）")

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("LLM_MODEL", "")

    print(f"  OPENAI_API_KEY: {'已配置 (' + api_key[:8] + '...)' if api_key else '❌ 未配置'}")
    print(f"  OPENAI_BASE_URL: {base_url or '❌ 未配置'}")
    print(f"  LLM_MODEL: {model or '❌ 未配置'}")

    if not api_key:
        print("\n  ⚠️  未检测到 API Key，跳过 DeepSeek API 测试")
        print("  请在项目根目录创建 .env 文件，内容如下：")
        print("  ─────────────────────────────")
        print("  OPENAI_API_KEY=sk-your-deepseek-key")
        print("  OPENAI_BASE_URL=https://api.deepseek.com")
        print("  LLM_MODEL=deepseek-chat")
        print("  ─────────────────────────────")
        return

    adapter = LLMAdapter()

    # 6a: 简单分类数据
    test_data_1 = "img001 cat\nimg002 dog\nimg003 cat\nimg004 bird"
    print(f"\n  [测试数据 1] 简单分类标签")
    print(f"  输入: {test_data_1}")
    data_obj = adapter.load("", raw_data=test_data_1)
    print(f"  parse_status: {data_obj.metadata.get('parse_status')}")
    print(f"  needs_human_review: {data_obj.metadata.get('needs_human_review')}")
    print(f"  sample_count: {data_obj.sample_count}")
    print(f"  label_names: {data_obj.label_names}")
    print(f"  task_type: {data_obj.task_type}")
    for s in data_obj.samples:
        print(f"    sample_id={s.sample_id}, label={s.primary_label}, bbox={s.annotations[0].bbox if s.annotations else None}")

    # 6b: 带边界框的检测数据
    test_data_2 = "photo_001 person 100 50 300 400\nphoto_002 car 200 150 500 350\nphoto_003 person 80 60 250 380"
    print(f"\n  [测试数据 2] 带边界框检测数据")
    print(f"  输入: {test_data_2}")
    data_obj = adapter.load("", raw_data=test_data_2)
    print(f"  parse_status: {data_obj.metadata.get('parse_status')}")
    print(f"  sample_count: {data_obj.sample_count}")
    print(f"  label_names: {data_obj.label_names}")
    print(f"  task_type: {data_obj.task_type}")
    print(f"  has_bboxes: {data_obj.has_bboxes}")

    # 6c: 半结构化混合数据（LLM 应能理解）
    test_data_3 = """image: DSC_0001.JPG | class: airplane | region: [120,80,340,260]
image: DSC_0002.JPG | class: automobile | region: [50,30,280,190]
image: DSC_0003.JPG | class: airplane
image: DSC_0004.JPG | class: truck | region: [200,100,450,300]"""
    print(f"\n  [测试数据 3] 半结构化混合数据（竖线分隔）")
    print(f"  输入: {test_data_3[:80]}...")
    data_obj = adapter.load("", raw_data=test_data_3)
    print(f"  parse_status: {data_obj.metadata.get('parse_status')}")
    print(f"  sample_count: {data_obj.sample_count}")
    print(f"  label_names: {data_obj.label_names}")
    print(f"  task_type: {data_obj.task_type}")

    # 6d: 不规则格式数据（LLM 的核心优势场景）
    test_data_4 = """编号001 猫 位置(120,80)-(300,250)
编号002 狗 位置(50,30)-(200,180)
编号003 猫"""
    print(f"\n  [测试数据 4] 不规则中文格式数据")
    print(f"  输入: {test_data_4}")
    data_obj = adapter.load("", raw_data=test_data_4)
    print(f"  parse_status: {data_obj.metadata.get('parse_status')}")
    print(f"  sample_count: {data_obj.sample_count}")
    print(f"  label_names: {data_obj.label_names}")
    print(f"  task_type: {data_obj.task_type}")

    print("\n  ✅ DeepSeek API 调用测试完成")


def test_adapter_manager_fallback():
    print_section("测试 7: AdapterManager 自动路由 + LLM 回退")

    mgr = AdapterManager()

    # 7a: 无法识别的格式 → LLM 回退
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False, mode="w")
    tmp.write("sample_A positive\nsample_B negative\nsample_C positive\n")
    tmp.close()

    data_obj = mgr.load(tmp.name)
    print(f"  [未知 .dat 文件] source_format={data_obj.source_format}")
    print(f"  sample_count={data_obj.sample_count}")
    print(f"  label_names={data_obj.label_names}")
    if data_obj.sample_count > 0:
        print(f"  labels={data_obj.get_labels()}")
    os.unlink(tmp.name)

    # 7b: 明确指定 LLM 格式
    tmp2 = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    tmp2.write("img001 cat 120 80 300 250\nimg002 dog\n")
    tmp2.close()

    data_obj = mgr.load(tmp2.name, format_hint="llm")
    print(f"\n  [指定 LLM 格式] source_format={data_obj.source_format}")
    print(f"  sample_count={data_obj.sample_count}")
    print(f"  label_names={data_obj.label_names}")
    os.unlink(tmp2.name)

    print("\n  ✅ AdapterManager 自动路由测试完成")


def test_load_bytes():
    print_section("测试 8: load_bytes 字节流加载")

    adapter = LLMAdapter()

    raw = b"img001 cat 120 80 300 250\nimg002 dog 50 30 200 180"
    data_obj = adapter.load_bytes(raw, filename="labels.txt")
    print(f"  source_format={data_obj.source_format}")
    print(f"  sample_count={data_obj.sample_count}")
    print(f"  label_names={data_obj.label_names}")
    if data_obj.sample_count > 0:
        print(f"  labels={data_obj.get_labels()}")

    print("\n  ✅ load_bytes 测试完成")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     LLM 适配器数据接入测试 - DeepSeek                   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_rule_based_fallback()
    test_schema_validation()
    test_build_data_object()
    test_json_extraction()
    test_flag_and_human_review()
    test_deepseek_api()
    test_adapter_manager_fallback()
    test_load_bytes()

    print("\n" + "=" * 60)
    print("  🎉 全部测试完成！")
    print("=" * 60)
