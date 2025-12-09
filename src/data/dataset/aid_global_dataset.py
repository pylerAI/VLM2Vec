from datasets import load_dataset
import os
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.utils import print_master

SUPPORTED_CATEGORIES = {"sexual", "violence", "hate", "politics", "controversy"}

def _build_image_relpath(video_id: str, image_ext: str = ".jpg") -> str:
    # IMAGE_ROOT/{video_id}.jpg, 여기서는 상대 경로만 생성 (ROOT는 data_args.image_dir)
    return f"{video_id}{image_ext}"

def _clean_text(x):
    return "" if x is None else str(x)

def _ensure_non_empty(s):
    return (s is not None) and (str(s).strip() != "")


def _build_explain_retrieval_batch(example, category: str, image_ext: str, prompt_template: str):
    """
    Base → data_prepare에 넘길 원시 컬럼(qry, qry_image_path, pos_text, pos_image_path, (opt) neg_*)
    """
    img_token = VLM_IMAGE_TOKENS[PHI3V]
    summary = _clean_text(example.get("summary", ""))
    pos = _clean_text(example.get(f"{category}_explanation", ""))

    prompt = prompt_template.format(category=category, summary=summary).strip()
    qry_text = f"{img_token}\n{prompt}"

    return {
        "qry": qry_text,
        "qry_image_path": _build_image_relpath(example["video_id"], image_ext),
        "pos_text": pos,
        "pos_image_path": None
    }

def _build_i2t_retrieval_batch(example, image_ext: str, prompt_template: str):
    img_token = VLM_IMAGE_TOKENS[PHI3V]
    summary = _clean_text(example.get("summary", ""))

    prompt = prompt_template.strip()
    qry_text = f"{img_token}\n{prompt}"

    return {
        "qry": qry_text,
        "qry_image_path": _build_image_relpath(example["video_id"], image_ext),
        "pos_text": summary,
        "pos_image_path": None,
    }

def _finalize_iterable_dataset(dataset, model_args, data_args, training_args, dataset_label: str):
    """
    기존 mmeb 파이프라인과 동일한 후처리:
    - IterableDataset 변환 + 샤딩
    - data_prepare 적용 (이미지 경로 결합/리사이즈 지시)
    - MULTIMODAL_FEATURES로 cast
    """
    # shard
    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)

    # data_prepare가 참조할 컨텍스트 채우기
    ctx = {
        "model_backbone": model_args.model_backbone,
        "image_resolution": data_args.image_resolution,
        "global_dataset_name": dataset_label,
        "image_dir": data_args.image_dir,  # IMAGE_ROOT
    }

    remove_columns = ["qry", "qry_image_path", "pos_image_path"]
    if "neg_image_path" in dataset.column_names:
        remove_columns.append("neg_image_path")

    dataset = dataset.map(
        lambda batch: data_prepare(batch, **ctx),
        batched=True,
        batch_size=2048,
        remove_columns=remove_columns,
        drop_last_batch=True,
    )
    dataset = dataset.cast(MULTIMODAL_FEATURES)
    return dataset


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_dir = kwargs['image_dir']
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']

    batch_size = len(batch_dict['qry'])
    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path in \
        zip(batch_dict['qry'], batch_dict['qry_image_path'],
            batch_dict['pos_text'], batch_dict['pos_image_path'],
            batch_dict.get('neg_text', [''] * batch_size), batch_dict.get('neg_image_path', [None] * batch_size)):
        if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
            print("empty inputs")
            continue
        if model_backbone != PHI3V:
            qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]) if neg_text else ''
        query_texts.append(qry_text)
        pos_texts.append(pos_text)
        neg_texts.append(neg_text)
        # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
        qry_image = {"bytes": [None], "paths": [os.path.join(image_dir, qry_image_path) if qry_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        pos_image = {"bytes": [None], "paths": [os.path.join(image_dir, pos_image_path) if pos_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        neg_image = {"bytes": [None], "paths": [os.path.join(image_dir, neg_image_path) if neg_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        query_images.append(qry_image)
        pos_images.append(pos_image)
        neg_images.append(neg_image)
    if len(query_texts) == 0:
        print('something went wrong')
    # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    return {"query_text": query_texts, "query_image": query_images,
            "pos_text": pos_texts, "pos_image": pos_images,
            "neg_text": neg_texts, "neg_image": neg_images}


DATASET_PARSER_NAME = "aid_global_explain"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_custom_explanation_dataset(model_args, data_args, training_args, *args, **kwargs):
    """
    Args (kwargs 기대):
      - parquet_path: str (필수)  예) "/data/my.parquet"
      - category: str in {"sexual","violence","hate","politics","controversy"} (필수)
      - image_ext: str (선택, 기본 ".jpg")
      - prompt_template: str (선택)
          기본: "Based on the video thumbnail and the description below, explain how {category} elements are visually or contextually represented in this video. Description: {summary}
      - split: str (선택, 기본 "train")
      - subset_name: str (선택, 로깅용 라벨, 기본 "local")
      - num_sample_per_subset: int (선택, data_args에도 동일 필드 있을 수 있음)
    """
    parquet_path = kwargs.get("parquet_path")
    category = kwargs.get("category")
    image_ext = kwargs.get("image_ext", ".jpg")
    prompt_template = kwargs.get(
        "prompt_template",
        "Based on the video thumbnail and the description below, explain how {category} elements are visually or contextually represented in this video. Description: {summary}",
    )
    split = kwargs.get("split", "train")
    subset_name = kwargs.get("subset_name", "local")

    assert parquet_path and isinstance(parquet_path, str), "parquet_path를 지정해야 합니다."

    ds = load_dataset("parquet", data_files={split: parquet_path}, split=split)

    # 필요한 열이 비어있는 샘플 제거 (video_id, pos_text 기반)
    ds = ds.filter(lambda x: _ensure_non_empty(x.get("video_id")) and _ensure_non_empty(x.get(f"{category}_explanation")))

    # 원시열을 표준 열로 변환
    ds = ds.map(
        lambda x: _build_explain_retrieval_batch(x, category=category, image_ext=image_ext, prompt_template=prompt_template),
        remove_columns=[],
    )

    # 샘플 수 제한
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < ds.num_rows:
        ds = ds.select(range(int(num_sample_per_subset)))

    num_rows = ds.num_rows
    label = f"aid_global_explain/{category}/{subset_name}"
    ds = _finalize_iterable_dataset(ds, model_args, data_args, training_args, dataset_label=label)

    # 통계 표시용
    setattr(ds, "num_rows", num_rows)
    print_master(f"Loaded {label} dataset with {num_rows} samples")
    return ds


@AutoPairDataset.register("aid_global_i2t")
def load_custom_i2t_dataset(model_args, data_args, training_args, *args, **kwargs):
    """
    Args (kwargs 기대):
      - parquet_path: str (필수)
      - image_ext: str (선택, 기본 ".jpg")
      - prompt_template: str (선택)
          기본: "Represent the given video thumbnail"
      - split: str (선택, 기본 "train")
      - subset_name: str (선택, 로깅용 라벨, 기본 "local")
      - num_sample_per_subset: int (선택)
    """
    parquet_path = kwargs.get("parquet_path")
    image_ext = kwargs.get("image_ext", ".jpg")
    prompt_template = kwargs.get("prompt_template", "Represent the given video thumbnail")
    split = kwargs.get("split", "train")
    subset_name = kwargs.get("subset_name", "local")

    assert parquet_path and isinstance(parquet_path, str), "parquet_path를 지정해야 합니다."

    ds = load_dataset("parquet", data_files={split: parquet_path}, split=split)

    # summary가 있는 샘플만 사용
    ds = ds.filter(lambda x: _ensure_non_empty(x.get("video_id")) and _ensure_non_empty(x.get("summary")))

    # 원시열 → 표준 열 매핑
    ds = ds.map(
        lambda x: _build_i2t_retrieval_batch(x, image_ext=image_ext, prompt_template=prompt_template),
        remove_columns=[],
    )

    # 샘플 수 제한
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < ds.num_rows:
        ds = ds.select(range(int(num_sample_per_subset)))

    num_rows = ds.num_rows
    label = f"aid_global_i2t/{subset_name}"
    ds = _finalize_iterable_dataset(ds, model_args, data_args, training_args, dataset_label=label)

    setattr(ds, "num_rows", num_rows)
    print_master(f"Loaded {label} dataset with {num_rows} samples")
    return ds