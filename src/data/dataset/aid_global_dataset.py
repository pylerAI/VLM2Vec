from datasets import load_dataset
import os
import numpy as np
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.utils import print_master

CATEGORY_DESCRIPTION = {
    "sexual": "Sexual elements are content intended to evoke sexual interest, from mild suggestiveness to explicit or illegal sexual acts.",
    "violence": "Violence elements are content depicting or implying physical or psychological harm—ranging from mild aggression to severe, graphic, or illegal acts such as abuse, warfare, or terror.",
    "hate": "Hate elements are content that targets individuals or groups with demeaning, discriminatory, or threatening expressions based on protected traits or social identity.",
    "politics": "Political elements are content expressing or referencing political actors, parties, elections, or policies, including opinions, advocacy, criticism, or satire about political positions.",
    "controversy": "Controversy elements are content that sensationalizes or heavily focuses on social issues, real incidents, or celebrity scandals in ways that may provoke conflict, confusion, or reputational harm.",
    "story": "Story elements are fictional or dramatized narratives with a clear beginning–development–ending structure that center on interpersonal conflict—such as affairs, divorce, family disputes, or betrayal—to evoke strong emotional reactions.",
    "drinking": "Drinking elements are content that depicts or encourages alcohol consumption—especially excessive, risky, illegal, or youth-related drinking—presented as entertainment or promotion.",
    "smoking": "Smoking elements are content that depicts or encourages the use, promotion, or acquisition of smoking products, including risky, illegal, or youth-targeted smoking behavior.",
}

def _build_image_relpath(video_id: str, image_ext: str = ".jpg") -> str:
    # IMAGE_ROOT/{video_id}.jpg, 여기서는 상대 경로만 생성 (ROOT는 data_args.image_dir)
    return f"{video_id}{image_ext}"

def _clean_text(x):
    return "" if x is None else str(x)

def _ensure_non_empty(s):
    return (s is not None) and (str(s).strip() != "")


def _build_explain_retrieval_batch(example, category: str, image_ext: str, prompt_template: str, add_category_description: bool = False):
    """
    Base → data_prepare에 넘길 원시 컬럼(qry, qry_image_path, pos_text, pos_image_path, (opt) neg_*)
    """
    # Category score for supervised-contrastive loss
    category_scores = []
    row_categories = [k for k in example.keys() if k.endswith("_score")]
    row_categories = sorted([k.replace("_score", "") for k in row_categories])
    for cat in row_categories:
        score = example.get(f"{cat}_score", None)
        if score is not None:
            score = int(score)
            score = max(0, min(5, score))  # Clamp to [0, 5]
        else:
            score = 0  # Default to 0 if not present
        category_scores.append(score)

    img_token = VLM_IMAGE_TOKENS[PHI3V]
    if "en_summary" in example:
        summary = _clean_text(example.get("en_summary", ""))
    else:
        summary = _clean_text(example.get("summary", ""))

    # category="random" 또는 "auto_select"일 때는 모든 explanation을 보존하여 data_prepare에서 매번 선택
    if row_categories and (category == "random" or category == "auto_select"):
        all_explanations = {}
        for cat in row_categories:
            explanation = _clean_text(example.get(f"{cat}_explanation", ""))
            if explanation:
                all_explanations[cat] = explanation

        return {
            "qry": None,  # data_prepare에서 동적으로 생성
            "qry_image_path": _build_image_relpath(example["video_id"], image_ext),
            "pos_text": None,  # data_prepare에서 동적으로 선택
            "pos_image_path": None,
            "category_score": category_scores,  # score 기반 확률 분포 계산에 필요
            "all_explanations": all_explanations,  # 모든 explanation 보존
            "row_categories": row_categories,  # 사용 가능한 category 목록
            "summary": summary,  # prompt 생성에 필요
            "prompt_template": prompt_template,  # prompt 생성에 필요
            "add_category_description": add_category_description,  # category description 추가 여부
            "use_random_category": (category == "random"),  # data_prepare에서 랜덤 선택 플래그
            "use_auto_select_category": (category == "auto_select"),  # data_prepare에서 score 기반 선택 플래그
        }
    else:
        # 고정된 category 사용
        pos = _clean_text(example.get(f"{category}_explanation", ""))
        prompt = prompt_template.format(category=category, summary=summary).strip()
        qry_text = f"{img_token}\n{prompt}"

    return {
        "qry": qry_text,
        "qry_image_path": _build_image_relpath(example["video_id"], image_ext),
        "pos_text": pos,
        "pos_image_path": None,
        "category_score": category_scores,
    }

def _normalize_explain_entry(entry: dict):
    return {
        "qry": entry.get("qry"),
        "qry_image_path": entry.get("qry_image_path"),
        "pos_text": entry.get("pos_text"),
        "pos_image_path": entry.get("pos_image_path"),
        "category_score": entry.get("category_score"),
        "all_explanations": entry.get("all_explanations"),
        "row_categories": entry.get("row_categories"),
        "summary": entry.get("summary"),
        "prompt_template": entry.get("prompt_template"),
        "add_category_description": entry.get("add_category_description"),
        "use_random_category": entry.get("use_random_category"),
        "use_auto_select_category": entry.get("use_auto_select_category"),
    }

def _build_explain_retrieval_batch_with_hard_negative(
    example,
    video_id_to_row: dict,
    category: str,
    image_ext: str,
    prompt_template: str,
    add_category_description: bool = False,
):
    base = _build_explain_retrieval_batch(
        example,
        category=category,
        image_ext=image_ext,
        prompt_template=prompt_template,
        add_category_description=add_category_description,
    )

    hard_negative_entry = {}
    hard_negative_ids = example.get("hard_negative")
    hard_negative_id = None
    if isinstance(hard_negative_ids, list) and len(hard_negative_ids) > 0:
        hard_negative_id = hard_negative_ids[0]
    elif isinstance(hard_negative_ids, str) and hard_negative_ids:
        hard_negative_id = hard_negative_ids

    if hard_negative_id:
        hard_negative_row = video_id_to_row.get(hard_negative_id)
        if hard_negative_row:
            hard_negative_entry = _build_explain_retrieval_batch(
                hard_negative_row,
                category=category,
                image_ext=image_ext,
                prompt_template=prompt_template,
                add_category_description=add_category_description,
            )

    hard_negative_entry = _normalize_explain_entry(hard_negative_entry)
    base.update({
        "hard_negative_qry": hard_negative_entry["qry"],
        "hard_negative_qry_image_path": hard_negative_entry["qry_image_path"],
        "hard_negative_pos_text": hard_negative_entry["pos_text"],
        "hard_negative_pos_image_path": hard_negative_entry["pos_image_path"],
        "hard_negative_category_score": hard_negative_entry["category_score"],
        "hard_negative_all_explanations": hard_negative_entry["all_explanations"],
        "hard_negative_row_categories": hard_negative_entry["row_categories"],
        "hard_negative_summary": hard_negative_entry["summary"],
        "hard_negative_prompt_template": hard_negative_entry["prompt_template"],
        "hard_negative_add_category_description": hard_negative_entry["add_category_description"],
        "hard_negative_use_random_category": hard_negative_entry["use_random_category"],
        "hard_negative_use_auto_select_category": hard_negative_entry["use_auto_select_category"],
    })
    return base

def _build_i2t_retrieval_batch(example, image_ext: str, prompt_template: str):
    img_token = VLM_IMAGE_TOKENS[PHI3V]
    summary = _clean_text(example.get("summary", ""))

    prompt = prompt_template.strip()
    qry_text = f"{img_token}\n{prompt}"

    # Category score for supervised-contrastive loss
    category_scores = []
    row_categories = [k for k in example.keys() if k.endswith("_score")]
    row_categories = sorted([k.replace("_score", "") for k in row_categories])
    for cat in row_categories:
        score = example.get(f"{cat}_score", None)
        if score is not None:
            score = int(score)
            score = max(0, min(5, score))  # Clamp to [0, 5]
        else:
            score = 0  # Default to 0 if not present
        category_scores.append(score)

    return {
        "qry": qry_text,
        "qry_image_path": _build_image_relpath(example["video_id"], image_ext),
        "pos_text": summary,
        "pos_image_path": None,
        "category_score": category_scores,
    }

def _finalize_iterable_dataset(
        dataset,
        model_args,
        data_args,
        training_args,
        dataset_label: str,
        image_dir: str,
    ):
    """
    기존 mmeb 파이프라인과 동일한 후처리:
    - IterableDataset 변환 + 샤딩
    - data_prepare 적용 (이미지 경로 결합/리사이즈 지시)
    - MULTIMODAL_FEATURES로 cast
    """
    # NOTE: https://github.com/TIGER-AI-Lab/VLM2Vec/issues/142
    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)

    # data_prepare가 참조할 컨텍스트 채우기
    ctx = {
        "model_backbone": model_args.model_backbone,
        "image_resolution": data_args.image_resolution,
        "global_dataset_name": dataset_label,
        "image_dir": image_dir,  # IMAGE_ROOT
    }

    remove_columns = ["qry", "qry_image_path", "pos_image_path"]
    if "neg_image_path" in dataset.column_names:
        remove_columns.append("neg_image_path")
    # category="random" 또는 "auto_select"인 경우 추가된 필드들도 제거
    if "all_explanations" in dataset.column_names:
        remove_columns.extend(["all_explanations", "row_categories", "summary", "prompt_template", "add_category_description", "use_random_category", "use_auto_select_category"])
    if "hard_negative_qry" in dataset.column_names:
        remove_columns.extend([
            "hard_negative_qry",
            "hard_negative_qry_image_path",
            "hard_negative_pos_text",
            "hard_negative_pos_image_path",
            "hard_negative_category_score",
            "hard_negative_all_explanations",
            "hard_negative_row_categories",
            "hard_negative_summary",
            "hard_negative_prompt_template",
            "hard_negative_add_category_description",
            "hard_negative_use_random_category",
            "hard_negative_use_auto_select_category",
        ])

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

    # batch_size 계산: qry_image_path를 기준으로 (항상 존재)
    batch_size = len(batch_dict.get('qry_image_path', []))
    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    category_scores = []

    def _get_list(key, default, prefix=""):
        return batch_dict.get(f"{prefix}{key}", [default] * batch_size)

    base_fields = {
        "qry": _get_list("qry", None),
        "qry_image_path": _get_list("qry_image_path", None),
        "pos_text": _get_list("pos_text", None),
        "pos_image_path": _get_list("pos_image_path", None),
        "neg_text": _get_list("neg_text", ""),
        "neg_image_path": _get_list("neg_image_path", None),
        "category_score": _get_list("category_score", None),
        "all_explanations": _get_list("all_explanations", None),
        "row_categories": _get_list("row_categories", None),
        "summary": _get_list("summary", None),
        "prompt_template": _get_list("prompt_template", None),
        "add_category_description": _get_list("add_category_description", False),
        "use_random_category": _get_list("use_random_category", False),
        "use_auto_select_category": _get_list("use_auto_select_category", False),
    }

    hard_negative_fields = {
        "qry": _get_list("qry", None, prefix="hard_negative_"),
        "qry_image_path": _get_list("qry_image_path", None, prefix="hard_negative_"),
        "pos_text": _get_list("pos_text", None, prefix="hard_negative_"),
        "pos_image_path": _get_list("pos_image_path", None, prefix="hard_negative_"),
        "category_score": _get_list("category_score", None, prefix="hard_negative_"),
        "all_explanations": _get_list("all_explanations", None, prefix="hard_negative_"),
        "row_categories": _get_list("row_categories", None, prefix="hard_negative_"),
        "summary": _get_list("summary", None, prefix="hard_negative_"),
        "prompt_template": _get_list("prompt_template", None, prefix="hard_negative_"),
        "add_category_description": _get_list("add_category_description", False, prefix="hard_negative_"),
        "use_random_category": _get_list("use_random_category", False, prefix="hard_negative_"),
        "use_auto_select_category": _get_list("use_auto_select_category", False, prefix="hard_negative_"),
    }

    img_token = VLM_IMAGE_TOKENS[PHI3V]

    def _process_entry(
        qry_text,
        qry_image_path,
        pos_text,
        pos_image_path,
        neg_text,
        neg_image_path,
        category_score,
        all_explanations,
        row_categories,
        summary,
        prompt_template,
        add_category_description,
        use_random_category,
        use_auto_select_category,
    ):
        # category="random"인 경우: 매번 랜덤하게 category 선택
        if use_random_category and all_explanations and row_categories:
            # 사용 가능한 category 중에서 랜덤 선택
            available_categories = [cat for cat in row_categories if cat in all_explanations and all_explanations[cat]]
            if available_categories:
                selected_category = np.random.choice(available_categories)
                pos_text = all_explanations[selected_category]
                # prompt 동적 생성
                if add_category_description and selected_category in CATEGORY_DESCRIPTION:
                    prompt = (
                        f"{CATEGORY_DESCRIPTION[selected_category]}\n"
                        + prompt_template.format(category=selected_category, summary=summary)
                    ).strip()
                else:
                    prompt = prompt_template.format(category=selected_category, summary=summary).strip()
                qry_text = f"{img_token}\n{prompt}"
            else:
                # 사용 가능한 explanation이 없으면 skip
                return None
        # category="auto_select"인 경우: score 기반 확률 분포로 매번 category 선택
        elif use_auto_select_category and all_explanations and row_categories and category_score is not None:
            # 사용 가능한 category와 해당 score 추출
            available_categories = []
            available_scores = []
            for i, cat in enumerate(row_categories):
                if cat in all_explanations and all_explanations[cat]:
                    available_categories.append(cat)
                    available_scores.append(category_score[i] if i < len(category_score) else 0)

            if available_categories:
                # score 기반 확률 분포 계산: (score + 1) / (sum(score) + len(categories))
                scores_array = np.array(available_scores)
                probabilities = (scores_array + 1) / (np.sum(scores_array) + len(available_categories))
                selected_category = np.random.choice(available_categories, p=probabilities)
                pos_text = all_explanations[selected_category]
                # prompt 동적 생성
                if add_category_description and selected_category in CATEGORY_DESCRIPTION:
                    prompt = (
                        f"{CATEGORY_DESCRIPTION[selected_category]}\n"
                        + prompt_template.format(category=selected_category, summary=summary)
                    ).strip()
                else:
                    prompt = prompt_template.format(category=selected_category, summary=summary).strip()
                qry_text = f"{img_token}\n{prompt}"
            else:
                # 사용 가능한 explanation이 없으면 skip
                return None
        else:
            # 기존 로직: 고정된 category 사용
            if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
                print("empty inputs")
                return None

        if model_backbone != PHI3V:
            qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]) if neg_text else ''

        # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
        qry_image = {"bytes": [None], "paths": [os.path.join(image_dir, qry_image_path) if qry_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        pos_image = {"bytes": [None], "paths": [os.path.join(image_dir, pos_image_path) if pos_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        neg_image = {"bytes": [None], "paths": [os.path.join(image_dir, neg_image_path) if neg_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        return qry_text, pos_text, neg_text, qry_image, pos_image, neg_image, category_score

    def _append_entry(entry):
        qry_text, pos_text, neg_text, qry_image, pos_image, neg_image, category_score = entry
        query_texts.append(qry_text)
        pos_texts.append(pos_text)
        neg_texts.append(neg_text)
        if category_score is not None:
            category_scores.append(category_score) # Multi-category score
        query_images.append(qry_image)
        pos_images.append(pos_image)
        neg_images.append(neg_image)

    for idx in range(batch_size):
        entry = _process_entry(
            base_fields["qry"][idx],
            base_fields["qry_image_path"][idx],
            base_fields["pos_text"][idx],
            base_fields["pos_image_path"][idx],
            base_fields["neg_text"][idx],
            base_fields["neg_image_path"][idx],
            base_fields["category_score"][idx],
            base_fields["all_explanations"][idx],
            base_fields["row_categories"][idx],
            base_fields["summary"][idx],
            base_fields["prompt_template"][idx],
            base_fields["add_category_description"][idx],
            base_fields["use_random_category"][idx],
            base_fields["use_auto_select_category"][idx],
        )
        if entry is None:
            continue
        _append_entry(entry)

        has_hard_negative = (
            hard_negative_fields["qry_image_path"][idx]
            or hard_negative_fields["qry"][idx]
            or hard_negative_fields["pos_text"][idx]
            or hard_negative_fields["pos_image_path"][idx]
            or (hard_negative_fields["all_explanations"][idx] and hard_negative_fields["row_categories"][idx])
        )
        if has_hard_negative:
            hard_negative_entry = _process_entry(
                hard_negative_fields["qry"][idx],
                hard_negative_fields["qry_image_path"][idx],
                hard_negative_fields["pos_text"][idx],
                hard_negative_fields["pos_image_path"][idx],
                "",
                None,
                hard_negative_fields["category_score"][idx],
                hard_negative_fields["all_explanations"][idx],
                hard_negative_fields["row_categories"][idx],
                hard_negative_fields["summary"][idx],
                hard_negative_fields["prompt_template"][idx],
                hard_negative_fields["add_category_description"][idx],
                hard_negative_fields["use_random_category"][idx],
                hard_negative_fields["use_auto_select_category"][idx],
            )
            if hard_negative_entry is None:
                continue
            _append_entry(hard_negative_entry)
    if len(query_texts) == 0:
        print('something went wrong')

    result = {
        "query_text": query_texts,
        "query_image": query_images,
        "pos_text": pos_texts,
        "pos_image": pos_images,
        "neg_text": neg_texts,
        "neg_image": neg_images,
    }
    if len(category_scores) > 0:
        result['category_score'] = category_scores

    return result


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
    category = kwargs.get("category", "")
    image_ext = kwargs.get("image_ext", ".jpg")
    image_dir = kwargs.get("image_dir")
    add_category_description = kwargs.get("add_category_description", False)
    use_general_template = kwargs.get("use_general_template", False)
    use_llm_summary = kwargs.get("use_llm_summary", False)

    # category="random" 또는 "auto_select"인 경우, add_category_description은 data_prepare에서 처리
    if category == "random" or category == "auto_select":
        if use_general_template:
            prompt_template = "Represent the given video thumbnail with the following metadata: {summary}"
        else:
            prompt_template = kwargs.get(
                "prompt_template",
                "Based on the video thumbnail and the description below, explain how {category} elements are visually or contextually represented in this video. Description: {summary}",
            )
    else:
        # 고정된 category인 경우에만 여기서 category description 추가
        if use_general_template:
            prompt_template = "Represent the given video thumbnail with the following metadata: {summary}"
        elif add_category_description:
            prompt_template = (
                f"{CATEGORY_DESCRIPTION[category]}\n"
                "Based on the video thumbnail and the description below, explain how {category} elements are visually or contextually represented in this video. "
                "Video Description: {summary}"
            )
        else:
            prompt_template = kwargs.get(
                "prompt_template",
                "Based on the video thumbnail and the description below, explain how {category} elements are visually or contextually represented in this video. Description: {summary}",
            )
    split = kwargs.get("split", "train")
    subset_name = kwargs.get("subset_name", "local")

    assert parquet_path and isinstance(parquet_path, str), "parquet_path를 지정해야 합니다."

    ds = load_dataset("parquet", data_files={split: parquet_path}, split=split)

    if use_llm_summary:
        # en_summary 대신 summary (llm-generated)를 input으로 사용하도록 변경
        if "en_summary" in ds.column_names:
            ds = ds.remove_columns("en_summary")

    # 필요한 열이 비어있는 샘플 제거 (video_id, pos_text 기반)
    if category == "random" or category == "auto_select":
        # random 또는 auto_select인 경우: 최소 하나의 category explanation이 있어야 함
        def has_any_explanation(x):
            if not _ensure_non_empty(x.get("video_id")):
                return False
            row_categories = [k for k in x.keys() if k.endswith("_score")]
            row_categories = [k.replace("_score", "") for k in row_categories]
            for cat in row_categories:
                if _ensure_non_empty(x.get(f"{cat}_explanation")):
                    return True
            return False
        ds = ds.filter(has_any_explanation)
    else:
        ds = ds.filter(lambda x: _ensure_non_empty(x.get("video_id")) and _ensure_non_empty(x.get(f"{category}_explanation")))

    video_id_to_row = {row["video_id"]: row for row in ds}

    # 원시열을 표준 열로 변환
    ds = ds.map(
        lambda x: _build_explain_retrieval_batch_with_hard_negative(
            x,
            video_id_to_row=video_id_to_row,
            category=category,
            image_ext=image_ext,
            prompt_template=prompt_template,
            add_category_description=add_category_description,
        ),
        remove_columns=[],
    )

    # 샘플 수 제한
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < ds.num_rows:
        ds = ds.select(range(int(num_sample_per_subset)))

    num_rows = ds.num_rows
    label = f"aid_global_explain/{category}/{subset_name}"
    ds = _finalize_iterable_dataset(ds, model_args, data_args, training_args, dataset_label=label, image_dir=image_dir)

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
    image_dir = kwargs.get("image_dir")
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
    ds = _finalize_iterable_dataset(ds, model_args, data_args, training_args, dataset_label=label, image_dir=image_dir)

    setattr(ds, "num_rows", num_rows)
    print_master(f"Loaded {label} dataset with {num_rows} samples")
    return ds