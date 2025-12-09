# merge_qwen2vl_vlm2vec_lora.py

import torch
from peft import PeftModel
from transformers import AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from src.arguments import ModelArguments
from src.model.model import MMEBModel

# 1) 기본 설정
BASE_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
# HF에서 그대로 쓰는 경우: "TIGER-Lab/VLM2Vec-Qwen2VL-7B"
# 로컬에 이미 clone 한 경우: 그 로컬 경로
LORA_REPO_OR_PATH = "/gpfs/private/iji/experiments/pyler_embeds/VLM2Vec/aid_global_qwen2vl-2B_1103_categorical/checkpoint-1000"
# LORA_SUBFOLDER = "lora"   # HF repo 구조상 LoRA가 lora/ 아래에 있음 :contentReference[oaicite:0]{index=0}

# 최종 merge 된 모델이 저장될 위치
OUTPUT_DIR = "/gpfs/public/artifacts/models/PylerAI/aid-global/experimental/VLM2Vec/1103_categorical_merged"

# 2) base Qwen2-VL 모델 로드
print(">> Loading base model:", BASE_MODEL_NAME)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",   # GPU 여러 개면 알아서 분산
)

# 3) LoRA 어댑터 로드
print(">> Loading LoRA adapter from:", LORA_REPO_OR_PATH)
lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_REPO_OR_PATH,
    # subfolder=LORA_SUBFOLDER,   # HF 구조 기준 (로컬이면 경로에 맞게 수정)
    torch_dtype=torch.bfloat16,
)

# 4) LoRA를 base에 merge
print(">> Merging LoRA into base model...")
merged_model = lora_model.merge_and_unload()  # 반환 타입은 Qwen2VLForConditionalGeneration

# 5) 저장 전에 CPU로 옮겨두는 게 안전
merged_model = merged_model.to("cpu")

print(">> Saving merged model to:", OUTPUT_DIR)
merged_model.save_pretrained(OUTPUT_DIR)

# 6) processor (Qwen2VLProcessor + tokenizer) 같이 저장
#    - 여기서는 base 모델에서 가져와서 저장
print(">> Saving processor (image processor + tokenizer)...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
processor.save_pretrained(OUTPUT_DIR)

print("✅ Done! Merged model + processor saved to", OUTPUT_DIR)

print("Checking the merged model...")

model_args = ModelArguments(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    checkpoint_path=OUTPUT_DIR,
    pooling="last",
    normalize=True,
    model_backbone="qwen2_vl",
    lora=False,
)

model = MMEBModel.load(model_args)
model = model.to("cuda", dtype=torch.bfloat16)
model.eval()

print("✅ Done! Merged model loaded and checked.")
