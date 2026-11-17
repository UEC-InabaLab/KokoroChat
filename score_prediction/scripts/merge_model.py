import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import os

# === 设置路径 ===
checkpoint_path = "llama-3-8b-optim_adamw_torch_fused_lr_0.0002_ws_100"  # LoRA checkpoint 路径
model_id = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"  # 原始模型
save_path = "merged_kokoro_chat_score_prediction"  # 保存路径

# === 量化设置（可选，视是否用于推理）===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# === 加载 tokenizer 和 base model ===
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# === 加载 LoRA adapter，并合并权重 ===
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model = model.merge_and_unload()  # 得到完整模型

# === 保存合并后的模型和 tokenizer ===
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"合并后的模型和 tokenizer 已保存到：{save_path}")
