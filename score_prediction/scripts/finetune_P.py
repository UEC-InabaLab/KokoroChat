import torch
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import wandb
import json

# === 1. 配置量化 ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 模型 ID
model_id = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

# === 2. 加载 Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=8192)

# === 3. 加载基模型 ===
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# === 4. 处理 pad_token_id ===
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

model.config.pad_token_id = tokenizer.pad_token_id

# === 5. 加载 JSON 数据集 ===
data_file = "datasets/finetune_data_P.json"

with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 6. 数据预处理 ===
def formatting_func(example):
    messages = [
        {'role': "system", 'content': example["instruction"]},
        {'role': "user", 'content': example["input"]},
        {'role': "assistant", 'content': example["output_1"]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

def update_dataset(example):
    return {"text": formatting_func(example)}

# 转换数据并创建 Dataset
dataset = Dataset.from_list([update_dataset(example) for example in data])

# === 7. 配置 LoRA ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens"],
)

# === 8. 初始化 wandb ===
wandb.init(project="llama3_sftqlora_P")

# === 9. 训练参数 ===
training_arguments = TrainingArguments(
    output_dir="./results",
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=5,
    optim="adamw_8bit",
    learning_rate=2e-4, 
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=5,
    group_by_length=True,
    report_to="wandb"
)

# === 10. 初始化 Trainer ===
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_arguments,
)

# === 11. 开始训练 ===
trainer.train()

# === 12. 保存 LoRA 适配器 ===
adapter_model_path = "llama-3-8b-lora-full-P"
trainer.model.save_pretrained(adapter_model_path)
tokenizer.save_pretrained(adapter_model_path)

print(f"LoRA 适配器已保存到 {adapter_model_path}")

# === 13. 合并 LoRA 适配器并保存完整模型 ===
print("正在合并 LoRA 适配器到原模型...")
full_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
full_model = PeftModel.from_pretrained(full_model, adapter_model_path)
full_model = full_model.merge_and_unload()  # **合并 LoRA**

# === 14. 保存完整模型 ===
full_model_path = "llama-3-8b-full-P"
full_model.save_pretrained(full_model_path)
tokenizer.save_pretrained(full_model_path)

print(f"完整模型已保存到 {full_model_path}")
