import torch
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import Dataset, DatasetDict
from trl import SFTTrainer
import wandb
import json
from sklearn.model_selection import train_test_split

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

# === 3. 处理 pad_token_id ===
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

# === 4. 加载 JSON 数据集 ===
data_file = "datasets/finetune_data_P.json"

with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 5. 数据预处理 ===
def formatting_func(example):
    messages = [
        {'role': "system", 'content': example["instruction"]},
        {'role': "user", 'content': example["input"]},
        {'role': "assistant", 'content': example["output_all"]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

def update_dataset(example):
    return {"text": formatting_func(example)}

# === 6. 数据集划分（80% 训练 + 20% 验证） ===
train_data, eval_data = train_test_split(data, test_size=0.11, random_state=42)

# 转换数据并创建 DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_list([update_dataset(example) for example in train_data]),
    "validation": Dataset.from_list([update_dataset(example) for example in eval_data])
})

# === 7. 训练配置列表 ===
configs = [
    {"optim": "adamw_torch_fused", "lr": 2e-4, "ws": 50},
    {"optim": "adamw_torch_fused", "lr": 2e-4, "ws": 100},
    {"optim": "adamw_8bit", "lr": 1e-4, "ws": 100}
]

# === 8. 训练循环 ===
for config in configs:
    optim = config["optim"]
    lr = config["lr"]
    warmup_steps = config["ws"]
    config_name = f"{optim}_lr_{lr}_ws_{warmup_steps}"
    
    print(f"\n===== 开始训练: {config_name} =====\n")

    # === 9. 初始化 wandb ===
    wandb.init(project="llama3_P", name=config_name, reinit=True)

    # === 10. 加载基模型（每次都重新加载） ===
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # === 11. 训练参数 ===
    training_arguments = TrainingArguments(
        output_dir=f"./results_all/{config_name}",
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        optim=optim,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=10,
        group_by_length=True,
        report_to="wandb",
        evaluation_strategy="epoch",  # **每个 epoch 进行评估**
        save_strategy="epoch",  # **每个 epoch 保存模型**
        save_total_limit=4,  # **最多保留 3 个最佳模型**
        load_best_model_at_end=True,  # **训练结束后加载最优模型**
        metric_for_best_model="loss",  # **按 loss 选择最佳模型**
        greater_is_better=False  # **loss 越小越好**
    )

    # === 12. 初始化 Trainer ===
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_arguments,
    )

    # === 13. 训练 ===
    trainer.train()

    # === 14. 保存完整模型 ===
    full_model_path = f"llama-3-8b-{config_name}"
    trainer.model.save_pretrained(full_model_path)
    tokenizer.save_pretrained(full_model_path)

    print(f"\n===== 训练完成: {config_name}，模型已保存至 {full_model_path} =====\n")

    # 关闭 wandb 运行
    wandb.finish()

print("\n===== 所有训练任务完成！ =====")
