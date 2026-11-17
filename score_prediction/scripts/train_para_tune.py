import torch
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import wandb
import json

# === 1. 设备设置 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. 量化配置 ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# === 3. 模型 ID ===
model_id = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

# === 4. 加载 Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=8192)

# === 5. 加载基模型 ===
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# === 6. 处理 pad_token_id ===
tokenizer.pad_token = tokenizer.eos_token  # 避免 OOV
model.config.pad_token_id = tokenizer.pad_token_id

# === 7. 加载 JSON 数据集 ===
data_file = "datasets/finetune_data_P.json"

with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 8. 数据预处理 ===
def formatting_func(example):
    messages = [
        {'role': "system", 'content': example["instruction"]},
        {'role': "user", 'content': example["input"]},
        {'role': "assistant", 'content': example["output_all"]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# === 9. 数据集拆分 (80% 训练集, 20% 验证集) ===
train_data, val_data = train_test_split(data, test_size=0.11, random_state=42)

# 转换 Hugging Face Dataset 格式
train_dataset = Dataset.from_list([{"text": formatting_func(example)} for example in train_data])
val_dataset = Dataset.from_list([{"text": formatting_func(example)} for example in val_data])

# === 10. LoRA 配置 ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens"],
)

# === 11. 定义 wandb 超参数搜索配置 ===
sweep_config = {
    "method": "grid",  # 也可以改成 "random" 进行随机搜索
    "metric": {"name": "eval_loss", "goal": "minimize"},  # 确保有 eval_loss 监控
    "parameters": {
        "optim": {
            "values": ["adamw_8bit"]
        },
        "learning_rate": {
            "values": [1e-4, 5e-5, 2e-5]
        },
        "warmup_steps": {
            "values": [50, 100]
        }
    }
}

# === 12. 创建 wandb Sweep (确保 project 名字正确) ===
sweep_id = wandb.sweep(sweep_config, project="llama3_P_para_tune")

# === 13. 训练函数 ===
def train():
    # 先初始化 wandb
    wandb.init(project="llama3_P_para_tune")

    # 获取 config
    config = wandb.config  

    # 更新 Run 名字
    wandb.run.name = f"optim_{config.optim}_lr_{config.learning_rate}_ws_{config.warmup_steps}"

    # === 14. 训练参数 ===
    training_arguments = TrainingArguments(
        output_dir="./results_para",
        bf16=True,
        per_device_train_batch_size=4,  # 调整 batch_size 适应大模型
        gradient_accumulation_steps=16,  # 确保 batch_size 仍然足够大
        num_train_epochs=4,
        optim=config.optim,  # 超参数优化器
        learning_rate=config.learning_rate,  # 超参数学习率
        warmup_steps=config.warmup_steps,  # 超参数 warmup
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=5,
        evaluation_strategy="epoch",  # 每个 epoch 进行评估
        save_strategy="no",  # 每个 epoch 保存模型
        save_total_limit=0,
        group_by_length=True,
        report_to="wandb"
    )

    # === 15. 初始化 Trainer ===
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 重要！让 wandb 监控 eval_loss
        peft_config=peft_config,
        args=training_arguments,
    )

    # === 16. 开始训练 ===
    trainer.train()

# === 17. 运行 wandb agent ===
if __name__ == "__main__":
    wandb.agent(sweep_id, function=train, project="llama3_P_para_tune")
