import torch
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig
from trl import SFTTrainer
import wandb
import json
import random
import numpy as np

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
tokenizer.pad_token = tokenizer.eos_token
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

# === 9. LoRA 配置 ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens"],
)

# === 10. wandb 超参数搜索配置 ===
sweep_config = {
    "method": "grid",
    "metric": {"name": "eval_loss", "goal": "minimize"},
    "parameters": {
        "optim": {"values": ["adamw_torch_fused"]},
        "learning_rate": {"values": [2e-4]},
        "warmup_steps": {"values": [100]}
    }
}

# === 11. 训练多种子 ===
seeds = [2, 12, 22, 32]
for seed in seeds:
    print(f"Training with seed {seed}")

    # === 设置全局随机种子 ===
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_data, val_data = train_test_split(data, test_size=0.11, random_state=seed)
    train_dataset = Dataset.from_list([{"text": formatting_func(example)} for example in train_data])
    val_dataset = Dataset.from_list([{"text": formatting_func(example)} for example in val_data])
    
    sweep_id = wandb.sweep(sweep_config, project=f"llama3_P_para_tune_seed_{seed}")
    
    def train():
        # === 确保在训练时随机种子保持一致 ===
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        wandb.init(project=f"llama3_P_para_tune_seed_{seed}")
        config = wandb.config  
        wandb.run.name = f"seed_{seed}_optim_{config.optim}_lr_{config.learning_rate}_ws_{config.warmup_steps}"
        
        training_arguments = TrainingArguments(
            seed=seed,  # 设定训练环境的随机种子
            output_dir=f"./results_{wandb.run.name}",
            bf16=True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            num_train_epochs=4,
            optim=config.optim,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            logging_steps=5,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=4,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            group_by_length=True,
            report_to="wandb"
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            args=training_arguments,
        )
        
        trainer.train()
        full_model_path = f"llama-3-8b-{wandb.run.name}"
        trainer.model.save_pretrained(full_model_path)
        tokenizer.save_pretrained(full_model_path)

    wandb.agent(sweep_id, function=train, project=f"llama3_P_para_tune_seed_{seed}")
