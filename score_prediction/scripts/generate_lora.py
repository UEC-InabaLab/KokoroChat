import torch
import json
import os
from torch import cuda, bfloat16
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# === 1. 训练文件夹列表 ===
train_dirs = [
    "results_seed_12_optim_adamw_torch_fused_lr_0.0002_ws_100",
    "results_seed_72_optim_adamw_torch_fused_lr_0.0002_ws_100",
    "results_seed_22_optim_adamw_torch_fused_lr_0.0002_ws_100",
    "results_seed_82_optim_adamw_torch_fused_lr_0.0002_ws_100",
    "results_seed_2_optim_adamw_torch_fused_lr_0.0002_ws_100",
    "results_seed_32_optim_adamw_torch_fused_lr_0.0002_ws_100",
]

# === 2. 读取测试数据 ===
test_data_path = "./datasets/test_data_P.json"
with open(test_data_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# === 3. 配置量化参数 ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 原始模型 ID
model_id = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

# === 4. 加载 tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_id)

# === 5. 遍历训练文件夹 ===
for train_dir in train_dirs:
    if not os.path.exists(train_dir):
        print(f"目录 {train_dir} 不存在，跳过。")
        continue

    # 获取所有 checkpoint 文件夹，按训练步数排序
    checkpoints = sorted(
        [os.path.join(train_dir, d) for d in os.listdir(train_dir) if d.startswith("-checkpoint-")],
        key=lambda x: int(x.split("-")[-1])  # 按照步数排序
    )

    # 遍历每个 checkpoint 进行推理
    for checkpoint in checkpoints:
        print(f"正在使用 {checkpoint} 进行推理...", flush=True)

        # === 6. 加载原始模型 ===
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # === 7. 加载 LoRA 适配器并合并 ===
        model = PeftModel.from_pretrained(base_model, checkpoint)
        model = model.merge_and_unload()  # 合并 LoRA 适配器，得到完整模型

        # 处理 pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = "[PAD]"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

        model.config.pad_token_id = tokenizer.pad_token_id

        # === 8. 遍历测试数据进行推理 ===
        num_ = 0
        results = []
        for entry in test_data:
            num_ += 1
            print(train_dir, checkpoint, num_, flush=True)
            messages = [
                {"role": "system", "content": "心理カウンセリングの会話について、点数をつけてください。"},
                {"role": "user", "content": entry["input"]},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),  # 确保 attention_mask 传入
                max_new_tokens=512,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,  # 解决 pad_token 警告
                do_sample=False,  # 关闭采样
                temperature=None,  # 移除 temperature，避免不必要的参数冲突
            )
            response = outputs[0][input_ids.shape[-1]:]
            response_text = tokenizer.decode(response, skip_special_tokens=True)

            results.append({
                "index": entry["index"],
                "input": entry["input"],
                "output_model": response_text,
                "output_original": entry["output_all"]  # 追加原始数据的 output
            })

        # === 9. 保存推理结果 ===
        output_results_path = f"./datasets/output/test_results_P_{train_dir}_{os.path.basename(checkpoint)}.json"
        with open(output_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"推理完成，结果已保存到 {output_results_path}\n")
