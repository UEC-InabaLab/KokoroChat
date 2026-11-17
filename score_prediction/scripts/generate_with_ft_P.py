import torch
import json
from torch import cuda, bfloat16
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# 配置量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "llama-3-8b-full-P-all"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 处理 pad_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

model.config.pad_token_id = tokenizer.pad_token_id

# 读取测试数据
test_data_path = "./datasets/test_data_P.json"
with open(test_data_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

num_ = 0

# 遍历测试数据进行推论
results = []
for entry in test_data:
    num_ += 1
    print(num_, flush=True)
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

# 保存推论结果
output_results_path = "./datasets/test_results_P_a.json"
with open(output_results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"推论完成，结果已保存到 {output_results_path}")
