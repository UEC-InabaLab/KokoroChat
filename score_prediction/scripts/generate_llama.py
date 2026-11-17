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

model_id = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

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
        {"role": "system", "content": "心理カウンセリングの会話について、各項目を0～5点で評価してください。\n必ず以下のフォーマットに従い、それ以外の内容を一切生成しないでください。\n出力は以下のように、各評価項目と点数のみを「, 」区切りで出力してください： \n【出力フォーマット】\n聴いてもらえた、わかってもらえたと感じた=1, 尊重されたと感じた=4, 新しい気づきや体験があった=3, 希望や期待を感じられた=3, 取り組みたかったことを扱えた=3, 一緒に考えながら取り組めた=3, やりとりのリズムがあっていた=2, 居心地のよいやりとりだった=3, 全体として適切でよかった=3, 今回の相談は価値があった=3, 相談開始の円滑さ=3, 相談終了のタイミング（不必要に聴きすぎていないか）、円滑さ=5, 受容・共感（相手の気持ちや考えを深く理解し、共感を持って対応できていたか）=4, 肯定・承認（クライアントの意見や感情を適切に受け止め、肯定的なフィードバックを与えられていたか）=4, 的確な質問による会話の促進（クライアントの気持ちや考えを引き出し、深めるための適切な問いかけができていたか）=4, 要約（クライアントの話を整理し、的確にまとめることで理解を助けることができたか）=4, 問題の明確化（相談の中で問題点が整理され、より明確になったか）=1, この相談での目標の明確化（クライアントの悩みや希望に沿って、相談の目的やゴールが適切に設定されたか）=3, 次の行動につながる提案（相談を踏まえて、具体的な行動や方向性を示すことができたか）=2, 勇気づけ・希望の喚起（クライアントが前向きな気持ちになり、行動を起こすための励ましがあったか）=3"},
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
output_results_path = "./gen_results/results_llama_.json"
with open(output_results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"推论完成，结果已保存到 {output_results_path}")
