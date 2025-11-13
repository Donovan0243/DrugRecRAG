"""SFT 微调训练脚本（使用 QLoRA）。

中文说明：
- 支持 Qwen2、ChatGLM3、Baichuan2 等模型
- 使用 QLoRA 进行高效微调（节省显存）
- 适用于 VM GPU 环境
- 输出格式：选项 B（JSON 输出，包含推理过程）

使用方法：
    # 在 VM 上运行
    python scripts/train_sft.py \
        --model_name Qwen/Qwen2-7B-Instruct \
        --train_file sft_data/train.jsonl \
        --dev_file sft_data/dev.jsonl \
        --output_dir ./models/gap-sft-7b \
        --lora_r 16 \
        --lora_alpha 32 \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --num_epochs 3
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    import torch
except ImportError as e:
    print(f"[error] 缺少依赖：{e}")
    print("[info] 请安装：pip install transformers peft datasets accelerate bitsandbytes torch")
    exit(1)


def format_prompt(sample: Dict, tokenizer, template_type: str = "qwen2") -> str:
    """格式化 prompt（根据模型类型选择模板）。
    
    中文：将 messages 格式转换为模型需要的 prompt 格式。
    """
    messages = sample.get("messages", [])
    
    if template_type == "qwen2":
        # Qwen2 格式
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    elif template_type == "chatml":
        # ChatML 格式（Qwen、ChatGLM）
        formatted = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return formatted
    else:
        # 简单拼接
        formatted = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        return formatted


def preprocess_function(
    examples: Dict,
    tokenizer,
    max_length: int = 2048,
    template_type: str = "qwen2"
) -> Dict:
    """预处理函数：将文本转换为 token ids。
    
    中文：处理训练数据，包括 tokenization 和 padding。
    """
    # 格式化 prompt
    texts = [
        format_prompt({"messages": msgs}, tokenizer, template_type)
        for msgs in examples["messages"]
    ]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # 设置 labels（用于计算 loss）
    labels = []
    for input_ids in tokenized["input_ids"]:
        labels.append(input_ids.copy())
    
    tokenized["labels"] = labels
    
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="SFT 微调训练脚本（QLoRA）")
    
    # 数据参数
    parser.add_argument("--train_file", type=str, required=True, help="训练集文件路径（JSONL）")
    parser.add_argument("--dev_file", type=str, required=True, help="验证集文件路径（JSONL）")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B-Instruct", 
                       help="基础模型名称（HuggingFace 模型 ID）")
    parser.add_argument("--template_type", type=str, default="qwen2",
                       choices=["qwen2", "chatml", "llama3"],
                       help="模板类型")
    
    # QLoRA 参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, nargs="+", 
                       default=["q_proj", "k_proj", "v_proj", "o_proj"],
                       help="LoRA 目标模块")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存检查点步数")
    parser.add_argument("--eval_steps", type=int, default=500, help="评估步数")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志步数")
    
    # 其他参数
    parser.add_argument("--bf16", action="store_true", help="使用 bf16 精度（A100/H100）")
    parser.add_argument("--fp16", action="store_true", help="使用 fp16 精度（V100/RTX）")
    parser.add_argument("--use_4bit", action="store_true", help="使用 4-bit 量化（QLoRA）")
    parser.add_argument("--use_8bit", action="store_true", help="使用 8-bit 量化（QLoRA）")
    
    args = parser.parse_args()
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("[warn] 未检测到 GPU，训练将非常慢。建议在 VM 上使用 GPU 运行。")
        device_map = "cpu"
    else:
        print(f"[info] 检测到 GPU：{torch.cuda.get_device_name(0)}")
        print(f"[info] GPU 显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device_map = "auto"
    
    # 加载 tokenizer
    print(f"[info] 加载 tokenizer：{args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print(f"[info] 加载模型：{args.model_name}")
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    
    # 量化设置
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("[info] 使用 4-bit 量化（QLoRA）")
    elif args.use_8bit:
        model_kwargs["load_in_8bit"] = True
        print("[info] 使用 8-bit 量化")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    
    # 准备 QLoRA
    if args.use_4bit or args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    print(f"[info] 加载数据集：{args.train_file}")
    dataset = load_dataset("json", data_files={
        "train": args.train_file,
        "dev": args.dev_file
    })
    
    # 预处理数据集
    print("[info] 预处理数据集...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.max_length, args.template_type),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=-1,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        dataloader_pin_memory=False,
        report_to="tensorboard",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        data_collator=data_collator,
    )
    
    # 开始训练
    print("[info] 开始训练...")
    trainer.train()
    
    # 保存模型
    print(f"[info] 保存模型到：{args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存训练配置
    config_file = Path(args.output_dir) / "training_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    print("[success] 训练完成！")


if __name__ == "__main__":
    main()

