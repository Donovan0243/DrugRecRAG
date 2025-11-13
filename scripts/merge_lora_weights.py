"""合并 LoRA 权重到基础模型（用于部署）。

中文说明：
- 将 QLoRA 训练的 LoRA 权重合并到基础模型中
- 合并后的模型可以直接用于 Ollama 或 vLLM 部署
- 合并后模型大小会增加（恢复为完整模型大小）

使用方法：
    python scripts/merge_lora_weights.py \
        --base_model Qwen/Qwen2-7B-Instruct \
        --lora_model ./models/gap-sft-7b \
        --output_dir ./models/gap-sft-7b-merged
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到基础模型")
    
    parser.add_argument("--base_model", type=str, required=True,
                       help="基础模型路径（HuggingFace 模型 ID 或本地路径）")
    parser.add_argument("--lora_model", type=str, required=True,
                       help="LoRA 模型路径（训练输出目录）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录（合并后的模型）")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"],
                       help="模型数据类型")
    
    args = parser.parse_args()
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("[warn] 未检测到 GPU，合并可能会很慢。")
        device_map = "cpu"
    else:
        print(f"[info] 检测到 GPU：{torch.cuda.get_device_name(0)}")
        device_map = "auto"
    
    # 数据类型
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # 加载基础模型
    print(f"[info] 加载基础模型：{args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # 加载 LoRA 权重
    print(f"[info] 加载 LoRA 权重：{args.lora_model}")
    model = PeftModel.from_pretrained(model, args.lora_model)
    
    # 合并权重
    print("[info] 合并 LoRA 权重...")
    model = model.merge_and_unload()
    
    # 加载 tokenizer
    print("[info] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # 保存合并后的模型
    print(f"[info] 保存合并后的模型到：{args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存配置信息
    import json
    config_info = {
        "base_model": args.base_model,
        "lora_model": args.lora_model,
        "dtype": args.dtype,
    }
    with open(f"{args.output_dir}/merge_config.json", 'w', encoding='utf-8') as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)
    
    print("[success] 合并完成！")


if __name__ == "__main__":
    main()

