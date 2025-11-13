"""数据转换脚本：将 JSONL 格式的训练数据转换为 SFT 微调格式。

中文说明：
- 输入：dialmed/train.txt, dialmed/dev.txt (JSONL 格式)
- 输出：sft_data/train.jsonl, sft_data/dev.jsonl (对话格式，选项 B：JSON 输出)
- 格式：采用 ChatML 或 Qwen2 格式（根据基础模型调整）
"""

import json
import os
from typing import Dict, List
from pathlib import Path


def extract_constraints(dialog: List[str]) -> Dict[str, List[str]]:
    """从对话中提取约束信息（过敏、怀孕、肝病史等）。
    
    中文：用于增强 prompt，让模型更好地理解约束条件。
    """
    constraints = {
        "allergies": [],
        "status": [],  # pregnant, infant, elderly, etc.
        "past_history": [],
        "taking_drugs": []
    }
    
    dialog_text = " ".join(dialog).lower()
    
    # 提取过敏信息
    if "过敏" in dialog_text or "过敏史" in dialog_text:
        # 简单提取，可以后续用 LLM 增强
        if "青霉素" in dialog_text:
            constraints["allergies"].append("青霉素")
        if "阿莫西林" in dialog_text:
            constraints["allergies"].append("阿莫西林")
    
    # 提取特殊人群
    if "怀孕" in dialog_text or "孕妇" in dialog_text or "哺乳期" in dialog_text:
        if "哺乳期" in dialog_text:
            constraints["status"].append("breastfeeding")
        else:
            constraints["status"].append("pregnant")
    
    if "个月" in dialog_text and ("婴儿" in dialog_text or "宝宝" in dialog_text):
        constraints["status"].append("infant")
    
    # 提取既往病史
    if "乙肝" in dialog_text or "肝炎" in dialog_text:
        constraints["past_history"].append("hepatitis")
    if "肝功" in dialog_text or "肝功能" in dialog_text:
        constraints["past_history"].append("liver_dysfunction")
    if "胃" in dialog_text and ("不好" in dialog_text or "反酸" in dialog_text):
        constraints["past_history"].append("gastric_problem")
    
    return constraints


def build_prompt(dialog: List[str], disease: List[str], constraints: Dict) -> str:
    """构建训练 prompt。
    
    中文：根据选项 B（JSON 输出），构建包含推理要求的 prompt。
    """
    # 将对话拼接成文本
    dialog_text = "\n".join(dialog)
    
    # 提取诊断信息
    disease_text = "、".join(disease) if disease else "未明确诊断"
    
    # 构建约束信息描述
    constraint_parts = []
    if constraints["allergies"]:
        constraint_parts.append(f"药物过敏：{', '.join(constraints['allergies'])}")
    if constraints["status"]:
        status_map = {
            "pregnant": "孕妇",
            "breastfeeding": "哺乳期",
            "infant": "婴幼儿",
            "elderly": "老年人"
        }
        status_text = [status_map.get(s, s) for s in constraints["status"]]
        constraint_parts.append(f"特殊人群：{', '.join(status_text)}")
    if constraints["past_history"]:
        history_map = {
            "hepatitis": "肝炎病史",
            "liver_dysfunction": "肝功能异常",
            "gastric_problem": "胃部问题"
        }
        history_text = [history_map.get(h, h) for h in constraints["past_history"]]
        constraint_parts.append(f"既往病史：{', '.join(history_text)}")
    
    constraint_text = "\n".join(constraint_parts) if constraint_parts else "无特殊约束"
    
    # 构建完整的 prompt
    prompt = f"""你是一个专业的医疗助手。请根据以下对话，分析患者情况并推荐最合适的药物。

对话历史：
{dialog_text}

诊断：{disease_text}

患者约束条件：
{constraint_text}

请以 JSON 格式返回您的推荐和理由。格式如下：
{{
  "drugs": ["药物1", "药物2"],
  "reasoning": "推荐理由..."
}}

要求：
1. 如果对话中包含 [MASK] 标记，请用推荐的药物填充。
2. 如果推荐多个药物，请按重要性排序。
3. reasoning 字段应该详细说明推荐理由，包括诊断依据、药物选择原因、以及如何处理约束条件。
"""
    
    return prompt


def build_completion(labels: List[str], dialog: List[str], disease: List[str], constraints: Dict) -> str:
    """构建训练 completion（模型应该输出的内容）。
    
    中文：生成选项 B 的 JSON 格式输出，包含推理过程。
    """
    # 生成推理理由（基于对话内容）
    dialog_text = " ".join(dialog)
    disease_text = "、".join(disease) if disease else "未明确诊断"
    
    # 构建推理理由
    reasoning_parts = []
    
    # 诊断信息
    if disease:
        reasoning_parts.append(f"患者诊断为{disease_text}，")
    
    # 约束处理
    if constraints["allergies"]:
        reasoning_parts.append(f"但患者对{', '.join(constraints['allergies'])}过敏，因此避免使用相关药物。")
    
    if constraints["status"]:
        status_map = {
            "pregnant": "孕妇",
            "breastfeeding": "哺乳期",
            "infant": "婴幼儿",
            "elderly": "老年人"
        }
        status_text = [status_map.get(s, s) for s in constraints["status"]]
        reasoning_parts.append(f"患者为{', '.join(status_text)}，需选择安全的药物。")
    
    if constraints["past_history"]:
        history_map = {
            "hepatitis": "肝炎病史",
            "liver_dysfunction": "肝功能异常",
            "gastric_problem": "胃部问题"
        }
        history_text = [history_map.get(h, h) for h in constraints["past_history"]]
        reasoning_parts.append(f"患者有{', '.join(history_text)}，需要考虑药物安全性。")
    
    # 药物推荐
    if len(labels) == 1:
        reasoning_parts.append(f"推荐使用{labels[0]}进行治疗。")
    else:
        drugs_text = "、".join(labels)
        reasoning_parts.append(f"推荐使用{drugs_text}进行联合治疗。")
    
    reasoning = "".join(reasoning_parts)
    if not reasoning:
        reasoning = f"根据对话内容，推荐使用{', '.join(labels)}。"
    
    # 构建 JSON 输出
    completion = {
        "drugs": labels,
        "reasoning": reasoning
    }
    
    return json.dumps(completion, ensure_ascii=False)


def convert_to_sft_format(
    input_file: str,
    output_file: str,
    template_type: str = "qwen2"  # 支持 qwen2, chatml, llama3
):
    """将原始 JSONL 数据转换为 SFT 训练格式。
    
    Args:
        input_file: 输入文件路径（JSONL 格式）
        output_file: 输出文件路径（JSONL 格式）
        template_type: 模板类型（qwen2, chatml, llama3）
    """
    output_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 提取字段
                labels = data.get("label", [])
                disease = data.get("disease", [])
                dialog = data.get("dialog", [])
                
                if not labels or not dialog:
                    print(f"[warn] 跳过第 {line_num} 行：缺少必要字段")
                    continue
                
                # 提取约束信息
                constraints = extract_constraints(dialog)
                
                # 构建 prompt 和 completion
                prompt = build_prompt(dialog, disease, constraints)
                completion = build_completion(labels, dialog, disease, constraints)
                
                # 根据模板类型构建训练样本
                if template_type == "qwen2":
                    # Qwen2 格式
                    sample = {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            },
                            {
                                "role": "assistant",
                                "content": completion
                            }
                        ]
                    }
                elif template_type == "chatml":
                    # ChatML 格式
                    sample = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "你是一个专业的医疗助手。"
                            },
                            {
                                "role": "user",
                                "content": prompt
                            },
                            {
                                "role": "assistant",
                                "content": completion
                            }
                        ]
                    }
                else:  # llama3
                    # LLaMA3 格式
                    sample = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "你是一个专业的医疗助手。"
                            },
                            {
                                "role": "user",
                                "content": prompt
                            },
                            {
                                "role": "assistant",
                                "content": completion
                            }
                        ]
                    }
                
                output_data.append(sample)
                
            except Exception as e:
                print(f"[error] 处理第 {line_num} 行时出错：{e}")
                continue
    
    # 写入输出文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in output_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"[success] 转换完成：{len(output_data)} 条样本")
    print(f"[info] 输出文件：{output_file}")


def main():
    """主函数：转换训练集和验证集。"""
    base_dir = Path(__file__).parent.parent
    dialmed_dir = base_dir / "dialmed"
    sft_data_dir = base_dir / "sft_data"
    
    # 创建输出目录
    sft_data_dir.mkdir(exist_ok=True)
    
    # 转换训练集
    print("=" * 50)
    print("转换训练集...")
    print("=" * 50)
    convert_to_sft_format(
        input_file=str(dialmed_dir / "train.txt"),
        output_file=str(sft_data_dir / "train.jsonl"),
        template_type="qwen2"  # 可根据基础模型调整
    )
    
    # 转换验证集
    print("\n" + "=" * 50)
    print("转换验证集...")
    print("=" * 50)
    convert_to_sft_format(
        input_file=str(dialmed_dir / "dev.txt"),
        output_file=str(sft_data_dir / "dev.jsonl"),
        template_type="qwen2"
    )
    
    print("\n" + "=" * 50)
    print("数据转换完成！")
    print("=" * 50)
    print(f"训练集：{sft_data_dir / 'train.jsonl'}")
    print(f"验证集：{sft_data_dir / 'dev.jsonl'}")


if __name__ == "__main__":
    main()

