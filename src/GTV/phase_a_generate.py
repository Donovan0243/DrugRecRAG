"""Phase A: 使用 SFT 模型生成候选药物。

中文说明：调用 SFT 模型（通过 Ollama/vLLM）生成候选药物推荐。
"""

from typing import Dict, List, Optional
import json
import time
import os
from ..inference import get_llm_caller
from ..config import llm as llm_cfg
from .prompts import phase_a_generate_prompt


def load_candidate_drugs_list(label_file: str = "dialmed/label.json") -> Optional[List[str]]:
    """加载候选药物列表（从 label.json）。
    
    Args:
        label_file: label.json 文件路径
    
    Returns:
        候选药物列表，如果文件不存在则返回 None
    """
    if not os.path.exists(label_file):
        return None
    
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            label_dict = json.load(f)
        candidate_drugs = list(label_dict.keys())
        print(f"[GTV][Phase A] 加载了 {len(candidate_drugs)} 个候选药物（来自 {label_file}）")
        return candidate_drugs
    except Exception as e:
        print(f"[GTV][Phase A][warn] 加载候选药物列表失败: {e}")
        return None


def phase_a_generate_drugs(
    dialogue_text: str,
    patient_state: Dict,
    trace: List[Dict] = None,
    use_candidate_list: bool = False,
    label_file: str = "dialmed/label.json"
) -> Dict:
    """Phase A: 使用 SFT 模型生成候选药物。
    
    Args:
        dialogue_text: 对话文本
        patient_state: 患者状态 JSON
        trace: 调试跟踪列表
        use_candidate_list: 是否使用候选药物列表（从 label.json 加载）
        label_file: label.json 文件路径
    
    Returns:
        {
            "drugs": List[str],  # 候选药物列表
            "reasoning": str,     # 推荐理由
            "raw_output": str     # 原始输出（用于调试）
        }
    """
    # 加载候选药物列表（如果启用）
    candidate_drugs_list = None
    if use_candidate_list:
        candidate_drugs_list = load_candidate_drugs_list(label_file)
        if candidate_drugs_list:
            print(f"[GTV][Phase A] 使用候选药物列表模式：{len(candidate_drugs_list)} 个候选药物")
        else:
            print(f"[GTV][Phase A][warn] 无法加载候选药物列表，使用自由生成模式")
    
    # 构建 prompt
    prompt = phase_a_generate_prompt(dialogue_text, patient_state, candidate_drugs_list)
    
    # 调用 SFT 模型（使用 expect_json=True 确保返回 JSON）
    caller = get_llm_caller(expect_json=True)
    
    t0 = time.time()
    raw_output = caller(prompt)
    dt = int((time.time() - t0) * 1000)
    
    try:
        print(f"[GTV][Phase A][{llm_cfg.provider}:{llm_cfg.model}] {dt} ms, in={len(prompt)} chars, out={len(str(raw_output))} chars")
    except Exception:
        pass
    
    if trace is not None:
        trace.append({
            "stage": "phase_a_generate",
            "prompt": prompt,
            "raw_output": raw_output
        })
    
    # 解析 JSON 输出
    default_result = {
        "drugs": [],
        "reasoning": "",
        "raw_output": raw_output
    }
    
    try:
        # 清理输出（去除 Markdown 代码块等）
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        
        # 提取 JSON 对象
        if "{" in cleaned and "}" in cleaned:
            start = cleaned.index("{")
            end = cleaned.rindex("}") + 1
            cleaned = cleaned[start:end]
        
        data = json.loads(cleaned)
        if isinstance(data, dict):
            result = {
                "drugs": data.get("drugs", []),
                "reasoning": data.get("reasoning", ""),
                "raw_output": raw_output
            }
            
            # 确保 drugs 是列表
            if not isinstance(result["drugs"], list):
                if isinstance(result["drugs"], str):
                    result["drugs"] = [result["drugs"]]
                else:
                    result["drugs"] = []
            
            print(f"[GTV][Phase A] 生成 {len(result['drugs'])} 个候选药物: {result['drugs']}")
            return result
            
    except Exception as e:
        try:
            print(f"[GTV][Phase A][warn] Failed to parse SFT model output: {e}")
            print(f"[GTV][Phase A][warn] Raw output: {raw_output[:500]}")
        except Exception:
            pass
    
    print(f"[GTV][Phase A][warn] 返回默认结果（空列表）")
    return default_result

