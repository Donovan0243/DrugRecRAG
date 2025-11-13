"""局部语义精筛调试脚本

用模拟KG返回的安全文本 + 模拟患者constraints，调用现有的
phase_b_safety_validation，观察是否能命中语义证据。

运行：
  python scripts/debug_phaseb_semantic.py
"""

from typing import List, Dict

# 导入现有 Phase B 实现
from src.retrieval import phase_b_safety_validation


class DummyKG:
    """最小KG桩：返回与药物直连的安全文本（字符串数组）。
    文本格式可以是任意自然语言句子，retrieval 会自动抽取『...』中的正文。
    """

    def __init__(self, data: Dict[str, Dict[str, List[str]]]):
        self.data = data

    # 注意：现有实现会同时取 precautions / contraindications / adverse_reactions
    def precautions_for_drugs(self, drug_names: List[str], k: int) -> List[str]:
        out = []
        for d in drug_names:
            out.extend(self.data.get(d, {}).get("precautions", [])[:k])
        return out

    def contraindications_for_drugs(self, drug_names: List[str], k: int) -> List[str]:
        out = []
        for d in drug_names:
            out.extend(self.data.get(d, {}).get("contraindications", [])[:k])
        return out

    def adverse_reactions_for_drugs(self, drug_names: List[str], k: int) -> List[str]:
        out = []
        for d in drug_names:
            out.extend(self.data.get(d, {}).get("adverseReactions", [])[:k])
        return out


def main():
    # 构造几种典型场景的安全文本（模拟 Neo4j 粗筛结果）
    # 保留两种形式：
    #  1) 原始中文句子（不含『』）
    #  2) KG样式：药物『X』—contraindications→『正文』
    dummy_data = {
        "特比萘芬": {
            "contraindications": [
                "药物『特比萘芬』—contraindications→『肝功能不全者禁用』",
                "已知对本品成分过敏者禁用",
            ],
            "precautions": [
                "治疗期间应监测肝功能",
            ],
            "adverseReactions": [
                "可能出现肝酶升高、皮疹等不良反应",
            ],
        },
        "布洛芬": {
            "contraindications": [
                "药物『布洛芬』—contraindications→『孕晚期禁用』",
                "对阿司匹林或其他NSAIDs过敏者禁用",
            ],
            "precautions": [
                "6个月以下婴儿不推荐使用",
            ],
            "adverseReactions": [
                "与华法林等抗凝药合用可能增加出血风险",
            ],
        },
        "阿莫西林": {
            "contraindications": [
                "对青霉素类药物过敏者禁用",
            ],
            "precautions": [],
            "adverseReactions": [],
        },
    }

    kg = DummyKG(dummy_data)

    # 构造候选药物与患者约束
    candidate_drugs = ["特比萘芬", "布洛芬", "阿莫西林"]

    # 场景A：既往病史 = 乙肝（期望命中 特比萘芬 的肝功能禁忌）
    constraints_A = {
        "allergies": [],
        "status": [],
        "past_history": ["乙肝"],
        "taking_drugs": [],
        "not_recommended_drugs": [],
    }
    print("\n=== 场景A：past_history=['乙肝'] ===")
    res_A = phase_b_safety_validation(candidate_drugs, {"constraints": constraints_A}, kg)
    for r in res_A:
        print(" ", r)

    # 场景B：过敏 = 青霉素（期望命中 阿莫西林 的过敏禁忌）
    constraints_B = {
        "allergies": ["青霉素"],
        "status": [],
        "past_history": [],
        "taking_drugs": [],
        "not_recommended_drugs": [],
    }
    print("\n=== 场景B：allergies=['青霉素'] ===")
    res_B = phase_b_safety_validation(candidate_drugs, {"constraints": constraints_B}, kg)
    for r in res_B:
        print(" ", r)

    # 场景C：特殊人群 = pregnant（期望命中 布洛芬 的孕期禁忌）
    constraints_C = {
        "allergies": [],
        "status": ["pregnant"],
        "past_history": [],
        "taking_drugs": [],
        "not_recommended_drugs": [],
    }
    print("\n=== 场景C：status=['pregnant'] ===")
    res_C = phase_b_safety_validation(candidate_drugs, {"constraints": constraints_C}, kg)
    for r in res_C:
        print(" ", r)

    # 场景D：相互作用 = 正在服用 华法林（期望命中 布洛芬 的合用风险）
    constraints_D = {
        "allergies": [],
        "status": [],
        "past_history": [],
        "taking_drugs": [{"name": "华法林", "status": "effective"}],
        "not_recommended_drugs": [],
    }
    print("\n=== 场景D：taking_drugs=['华法林'] ===")
    res_D = phase_b_safety_validation(candidate_drugs, {"constraints": constraints_D}, kg)
    for r in res_D:
        print(" ", r)


if __name__ == "__main__":
    main()


