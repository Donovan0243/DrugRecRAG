"""
测试修正后的语义匹配逻辑

流程：
1. 在线编码患者的过敏（如"青霉素"）
2. 通过 Neo4j 查找药物的安全文本（如"对青霉素类药物过敏者禁用"）
3. 从 FAISS 索引中获取这些安全文本的离线向量
4. 计算患者查询向量与安全文本向量的相似度
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.retrieval import phase_b_safety_validation

# 模拟 KG
class DummyKG:
    def precautions_for_drugs(self, drug_names, k=20):
        return []
    
    def contraindications_for_drugs(self, drug_names, k=20):
        if "阿莫西林" in drug_names:
            return ["药物『阿莫西林』—contraindications→Subject『对青霉素类药物过敏者禁用』"]
        return []
    
    def adverse_reactions_for_drugs(self, drug_names, k=20):
        return []

# 测试用例
patient_state = {
    "constraints": {
        "allergies": ["青霉素"]
    }
}

candidate_drugs = ["阿莫西林"]

print("=" * 80)
print("测试修正后的语义匹配逻辑")
print("=" * 80)
print(f"\n患者状态: {patient_state}")
print(f"候选药物: {candidate_drugs}\n")

print("流程说明：")
print("1. 在线编码患者过敏：'青霉素'")
print("2. Neo4j 粗筛：查询阿莫西林的安全文本 → '对青霉素类药物过敏者禁用'")
print("3. 从 FAISS 索引获取该安全文本的离线向量")
print("4. 计算相似度：患者向量 vs 安全文本向量\n")

kg = DummyKG()
results = phase_b_safety_validation(candidate_drugs, patient_state, kg)

print("=" * 80)
print("验证结果：")
print("=" * 80)
if results:
    for r in results:
        print(f"  ✅ {r}")
else:
    print("  ❌ 未命中（可能原因：FAISS 索引中没有该安全文本，或相似度未达到阈值）")

