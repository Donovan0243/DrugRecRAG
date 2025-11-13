#!/usr/bin/env python3
"""手动查询药物的安全文本（precautions、contraindications、adverse_reactions）"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.kg import CMeKGKG, DiseaseKBKG, MultiKG
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASS, NEO4J_DATABASES, KG_INTEGRATION_STRATEGY

def check_drug_safety(drug_name: str, kg):
    """查询药物的安全文本"""
    print(f"药物名称: {drug_name}")
    print("-" * 80)
    
    try:
        # 1. 查询 precautions（注意事项）
        print("=" * 80)
        print("1. Precautions（注意事项）")
        print("=" * 80)
        precautions = kg.precautions_for_drugs([drug_name], k=20)
        if precautions:
            print(f"✅ 找到 {len(precautions)} 个 Precautions：")
            for i, p in enumerate(precautions, 1):
                print(f"  {i}. {p}")
        else:
            print("❌ 未找到 Precautions")
        print()
        
        # 2. 查询 contraindications（禁忌症）
        print("=" * 80)
        print("2. Contraindications（禁忌症）")
        print("=" * 80)
        contraindications = kg.contraindications_for_drugs([drug_name], k=20)
        if contraindications:
            print(f"✅ 找到 {len(contraindications)} 个 Contraindications：")
            for i, c in enumerate(contraindications, 1):
                print(f"  {i}. {c}")
        else:
            print("❌ 未找到 Contraindications")
        print()
        
        # 3. 查询 adverse_reactions（不良反应）
        print("=" * 80)
        print("3. Adverse Reactions（不良反应）")
        print("=" * 80)
        adverse_reactions = kg.adverse_reactions_for_drugs([drug_name], k=20)
        if adverse_reactions:
            print(f"✅ 找到 {len(adverse_reactions)} 个 Adverse Reactions：")
            for i, a in enumerate(adverse_reactions, 1):
                print(f"  {i}. {a}")
        else:
            print("❌ 未找到 Adverse Reactions")
        print()
        
    except Exception as e:
        print(f"❌ 查询出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    drug_name = "阿奇霉素片"
    if len(sys.argv) > 1:
        drug_name = sys.argv[1]
    
    print(f"查询药物安全文本: {drug_name}")
    print("=" * 80)
    print()
    
    # 初始化 KG
    try:
        kgs = {}
        if "cmekg" in NEO4J_DATABASES:
            kgs["cmekg"] = CMeKGKG(NEO4J_URI, NEO4J_USER, NEO4J_PASS, NEO4J_DATABASES["cmekg"])
        if "diseasekb" in NEO4J_DATABASES:
            kgs["diseasekb"] = DiseaseKBKG(NEO4J_URI, NEO4J_USER, NEO4J_PASS, NEO4J_DATABASES["diseasekb"])
        
        if not kgs:
            print("❌ 无法连接到任何 KG")
            sys.exit(1)
        
        kg = MultiKG(kgs, strategy=KG_INTEGRATION_STRATEGY)
        
        # 查询
        check_drug_safety(drug_name, kg)
        
    except Exception as e:
        print(f"❌ 初始化 KG 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

