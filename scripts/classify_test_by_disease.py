#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据疾病字段将 test.txt 分类到不同科室
"""

import json
import os

# 定义科室和对应的疾病
DEPARTMENT_DISEASES = {
    "respiratory": {
        "name": "呼吸科",
        "diseases": [
            "上呼吸道感染",
            "鼻炎",
            "过敏性鼻炎",
            "肺炎"
        ]
    },
    "gastroenterology": {
        "name": "消化科",
        "diseases": [
            "胃炎",
            "急性胃肠炎",  # 对应肠胃炎
            "肠炎",
            "病毒性肝炎",
            "食管炎",
            "脂肪肝",
            "肝炎",
            "十二指肠炎",
            "肝硬化"
        ]
    },
    "dermatology": {
        "name": "皮肤科",
        "diseases": [
            "皮炎",
            "过敏性皮炎",
            "脂溢性皮炎"
        ]
    }
}

# 输入文件路径
INPUT_FILE = "dialmed/test.txt"

# 输出文件路径
OUTPUT_DIR = "dialmed"
OUTPUT_FILES = {
    "respiratory": os.path.join(OUTPUT_DIR, "test_respiratory.txt"),
    "gastroenterology": os.path.join(OUTPUT_DIR, "test_gastroenterology.txt"),
    "dermatology": os.path.join(OUTPUT_DIR, "test_dermatology.txt"),
    "other": os.path.join(OUTPUT_DIR, "test_other.txt")
}


def classify_disease(disease_list):
    """
    根据疾病列表判断属于哪个科室
    
    Args:
        disease_list: 疾病列表，如 ["上呼吸道感染"]
    
    Returns:
        str: 科室名称（respiratory, gastroenterology, dermatology, other）
    """
    # 遍历所有疾病，找到第一个匹配的科室
    for disease in disease_list:
        for dept_key, dept_info in DEPARTMENT_DISEASES.items():
            if disease in dept_info["diseases"]:
                return dept_key
    
    # 如果没有匹配的科室，返回 other
    return "other"


def main():
    """主函数"""
    # 统计信息
    stats = {
        "respiratory": 0,
        "gastroenterology": 0,
        "dermatology": 0,
        "other": 0,
        "total": 0
    }
    
    # 打开输出文件
    output_files = {}
    for dept_key, filepath in OUTPUT_FILES.items():
        output_files[dept_key] = open(filepath, "w", encoding="utf-8")
    
    try:
        # 读取输入文件并分类
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    disease_list = data.get("disease", [])
                    
                    # 分类
                    dept = classify_disease(disease_list)
                    
                    # 写入对应的文件
                    output_files[dept].write(line + "\n")
                    stats[dept] += 1
                    stats["total"] += 1
                    
                except json.JSONDecodeError as e:
                    print(f"警告: 无法解析JSON行: {line[:50]}... 错误: {e}")
                    continue
    
    finally:
        # 关闭所有输出文件
        for f in output_files.values():
            f.close()
    
    # 打印统计信息
    print("=" * 50)
    print("分类完成！统计信息：")
    print("=" * 50)
    print(f"总数据量: {stats['total']}")
    print(f"\n呼吸科 (Respiratory): {stats['respiratory']} 条")
    print(f"  疾病: {', '.join(DEPARTMENT_DISEASES['respiratory']['diseases'])}")
    print(f"\n消化科 (Gastroenterology): {stats['gastroenterology']} 条")
    print(f"  疾病: {', '.join(DEPARTMENT_DISEASES['gastroenterology']['diseases'])}")
    print(f"\n皮肤科 (Dermatology): {stats['dermatology']} 条")
    print(f"  疾病: {', '.join(DEPARTMENT_DISEASES['dermatology']['diseases'])}")
    print(f"\n其他: {stats['other']} 条")
    print("=" * 50)
    print(f"\n输出文件:")
    for dept_key, filepath in OUTPUT_FILES.items():
        print(f"  - {filepath}")


if __name__ == "__main__":
    main()

