"""Patient-centric graph construction and linearization.

中文说明：根据抽取结果构建患者中心图（轻量实现），并做实体归一化与线性化输出。
"""

from typing import Dict, List, Tuple
from .kg import DiseaseKBKG, CMeKGKG, MultiKG


def normalize_entity(name: str, synonyms: Dict[str, str]) -> str:
    """Normalize entity name using a simple synonyms map.

    中文：使用同义词映射做简单归一化；未命中则返回原名。
    """
    key = name.strip().lower()
    return synonyms.get(key, name)


def build_patient_graph(
    patient_id: str,
    entity_states: Dict[str, Dict],
    synonyms: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    """Return a list of triples (subject, predicate, object) as the patient-centric graph.

    中文：最小化实现，直接返回三元组列表。
    """
    triples: List[Tuple[str, str, str]] = []
    patient = f"Patient:{patient_id}"
    for ent, meta in entity_states.items():
        ent_norm = normalize_entity(ent, synonyms)
        triples.append((patient, "has_concept", ent_norm))
        if meta.get("main-state"):
            triples.append((ent_norm, "state", str(meta["main-state"])))
    return triples


def linearize_triples(triples: List[Tuple[str, str, str]]) -> str:
    """Convert triples to a compact text block.

    中文：线性化为文本，便于注入到最终提示中。
    """
    lines = [f"({s}, {p}, {o})" for s, p, o in triples]
    return "\n".join(lines)


def split_concepts_by_labels(linked_states: Dict[str, Dict]):
    """Classify concepts using KG labels after entity linking.

    中文：基于实体链接得到的 kg_label 将概念划分为疾病/症状/药物，回退到名称启发式。
    """
    diseases, symptoms, drugs = set(), set(), set()
    for ent, meta in (linked_states or {}).items():
        label = (meta.get("kg_label") or "").lower()
        if label == "disease":
            diseases.add(meta.get("kg_name", ent))
        elif label == "symptom":
            symptoms.add(meta.get("kg_name", ent))
        elif label == "drug":
            drugs.add(meta.get("kg_name", ent))
        else:
            # 回退：简单启发式
            name = (meta.get("kg_name") or ent).lower()
            if any(k in name for k in ["炎", "感冒", "综合征", "感染", "病", "症"]):
                diseases.add(name)
            elif any(k in name for k in ["痛", "痒", "咳", "热", "便", "恶心", "鼻塞", "腹泻"]):
                symptoms.add(name)
            else:
                drugs.add(name)
    return list(diseases), list(symptoms), list(drugs)


def link_entities_to_kg(entity_states: Dict[str, Dict], kg) -> Dict[str, Dict]:
    """Link extracted entities to KG nodes; return mapping with resolved fields.

    中文：对每个实体做 Neo4j 精确/模糊匹配，附加 {kg_id, kg_label, kg_name} 信息。
    """
    linked: Dict[str, Dict] = {}
    for ent, meta in entity_states.items():
        resolved = {}
        try:
            resolved = kg.resolve_entity(ent)
        except Exception:
            resolved = {}
        meta2 = dict(meta)
        if resolved:
            meta2.update({
                "kg_id": str(resolved.get("id")),
                "kg_label": resolved.get("label"),
                "kg_name": resolved.get("name", ent),
                "kg_source": resolved.get("kg_source"),  # 添加来源KG信息
            })
        linked[ent] = meta2
    return linked


def augment_patient_graph_with_kg(
    triples: List[Tuple[str, str, str]], kg, topk: int = 5, linked_states: Dict[str, Dict] = None
) -> List[Tuple[str, str, str]]:
    """Augment patient graph with KG neighborhood facts and return merged triples.

    中文：从患者概念推断疾病/症状集合，查询 diseasekb 的邻域事实并写回到图中。
    写回关系使用图谱原生关系名（common_drug、recommand_drug、need_check、*_eat）。
    """
    diseases, symptoms, _ = split_concepts_by_labels(linked_states or {})
    added: List[Tuple[str, str, str]] = []
    try:
        added += kg.drugs_for_diseases_struct(diseases, topk)
        added += kg.drugs_for_symptoms_struct(symptoms, topk)
        added += kg.checks_for_diseases_struct(diseases, topk)
        added += kg.diet_for_diseases_struct(diseases, topk)
    except Exception:
        pass
    # 去重合并
    seen = set(triples)
    merged = list(triples)
    for t in added:
        if t not in seen:
            seen.add(t)
            merged.append(t)
    return merged


def augment_patient_graph_with_kg_by_type(
    triples: List[Tuple[str, str, str]], kg, relation_type: str, topk: int = 5, linked_states: Dict[str, Dict] = None
) -> List[Tuple[str, str, str]]:
    """按类型检索对应的KG事实，但不写回患者图，返回仅包含KG事实的三元组列表。
    
    中文：保持 Gp（患者中心图）纯净，只包含对话事实；本函数仅返回与 relation_type 相关的 KG 三元组，
    供 NP/PP 生成使用，不再将其合并回患者图。
    
    Args:
        triples: 原始患者图三元组（不会被修改）
        kg: KG实例
        relation_type: 关系类型（treatment/check/diet/symptom2disease）
        topk: 每种类型的查询数量上限
        linked_states: 实体链接后的状态
    
    Returns:
        仅包含 KG 检索到的三元组列表（不含患者图原有三元组）
    """
    # 使用改进版的split_concepts_by_labels，它会包含链接失败的原始名称
    from .retrieval import _split_concepts_by_labels
    diseases, symptoms, _ = _split_concepts_by_labels(linked_states or {})
    added: List[Tuple[str, str, str]] = []
    
    # 调试日志：显示实际查询的疾病列表
    if diseases:
        print(f"[graph][augment_by_type] 查询疾病列表: {diseases}")
    
    try:
        if relation_type == "treatment":
            # 查询药物关系（会查询所有传入的疾病名称，包括链接失败的原始名称）
            added += kg.drugs_for_diseases_struct(diseases, topk)
            if not added:
                added += kg.drugs_for_symptoms_struct(symptoms, topk)
        elif relation_type == "check":
            # 查询检查关系
            added += kg.checks_for_diseases_struct(diseases, topk)
        elif relation_type == "diet":
            # 查询饮食关系
            added += kg.diet_for_diseases_struct(diseases, topk)
        elif relation_type == "symptom2disease":
            # 查询症状→疾病→药物关系
            added += kg.drugs_for_symptoms_struct(symptoms, topk)
        # 如果类型不在预期范围内，默认不查询（或查询所有类型，根据需求调整）
    except Exception:
        pass
    
    # 仅返回 KG 事实（不写回患者图）
    # 去重
    seen = set()
    unique_added: List[Tuple[str, str, str]] = []
    for t in added:
        if t not in seen:
            seen.add(t)
            unique_added.append(t)
    return unique_added


