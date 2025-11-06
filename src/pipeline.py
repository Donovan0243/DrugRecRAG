"""End-to-end GAP pipeline (minimal, runnable).

中文说明：以最少依赖串起：抽取→建图→NP/PP→最终提示→推理。
"""

from typing import Dict, List, Tuple
import re
from .config import pipeline as pipe_cfg
from . import extraction, graph, retrieval, inference
from .kg import MultiKG, DiseaseKBKG, CMeKGKG
from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASS, NEO4J_DATABASE, NEO4J_DATABASES, KG_INTEGRATION_STRATEGY


def _extract_candidates(
    graph_text: str,
    np_facts: List[str],
    pp_items: List[str],
    linked_states: Dict[str, Dict] = None
) -> Tuple[List[str], List[str]]:
    """从 graph_text、NP/PP 和 linked_states 中提取候选疾病和药物。
    
    中文：提取逻辑：
    - candidate_medications（候选药物）：强制约束，只从知识图谱的药物中推荐
      1. graph_text 中的药物三元组（最重要）
      2. PP 文本中解析的药物
      3. NP 文本中解析的药物（如果是 treatment）
    
    - candidate_diseases（候选疾病）：提供上下文信息
      1. linked_states 中医生诊断的疾病（doctor-positive + Disease label）
      2. linked_states 中其他已链接的疾病
      3. NP/PP 中解析的疾病名
    
    返回：(候选疾病列表, 候选药物列表)
    """
    diseases = set()
    medications = set()
    
    # ============ 提取候选药物（强制约束） ============
    
    # 1. 从 graph_text 三元组中提取药物（最重要）
    # 模式：(Disease, drugTherapy, Drug) 或 (Disease, common_drug, Drug) 等
    drug_relations = ['drugTherapy', 'treatment', 'Treatment', 'common_drug', 'recommand_drug', 'commonDrug', 'recommandDrug']
    drug_pattern = re.compile(rf'\(([^,]+),\s*(?:{"|".join(drug_relations)}),\s*([^)]+)\)')
    for match in drug_pattern.finditer(graph_text):
        disease_name = match.group(1).strip()
        drug_name = match.group(2).strip()
        # 排除 Patient 节点
        if not drug_name.startswith("Patient:") and drug_name:
            medications.add(drug_name)
    
    # 2. 从 PP 文本中提取药物（正则匹配）
    # 模式：...→药物『XXX』 或 [E0001] ...—drugTherapy→药物『XXX』
    pp_pattern_drug = re.compile(r"药物『([^』]+)』")
    for pp_text in pp_items:
        for match in pp_pattern_drug.finditer(pp_text):
            drug_name = match.group(1).strip()
            if drug_name:
                medications.add(drug_name)
    
    # 3. 从 NP 文本中提取药物（如果是 treatment 类型）
    # 模式：疾病『XXX』—drugTherapy→药物『YYY』
    np_pattern_drug = re.compile(r"药物『([^』]+)』")
    for np_text in np_facts:
        for match in np_pattern_drug.finditer(np_text):
            drug_name = match.group(1).strip()
            if drug_name:
                medications.add(drug_name)
    
    # ============ 提取候选疾病（提供上下文） ============
    
    # 1. 从 linked_states 中提取疾病（优先级最高）
    if linked_states:
        doctor_diseases = set()  # 医生诊断的疾病（优先级最高）
        other_diseases = set()    # 其他已链接的疾病
        
        for ent, meta in linked_states.items():
            label = (meta.get("kg_label") or "").lower()
            if label == "disease":
                disease_name = meta.get("kg_name", ent)
                main_state = meta.get("main-state", "").lower()
                
                # 优先：医生诊断的疾病
                if main_state == "doctor-positive":
                    doctor_diseases.add(disease_name)
                else:
                    other_diseases.add(disease_name)
        
        # 先添加医生诊断的疾病，再添加其他疾病
        diseases.update(doctor_diseases)
        diseases.update(other_diseases)
    
    # 2. 从 NP/PP 文本中提取疾病（补充上下文）
    # 模式：疾病『XXX』
    disease_pattern = re.compile(r"疾病『([^』]+)』")
    all_text = "\n".join(np_facts + pp_items)
    for match in disease_pattern.finditer(all_text):
        disease_name = match.group(1).strip()
        if disease_name:
            diseases.add(disease_name)
    
    # 3. 从 graph_text 三元组中提取疾病（补充）
    # 只提取那些作为疾病节点出现在药物关系中的
    for match in drug_pattern.finditer(graph_text):
        disease_name = match.group(1).strip()
        # 排除 Patient 节点和通用节点
        if disease_name and not disease_name.startswith("Patient:"):
            # 简单启发式：如果包含常见疾病关键词，当作疾病
            if any(kw in disease_name for kw in ["炎", "病", "症", "感染", "溃疡", "综合征", "综合症"]):
                diseases.add(disease_name)
    
    return sorted(list(diseases)), sorted(list(medications))


def run_gap(dialogue_text: str, kg_triples: List[tuple] = None) -> Dict:
    """Run the minimal GAP pipeline on a single dialogue text.

    中文：对单条对话执行 GAP。KG 可传入三元组列表，未提供则用空 KG。
    返回：{"recommendations": [...], "graph": str, "np": [...], "pp": [...]}。
    """
    # 1) 准备 LLM 调用器（纯文本阶段）与调试跟踪容器
    llm_call = inference.get_llm_caller(expect_json=False)
    trace: List[Dict] = []

    # 2) 概念抽取 + 状态判定
    entities = extraction.extract_concepts(dialogue_text, llm_call, trace=trace)
    entity_states = extraction.extract_states(dialogue_text, entities, llm_call, trace=trace)

    # 3) 构建患者图（含归一化）
    synonyms = {}  # 中文：可扩展为从词表加载
    triples = graph.build_patient_graph("P1", entity_states, synonyms)
    graph_text = graph.linearize_triples(triples)

    # 4) 加载 KG（必须连接到两个 Neo4j 图）
    kg = None
    try:
        if NEO4J_URI:
            # 中文：尝试创建MultiKG（多KG集成）
            kgs = {}
            for name, db_config in NEO4J_DATABASES.items():
                try:
                    if name == "cmekg":
                        kgs[name] = CMeKGKG(
                            db_config["uri"],
                            db_config["user"],
                            db_config["password"],
                            db_config["database"],
                        )
                    elif name == "diseasekb":
                        kgs[name] = DiseaseKBKG(
                            db_config["uri"],
                            db_config["user"],
                            db_config["password"],
                            db_config["database"],
                        )
                    else:
                        # 默认按DiseaseKB处理
                        kgs[name] = DiseaseKBKG(
                            db_config["uri"],
                            db_config["user"],
                            db_config["password"],
                            db_config["database"],
                        )
                except Exception as e:
                    print(f"[pipeline][warn] Failed to connect to {name}: {e}")
            
            # 要求两个库都可用
            required = ["cmekg", "diseasekb"]
            missing = [name for name in required if name not in kgs]
            if missing:
                raise RuntimeError(f"Neo4j databases not connected: {missing}. Please ensure both CMeKG and DiseaseKB are available.")
            
            # 初始化全文索引（用于近似匹配）
            try:
                for name, kg_instance in kgs.items():
                    kg_instance.ensure_fulltext_indexes()
            except Exception as e:
                print(f"[pipeline][warn] Failed to create fulltext indexes: {e}")
            
            kg = MultiKG(kgs, strategy=KG_INTEGRATION_STRATEGY)
    except Exception as e:
        print(f"[pipeline][warn] KG initialization failed: {e}")
        kg = None
    
    if kg is None:
        raise RuntimeError("KG initialization failed. Both CMeKG and DiseaseKB must be connected.")

    # 5) 实体链接（EL）到 KG（仅当 Neo4j/MultiKG 可用）
    linked_states = None
    if isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
        # 将实体与 KG 标准节点对齐（但不查询KG事实，等LLM判断类型后再查询）
        linked_states = graph.link_entities_to_kg(entity_states, kg)
        # 暂时不查询KG事实，只保留原始患者图
        graph_text = graph.linearize_triples(triples)

    # 5.5) 构造实体链接映射，便于调试
    el_mappings: List[Dict] = []
    if linked_states:
        for original_name, meta in linked_states.items():
            el_mappings.append({
                "original": original_name,
                "kg_name": meta.get("kg_name"),
                "kg_label": meta.get("kg_label"),
                "kg_id": meta.get("kg_id"),
                "kg_source": meta.get("kg_source"),
                "main_state": meta.get("main-state"),
                # 新增：输出槽位里的既往史字段，便于调试观察
                "past_medical_history": meta.get("past-medical-history"),
            })

    # 6) 先问LLM要查什么类型（使用原始graph，不包含KG事实）
    relation_type = None
    if isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
        from .inference import get_llm_caller
        caller = get_llm_caller(expect_json=False)
        context_text = "\n".join({s for s, p, o in triples if p == "has_concept"})
        original_graph_text = graph.linearize_triples(triples)
        from .prompts import relation_type_classifier
        cls_prompt = relation_type_classifier(context_text, original_graph_text)
        relation_type = caller(cls_prompt).strip().lower()
        # 去掉可能的首尾引号
        if (relation_type.startswith('"') and relation_type.endswith('"')) or (relation_type.startswith("'") and relation_type.endswith("'")):
            relation_type = relation_type[1:-1].strip()
        if trace is not None:
            trace.append({"stage": "relation_type", "prompt": cls_prompt, "output": relation_type})
    
    # 6.5) 根据LLM判断的类型，查询对应的KG事实（不写回患者图）
    kg_triples: List[Tuple[str, str, str]] = []
    if isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)) and relation_type:
        kg_triples = graph.augment_patient_graph_with_kg_by_type(triples, kg, relation_type, topk=5, linked_states=linked_states)
        # 保持 graph_text 只来源于原始患者图（Gp 纯净）
        graph_text = graph.linearize_triples(triples)

    # 7) 生成 NP/PP
    # NP：负责"发现"候选药物（动态触发，1-hop查询）
    np_facts = retrieval.build_np(kg_triples or triples, kg, linked_states=linked_states, relation_type=relation_type, trace=trace)
    # PP：负责"验证"候选药物的安全性（Schema匹配 + 多源验证）
    pp_items = retrieval.build_pp(kg_triples or triples, kg, linked_states=linked_states, np_facts=np_facts, trace=trace, dialogue_text=dialogue_text)

    # 7.5) 从 graph_text（纯Gp）、NP/PP、linked_states 中提取候选疾病和药物
    candidate_diseases, candidate_medications = _extract_candidates(
        graph_text, np_facts, pp_items, linked_states
    )

    # 7) 最终推理
    inf = inference.run_final_inference(
        context=dialogue_text,
        graph_text=graph_text,
        np_facts=np_facts,
        pp_items=pp_items,
        candidate_diseases=candidate_diseases,
        candidate_medications=candidate_medications,
    )

    result = {
        "recommendations": inf.get("recommendations", []),
        "llm_raw": inf.get("llm_raw", ""),
        "final_prompt": inf.get("final_prompt", ""),
        "graph": graph_text,
        "el_mappings": el_mappings,
        "np": np_facts,
        "pp": pp_items,
        "trace": trace,
    }
    # 中文：关闭 Neo4j/MultiKG 连接（如适用）
    if isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
        try:
            kg.close()
        except Exception:
            pass
    return result


