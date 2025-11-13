"""新版检索模块：Phase A（候选生成）和 Phase B（安全验证）。

中文说明：基于结构化患者状态JSON，执行候选生成和安全验证。
"""

from typing import Dict, List, Tuple
from .kg import MultiKG, DiseaseKBKG, CMeKGKG
from .query import _get_vector_searcher  # 使用离线 FAISS 索引（data/embeddings）
import numpy as np
try:
    from .embedding.encoder import EmbeddingEncoder  # 复用已有编码器（与离线索引同模）
except Exception:
    EmbeddingEncoder = None  # 运行期无可用编码器则降级


def phase_a_candidate_generation(patient_state: Dict, kg) -> List[str]:
    """Phase A: 基于problems生成候选药物列表（NP-like）。
    
    改进版（推理逻辑）：优先使用疾病查询，避免噪音。
    - 如果疾病查询有结果 → 直接返回，不再查询症状（避免噪音）
    - 如果疾病查询为空 → 才查询症状（作为fallback）
    
    这样做的目的是：
    1. 提高准确率：避免"感冒+头痛"查询返回"高血压"药物
    2. 提高召回率：症状查询支持1跳和2跳路径
    
    Args:
        patient_state: 结构化患者状态JSON
        kg: 知识图谱实例
    
    Returns:
        候选药物列表（已排序）
    """
    if not isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
        return []
    
    candidate_drugs = set()

    from .config import retrieval as retrieval_cfg
    K = max(1, int(getattr(retrieval_cfg, "phase_a_k", 10)))

    # 辅助函数：从事实文本解析药物名
    def _collect_drugs_from_facts(facts: List[str]) -> List[str]:
        collected = []
        for fact in facts or []:
            if "药物『" in fact:
                start = fact.index("药物『") + 3
                end = fact.index("』", start) if "』" in fact[start:] else len(fact)
                drug_name = fact[start:end].strip()
                if drug_name:
                    collected.append(drug_name)
        return collected
    
    # Step1: 诊断疾病 + 原始疾病名称（兜底）
    diagnosed = patient_state.get("problems", {}).get("diagnosed", []) or []
    raw_diagnosed = patient_state.get("problems_raw", {}).get("diagnosed", []) or []
    # 如果 patient_state 中没有保留原始字段，则使用输入的 diagnosed 作为原始
    if not raw_diagnosed:
        raw_diagnosed = patient_state.get("problems", {}).get("diagnosed_raw", []) or []
    if not raw_diagnosed:
        raw_diagnosed = diagnosed

    disease_queries = []
    disease_seen = set()
    for name in list(diagnosed) + list(raw_diagnosed):
        name = (name or "").strip()
        if name and name not in disease_seen:
            disease_seen.add(name)
            disease_queries.append(name)

    if disease_queries:
        try:
            facts = kg.drugs_for_diseases(disease_queries, k=K)
            for drug_name in _collect_drugs_from_facts(facts):
                candidate_drugs.add(drug_name)
        except Exception as e:
            print(f"[retrieval][phase_a][warn] Failed to query drugs for diseases: {e}")

    # Step2: 告诉症状（不再因为疾病命中而跳过）
    symptoms = patient_state.get("problems", {}).get("symptoms", []) or []
    raw_symptoms = patient_state.get("problems_raw", {}).get("symptoms", []) or []
    if not raw_symptoms:
        raw_symptoms = patient_state.get("problems", {}).get("symptoms_raw", []) or []
    if not raw_symptoms:
        raw_symptoms = symptoms

    symptom_queries = []
    symptom_seen = set()
    for name in list(symptoms) + list(raw_symptoms):
        name = (name or "").strip()
        if name and name not in symptom_seen:
            symptom_seen.add(name)
            symptom_queries.append(name)

    if symptom_queries:
        try:
            facts = kg.drugs_for_symptoms(symptom_queries, k=K)
            for drug_name in _collect_drugs_from_facts(facts):
                candidate_drugs.add(drug_name)
        except Exception as e:
            print(f"[retrieval][phase_a][warn] Failed to query drugs for symptoms: {e}")
    
    return sorted(list(candidate_drugs))


def _phase_b_collect_all_safety_info(
    candidate_drugs: List[str],
    kg,
    trace: List[Dict] = None
) -> List[str]:
    """Phase B 无脑收集模式：直接收集所有候选药物的安全信息，不做匹配判断。
    
    中文：这个模式会收集所有候选药物的所有安全信息（注意事项、禁忌、不良反应），
    不做任何匹配判断，让最终的 LLM 自己去判断这些信息是否与患者约束相关。
    
    Args:
        candidate_drugs: 候选药物列表
        kg: 知识图谱实例
        trace: 调试跟踪列表
    
    Returns:
        安全信息列表（带证据ID）
    """
    validation_results = []
    evidence_id = 1
    
    # 提取 KG 格式串中的正文
    def _extract_text(s: str) -> str:
        if not isinstance(s, str):
            return ""
        try:
            if "『" in s and "』" in s:
                start = s.rfind("『") + 1
                end = s.rfind("』")
                if 0 < start < end:
                    return s[start:end].strip()
            if "→" in s:
                return s.split("→", 1)[-1].strip().strip("『』 ")
        except Exception:
            pass
        return s.strip()
    
    print(f"[phase_b][collect_all] 无脑收集模式：开始收集 {len(candidate_drugs)} 个候选药物的所有安全信息")
    
    # 对于每个候选药物，收集它的所有安全信息
    for drug in candidate_drugs:
        try:
            # 查询该药物的所有安全信息
            precautions = kg.precautions_for_drugs([drug], k=50)  # 增加 k 值以获取更多信息
            contraindications = kg.contraindications_for_drugs([drug], k=50)
            adverse_reactions = kg.adverse_reactions_for_drugs([drug], k=50)
            
            # 收集所有安全文本
            safety_items: List[Tuple[str, str]] = []
            for src, arr in (
                ("precautions", precautions),
                ("contraindications", contraindications),
                ("adverseReactions", adverse_reactions)
            ):
                for t in arr or []:
                    text = _extract_text(t)
                    if text:
                        safety_items.append((src, text))
            
            # 将所有安全信息添加到结果中（不做匹配判断）
            for src_label, text in safety_items:
                validation_results.append(
                    f"[E{evidence_id:04d}] 药物『{drug}』—{src_label}→{text}"
                )
                evidence_id += 1
            
            print(f"[phase_b][collect_all] 药物『{drug}』: 收集到 {len(safety_items)} 条安全信息")
        except Exception as e:
            print(f"[phase_b][collect_all] 收集药物『{drug}』的安全信息失败: {e}")
            continue
    
    print(f"[phase_b][collect_all] 总共收集到 {len(validation_results)} 条安全信息")
    
    if trace is not None:
        trace.append({
            "stage": "phase_b",
            "mode": "collect_all",
            "candidate_drugs": candidate_drugs,
            "validation_results": validation_results
        })
    
    return validation_results


def phase_b_safety_validation(
    candidate_drugs: List[str],
    patient_state: Dict,
    kg,
    trace: List[Dict] = None
) -> List[str]:
    """Phase B: 基于constraints验证候选药物的安全性（PP-like）。
    
    Args:
        candidate_drugs: 候选药物列表
        patient_state: 结构化患者状态JSON
        kg: 知识图谱实例
        trace: 调试跟踪列表
    
    Returns:
        安全验证结果列表（带证据ID）
    """
    # 允许鸭子类型KG：只要提供所需方法即可（便于本地桩调试）
    required_methods = [
        "precautions_for_drugs",
        "contraindications_for_drugs",
        "adverse_reactions_for_drugs",
    ]
    if not candidate_drugs or not all(hasattr(kg, m) for m in required_methods):
        return []
    
    # 检查是否启用"无脑收集"模式（直接收集所有安全信息，不做匹配判断）
    from .config import retrieval as retrieval_cfg
    if retrieval_cfg.phase_b_collect_all:
        return _phase_b_collect_all_safety_info(candidate_drugs, kg, trace)
    
    constraints = patient_state.get("constraints", {})
    validation_results = []
    evidence_id = 1
    # 统一相似度阈值（可迁移到 config）
    SIM_THRESHOLD = 0.50
    # 使用离线向量索引（FAISS）：data/embeddings/{kg}.index + *_mapping.json
    # 流程：1. 在线编码患者查询词 2. 从 FAISS 获取安全文本的离线向量 3. 计算相似度
    # 如果 FAISS 未命中，回退到字符串包含匹配
    encoder = None  # 用于在线编码患者查询词（安全文本使用离线向量）
    searcher_cmekg = None
    searcher_dkb = None
    try:
        # 两库均尝试加载（若配置关闭或文件缺失，将返回 None）
        searcher_cmekg = _get_vector_searcher("cmekg")
    except Exception:
        searcher_cmekg = None
    try:
        searcher_dkb = _get_vector_searcher("diseasekb")
    except Exception:
        searcher_dkb = None
    # 编码器用于在线编码患者查询词（所有节点都已离线embedding，安全文本使用离线向量）
    if 'EmbeddingEncoder' in globals() and EmbeddingEncoder is not None:
        try:
            encoder = EmbeddingEncoder()
        except Exception:
            encoder = None

    # 提取 KG 格式串中的正文，例如：
    # "药物『X』—contraindications→『正文』" → 返回 "正文"
    def _extract_text(s: str) -> str:
        if not isinstance(s, str):
            return ""
        # 优先取最后一对『...』内的内容
        try:
            if "『" in s and "』" in s:
                start = s.rfind("『") + 1
                end = s.rfind("』")
                if 0 < start < end:
                    return s[start:end].strip()
            # 次优：按箭头切分取右侧
            if "→" in s:
                return s.split("→", 1)[-1].strip().strip("『』 ")
        except Exception:
            pass
        return s.strip()

    # 使用离线向量索引进行语义匹配：返回是否命中
    # 正确流程：1. 在线编码患者查询词 2. 从 FAISS 获取安全文本的离线向量 3. 计算相似度
    def _semantic_hit_via_index(query_term: str, safety_texts: List[str], constraint_type: str = "约束") -> tuple:
        """
        使用离线向量索引进行语义匹配。
        
        流程：
        1. 在线编码患者查询词（如"青霉素"）
        2. 从 FAISS 索引中获取安全文本名称对应的离线向量
        3. 计算查询向量与安全文本向量的相似度
        4. 如果相似度 >= 阈值，返回命中
        
        Args:
            query_term: 患者查询词（如"青霉素"、"乙肝"）
            safety_texts: 从 Neo4j 获取的安全文本名称列表（如["对青霉素类药物过敏者禁用"]）
            constraint_type: 约束类型（用于日志输出，如"过敏"、"既往病史"）
        
        Returns:
            (hit, matched_name, score) 元组
        """
        if not query_term or not safety_texts or not encoder:
            if not encoder:
                print(f"[phase_b][faiss][{constraint_type}] 编码器不可用，跳过 FAISS 匹配")
            return (False, None, 0.0)
        
        print(f"[phase_b][faiss][{constraint_type}] 查询词: '{query_term}'")
        print(f"[phase_b][faiss][{constraint_type}] Neo4j 安全文本数量: {len(safety_texts)}")
        
        # 1. 在线编码患者查询词
        try:
            import numpy as np
            query_vec = np.array(encoder.encode_single(query_term), dtype=np.float32)
            # 归一化（用于余弦相似度）
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            print(f"[phase_b][faiss][{constraint_type}] 在线编码查询词完成，向量维度: {query_vec.shape[0]}")
        except Exception as e:
            print(f"[phase_b][faiss][{constraint_type}] 在线编码查询词失败: {e}")
            return (False, None, 0.0)
        
        # 2. 从 FAISS 索引中获取安全文本的离线向量（优先 CMeKG，再 DiseaseKB）
        best = (False, None, 0.0)
        
        for kg_name, searcher in [("cmekg", searcher_cmekg), ("diseasekb", searcher_dkb)]:
            if searcher is None:
                print(f"[phase_b][faiss][{constraint_type}] {kg_name} 索引不可用，跳过")
                continue
            try:
                print(f"[phase_b][faiss][{constraint_type}] 从 {kg_name} 索引获取安全文本向量...")
                # 从 FAISS 索引中获取这些安全文本名称对应的向量
                safety_vectors = searcher.get_vectors_by_names(safety_texts)
                
                found_count = sum(1 for v in safety_vectors if v is not None)
                print(f"[phase_b][faiss][{constraint_type}] {kg_name} 索引中找到 {found_count}/{len(safety_texts)} 个安全文本向量")
                
                # 3. 计算相似度
                for i, safety_vec in enumerate(safety_vectors):
                    if safety_vec is None:
                        continue
                    
                    safety_name = safety_texts[i]
                    
                    # 归一化安全文本向量
                    safety_norm = safety_vec / (np.linalg.norm(safety_vec) + 1e-8)
                    
                    # 计算余弦相似度（内积，因为已归一化）
                    similarity = float(np.dot(query_norm, safety_norm))
                    
                    print(f"[phase_b][faiss][{constraint_type}]   '{query_term}' vs '{safety_name[:50]}...' = {similarity:.4f} {'✅ 命中' if similarity >= SIM_THRESHOLD else '❌ 未命中'}")
                    
                    # 如果相似度 >= 阈值且比之前的最佳结果更好，更新
                    if similarity >= SIM_THRESHOLD and similarity > best[2]:
                        best = (True, safety_name, similarity)
                        print(f"[phase_b][faiss][{constraint_type}] 更新最佳匹配: '{safety_name}' (相似度: {similarity:.4f})")
            except Exception as e:
                print(f"[phase_b][faiss][{constraint_type}] {kg_name} 索引处理失败: {e}")
                continue
        
        if best[0]:
            print(f"[phase_b][faiss][{constraint_type}] ✅ 最终命中: '{best[1]}' (相似度: {best[2]:.4f})")
        else:
            print(f"[phase_b][faiss][{constraint_type}] ❌ 未命中（所有相似度 < {SIM_THRESHOLD}），将回退到字符串匹配")
        
        return best
    
    # 1. 检查过敏（Neo4j 粗筛 + 向量精筛）
    allergies = constraints.get("allergies", [])
    if allergies:
        try:
            # 预处理过敏词
            query_terms = [a for a in allergies if isinstance(a, str) and a.strip()]

            for drug in candidate_drugs:
                print(f"[phase_b][过敏] 检查药物『{drug}』")
                precautions = kg.precautions_for_drugs([drug], k=20)
                contraindications = kg.contraindications_for_drugs([drug], k=20)
                adverse_reactions = kg.adverse_reactions_for_drugs([drug], k=20)

                safety_items: List[Tuple[str, str]] = []
                for src, arr in (("precautions", precautions), ("contraindications", contraindications), ("adverseReactions", adverse_reactions)):
                    for t in arr or []:
                        text = _extract_text(t)
                        if text:
                            safety_items.append((src, text))
                
                print(f"[phase_b][过敏] Neo4j 粗筛结果: precautions={len(precautions)}, contraindications={len(contraindications)}, adverseReactions={len(adverse_reactions)}, 提取后安全文本: {len(safety_items)}")
                
                if not safety_items or not query_terms:
                    if not safety_items:
                        print(f"[phase_b][过敏] 药物『{drug}』无安全文本，跳过")
                    continue

                # 先尝试离线索引匹配（精确名称重合）
                print(f"[phase_b][过敏] 开始 FAISS 语义匹配，安全文本数量: {len(safety_items)}")
                hit, hit_name, hit_score = (False, None, 0.0)
                try:
                    hit, hit_name, hit_score = _semantic_hit_via_index(
                        query_term=query_terms[0] if len(query_terms) == 1 else " ".join(query_terms),
                        safety_texts=[t for _, t in safety_items],
                        constraint_type="过敏"
                    )
                except Exception as e:
                    print(f"[phase_b][过敏] FAISS 匹配异常: {e}")
                    pass
                if hit:
                    # 命中即记证据
                    src_label = next((src for src, t in safety_items if t == hit_name), "precautions")
                    validation_results.append(
                        f"[E{evidence_id:04d}] 药物『{drug}』—{src_label}→{hit_name} (faiss.sim={hit_score:.2f} vs 过敏)"
                    )
                    evidence_id += 1
                    continue

                # 兜底：字符串包含（FAISS 未命中时）
                for src_label, text in safety_items:
                    for q_text in query_terms:
                        if q_text in text or any(k in text for k in ["过敏", "禁忌", "禁用"]):
                            validation_results.append(f"[E{evidence_id:04d}] 药物『{drug}』—{src_label}→{text}")
                            evidence_id += 1
                            break
        except Exception as e:
            print(f"[retrieval_v2][phase_b][warn] Failed to check allergies (semantic): {e}")
    
    # 2. 检查特殊人群状态（Neo4j 粗筛 + 向量精筛）
    status = constraints.get("status", [])
    if status:
        try:
            status_keywords = {
                "pregnant": ["孕", "怀孕", "妊娠", "孕妇", "哺乳", "哺乳期"],
                "infant": ["婴儿", "新生儿", "幼儿", "儿童", "小孩", "婴幼儿", "宝宝"],
                "elderly": ["老年", "老人", "高龄", "年长"]
            }
            query_terms = []
            for s in status:
                query_terms.extend(status_keywords.get(s, []))
            query_terms = [t for t in query_terms if t]

            for drug in candidate_drugs:
                precautions = kg.precautions_for_drugs([drug], k=20)
                contraindications = kg.contraindications_for_drugs([drug], k=20)
                safety_items: List[Tuple[str, str]] = []
                for src, arr in (("precautions", precautions), ("contraindications", contraindications)):
                    for t in arr or []:
                        text = _extract_text(t)
                        if text:
                            safety_items.append((src, text))
                if not safety_items or not query_terms:
                    continue

                # 先尝试离线索引匹配
                print(f"[phase_b][特殊人群] 检查药物『{drug}』，安全文本数量: {len(safety_items)}")
                hit, hit_name, hit_score = (False, None, 0.0)
                try:
                    hit, hit_name, hit_score = _semantic_hit_via_index(
                        query_term=" ".join(query_terms),
                        safety_texts=[t for _, t in safety_items],
                        constraint_type="特殊人群"
                    )
                except Exception as e:
                    print(f"[phase_b][特殊人群] FAISS 匹配异常: {e}")
                    pass
                if hit:
                    src_label = next((src for src, t in safety_items if t == hit_name), "precautions")
                    validation_results.append(
                        f"[E{evidence_id:04d}] 药物『{drug}』—{src_label}→{hit_name} (faiss.sim={hit_score:.2f} vs 特殊人群)"
                    )
                    evidence_id += 1
                    continue

                # 兜底：字符串包含（FAISS 未命中时）
                for src_label, text in safety_items:
                    if any(kw in text for kw in query_terms):
                        validation_results.append(f"[E{evidence_id:04d}] 药物『{drug}』—{src_label}→{text}")
                        evidence_id += 1
        except Exception as e:
            print(f"[retrieval_v2][phase_b][warn] Failed to check status (semantic): {e}")
    
    # 3. 检查既往病史（Neo4j 粗筛 + 向量精筛）
    past_history = constraints.get("past_history", [])
    if past_history:
        try:
            # 预处理既往病史文本
            history_texts: List[str] = [h for h in past_history if isinstance(h, str) and h.strip()]

            for drug in candidate_drugs:
                # Neo4j 粗筛：取该药物直连的安全文本（注意/禁忌/不良反应）
                precautions = kg.precautions_for_drugs([drug], k=20)
                contraindications = kg.contraindications_for_drugs([drug], k=20)
                adverse_reactions = kg.adverse_reactions_for_drugs([drug], k=20)

                # 整理候选安全文本
                safety_items: List[Tuple[str, str]] = []  # (src, text)
                for src, arr in (("precautions", precautions), ("contraindications", contraindications), ("adverseReactions", adverse_reactions)):
                    for t in arr or []:
                        text = _extract_text(t)
                        if text:
                            safety_items.append((src, text))
                if not safety_items or not history_texts:
                    continue

                # 语义精筛
                # 先尝试离线索引匹配
                print(f"[phase_b][既往病史] 检查药物『{drug}』，安全文本数量: {len(safety_items)}")
                hit, hit_name, hit_score = (False, None, 0.0)
                try:
                    hit, hit_name, hit_score = _semantic_hit_via_index(
                        query_term=history_texts[0] if len(history_texts) == 1 else " ".join(history_texts),
                        safety_texts=[t for _, t in safety_items],
                        constraint_type="既往病史"
                    )
                except Exception as e:
                    print(f"[phase_b][既往病史] FAISS 匹配异常: {e}")
                    pass
                if hit:
                    src_label = next((src for src, t in safety_items if t == hit_name), "contraindications")
                    validation_results.append(
                        f"[E{evidence_id:04d}] 药物『{drug}』—{src_label}→{hit_name} (faiss.sim={hit_score:.2f} vs 既往病史)"
                    )
                    evidence_id += 1
                    continue

                # 兜底：字符串包含（FAISS 未命中时）
                for src_label, text in safety_items:
                    for h_text in history_texts:
                        if h_text in text:
                            validation_results.append(
                                f"[E{evidence_id:04d}] 药物『{drug}』—{src_label}→{text} (string match vs 既往病史『{h_text}』)"
                            )
                            evidence_id += 1
                            break
        except Exception as e:
            print(f"[retrieval][phase_b][warn] Failed to check past history (semantic): {e}")
    
    # 4. 检查药物相互作用（正在服用的药物：Neo4j 粗筛 + 向量精筛）
    taking_drugs = constraints.get("taking_drugs", [])
    if taking_drugs:
        try:
            taking_drug_names = [d.get("name", "") if isinstance(d, dict) else str(d) for d in taking_drugs]
            taking_drug_names = [n for n in taking_drug_names if n]

            for drug in candidate_drugs:
                adverse_reactions = kg.adverse_reactions_for_drugs([drug], k=20)
                contraindications = kg.contraindications_for_drugs([drug], k=20)
                safety_items: List[Tuple[str, str]] = []
                for src, arr in (("adverseReactions", adverse_reactions), ("contraindications", contraindications)):
                    for t in arr or []:
                        text = _extract_text(t)
                        if text:
                            safety_items.append((src, text))
                if not safety_items or not taking_drug_names:
                    continue

                # 对每个正在服用的药物分别进行匹配（而不是拼接成一个字符串）
                print(f"[phase_b][正在服用] 检查药物『{drug}』，安全文本数量: {len(safety_items)}，正在服用药物数量: {len(taking_drug_names)}")
                
                # 遍历每个正在服用的药物，分别进行语义匹配
                for taking_drug_name in taking_drug_names:
                    hit, hit_name, hit_score = (False, None, 0.0)
                    try:
                        hit, hit_name, hit_score = _semantic_hit_via_index(
                            query_term=taking_drug_name,  # 单独匹配每个药物
                            safety_texts=[t for _, t in safety_items],
                            constraint_type="正在服用"
                        )
                    except Exception as e:
                        print(f"[phase_b][正在服用] FAISS 匹配异常（药物『{taking_drug_name}』）: {e}")
                        pass
                    
                    if hit:
                        src_label = next((src for src, t in safety_items if t == hit_name), "adverseReactions")
                        validation_results.append(
                            f"[E{evidence_id:04d}] 药物『{drug}』—{src_label}→{hit_name} (faiss.sim={hit_score:.2f} vs 正在服用『{taking_drug_name}』)"
                        )
                        evidence_id += 1
                        continue  # 找到匹配后，继续检查下一个正在服用的药物

                    # 兜底：字符串包含（FAISS 未命中时）
                    for src_label, text in safety_items:
                        if taking_drug_name in text or "相互作用" in text:
                            validation_results.append(
                                f"[E{evidence_id:04d}] 药物『{drug}』—{src_label}→{text} (string match vs 正在服用『{taking_drug_name}』)"
                            )
                            evidence_id += 1
                            break  # 找到一个匹配就跳出内层循环
        except Exception as e:
            print(f"[retrieval_v2][phase_b][warn] Failed to check drug interactions (semantic): {e}")
    
    # 5. 检查不推荐的药物
    not_recommended = constraints.get("not_recommended_drugs", [])
    if not_recommended:
        for drug in candidate_drugs:
            for nr_drug in not_recommended:
                if nr_drug in drug or drug in nr_drug:
                    # 如果候选药物中包含不推荐的药物，添加警告
                    validation_results.append(f"[E{evidence_id:04d}] [警告] 药物『{drug}』与不推荐药物『{nr_drug}』相关，需谨慎使用。")
                    evidence_id += 1
    
    if trace is not None:
        trace.append({
            "stage": "phase_b",
            "candidate_drugs": candidate_drugs,
            "constraints": constraints,
            "validation_results": validation_results
        })
    
    return validation_results

