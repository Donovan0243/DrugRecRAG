"""Cypher queries for different knowledge graphs.

中文说明：定义不同知识图谱的Cypher查询语句。
- DiseaseKB查询：针对diseasekb数据库的查询
- CMeKG查询：针对cmekg-v5.2-no-constraints数据库的查询
"""

from typing import List, Tuple, Sequence, Dict, Optional
from neo4j import Session
from difflib import SequenceMatcher
import os

# 向量检索模块（可选）
_vector_searchers = {}  # KG名称 -> VectorEntitySearch实例的缓存


def _get_vector_searcher(kg_name: str):
    """获取向量检索器（懒加载，带缓存）。
    
    Args:
        kg_name: KG名称（"cmekg" 或 "diseasekb"）
    
    Returns:
        VectorEntitySearch实例，如果索引不存在则返回None
    """
    from .config import vector_search
    
    if not vector_search.enabled:
        return None
    
    # 检查缓存
    if kg_name in _vector_searchers:
        return _vector_searchers[kg_name]
    
    # 检查索引文件是否存在
    index_path = os.path.join(vector_search.index_dir, f"{kg_name}.index")
    mapping_path = os.path.join(vector_search.index_dir, f"{kg_name}_mapping.json")
    
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        # 索引不存在，返回None（不使用向量检索）
        print(f"[query][warn] 向量检索索引文件不存在 ({kg_name}): index={os.path.exists(index_path)}, mapping={os.path.exists(mapping_path)}")
        return None
    
    try:
        from .embedding import VectorEntitySearch
        searcher = VectorEntitySearch(index_path, mapping_path)
        _vector_searchers[kg_name] = searcher
        print(f"[query][info] 向量检索器加载成功 ({kg_name}): {index_path}")
        return searcher
    except Exception as e:
        # 加载失败，记录错误但不中断
        print(f"[query][warn] 向量检索器加载失败 ({kg_name}): {e}")
        return None


# ============ 实体链接辅助函数 ============

# 同义词映射表（可扩展）
SYNONYMS = {
    "拉肚子": ["腹泻", "泻", "腹泻病"],
    "肚子疼": ["腹痛", "肚子痛", "腹部疼痛"],
    "急性胃肠炎": ["胃肠炎", "急性肠胃炎"],
    # 可以继续扩展...
}

def _similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度（0-1之间，1表示完全相同）"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _find_by_edit_distance(sess: Session, query: str, threshold: float = 0.7, allowed_labels: List[str] = None) -> Optional[Dict]:
    """使用编辑距离（相似度）匹配实体
    
    Args:
        sess: Neo4j session
        query: 查询字符串
        threshold: 相似度阈值
        allowed_labels: 允许的节点标签列表（如 ["Disease", "Symptom"]），None 表示不限制
    """
    # 先获取所有可能的候选节点（限制范围以避免性能问题）
    prefix = query[:2] if len(query) >= 2 else query  # 取前2个字符作为前缀过滤
    
    if allowed_labels:
        allowed_labels_lower = [l.lower() for l in allowed_labels]
        cypher = """
        MATCH (n)
        WHERE (toLower(n.name) CONTAINS $prefix OR toLower($query) CONTAINS toLower(n.name))
          AND ANY(label IN labels(n) WHERE toLower(label) IN $allowed_labels)
        RETURN id(n) AS id, labels(n)[0] AS label, n.name AS name
        LIMIT 100
        """
        recs = sess.run(cypher, query=query, prefix=prefix, allowed_labels=allowed_labels_lower)
    else:
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS $prefix OR toLower($query) CONTAINS toLower(n.name)
        RETURN id(n) AS id, labels(n)[0] AS label, n.name AS name
        LIMIT 100
        """
        recs = sess.run(cypher, query=query, prefix=prefix)
    
    best_match = None
    best_score = 0.0
    
    for rec in recs:
        node_name = rec["name"]
        score = _similarity(query, node_name)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = {
                "id": rec["id"],
                "label": rec["label"],
                "name": node_name,
                "score": score
            }
    
    return best_match

# ============ DiseaseKB 查询 ============

def query_diseasekb_drugs_for_diseases(sess: Session, disease_names: List[str], k: int) -> List[str]:
    """DiseaseKB: 疾病→药物（common_drug / recommand_drug）
    
    中文：查询DiseaseKB中疾病对应的药物。
    改进版：支持精确匹配 + 模糊匹配（CONTAINS），提高召回率。
    """
    if not disease_names:
        return []
    
    names = [n.strip().lower() for n in disease_names]
    all_results = []
    seen = set()
    
    # 策略1：精确匹配（优先级高）
    cypher_exact = """
    MATCH (d:Disease)-[r]->(m:Drug)
    WHERE toLower(d.name) IN $names AND type(r) IN ['common_drug','recommand_drug']
    RETURN d.name AS d, type(r) AS rel, m.name AS m
    LIMIT $k
    """
    recs_exact = list(sess.run(cypher_exact, names=names, k=k))
    for row in recs_exact:
        key = (row['d'], row['rel'], row['m'])
        if key not in seen:
            seen.add(key)
            all_results.append(row)
    
    # 策略2：如果精确匹配结果不足，尝试模糊匹配（CONTAINS）
    if len(all_results) < k:
        for name in names:
            if len(name) >= 2:  # 至少2个字符才做模糊匹配
                cypher_fuzzy = """
                MATCH (d:Disease)-[r]->(m:Drug)
                WHERE toLower(d.name) CONTAINS $q 
                  AND type(r) IN ['common_drug','recommand_drug']
                  AND NOT toLower(d.name) IN $exact_names
                RETURN d.name AS d, type(r) AS rel, m.name AS m
                LIMIT $remaining
                """
                remaining = min(k - len(all_results), k // 2)  # 限制模糊匹配数量
                if remaining > 0:
                    recs_fuzzy = sess.run(cypher_fuzzy, q=name, exact_names=names, remaining=remaining)
                    for row in recs_fuzzy:
                        key = (row['d'], row['rel'], row['m'])
                        if key not in seen and len(all_results) < k:
                            seen.add(key)
                            all_results.append(row)
    
    return [f"疾病『{row['d']}』—{row['rel']}→药物『{row['m']}』" for row in all_results[:k]]


def query_diseasekb_drugs_for_symptoms(sess: Session, symptom_names: List[str], k: int) -> List[str]:
    """DiseaseKB: 症状→药物（改进版：同时查询1跳和2跳路径）
    
    中文：查询DiseaseKB中症状相关的药物。
    - 1跳路径：症状直接治疗药物（Symptom → Drug）
    - 2跳路径：症状→疾病→药物（Symptom → Disease → Drug）
    
    合并结果以提高召回率。
    """
    if not symptom_names:
        return []
    
    names = [n.strip().lower() for n in symptom_names]
    all_results = []
    seen = set()
    
    # 策略1：1跳路径（症状直接治疗药物）- 如果KG中存在这种关系
    try:
        cypher_1hop = """
        MATCH (s:Symptom)-[r]->(m:Drug)
        WHERE toLower(s.name) IN $names 
          AND type(r) IN ['treatment', 'common_drug', 'recommand_drug']
        RETURN s.name AS s, "N/A" AS d, type(r) AS rel, m.name AS m
        LIMIT $k
        """
        recs_1hop = list(sess.run(cypher_1hop, names=names, k=k))
        for row in recs_1hop:
            key = (row['s'], row['m'])
            if key not in seen:
                seen.add(key)
                all_results.append(row)
    except Exception:
        # 如果1跳路径不存在，跳过
        pass
    
    # 策略2：2跳路径（症状→疾病→药物）
    # 改进：优先查询同时包含所有症状的疾病，避免误匹配
    if len(names) > 1:
        # 多个症状：查询同时包含所有症状的疾病（准确匹配）
        cypher_2hop_strict = """
        MATCH (d:Disease)-[r]->(m:Drug)
        WHERE type(r) IN ['common_drug','recommand_drug']
          AND ALL(symptom IN $names WHERE EXISTS {
            MATCH (s:Symptom)-[:acompany_with]->(d)
            WHERE toLower(s.name) = symptom
          })
        RETURN d.name AS d, type(r) AS rel, m.name AS m
        LIMIT $k
        """
        try:
            recs_2hop_strict = list(sess.run(cypher_2hop_strict, names=names, k=k))
            for row in recs_2hop_strict:
                key = ("combined", row['m'])  # 使用"combined"表示多症状组合
                if key not in seen and len(all_results) < k:
                    seen.add(key)
                    all_results.append({
                        's': '+'.join(names),  # 组合症状
                        'd': row['d'],
                        'rel': row['rel'],
                        'm': row['m']
                    })
        except Exception:
            # 如果严格匹配失败，降级到宽松匹配
            pass
    
    # 如果严格匹配没有结果，或者只有单个症状，使用宽松匹配（原有逻辑）
    if len(all_results) < k:
        cypher_2hop = """
        MATCH (s:Symptom)-[:acompany_with]->(d:Disease)-[r]->(m:Drug)
        WHERE toLower(s.name) IN $names 
          AND type(r) IN ['common_drug','recommand_drug']
        RETURN s.name AS s, d.name AS d, type(r) AS rel, m.name AS m
        LIMIT $k
        """
        recs_2hop = list(sess.run(cypher_2hop, names=names, k=k))
        for row in recs_2hop:
            key = (row['s'], row['m'])
            if key not in seen and len(all_results) < k:
                seen.add(key)
                all_results.append(row)
    
    # 格式化返回结果
    results = []
    for row in all_results[:k]:
        if row['d'] == "N/A":
            # 1跳路径：症状直接治疗
            results.append(f"症状『{row['s']}』—{row['rel']}→药物『{row['m']}』")
        else:
            # 2跳路径：症状→疾病→药物
            results.append(f"症状『{row['s']}』常伴疾病『{row['d']}』—{row['rel']}→药物『{row['m']}』")
    
    return results


def query_diseasekb_checks_for_diseases(sess: Session, disease_names: List[str], k: int) -> List[str]:
    """DiseaseKB: 疾病→需要检查（need_check）
    
    中文：查询DiseaseKB中疾病对应的检查项目。
    """
    if not disease_names:
        return []
    cypher = """
    MATCH (d:Disease)-[:need_check]->(c:Check)
    WHERE toLower(d.name) IN $names
    RETURN d.name AS d, c.name AS c
    LIMIT $k
    """
    names = [n.strip().lower() for n in disease_names]
    recs = sess.run(cypher, names=names, k=k)
    return [f"疾病『{row['d']}』建议检查『{row['c']}』" for row in recs]


def query_diseasekb_diet_for_diseases(sess: Session, disease_names: List[str], k: int) -> List[str]:
    """DiseaseKB: 疾病→饮食（do_eat / no_eat / recommand_eat）
    
    中文：查询DiseaseKB中疾病对应的饮食建议。
    """
    if not disease_names:
        return []
    cypher = """
    MATCH (d:Disease)-[r]->(f:Food)
    WHERE toLower(d.name) IN $names AND type(r) IN ['do_eat','no_eat','recommand_eat']
    RETURN d.name AS d, type(r) AS rel, f.name AS f
    LIMIT $k
    """
    names = [n.strip().lower() for n in disease_names]
    recs = sess.run(cypher, names=names, k=k)
    return [f"疾病『{row['d']}』—{row['rel']}→食物『{row['f']}』" for row in recs]


def query_diseasekb_drugs_for_diseases_struct(sess: Session, disease_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    """DiseaseKB: 疾病→药物（结构化三元组）"""
    if not disease_names:
        return []
    cypher = """
    MATCH (d:Disease)-[r]->(m:Drug)
    WHERE toLower(d.name) IN $names AND type(r) IN ['common_drug','recommand_drug']
    RETURN d.name AS d, type(r) AS rel, m.name AS m
    LIMIT $k
    """
    names = [n.strip().lower() for n in disease_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['d'], row['rel'], row['m']) for row in recs]


def query_diseasekb_drugs_for_symptoms_struct(sess: Session, symptom_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    """DiseaseKB: 症状→疾病→药物（结构化三元组）"""
    if not symptom_names:
        return []
    cypher = """
    MATCH (s:Symptom)-[:acompany_with]->(d:Disease)-[r]->(m:Drug)
    WHERE toLower(s.name) IN $names AND type(r) IN ['common_drug','recommand_drug']
    RETURN d.name AS d, type(r) AS rel, m.name AS m
    LIMIT $k
    """
    names = [n.strip().lower() for n in symptom_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['d'], row['rel'], row['m']) for row in recs]


def query_diseasekb_checks_for_diseases_struct(sess: Session, disease_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    """DiseaseKB: 疾病→检查（结构化三元组）"""
    if not disease_names:
        return []
    cypher = """
    MATCH (d:Disease)-[:need_check]->(c:Check)
    WHERE toLower(d.name) IN $names
    RETURN d.name AS d, 'need_check' AS rel, c.name AS c
    LIMIT $k
    """
    names = [n.strip().lower() for n in disease_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['d'], row['rel'], row['c']) for row in recs]


def query_diseasekb_diet_for_diseases_struct(sess: Session, disease_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    """DiseaseKB: 疾病→饮食（结构化三元组）"""
    if not disease_names:
        return []
    cypher = """
    MATCH (d:Disease)-[r]->(f:Food)
    WHERE toLower(d.name) IN $names AND type(r) IN ['do_eat','no_eat','recommand_eat']
    RETURN d.name AS d, type(r) AS rel, f.name AS f
    LIMIT $k
    """
    names = [n.strip().lower() for n in disease_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['d'], row['rel'], row['f']) for row in recs]


def query_diseasekb_resolve_entity(sess: Session, name: str, use_fulltext: bool = True, use_vector: bool = True, allowed_labels: List[str] = None) -> Dict:
    """DiseaseKB: 实体链接（精确/向量检索/编辑距离/全文索引匹配）
    
    改进版：先精确与向量检索，去掉同义词表。
    匹配策略：精确匹配 → 向量检索 → 编辑距离匹配 → 全文索引匹配 → 失败返回空
    
    Args:
        name: 实体名称
        use_fulltext: 是否使用全文索引
        use_vector: 是否使用向量检索
        allowed_labels: 允许的节点标签列表（如 ["Disease", "Symptom"]），None 表示不限制
    """
    q = (name or "").strip()
    if not q:
        return {}
    q_lower = q.lower()
    
    # 标准化 allowed_labels（转为小写，便于比较）
    allowed_labels_lower = None
    if allowed_labels:
        allowed_labels_lower = [l.lower() for l in allowed_labels]
        print(f"[query][diseasekb] 只匹配以下标签: {allowed_labels}")
    
    # 策略1：精确匹配（忽略大小写）- 最快，优先使用
    if allowed_labels:
        # 如果指定了 allowed_labels，在 Cypher 中过滤
        cypher_exact = """
        MATCH (n)
        WHERE toLower(n.name) = $q
          AND ANY(label IN labels(n) WHERE toLower(label) IN $allowed_labels)
        RETURN id(n) AS id, labels(n)[0] AS label, n.name AS name
        LIMIT 1
        """
        rec = sess.run(cypher_exact, q=q_lower, allowed_labels=allowed_labels_lower).single()
    else:
        cypher_exact = """
        MATCH (n)
        WHERE toLower(n.name) = $q
        RETURN id(n) AS id, labels(n)[0] AS label, n.name AS name
        LIMIT 1
        """
        rec = sess.run(cypher_exact, q=q_lower).single()
    
    if rec:
        print(f"[query][diseasekb][exact] 精确匹配成功: '{q}' -> '{rec['name']}' (label: {rec['label']})")
        return {"id": rec["id"], "label": rec["label"], "name": rec["name"]}
    else:
        print(f"[query][diseasekb][exact] 精确匹配失败: '{q}'")
    
    # 策略2：向量检索（如果启用且索引存在）- 效果好，能找到语义相似的实体
    if use_vector:
        print(f"[query][diseasekb][vector] 尝试向量检索: '{q}'")
        vector_searcher = _get_vector_searcher("diseasekb")
        if vector_searcher:
            from .config import vector_search
            candidates = vector_searcher.search(q, topk=vector_search.topk, threshold=vector_search.threshold)
            if not candidates:
                print(f"[query][diseasekb][vector] 未找到超过阈值({vector_search.threshold})的候选")
            if candidates:
                # 如果指定了 allowed_labels，过滤候选
                if allowed_labels_lower:
                    filtered_candidates = [
                        c for c in candidates 
                        if c.get("label", "").lower() in allowed_labels_lower
                    ]
                    if filtered_candidates:
                        candidates = filtered_candidates
                        print(f"[query][diseasekb][vector] 过滤后剩余 {len(candidates)} 个候选（只保留 {allowed_labels} 类型）")
                    else:
                        print(f"[query][diseasekb][vector] 所有候选都不在允许的标签中，跳过向量检索")
                        candidates = []
                
                if candidates:
                    # 调试日志：显示向量检索结果
                    print(f"[query][diseasekb][vector] 查询'{q}' -> 找到{len(candidates)}个候选，最佳: {candidates[0]['name']} (相似度: {candidates[0].get('score', candidates[0].get('similarity', 'N/A')):.4f}, label: {candidates[0].get('label', 'N/A')})")
                    best = candidates[0]
                    # 再次过滤人群关键词
                    matched_name = best["name"]
                    population_keywords = ["孕", "婴", "新生儿", "儿童", "老年", "老人"]
                    query_has_pop = any(kw in q for kw in population_keywords)
                    match_has_pop = any(kw in matched_name for kw in population_keywords)
                    
                    if not query_has_pop and match_has_pop:
                        # 查询不涉及人群，但匹配结果涉及人群，尝试下一个候选
                        if len(candidates) > 1:
                            best = candidates[1]
                            matched_name = best["name"]
                            match_has_pop = any(kw in matched_name for kw in population_keywords)
                            if not query_has_pop and match_has_pop:
                                # 如果下一个候选也有问题，跳过向量检索，继续其他策略
                                pass
                            else:
                                return {"id": best["id"], "label": best["label"], "name": best["name"]}
                    else:
                        # 人群关键词匹配，或没有人群关键词问题，直接返回
                        return {"id": best["id"], "label": best["label"], "name": best["name"]}
    
    # 策略3：编辑距离匹配（相似度阈值 >= 0.7）
    # 同时过滤掉包含人群关键词的节点（除非查询本身包含这些词）
    match = _find_by_edit_distance(sess, q, threshold=0.7, allowed_labels=allowed_labels)
    if match:
        print(f"[query][cmekg][edit_distance] 编辑距离匹配成功: '{q}' -> '{match['name']}' (相似度: {match.get('score', 'N/A')}, label: {match.get('label', 'N/A')})")
        matched_name = match["name"]
        # 过滤人群关键词：如果查询不包含"孕/婴/新生儿/儿童/老年"，则拒绝包含这些词的匹配
        population_keywords = ["孕", "婴", "新生儿", "儿童", "老年", "老人"]
        query_has_pop = any(kw in q for kw in population_keywords)
        match_has_pop = any(kw in matched_name for kw in population_keywords)
        
        if not query_has_pop and match_has_pop:
            # 查询不涉及人群，但匹配结果涉及人群，拒绝这个匹配
            return {}
        
        return {"id": match["id"], "label": match["label"], "name": match["name"]}
    
    # 策略4：全文索引匹配（作为最后备选，仅在启用时使用）
    if use_fulltext:
        try:
            # 根据 allowed_labels 决定查询哪些索引
            indices_to_try = []
            if not allowed_labels_lower:
                # 没有限制，尝试所有索引
                indices_to_try = [
                    ("disease_fulltext", "Disease"),
                    ("symptom_fulltext", "Symptom"),
                    ("drug_fulltext", "Drug")
                ]
            else:
                # 只查询允许的标签对应的索引
                if "disease" in allowed_labels_lower:
                    indices_to_try.append(("disease_fulltext", "Disease"))
                if "symptom" in allowed_labels_lower:
                    indices_to_try.append(("symptom_fulltext", "Symptom"))
                if "drug" in allowed_labels_lower:
                    indices_to_try.append(("drug_fulltext", "Drug"))
            
            for index_name, label_type in indices_to_try:
                cypher_fulltext = f"""
                CALL db.index.fulltext.queryNodes('{index_name}', $query)
                YIELD node, score
                WHERE score > 0.5
                RETURN id(node) AS id, labels(node)[0] AS label, node.name AS name, score
                ORDER BY score DESC
                LIMIT 1
                """
                rec = sess.run(cypher_fulltext, query=q).single()
                if rec:
                    # 再次过滤人群关键词
                    matched_name = rec["name"]
                    population_keywords = ["孕", "婴", "新生儿", "儿童", "老年", "老人"]
                    query_has_pop = any(kw in q for kw in population_keywords)
                    match_has_pop = any(kw in matched_name for kw in population_keywords)
                    
                    if not query_has_pop and match_has_pop:
                        continue  # 跳过这个匹配，尝试下一个索引
                    
                    return {"id": rec["id"], "label": rec["label"], "name": rec["name"]}
        except Exception:
            # 全文索引可能不存在，忽略错误
            pass
    
    # 所有策略都失败，返回空（不强制映射）
    return {}


# ============ CMeKG 查询 ============

def query_cmekg_drugs_for_diseases(sess: Session, disease_names: List[str], k: int) -> List[str]:
    """CMeKG: 疾病→药物（drugTherapy / treatment）
    
    中文：查询CMeKG中疾病对应的药物。CMeKG使用drugTherapy或treatment关系。
    改进版：支持精确匹配 + 模糊匹配（CONTAINS），提高召回率。
    """
    if not disease_names:
        return []
    
    names = [n.strip().lower() for n in disease_names]
    all_results = []
    seen = set()
    
    # 策略1：精确匹配（优先级高）
    cypher_exact = """
    MATCH (d:Disease)-[r]->(m:Drug)
    WHERE toLower(d.name) IN $names 
      AND type(r) IN ['drugTherapy', 'treatment', 'Treatment']
    RETURN d.name AS d, type(r) AS rel, m.name AS m
    LIMIT $k
    """
    recs_exact = list(sess.run(cypher_exact, names=names, k=k))
    for row in recs_exact:
        key = (row['d'], row['rel'], row['m'])
        if key not in seen:
            seen.add(key)
            all_results.append(row)
    
    # 策略2：如果精确匹配结果不足，尝试模糊匹配（CONTAINS）
    if len(all_results) < k:
        for name in names:
            if len(name) >= 2:  # 至少2个字符才做模糊匹配
                cypher_fuzzy = """
                MATCH (d:Disease)-[r]->(m:Drug)
                WHERE toLower(d.name) CONTAINS $q 
                  AND type(r) IN ['drugTherapy', 'treatment', 'Treatment']
                  AND NOT toLower(d.name) IN $exact_names
                RETURN d.name AS d, type(r) AS rel, m.name AS m
                LIMIT $remaining
                """
                remaining = min(k - len(all_results), k // 2)  # 限制模糊匹配数量
                if remaining > 0:
                    recs_fuzzy = sess.run(cypher_fuzzy, q=name, exact_names=names, remaining=remaining)
                    for row in recs_fuzzy:
                        key = (row['d'], row['rel'], row['m'])
                        if key not in seen and len(all_results) < k:
                            seen.add(key)
                            all_results.append(row)
    
    return [f"疾病『{row['d']}』—{row['rel']}→药物『{row['m']}』" for row in all_results[:k]]


def query_cmekg_drugs_for_symptoms(sess: Session, symptom_names: List[str], k: int) -> List[str]:
    """CMeKG: 症状→药物（改进版：同时查询1跳和2跳路径）
    
    中文：查询CMeKG中症状相关的药物。
    - 1跳路径：症状直接治疗药物（Symptom → Drug）
    - 2跳路径：症状→疾病→药物（Symptom → Disease → Drug）
    
    合并结果以提高召回率。
    """
    if not symptom_names:
        return []
    
    names = [n.strip().lower() for n in symptom_names]
    all_results = []
    seen = set()
    
    # 策略1：1跳路径（症状直接治疗药物）- 如果KG中存在这种关系
    try:
        cypher_1hop = """
        MATCH (s:Symptom)-[r]->(m:Drug)
        WHERE toLower(s.name) IN $names 
          AND type(r) IN ['drugTherapy', 'treatment', 'Treatment']
        RETURN s.name AS s, "N/A" AS d, type(r) AS rel, m.name AS m
        LIMIT $k
        """
        recs_1hop = list(sess.run(cypher_1hop, names=names, k=k))
        for row in recs_1hop:
            key = (row['s'], row['m'])
            if key not in seen:
                seen.add(key)
                all_results.append(row)
    except Exception:
        # 如果1跳路径不存在，跳过
        pass
    
    # 策略2：2跳路径（症状→疾病→药物）
    # 改进：优先查询同时包含所有症状的疾病，避免误匹配
    if len(names) > 1:
        # 多个症状：查询同时包含所有症状的疾病（准确匹配）
        cypher_2hop_strict = """
        MATCH (d:Disease)-[r2]->(m:Drug)
        WHERE type(r2) IN ['drugTherapy', 'treatment', 'Treatment']
          AND ALL(symptom IN $names WHERE EXISTS {
            MATCH (s:Symptom)-[r1]->(d)
            WHERE toLower(s.name) = symptom
              AND type(r1) IN ['relatedDisease', 'relatedSymptom', 'relatedTo']
          })
        RETURN d.name AS d, type(r2) AS rel, m.name AS m
        LIMIT $k
        """
        try:
            recs_2hop_strict = list(sess.run(cypher_2hop_strict, names=names, k=k))
            for row in recs_2hop_strict:
                key = ("combined", row['m'])  # 使用"combined"表示多症状组合
                if key not in seen and len(all_results) < k:
                    seen.add(key)
                    all_results.append({
                        's': '+'.join(names),  # 组合症状
                        'd': row['d'],
                        'rel': row['rel'],
                        'm': row['m']
                    })
        except Exception:
            # 如果严格匹配失败，降级到宽松匹配
            pass
    
    # 如果严格匹配没有结果，或者只有单个症状，使用宽松匹配（原有逻辑）
    if len(all_results) < k:
        cypher_2hop = """
        MATCH (s:Symptom)-[r1]->(d:Disease)-[r2]->(m:Drug)
        WHERE toLower(s.name) IN $names 
          AND type(r1) IN ['relatedDisease', 'relatedSymptom', 'relatedTo']
          AND type(r2) IN ['drugTherapy', 'treatment', 'Treatment']
        RETURN s.name AS s, d.name AS d, type(r2) AS rel, m.name AS m
        LIMIT $k
        """
        recs_2hop = list(sess.run(cypher_2hop, names=names, k=k))
        for row in recs_2hop:
            key = (row['s'], row['m'])
            if key not in seen and len(all_results) < k:
                seen.add(key)
                all_results.append(row)
    
    # 格式化返回结果
    results = []
    for row in all_results[:k]:
        if row['d'] == "N/A":
            # 1跳路径：症状直接治疗
            results.append(f"症状『{row['s']}』—{row['rel']}→药物『{row['m']}』")
        else:
            # 2跳路径：症状→疾病→药物
            results.append(f"症状『{row['s']}』相关疾病『{row['d']}』—{row['rel']}→药物『{row['m']}』")
    
    return results


def query_cmekg_checks_for_diseases(sess: Session, disease_names: List[str], k: int) -> List[str]:
    """CMeKG: 疾病→需要检查（check / auxiliaryExamination）
    
    中文：查询CMeKG中疾病对应的检查项目。CMeKG使用check或auxiliaryExamination关系。
    """
    if not disease_names:
        return []
    cypher = """
    MATCH (d:Disease)-[r]->(c)
    WHERE toLower(d.name) IN $names 
      AND type(r) IN ['check', 'auxiliaryExamination', 'Check', 'AuxiliaryExamination']
      AND (c:Check OR c:AuxiliaryExamination OR c:CheckSubject)
    RETURN d.name AS d, c.name AS c
    LIMIT $k
    """
    names = [n.strip().lower() for n in disease_names]
    recs = sess.run(cypher, names=names, k=k)
    return [f"疾病『{row['d']}』建议检查『{row['c']}』" for row in recs]


def query_cmekg_diet_for_diseases(sess: Session, disease_names: List[str], k: int) -> List[str]:
    """CMeKG: 疾病→饮食（如果有Food节点）
    
    中文：查询CMeKG中疾病对应的饮食建议。CMeKG可能没有Food节点，此查询可能返回空。
    """
    if not disease_names:
        return []
    # 中文：CMeKG可能没有专门的Food节点，先尝试查找
    cypher = """
    MATCH (d:Disease)-[r]->(f)
    WHERE toLower(d.name) IN $names 
      AND (f:Food OR type(r) IN ['diet', 'recommand_eat', 'do_eat', 'no_eat'])
    RETURN d.name AS d, type(r) AS rel, f.name AS f
    LIMIT $k
    """
    names = [n.strip().lower() for n in disease_names]
    try:
        recs = sess.run(cypher, names=names, k=k)
        return [f"疾病『{row['d']}』—{row['rel']}→食物『{row['f']}』" for row in recs]
    except Exception:
        # 中文：如果Food节点不存在，返回空列表
        return []


def query_cmekg_drugs_for_diseases_struct(sess: Session, disease_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    """CMeKG: 疾病→药物（结构化三元组）"""
    if not disease_names:
        return []
    cypher = """
    MATCH (d:Disease)-[r]->(m:Drug)
    WHERE toLower(d.name) IN $names 
      AND type(r) IN ['drugTherapy', 'treatment', 'Treatment', 'drugTherapy']
    RETURN d.name AS d, type(r) AS rel, m.name AS m
    LIMIT $k
    """
    names = [n.strip().lower() for n in disease_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['d'], row['rel'], row['m']) for row in recs]


def query_cmekg_drugs_for_symptoms_struct(sess: Session, symptom_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    """CMeKG: 症状→疾病→药物（结构化三元组）"""
    if not symptom_names:
        return []
    cypher = """
    MATCH (s:Symptom)-[r1]->(d:Disease)-[r2]->(m:Drug)
    WHERE toLower(s.name) IN $names 
      AND type(r1) IN ['relatedDisease', 'relatedSymptom', 'relatedTo']
      AND type(r2) IN ['drugTherapy', 'treatment', 'Treatment', 'drugTherapy']
    RETURN d.name AS d, type(r2) AS rel, m.name AS m
    LIMIT $k
    """
    names = [n.strip().lower() for n in symptom_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['d'], row['rel'], row['m']) for row in recs]


def query_cmekg_checks_for_diseases_struct(sess: Session, disease_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    """CMeKG: 疾病→检查（结构化三元组）"""
    if not disease_names:
        return []
    cypher = """
    MATCH (d:Disease)-[r]->(c)
    WHERE toLower(d.name) IN $names 
      AND type(r) IN ['check', 'auxiliaryExamination', 'Check', 'AuxiliaryExamination']
      AND (c:Check OR c:AuxiliaryExamination OR c:CheckSubject)
    RETURN d.name AS d, type(r) AS rel, c.name AS c
    LIMIT $k
    """
    names = [n.strip().lower() for n in disease_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['d'], row['rel'], row['c']) for row in recs]


def query_cmekg_diet_for_diseases_struct(sess: Session, disease_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    """CMeKG: 疾病→饮食（结构化三元组）"""
    if not disease_names:
        return []
    cypher = """
    MATCH (d:Disease)-[r]->(f)
    WHERE toLower(d.name) IN $names 
      AND (f:Food OR type(r) IN ['diet', 'recommand_eat', 'do_eat', 'no_eat'])
    RETURN d.name AS d, type(r) AS rel, f.name AS f
    LIMIT $k
    """
    names = [n.strip().lower() for n in disease_names]
    try:
        recs = sess.run(cypher, names=names, k=k)
        return [(row['d'], row['rel'], row['f']) for row in recs]
    except Exception:
        return []


def query_cmekg_resolve_entity(sess: Session, name: str, use_fulltext: bool = True, use_vector: bool = True, allowed_labels: List[str] = None) -> Dict:
    """CMeKG: 实体链接（精确/向量检索/编辑距离/全文索引匹配）
    
    改进版：先精确与向量检索，去掉同义词表。
    匹配策略：精确匹配 → 向量检索 → 编辑距离匹配 → 全文索引匹配 → 失败返回空
    
    Args:
        name: 实体名称
        use_fulltext: 是否使用全文索引
        use_vector: 是否使用向量检索
        allowed_labels: 允许的节点标签列表（如 ["Disease", "Symptom"]），None 表示不限制
    """
    q = (name or "").strip()
    if not q:
        return {}
    q_lower = q.lower()
    
    # 标准化 allowed_labels（转为小写，便于比较）
    allowed_labels_lower = None
    if allowed_labels:
        allowed_labels_lower = [l.lower() for l in allowed_labels]
        print(f"[query][cmekg] 只匹配以下标签: {allowed_labels}")
    
    # 策略1：精确匹配（忽略大小写）- 最快，优先使用
    if allowed_labels:
        # 如果指定了 allowed_labels，在 Cypher 中过滤
        cypher_exact = """
        MATCH (n)
        WHERE toLower(n.name) = $q
          AND ANY(label IN labels(n) WHERE toLower(label) IN $allowed_labels)
        RETURN id(n) AS id, labels(n)[0] AS label, n.name AS name
        LIMIT 1
        """
        rec = sess.run(cypher_exact, q=q_lower, allowed_labels=allowed_labels_lower).single()
    else:
        cypher_exact = """
        MATCH (n)
        WHERE toLower(n.name) = $q
        RETURN id(n) AS id, labels(n)[0] AS label, n.name AS name
        LIMIT 1
        """
        rec = sess.run(cypher_exact, q=q_lower).single()
    
    if rec:
        print(f"[query][cmekg][exact] 精确匹配成功: '{q}' -> '{rec['name']}' (label: {rec['label']})")
        return {"id": rec["id"], "label": rec["label"], "name": rec["name"]}
    else:
        print(f"[query][cmekg][exact] 精确匹配失败: '{q}'")
    
    # 策略2：向量检索（如果启用且索引存在）- 效果好，能找到语义相似的实体
    if use_vector:
        print(f"[query][cmekg][vector] 尝试向量检索: '{q}'")
        vector_searcher = _get_vector_searcher("cmekg")
        if vector_searcher:
            from .config import vector_search
            candidates = vector_searcher.search(q, topk=vector_search.topk, threshold=vector_search.threshold)
            if not candidates:
                print(f"[query][cmekg][vector] 未找到超过阈值({vector_search.threshold})的候选")
            if candidates:
                # 如果指定了 allowed_labels，过滤候选
                if allowed_labels_lower:
                    filtered_candidates = [
                        c for c in candidates 
                        if c.get("label", "").lower() in allowed_labels_lower
                    ]
                    if filtered_candidates:
                        candidates = filtered_candidates
                        print(f"[query][cmekg][vector] 过滤后剩余 {len(candidates)} 个候选（只保留 {allowed_labels} 类型）")
                    else:
                        print(f"[query][cmekg][vector] 所有候选都不在允许的标签中，跳过向量检索")
                        candidates = []
                
                if candidates:
                    # 调试日志：显示向量检索结果
                    print(f"[query][cmekg][vector] 查询'{q}' -> 找到{len(candidates)}个候选，最佳: {candidates[0]['name']} (相似度: {candidates[0].get('score', candidates[0].get('similarity', 'N/A')):.4f}, label: {candidates[0].get('label', 'N/A')})")
                    best = candidates[0]
                    # 再次过滤人群关键词
                    matched_name = best["name"]
                    population_keywords = ["孕", "婴", "新生儿", "儿童", "老年", "老人"]
                    query_has_pop = any(kw in q for kw in population_keywords)
                    match_has_pop = any(kw in matched_name for kw in population_keywords)
                    
                    if not query_has_pop and match_has_pop:
                        # 查询不涉及人群，但匹配结果涉及人群，尝试下一个候选
                        if len(candidates) > 1:
                            best = candidates[1]
                            matched_name = best["name"]
                            match_has_pop = any(kw in matched_name for kw in population_keywords)
                            if not query_has_pop and match_has_pop:
                                # 如果下一个候选也有问题，跳过向量检索，继续其他策略
                                pass
                            else:
                                return {"id": best["id"], "label": best["label"], "name": best["name"]}
                    else:
                        # 人群关键词匹配，或没有人群关键词问题，直接返回
                        return {"id": best["id"], "label": best["label"], "name": best["name"]}
    
    # 策略3：编辑距离匹配（相似度阈值 >= 0.7）
    # 同时过滤掉包含人群关键词的节点（除非查询本身包含这些词）
    match = _find_by_edit_distance(sess, q, threshold=0.7, allowed_labels=allowed_labels)
    if match:
        print(f"[query][cmekg][edit_distance] 编辑距离匹配成功: '{q}' -> '{match['name']}' (相似度: {match.get('score', 'N/A')}, label: {match.get('label', 'N/A')})")
        matched_name = match["name"]
        # 过滤人群关键词：如果查询不包含"孕/婴/新生儿/儿童/老年"，则拒绝包含这些词的匹配
        population_keywords = ["孕", "婴", "新生儿", "儿童", "老年", "老人"]
        query_has_pop = any(kw in q for kw in population_keywords)
        match_has_pop = any(kw in matched_name for kw in population_keywords)
        
        if not query_has_pop and match_has_pop:
            # 查询不涉及人群，但匹配结果涉及人群，拒绝这个匹配
            return {}
        
        return {"id": match["id"], "label": match["label"], "name": match["name"]}
    
    # 策略4：全文索引匹配（作为最后备选，仅在启用时使用）
    if use_fulltext:
        try:
            # 根据 allowed_labels 决定查询哪些索引
            indices_to_try = []
            if not allowed_labels_lower:
                # 没有限制，尝试所有索引
                indices_to_try = [
                    ("disease_fulltext", "Disease"),
                    ("symptom_fulltext", "Symptom"),
                    ("drug_fulltext", "Drug")
                ]
            else:
                # 只查询允许的标签对应的索引
                if "disease" in allowed_labels_lower:
                    indices_to_try.append(("disease_fulltext", "Disease"))
                if "symptom" in allowed_labels_lower:
                    indices_to_try.append(("symptom_fulltext", "Symptom"))
                if "drug" in allowed_labels_lower:
                    indices_to_try.append(("drug_fulltext", "Drug"))
            
            for index_name, label_type in indices_to_try:
                cypher_fulltext = f"""
                CALL db.index.fulltext.queryNodes('{index_name}', $query)
                YIELD node, score
                WHERE score > 0.5
                RETURN id(node) AS id, labels(node)[0] AS label, node.name AS name, score
                ORDER BY score DESC
                LIMIT 1
                """
                rec = sess.run(cypher_fulltext, query=q).single()
                if rec:
                    # 再次过滤人群关键词
                    matched_name = rec["name"]
                    population_keywords = ["孕", "婴", "新生儿", "儿童", "老年", "老人"]
                    query_has_pop = any(kw in q for kw in population_keywords)
                    match_has_pop = any(kw in matched_name for kw in population_keywords)
                    
                    if not query_has_pop and match_has_pop:
                        continue  # 跳过这个匹配，尝试下一个索引
                    
                    return {"id": rec["id"], "label": rec["label"], "name": rec["name"]}
        except Exception:
            # 全文索引可能不存在，忽略错误
            pass
    
    # 所有策略都失败，返回空（不强制映射）
    return {}

# ============ CMeKG 安全相关查询（新增） ============

def query_cmekg_contraindications_for_drugs(sess: Session, drug_names: List[str], k: int) -> List[str]:
    """CMeKG: 药物 → 禁忌（contraindications）

    中文：查询药物的禁忌信息（对象可能为 Subject/MultipleGroups/Indications 等）。
    """
    if not drug_names:
        return []
    cypher = """
    MATCH (m:Drug)-[:contraindications]->(x)
    WHERE toLower(m.name) IN $names
    RETURN m.name AS m, labels(x) AS x_labels, x.name AS x_name
    LIMIT $k
    """
    names = [n.strip().lower() for n in drug_names]
    recs = sess.run(cypher, names=names, k=k)
    out: List[str] = []
    for row in recs:
        x_label = row["x_labels"][0] if row["x_labels"] else "Entity"
        out.append(f"药物『{row['m']}』—contraindications→{x_label}『{row.get('x_name') or ''}』")
    return out


def query_cmekg_contraindications_for_drugs_struct(sess: Session, drug_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    if not drug_names:
        return []
    cypher = """
    MATCH (m:Drug)-[:contraindications]->(x)
    WHERE toLower(m.name) IN $names
    RETURN m.name AS m, 'contraindications' AS rel, x.name AS x
    LIMIT $k
    """
    names = [n.strip().lower() for n in drug_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['m'], row['rel'], row.get('x') or '') for row in recs]


def query_cmekg_adverse_reactions_for_drugs(sess: Session, drug_names: List[str], k: int) -> List[str]:
    """CMeKG: 药物 → 不良反应（adverseReactions）"""
    if not drug_names:
        return []
    cypher = """
    MATCH (m:Drug)-[:adverseReactions]->(ar:AdverseReactions)
    WHERE toLower(m.name) IN $names
    RETURN m.name AS m, ar.name AS ar
    LIMIT $k
    """
    names = [n.strip().lower() for n in drug_names]
    recs = sess.run(cypher, names=names, k=k)
    return [f"药物『{row['m']}』—adverseReactions→『{row.get('ar') or ''}』" for row in recs]


def query_cmekg_adverse_reactions_for_drugs_struct(sess: Session, drug_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    if not drug_names:
        return []
    cypher = """
    MATCH (m:Drug)-[:adverseReactions]->(ar:AdverseReactions)
    WHERE toLower(m.name) IN $names
    RETURN m.name AS m, 'adverseReactions' AS rel, ar.name AS ar
    LIMIT $k
    """
    names = [n.strip().lower() for n in drug_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['m'], row['rel'], row.get('ar') or '') for row in recs]


def query_cmekg_indications_for_drugs(sess: Session, drug_names: List[str], k: int) -> List[str]:
    """CMeKG: 药物 → 适应证（indications）"""
    if not drug_names:
        return []
    cypher = """
    MATCH (m:Drug)-[:indications]->(ind:Indications)
    WHERE toLower(m.name) IN $names
    RETURN m.name AS m, ind.name AS ind
    LIMIT $k
    """
    names = [n.strip().lower() for n in drug_names]
    recs = sess.run(cypher, names=names, k=k)
    return [f"药物『{row['m']}』—indications→『{row.get('ind') or ''}』" for row in recs]


def query_cmekg_indications_for_drugs_struct(sess: Session, drug_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    if not drug_names:
        return []
    cypher = """
    MATCH (m:Drug)-[:indications]->(ind:Indications)
    WHERE toLower(m.name) IN $names
    RETURN m.name AS m, 'indications' AS rel, ind.name AS ind
    LIMIT $k
    """
    names = [n.strip().lower() for n in drug_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['m'], row['rel'], row.get('ind') or '') for row in recs]


def query_cmekg_precautions_for_drugs(sess: Session, drug_names: List[str], k: int) -> List[str]:
    """CMeKG: 药物 → 注意事项（precautions）"""
    if not drug_names:
        return []
    cypher = """
    MATCH (m:Drug)-[:precautions]->(p:Precautions)
    WHERE toLower(m.name) IN $names
    RETURN m.name AS m, p.name AS p
    LIMIT $k
    """
    names = [n.strip().lower() for n in drug_names]
    recs = sess.run(cypher, names=names, k=k)
    return [f"药物『{row['m']}』—precautions→『{row.get('p') or ''}』" for row in recs]


def query_cmekg_precautions_for_drugs_struct(sess: Session, drug_names: List[str], k: int) -> List[Tuple[str, str, str]]:
    if not drug_names:
        return []
    cypher = """
    MATCH (m:Drug)-[:precautions]->(p:Precautions)
    WHERE toLower(m.name) IN $names
    RETURN m.name AS m, 'precautions' AS rel, p.name AS p
    LIMIT $k
    """
    names = [n.strip().lower() for n in drug_names]
    recs = sess.run(cypher, names=names, k=k)
    return [(row['m'], row['rel'], row.get('p') or '') for row in recs]

