"""Lightweight KG loader and query helpers.

中文说明：提供最小的知识图谱数据结构与查询函数，支持 NP/PP 所需操作。
"""

from typing import Dict, List, Tuple, Iterable, Sequence
from contextlib import contextmanager
import os
try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None  # 中文：没有驱动时允许导入通过，运行时报错提示安装

# 导入查询函数
try:
    from . import query
except Exception:
    query = None  # 中文：查询模块导入失败时的回退


class _BaseNeo4jKG:
    """公共的 Neo4j 连接与会话管理基类。

    中文：仅封装连接/会话，具体查询由子类（DiseaseKBKG/CMeKGKG）调用 query.py。
    """

    def __init__(self, uri: str, user: str, password: str, database: str = None):
        if GraphDatabase is None:
            raise RuntimeError("neo4j driver not installed. Please `pip install neo4j`.")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self):
        self._driver.close()

    @contextmanager
    def _session(self):
        if self._database:
            with self._driver.session(database=self._database) as sess:
                yield sess
        else:
            with self._driver.session() as sess:
                yield sess

    def topk_relation_facts(self, nodes: Sequence[str], relation: str, k: int) -> List[str]:
        # 中文：在 Neo4j 中执行关系检索，返回 (s)-[:REL]->(o) 的文本表示
        if not nodes:
            return []
        cypher = f"""
        MATCH (s)-[r:`{relation}`]->(o)
        WHERE toLower(s.name) IN $names
        RETURN s.name AS s, type(r) AS p, o.name AS o
        LIMIT $k
        """
        names = [n.strip().lower() for n in nodes]
        with self._session() as sess:
            recs = sess.run(cypher, names=names, k=k)
            results = []
            seen = set()
            for row in recs:
                text = f"({row['s']}, {row['p']}, {row['o']})"
                if text not in seen:
                    seen.add(text)
                    results.append(text)
            return results

    def ensure_fulltext_indexes(self):
        """确保全文索引存在（用于近似匹配）。
        
        中文：为Disease、Symptom、Drug节点创建全文索引（如果不存在），
        用于后续的近似匹配查询。
        """
        with self._session() as sess:
            # 检查并创建Disease节点的全文索引
            try:
                sess.run("""
                CALL db.index.fulltext.createNodeIndex(
                    'disease_fulltext',
                    ['Disease'],
                    ['name']
                )
                """)
            except Exception:
                # 索引可能已存在，忽略
                pass
            
            # 检查并创建Symptom节点的全文索引
            try:
                sess.run("""
                CALL db.index.fulltext.createNodeIndex(
                    'symptom_fulltext',
                    ['Symptom'],
                    ['name']
                )
                """)
            except Exception:
                pass
            
            # 检查并创建Drug节点的全文索引
            try:
                sess.run("""
                CALL db.index.fulltext.createNodeIndex(
                    'drug_fulltext',
                    ['Drug'],
                    ['name']
                )
                """)
            except Exception:
                pass

class DiseaseKBKG(_BaseNeo4jKG):
    """DiseaseKB 单库适配。

    中文：对接 diseasekb 数据库，所有查询使用 query.py 中的 DiseaseKB 版本。
    """

    def drugs_for_diseases(self, disease_names: Sequence[str], k: int) -> List[str]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_diseasekb_drugs_for_diseases(sess, list(disease_names), k) if query else []

    def drugs_for_symptoms(self, symptom_names: Sequence[str], k: int) -> List[str]:
        if not symptom_names:
            return []
        with self._session() as sess:
            return query.query_diseasekb_drugs_for_symptoms(sess, list(symptom_names), k) if query else []

    def checks_for_diseases(self, disease_names: Sequence[str], k: int) -> List[str]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_diseasekb_checks_for_diseases(sess, list(disease_names), k) if query else []

    def diet_for_diseases(self, disease_names: Sequence[str], k: int) -> List[str]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_diseasekb_diet_for_diseases(sess, list(disease_names), k) if query else []

    def resolve_entity(self, name: str) -> Dict:
        with self._session() as sess:
            return query.query_diseasekb_resolve_entity(sess, name) if query else {}

    # 结构化三元组
    def drugs_for_diseases_struct(self, disease_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_diseasekb_drugs_for_diseases_struct(sess, list(disease_names), k) if query else []

    def drugs_for_symptoms_struct(self, symptom_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not symptom_names:
            return []
        with self._session() as sess:
            return query.query_diseasekb_drugs_for_symptoms_struct(sess, list(symptom_names), k) if query else []

    def checks_for_diseases_struct(self, disease_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_diseasekb_checks_for_diseases_struct(sess, list(disease_names), k) if query else []

    def diet_for_diseases_struct(self, disease_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_diseasekb_diet_for_diseases_struct(sess, list(disease_names), k) if query else []


class CMeKGKG(_BaseNeo4jKG):
    """CMeKG 单库适配。

    中文：对接 cmekg-v5.2-no-constraints 数据库，所有查询使用 query.py 中的 CMeKG 版本。
    """

    def drugs_for_diseases(self, disease_names: Sequence[str], k: int) -> List[str]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_drugs_for_diseases(sess, list(disease_names), k) if query else []

    def drugs_for_symptoms(self, symptom_names: Sequence[str], k: int) -> List[str]:
        if not symptom_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_drugs_for_symptoms(sess, list(symptom_names), k) if query else []

    def checks_for_diseases(self, disease_names: Sequence[str], k: int) -> List[str]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_checks_for_diseases(sess, list(disease_names), k) if query else []

    def diet_for_diseases(self, disease_names: Sequence[str], k: int) -> List[str]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_diet_for_diseases(sess, list(disease_names), k) if query else []

    def resolve_entity(self, name: str) -> Dict:
        with self._session() as sess:
            return query.query_cmekg_resolve_entity(sess, name) if query else {}

    # 结构化三元组
    def drugs_for_diseases_struct(self, disease_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_drugs_for_diseases_struct(sess, list(disease_names), k) if query else []

    def drugs_for_symptoms_struct(self, symptom_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not symptom_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_drugs_for_symptoms_struct(sess, list(symptom_names), k) if query else []

    def checks_for_diseases_struct(self, disease_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_checks_for_diseases_struct(sess, list(disease_names), k) if query else []

    def diet_for_diseases_struct(self, disease_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not disease_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_diet_for_diseases_struct(sess, list(disease_names), k) if query else []

    # 安全相关
    def contraindications_for_drugs(self, drug_names: Sequence[str], k: int) -> List[str]:
        if not drug_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_contraindications_for_drugs(sess, list(drug_names), k) if query else []

    def contraindications_for_drugs_struct(self, drug_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not drug_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_contraindications_for_drugs_struct(sess, list(drug_names), k) if query else []

    def adverse_reactions_for_drugs(self, drug_names: Sequence[str], k: int) -> List[str]:
        if not drug_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_adverse_reactions_for_drugs(sess, list(drug_names), k) if query else []

    def adverse_reactions_for_drugs_struct(self, drug_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not drug_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_adverse_reactions_for_drugs_struct(sess, list(drug_names), k) if query else []

    def indications_for_drugs(self, drug_names: Sequence[str], k: int) -> List[str]:
        if not drug_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_indications_for_drugs(sess, list(drug_names), k) if query else []

    def indications_for_drugs_struct(self, drug_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not drug_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_indications_for_drugs_struct(sess, list(drug_names), k) if query else []

    def precautions_for_drugs(self, drug_names: Sequence[str], k: int) -> List[str]:
        if not drug_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_precautions_for_drugs(sess, list(drug_names), k) if query else []

    def precautions_for_drugs_struct(self, drug_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        if not drug_names:
            return []
        with self._session() as sess:
            return query.query_cmekg_precautions_for_drugs_struct(sess, list(drug_names), k) if query else []


class MultiKG:
    """Multi-KG adapter that queries multiple Neo4j databases and merges results.
    
    中文：多KG适配器，支持同时查询多个Neo4j数据库并合并结果。
    支持策略：union（合并）、primary_fallback（主+回退）、cmekg_only、diseasekb_only
    """
    
    def __init__(self, kgs: Dict[str, _BaseNeo4jKG], strategy: str = "union"):
        """
        Args:
            kgs: 字典，键为KG名称（如'cmekg', 'diseasekb'），值为单库适配实例（CMeKGKG/DiseaseKBKG）
            strategy: 集成策略，'union'（合并）、'primary_fallback'（主+回退）、'cmekg_only'、'diseasekb_only'
        """
        self._kgs = kgs
        self._strategy = strategy.lower()
        # 中文：确定主KG（CMeKG）和辅KG（DiseaseKB）
        self._primary = kgs.get("cmekg")
        self._fallback = kgs.get("diseasekb")
    
    def close(self):
        # 中文：关闭所有KG连接
        for kg in self._kgs.values():
            try:
                kg.close()
            except Exception:
                pass
    
    def _query_union(self, query_func, *args, **kwargs):
        # 中文：Union策略：对所有KG执行查询，合并结果并去重
        all_results = []
        for name, kg in self._kgs.items():
            try:
                results = query_func(kg, *args, **kwargs)
                if results:
                    all_results.extend(results)
            except Exception as e:
                # 中文：一个KG查询失败不应影响其他KG
                print(f"[MultiKG][warn] {name} query failed: {e}")
                continue
        # 去重（基于结果字符串）
        seen = set()
        unique_results = []
        for r in all_results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)
        return unique_results
    
    def _query_primary_fallback(self, query_func, *args, **kwargs):
        # 中文：Primary+Fallback策略：先查主KG（CMeKG），结果不足或为空时查辅KG（DiseaseKB）
        if self._primary:
            try:
                primary_results = query_func(self._primary, *args, **kwargs)
                if primary_results:
                    return primary_results
            except Exception as e:
                print(f"[MultiKG][warn] primary KG query failed: {e}")
        
        # 中文：主KG结果为空或失败时，使用辅KG
        if self._fallback:
            try:
                fallback_results = query_func(self._fallback, *args, **kwargs)
                if fallback_results:
                    return fallback_results
            except Exception as e:
                print(f"[MultiKG][warn] fallback KG query failed: {e}")
        
        return []
    
    def _query_single(self, kg_name: str, query_func, *args, **kwargs):
        # 中文：单KG策略：只查询指定的KG
        kg = self._kgs.get(kg_name)
        if kg:
            try:
                return query_func(kg, *args, **kwargs)
            except Exception as e:
                print(f"[MultiKG][warn] {kg_name} query failed: {e}")
        return []
    
    def _execute_query(self, query_func, *args, **kwargs):
        # 中文：根据策略执行查询（旧接口，兼容性保留）
        if self._strategy == "union":
            return self._query_union(query_func, *args, **kwargs)
        elif self._strategy == "primary_fallback":
            return self._query_primary_fallback(query_func, *args, **kwargs)
        elif self._strategy == "cmekg_only":
            return self._query_single("cmekg", query_func, *args, **kwargs)
        elif self._strategy == "diseasekb_only":
            return self._query_single("diseasekb", query_func, *args, **kwargs)
        else:
            # 中文：未知策略，默认使用union
            return self._query_union(query_func, *args, **kwargs)
    
    def _execute_query_by_kg(self, query_func, *args, **kwargs):
        # 中文：根据策略执行查询（新接口，支持按KG类型分发查询）
        # query_func 接收 (kg_name: str, kg: _BaseNeo4jKG, ...) 参数
        if self._strategy == "union":
            # 中文：Union策略：对所有KG执行查询，合并结果并去重
            all_results = []
            for kg_name, kg in self._kgs.items():
                try:
                    results = query_func(kg_name, kg, *args, **kwargs)
                    if results:
                        all_results.extend(results)
                except Exception as e:
                    print(f"[MultiKG][warn] {kg_name} query failed: {e}")
                    continue
            # 去重（基于结果字符串或元组）
            seen = set()
            unique_results = []
            for r in all_results:
                if r not in seen:
                    seen.add(r)
                    unique_results.append(r)
            return unique_results
        elif self._strategy == "primary_fallback":
            # 中文：Primary+Fallback策略：先查主KG（CMeKG），结果不足或为空时查辅KG（DiseaseKB）
            if self._primary:
                try:
                    primary_results = query_func("cmekg", self._primary, *args, **kwargs)
                    if primary_results:
                        return primary_results
                except Exception as e:
                    print(f"[MultiKG][warn] primary KG query failed: {e}")
            if self._fallback:
                try:
                    fallback_results = query_func("diseasekb", self._fallback, *args, **kwargs)
                    if fallback_results:
                        return fallback_results
                except Exception as e:
                    print(f"[MultiKG][warn] fallback KG query failed: {e}")
            return []
        elif self._strategy == "cmekg_only":
            # 中文：只查CMeKG
            if self._primary:
                try:
                    return query_func("cmekg", self._primary, *args, **kwargs)
                except Exception as e:
                    print(f"[MultiKG][warn] cmekg query failed: {e}")
            return []
        elif self._strategy == "diseasekb_only":
            # 中文：只查DiseaseKB
            if self._fallback:
                try:
                    return query_func("diseasekb", self._fallback, *args, **kwargs)
                except Exception as e:
                    print(f"[MultiKG][warn] diseasekb query failed: {e}")
            return []
        else:
            # 中文：未知策略，默认使用union
            all_results = []
            for kg_name, kg in self._kgs.items():
                try:
                    results = query_func(kg_name, kg, *args, **kwargs)
                    if results:
                        all_results.extend(results)
                except Exception as e:
                    print(f"[MultiKG][warn] {kg_name} query failed: {e}")
                    continue
            seen = set()
            unique_results = []
            for r in all_results:
                if r not in seen:
                    seen.add(r)
                    unique_results.append(r)
            return unique_results
    
    # ---------- 代理所有单库KG适配的方法 ----------
    
    def topk_relation_facts(self, nodes: Sequence[str], relation: str, k: int) -> List[str]:
        # 中文：关系事实检索（代理到各个KG）
        def _query(kg: _BaseNeo4jKG, *args, **kwargs):
            return kg.topk_relation_facts(*args, **kwargs)
        return self._execute_query(_query, nodes, relation, k)
    
    def drugs_for_diseases(self, disease_names: Sequence[str], k: int) -> List[str]:
        # 中文：疾病→药物（使用query.py中的查询函数）
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[str]:
            with kg._session() as sess:
                if kg_name == "cmekg":
                    return query.query_cmekg_drugs_for_diseases(sess, names, k) if query else []
                elif kg_name == "diseasekb":
                    return query.query_diseasekb_drugs_for_diseases(sess, names, k) if query else []
                else:
                    # 回退到原来的Neo4jKG方法
                    return kg.drugs_for_diseases(names, k)
        return self._execute_query_by_kg(_query_by_kg, disease_names, k)

    # ---------- 安全相关（仅 CMeKG 有效；DiseaseKB 返回空） ----------
    def contraindications_for_drugs(self, drug_names: Sequence[str], k: int) -> List[str]:
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[str]:
            if kg_name == "cmekg" and hasattr(kg, "contraindications_for_drugs"):
                return getattr(kg, "contraindications_for_drugs")(names, k)
            return []
        return self._execute_query_by_kg(_query_by_kg, drug_names, k)

    def adverse_reactions_for_drugs(self, drug_names: Sequence[str], k: int) -> List[str]:
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[str]:
            if kg_name == "cmekg" and hasattr(kg, "adverse_reactions_for_drugs"):
                return getattr(kg, "adverse_reactions_for_drugs")(names, k)
            return []
        return self._execute_query_by_kg(_query_by_kg, drug_names, k)

    def indications_for_drugs(self, drug_names: Sequence[str], k: int) -> List[str]:
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[str]:
            if kg_name == "cmekg" and hasattr(kg, "indications_for_drugs"):
                return getattr(kg, "indications_for_drugs")(names, k)
            return []
        return self._execute_query_by_kg(_query_by_kg, drug_names, k)

    def precautions_for_drugs(self, drug_names: Sequence[str], k: int) -> List[str]:
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[str]:
            if kg_name == "cmekg" and hasattr(kg, "precautions_for_drugs"):
                return getattr(kg, "precautions_for_drugs")(names, k)
            return []
        return self._execute_query_by_kg(_query_by_kg, drug_names, k)
    
    def drugs_for_symptoms(self, symptom_names: Sequence[str], k: int) -> List[str]:
        # 中文：症状→疾病→药物（使用query.py中的查询函数）
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[str]:
            with kg._session() as sess:
                if kg_name == "cmekg":
                    return query.query_cmekg_drugs_for_symptoms(sess, names, k) if query else []
                elif kg_name == "diseasekb":
                    return query.query_diseasekb_drugs_for_symptoms(sess, names, k) if query else []
                else:
                    return kg.drugs_for_symptoms(names, k)
        return self._execute_query_by_kg(_query_by_kg, symptom_names, k)
    
    def checks_for_diseases(self, disease_names: Sequence[str], k: int) -> List[str]:
        # 中文：疾病→检查（使用query.py中的查询函数）
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[str]:
            with kg._session() as sess:
                if kg_name == "cmekg":
                    return query.query_cmekg_checks_for_diseases(sess, names, k) if query else []
                elif kg_name == "diseasekb":
                    return query.query_diseasekb_checks_for_diseases(sess, names, k) if query else []
                else:
                    return kg.checks_for_diseases(names, k)
        return self._execute_query_by_kg(_query_by_kg, disease_names, k)
    
    def diet_for_diseases(self, disease_names: Sequence[str], k: int) -> List[str]:
        # 中文：疾病→饮食（使用query.py中的查询函数）
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[str]:
            with kg._session() as sess:
                if kg_name == "cmekg":
                    return query.query_cmekg_diet_for_diseases(sess, names, k) if query else []
                elif kg_name == "diseasekb":
                    return query.query_diseasekb_diet_for_diseases(sess, names, k) if query else []
                else:
                    return kg.diet_for_diseases(names, k)
        return self._execute_query_by_kg(_query_by_kg, disease_names, k)
    
    def resolve_entity(self, name: str) -> Dict:
        """Resolve entity according to integration strategy.
        
        中文：按集成策略进行实体链接：
        - union：同时查询两个KG，若都命中优先返回CMeKG；否则返回命中的那个
        - primary_fallback：先CMeKG，失败再DiseaseKB（原行为）
        - cmekg_only / diseasekb_only：仅查询指定KG
        """
        strategy = self._strategy

        # cmekg_only
        if strategy == "cmekg_only" and self._primary:
            try:
                with self._primary._session() as sess:
                    return query.query_cmekg_resolve_entity(sess, name) if query else {}
            except Exception:
                return {}

        # diseasekb_only
        if strategy == "diseasekb_only" and self._fallback:
            try:
                with self._fallback._session() as sess:
                    return query.query_diseasekb_resolve_entity(sess, name) if query else {}
            except Exception:
                return {}

        # union：两个库都查，优先CMeKG
        if strategy == "union":
            q_lower = (name or "").strip().lower()

            def _exact_match(sess, q: str) -> Dict:
                cypher_exact = """
                MATCH (n)
                WHERE toLower(n.name) = $q
                RETURN id(n) AS id, labels(n)[0] AS label, n.name AS name
                LIMIT 1
                """
                rec = sess.run(cypher_exact, q=q).single()
                return {"id": rec["id"], "label": rec["label"], "name": rec["name"]} if rec else {}

            # 1) 先跨库做精确匹配（谁精确命中用谁；若两者都精确命中，优先CMeKG）
            cmekg_exact, dkb_exact = {}, {}
            try:
                if self._primary:
                    with self._primary._session() as sess:
                        cmekg_exact = _exact_match(sess, q_lower)
                        if cmekg_exact:
                            cmekg_exact.setdefault("kg_source", "cmekg")
            except Exception:
                cmekg_exact = {}
            try:
                if self._fallback:
                    with self._fallback._session() as sess:
                        dkb_exact = _exact_match(sess, q_lower)
                        if dkb_exact:
                            dkb_exact.setdefault("kg_source", "diseasekb")
            except Exception:
                dkb_exact = {}

            if cmekg_exact:
                return cmekg_exact
            if dkb_exact:
                return dkb_exact

            # 2) 无精确命中，再分别走各自的多策略（含向量/模糊/全文）
            cmekg_res, dkb_res = {}, {}
            try:
                if self._primary:
                    with self._primary._session() as sess:
                        cmekg_res = query.query_cmekg_resolve_entity(sess, name) if query else {}
                        if cmekg_res:
                            cmekg_res.setdefault("kg_source", "cmekg")
            except Exception:
                cmekg_res = {}
            try:
                if self._fallback:
                    with self._fallback._session() as sess:
                        dkb_res = query.query_diseasekb_resolve_entity(sess, name) if query else {}
                        if dkb_res:
                            dkb_res.setdefault("kg_source", "diseasekb")
            except Exception:
                dkb_res = {}
            if cmekg_res:
                return cmekg_res
            if dkb_res:
                return dkb_res
            return {}

        # primary_fallback（默认）
        if self._primary:
            try:
                with self._primary._session() as sess:
                    if query:
                        result = query.query_cmekg_resolve_entity(sess, name)
                    else:
                        result = self._primary.resolve_entity(name)
                    if result:
                        result.setdefault("kg_source", "cmekg")
                        return result
            except Exception:
                pass
        if self._fallback:
            try:
                with self._fallback._session() as sess:
                    if query:
                        result = query.query_diseasekb_resolve_entity(sess, name)
                    else:
                        result = self._fallback.resolve_entity(name)
                    if result:
                        result.setdefault("kg_source", "diseasekb")
                        return result
            except Exception:
                pass
        return {}
    
    def drugs_for_diseases_struct(self, disease_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        # 中文：疾病→药物（结构化三元组，使用query.py中的查询函数）
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[Tuple[str, str, str]]:
            with kg._session() as sess:
                if kg_name == "cmekg":
                    return query.query_cmekg_drugs_for_diseases_struct(sess, names, k) if query else []
                elif kg_name == "diseasekb":
                    return query.query_diseasekb_drugs_for_diseases_struct(sess, names, k) if query else []
                else:
                    return kg.drugs_for_diseases_struct(names, k)
        results = self._execute_query_by_kg(_query_by_kg, disease_names, k)
        # 中文：去重结构化三元组
        seen = set()
        unique_results = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)
        return unique_results
    
    def drugs_for_symptoms_struct(self, symptom_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        # 中文：症状→疾病→药物（结构化三元组，使用query.py中的查询函数）
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[Tuple[str, str, str]]:
            with kg._session() as sess:
                if kg_name == "cmekg":
                    return query.query_cmekg_drugs_for_symptoms_struct(sess, names, k) if query else []
                elif kg_name == "diseasekb":
                    return query.query_diseasekb_drugs_for_symptoms_struct(sess, names, k) if query else []
                else:
                    return kg.drugs_for_symptoms_struct(names, k)
        results = self._execute_query_by_kg(_query_by_kg, symptom_names, k)
        seen = set()
        unique_results = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)
        return unique_results
    
    def checks_for_diseases_struct(self, disease_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        # 中文：疾病→检查（结构化三元组，使用query.py中的查询函数）
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[Tuple[str, str, str]]:
            with kg._session() as sess:
                if kg_name == "cmekg":
                    return query.query_cmekg_checks_for_diseases_struct(sess, names, k) if query else []
                elif kg_name == "diseasekb":
                    return query.query_diseasekb_checks_for_diseases_struct(sess, names, k) if query else []
                else:
                    return kg.checks_for_diseases_struct(names, k)
        results = self._execute_query_by_kg(_query_by_kg, disease_names, k)
        seen = set()
        unique_results = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)
        return unique_results
    
    def diet_for_diseases_struct(self, disease_names: Sequence[str], k: int) -> List[Tuple[str, str, str]]:
        # 中文：疾病→饮食（结构化三元组，使用query.py中的查询函数）
        def _query_by_kg(kg_name: str, kg: _BaseNeo4jKG, names: List[str], k: int) -> List[Tuple[str, str, str]]:
            with kg._session() as sess:
                if kg_name == "cmekg":
                    return query.query_cmekg_diet_for_diseases_struct(sess, names, k) if query else []
                elif kg_name == "diseasekb":
                    return query.query_diseasekb_diet_for_diseases_struct(sess, names, k) if query else []
                else:
                    return kg.diet_for_diseases_struct(names, k)
        results = self._execute_query_by_kg(_query_by_kg, disease_names, k)
        seen = set()
        unique_results = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)
        return unique_results


