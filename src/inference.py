"""LLM inference utilities.

中文说明：提供LLM调用器，支持不同提供商的API调用。
"""

import json
from typing import List, Dict
import requests
from .config import llm as llm_cfg
import time


def _dummy_llm_call(prompt: str) -> str:
    """A placeholder LLM call returning a fixed JSON answer.

    中文：占位 LLM 返回固定 JSON，便于端到端打通。
    """
    return json.dumps(
        [
            {
                "drug": "Loratadine",
                "dosage": "10mg",
                "regimen": "once daily",
                "rationale": "Recommended based on symptoms and safety considerations",
                "cited_evidence_ids": ["e1", "e2"],
            }
        ]
    )


def get_llm_caller(expect_json: bool = False):
    """Return a callable for LLM invocation based on config.

    中文：根据配置返回 LLM 调用器；默认使用占位实现。
    """
    if llm_cfg.provider == "ollama":
        def _caller(prompt: str) -> str:
            # 中文：使用 OpenAI 兼容接口 /chat/completions 调用 Ollama 服务
            url = f"{llm_cfg.base_url.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {llm_cfg.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": llm_cfg.model,
                "temperature": llm_cfg.temperature,
                "max_tokens": llm_cfg.max_tokens,
                "messages": [
                    {"role": "system", "content": "You are a clinical medication recommendation assistant."},
                    {"role": "user", "content": prompt},
                ],
            }
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    try:
                        print(f"[LLM][ollama][warn] empty content, status={resp.status_code}, body_len={len(resp.text)}")
                    except Exception:
                        pass
                return content
            except Exception as e:
                try:
                    status = getattr(resp, "status_code", "?") if 'resp' in locals() else "?"
                    text_sample = (resp.text[:500] + "...") if 'resp' in locals() and hasattr(resp, 'text') else "<no response>"
                    print(f"[LLM][ollama][error] model={llm_cfg.model} status={status} err={e}\nresp_body_sample={text_sample}")
                except Exception:
                    pass
                return _dummy_llm_call(prompt)

        return _caller

    if llm_cfg.provider == "gemini":
        # 中文：使用 Google Gemini 官方 REST 接口
        def _caller(prompt: str) -> str:
            base = llm_cfg.base_url.strip() if llm_cfg.base_url else "https://generativelanguage.googleapis.com/v1beta"
            url = f"{base.rstrip('/')}/models/{llm_cfg.model}:generateContent?key={llm_cfg.api_key}"
            headers = {"Content-Type": "application/json"}
            body = {
                "systemInstruction": {
                    "parts": [{"text": "You are a clinical medication recommendation assistant."}]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": llm_cfg.temperature,
                    "maxOutputTokens": llm_cfg.max_tokens,
                },
                # 放宽安全阈值，避免医学内容被拦截
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                ],
            }
            # 仅当期望 JSON 时启用 JSON MIME（避免干扰纯文本阶段）
            if expect_json:
                body["generationConfig"]["responseMimeType"] = "application/json"
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                # 记录安全/拦截信息，便于定位“200 无 candidates”的原因
                try:
                    pf = data.get("promptFeedback") or {}
                    if pf:
                        print(f"[LLM][gemini][info] promptFeedback={pf}")
                    safety = data.get("candidates", [{}])[0].get("safetyRatings") if data.get("candidates") else None
                    if safety:
                        print(f"[LLM][gemini][info] safetyRatings={safety}")
                except Exception:
                    pass
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        if not text:
                            try:
                                print(f"[LLM][gemini][warn] empty parts text, status={resp.status_code}")
                            except Exception:
                                pass
                        return text
                try:
                    print(f"[LLM][gemini][warn] no candidates returned, status={resp.status_code}, body_len={len(resp.text)}")
                    # 打印原始响应文本样本与可见字段，直观看到服务端返回
                    print(f"[LLM][gemini][warn] resp_text_sample={resp.text[:2000]}")
                    try:
                        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
                        print(f"[LLM][gemini][warn] resp_keys={keys}")
                    except Exception:
                        pass
                    # 若存在 error 字段也打印样本
                    err = data.get("error") if isinstance(data, dict) else None
                    if err:
                        print(f"[LLM][gemini][warn] error_obj={err}")
                except Exception:
                    pass
                return _dummy_llm_call(prompt)
            except Exception as e:
                try:
                    status = getattr(resp, "status_code", "?") if 'resp' in locals() else "?"
                    text_sample = (resp.text[:500] + "...") if 'resp' in locals() and hasattr(resp, 'text') else "<no response>"
                    print(f"[LLM][gemini][error] model={llm_cfg.model} url={url} expect_json={expect_json} status={status} err={e}\nresp_body_sample={text_sample}")
                except Exception:
                    pass
                return _dummy_llm_call(prompt)

        return _caller

    return _dummy_llm_call

