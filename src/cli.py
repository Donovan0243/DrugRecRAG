"""Simple CLI for running the GAP pipeline.

中文说明：使用命令行传入对话文本，运行端到端 GAP 流程并输出 JSON 结果。
"""

import argparse
import json
import sys

from .pipeline import run_gap
from .eval_runner import run_eval_dialmed


def main():
    parser = argparse.ArgumentParser(description="Run GAP minimal pipeline")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--dialog", type=str, help="Dialogue text (if omitted, read from STDIN)")
    mode.add_argument("--eval", type=str, help="Path to DialMed test.txt (JSONL)")
    mode.add_argument("--normalize", type=str, help="Normalize DialMed JSONL labels with mapping file")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for evaluation")
    parser.add_argument("--out", type=str, default=None, help="Write per-sample results to JSONL path")
    parser.add_argument("--progress", action="store_true", help="Show progress during evaluation")
    parser.add_argument("--progress_every", type=int, default=1, help="Progress print frequency (default 1)")
    parser.add_argument("--trace", action="store_true", help="Include final_prompt/trace in JSONL (llm_raw已默认写入)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input file labels with normalized labels")

    args = parser.parse_args()

    if args.normalize:
        # 默认映射路径
        mapping_path = "appendix/medication_normalization.json"
        if args.overwrite:
            from .normalize import normalize_dialmed_overwrite
            normalize_dialmed_overwrite(args.normalize, mapping_path)
            print(json.dumps({"status": "ok", "inplace": True, "input": args.normalize}, ensure_ascii=False))
            return
        else:
            from .normalize import normalize_dialmed_jsonl
            if not args.out:
                raise SystemExit("--normalize requires --out or use --overwrite for in-place")
            normalize_dialmed_jsonl(args.normalize, args.out, mapping_path)
            print(json.dumps({"status": "ok", "input": args.normalize, "output": args.out}, ensure_ascii=False))
            return

    if args.eval:
        metrics = run_eval_dialmed(
            args.eval,
            limit=args.limit,
            out_path=args.out,
            show_progress=args.progress,
            progress_every=args.progress_every,
            include_trace=args.trace,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    dialogue = args.dialog
    if not dialogue:
        dialogue = sys.stdin.read()
    
    result = run_gap(dialogue_text=dialogue)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


