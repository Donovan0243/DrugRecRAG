"""GTV (Generate-then-Verify) CLI.

中文说明：GTV 流程的命令行接口。
"""

import argparse
import json
import sys

from .pipeline import run_gtv
from .eval_runner import run_eval_dialmed_gtv


def main():
    parser = argparse.ArgumentParser(description="Run GTV (Generate-then-Verify) pipeline")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--dialog", type=str, help="Dialogue text (if omitted, read from STDIN)")
    mode.add_argument("--eval", type=str, help="Path to DialMed test.txt (JSONL)")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for evaluation")
    parser.add_argument("--out", type=str, default=None, help="Write per-sample results to JSONL path")
    parser.add_argument("--progress", action="store_true", help="Show progress during evaluation")
    parser.add_argument("--progress_every", type=int, default=1, help="Progress print frequency (default 1)")
    parser.add_argument("--trace", action="store_true", help="Include final_prompt/trace in JSONL")
    parser.add_argument("--debug", action="store_true", help="Debug mode: only run Phase A (SFT model generation), skip Phase B verification and final reasoning")
    parser.add_argument("--use-candidate-list", action="store_true", help="Use candidate drug list: include all possible drugs (from label.json) in prompt, let model choose from the list")

    args = parser.parse_args()
    
    # 如果使用了 --use-candidate-list，设置环境变量
    if args.use_candidate_list:
        import os
        os.environ["GTV_USE_CANDIDATE_LIST"] = "true"

    if args.eval:
        # 评估模式：使用 GTV 评估运行器
        metrics = run_eval_dialmed_gtv(
            args.eval,
            limit=args.limit,
            out_path=args.out,
            show_progress=args.progress,
            progress_every=args.progress_every,
            include_trace=args.trace,
            debug_mode=args.debug,
            use_candidate_list=args.use_candidate_list,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    dialogue = args.dialog
    if not dialogue:
        dialogue = sys.stdin.read()
    
    result = run_gtv(dialogue_text=dialogue, debug_mode=args.debug, use_candidate_list=args.use_candidate_list)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

