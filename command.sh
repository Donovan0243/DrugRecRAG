# GTV 测评指令
python -m src.GTV.gtv_cli \
    --eval dialmed/test.txt \
    --out runs/gtv_pred_100.jsonl \
    --limit 4 \
    --progress \
    --trace