# Models

## Injection Detector (Layer 2)

Base model: `bert-base-multilingual-cased`

微調為二元分類：
- `0` — BENIGN（正常查詢）
- `1` — INJECTION（prompt injection 嘗試）

### 1. 產生訓練資料（純本地，不需 API）

```bash
python scripts/build_detector_dataset.py
```

輸出 `data/processed/attack_sets/benign.json`、`all_attacks.json`，以及一份
**held-out、非模板化**的 `ood_eval.json`（schema 與洩漏防護說明見
`data/processed/attack_sets/README.md`）。

### 2. 訓練

```bash
python scripts/train_detector.py --config configs/defense.yaml --no-wandb
```

- `--no-wandb`：停用 Weights & Biases（也可設環境變數 `WANDB_DISABLED=true`）。
- 無 GPU 時會自動關閉 `fp16`，可在 CPU 上訓練（資料量約 1.2k 筆、mBERT、5 epoch，
  CPU 約數十分鐘；有 CUDA 會自動使用）。
- 超參數（epoch / batch size / lr / split 比例）讀自 `configs/defense.yaml` 的 `detector:` 區塊。
- train/val/test 用 **group-aware split**（依 `group_id` 切，而非逐列隨機）：同一個
  基礎樣本增強出的所有變體只會落在同一邊，避免變體互相洩漏到不同 split。

Checkpoints 寫到 `models/detector/checkpoints/`；最佳（依 validation F1）存到
`models/detector/checkpoints/best_model/`，並附一份 `metrics.json`，內含兩組指標：

```json
{
  "test": { "eval_f1": 1.0, "eval_precision": 1.0, "eval_recall": 1.0, ... },
  "ood":  { "ood_f1": 0.83, "ood_precision": 1.0, "ood_recall": 0.71, ... }
}
```

- **`test`**：in-distribution（跟訓練資料同模板家族的 held-out 樣本）。
- **`ood`**：`ood_eval.json`——手寫、非模板化的釋義句，訓練從未見過類似寫法。

**寫報告 / 論文時請引用 `ood` 的數字，不要只放 `test`。** `test` 分數容易因為模板
高度重複而虛高（第一版訓練資料就出現過 `eval_f1 = 1.0` 但其實是資料洩漏，見
`data/processed/attack_sets/README.md`）；`test` 和 `ood` 的差距本身就是一個
值得在論文中討論的發現——例如目前 `ood_precision=1.0` 但 `ood_recall≈0.71`，
代表偵測器對「換句話說」過的攻擊召回率會下降，但沒有因此誤判良性輸入。
若沒有 `ood_eval.json`（例如用舊版資料集），訓練仍會照跑，只是 log 會警告
拿不到這組誠實指標。

### 3. 接上防禦管線

`DefensePipeline.from_config()` 會在 `models/detector/checkpoints/best_model/`
存在時自動載入 L2；不存在則整條管線退回 L1+L3+L4+L5。

### 目錄結構

```
models/
└── detector/
    ├── train.py               # 訓練邏輯
    └── checkpoints/           # Git-ignored（大檔）
        └── best_model/        # 推論用最終模型 + metrics.json
```
