# RAG 系統 Prompt Injection 攻擊與防禦框架

大學部資工系專題 — 在 RAG 問答系統上研究 Prompt Injection 攻擊，並設計、評估一套五層防禦管線。

## 架構

```
使用者查詢
   │
   ▼
┌─────────────────────── DefensePipeline (src/defense/pipeline.py) ───────────────────────┐
│ L1 InputSanitizer      正規化 + 長度限制 + 關鍵字/正則黑名單 (configs/blacklist.txt)     │
│ L2 InjectionDetector   微調 mBERT 二元分類器 (需先訓練，見下方步驟 3–4)                 │
│ ── 檢索 ──                                                                               │
│ L1/L2 ContextScan      對檢索回來的每份文件再掃一次 (L1 結構特徵 + L2 滑動視窗)，       │
│                        命中者只丟棄該文件、不擋下查詢 (間接注入防線)                     │
│ L3 PromptSegmentation  [SYSTEM]/[CONTEXT]/[USER] 標籤隔離 + 邊界規則系統提示            │
│ L4 ControlledGeneration temperature=0、max_tokens、(選用) JSON 結構化輸出、Llama Guard   │
│ L5 OutputValidator     正則樣式 + 格式合規檢查 + 獨立模型 (audit_model) LLM 稽核         │
└────────────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
最終回應 / [BLOCKED]
```

- **攻擊模組** `src/attack/` — `direct` / `indirect` / `jailbreak` / `prompt_leaking` / `multilingual_obfuscation` 產生器，統一入口 `generate_samples()`；`dataset_loader` 載入自建繁中資料集與外部資料集 (AdvBench / JailbreakBench)；`indirect_scenarios` 以 `PoisoningRetriever` 做端到端間接注入 (送出良性觸發查詢、payload 經檢索文件進入 context)。
- **評估框架** `src/evaluation/` — ASR (含 user-input vs knowledge-base 分列) / FPR / F1 / 延遲增量 (vs baseline) 指標，報告自動對照計畫書目標 (ASR<20%、F1>0.90、FPR<5%、延遲增量<500ms)。
- **API** `api/` — FastAPI：查詢、文件上傳、A/B 比較、背景評估 (見下方「API」)。
- **實驗記錄** — `run_evaluation.py` 與所有 Experience 腳本會把結果記到 W&B (`configs/default.yaml` 的 `wandb`；未設 `WANDB_API_KEY` 時自動略過，或加 `--no-wandb`)。
- **實驗** `scripts/Experience_*/` — 全部走 `DefensePipeline` + `configs/*.yaml`，共用 `scripts/experiment_utils.py`。

## 環境設定

```bash
# 有 NVIDIA GPU 時先裝 CUDA 版 torch（否則 pip 會裝 CPU 版，L2 訓練 ~2 小時 vs GPU ~4 分鐘）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cp .env.example .env    # 若無範例檔，手動建立
```

> 模組分工與資料流的完整說明見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

`.env` 需包含：

```
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...   # 若 configs/default.yaml 的 llm.provider 改用 anthropic
# WANDB_API_KEY=...        # 若要用 wandb 記錄 L2 訓練；否則訓練時加 --no-wandb
```

## 端到端執行順序

| 步驟 | 指令 | 說明 | 需 API |
|------|------|------|:---:|
| 1 | `python scripts/setup_vector_store.py` | 由 `data/raw/documents/` 建 FAISS 索引 | ✅ (embeddings) |
| 1b | `python scripts/download_external_datasets.py` | 下載 AdvBench (520) / JailbreakBench (100) 到 `data/raw/attack_datasets/`（AdvBench 在 HF 需登入，會自動改抓原作者 GitHub CSV） | ❌ |
| 2 | `python scripts/build_detector_dataset.py` | 產生 `benign.json` / `all_attacks.json`（純本地） | ❌ |
| 3 | `python scripts/train_detector.py --config configs/defense.yaml --no-wandb` | 微調 L2 偵測器 → `models/detector/checkpoints/best_model/`（含文件層級資料；GPU 約 4 分鐘、CPU 約 2 小時）。**`build_detector_dataset` 有變動後一定要重訓**，`metrics.json` 應有 `ood_doc` 欄位 | ❌ |
| 3b | `python scripts/eval_context_scan.py` | 離線評估文件掃描：乾淨文件誤丟率（訓練過 vs 保留文件）、污染文件偵測率、延遲、**L2 threshold 掃描**；`--model-path` 可比較不同 checkpoint | ❌ |
| 4 | `python scripts/Experience_Init/run_veryBaseAttack.py` | 架構煙霧測試：無防禦下的直接注入 ASR | ✅ |
| 5 | `python scripts/Experience_1/run_indirect_baseline.py` | 間接注入基線（乾淨 vs 污染知識庫） | ✅ |
| 6 | `python scripts/Experience_2/run_indirect_defense.py` | 間接注入：baseline vs 完整 DefensePipeline（含乾淨文件誤丟率） | ✅ |
| 7 | `python scripts/Experience_3/run_ablation.py` | 消融：逐層開關（含 L1/L2）；`--configs` 可只跑部分配置 | ✅ |
| 8 | `python scripts/Experience_4/run_fpr_test.py` | FPR 誤攔測試（完整管線，含 L2；93 筆**未進 L2 訓練**的良性查詢；`--train-sample N` 另報訓練分布樣本） | ✅ |
| 9 | `python scripts/Experience_5/run_chinese_attack_set.py` | 自建繁中攻擊資料集：baseline vs 防禦 | ✅ |
| 10 | `python scripts/run_evaluation.py --include-external --max-external 50` | 整合評估：直接類攻擊 + AdvBench/JBB + **端到端間接注入情境**（`--no-indirect` 可關）+ held-out 良性查詢；報告含 ASR 分向量、FPR、detector_f1（注意：此處為 in-distribution，泛化請引用 `metrics.json` 的 `ood`/`ood_doc`）、**延遲增量**、逐層攔截、計畫書目標達成 ✅/❌。`--max-attacks` / `--max-indirect` 控制成本 | ✅ |
| 11 | `python scripts/build_results_dashboard.py` | 把上面所有結果 + 偵測器 metrics 彙整成 `dashboard_data.json`（純本地） | ❌ |

> **實驗規模**：實驗 1–3 預設 `--queries extended`（44 個觸發查詢，涵蓋全部 13 份文件）× `--seeds 42 43 44`（3 個不同毒化的知識庫）= 每個配置 132 次試驗，所有比率附 Wilson 95% 信賴區間。`--queries original --seeds 42` 可重現舊版的 8 次試驗。
>
> **良性查詢**：FPR 一律使用 `src/evaluation/benign.py` 的 held-out 查詢（kb / general / edge / multilingual），`tests/test_detector/test_build_dataset.py` 會檢查它們不在 L2 訓練資料中。
>
> 知識庫文件 (`data/raw/documents/`) 有變動時需重跑步驟 1–3。`03_rag_system_security`、`08_taiwan_travel_guide`、`13_prompt_engineering_guide` 為 L2 的保留文件（`DOC_HOLDOUT`），不進訓練，只用於 `ood_doc` 評估；實驗 1–3 用的 `MARKER_PAYLOADS` 也不進任何訓練資料（`tests/test_detector/test_build_dataset.py` 會檢查）。
>
> L2 未訓練時（步驟 3 未執行）整條管線仍可運作，會退回 **L1+L3+L4+L5**，實驗腳本會印出 `L2 detector checkpoint not found`。

結果輸出於 `experiments/results/`（該資料夾被 `.gitignore` 忽略）。

## 結果視覺化

`python scripts/build_results_dashboard.py` 會讀 `experiments/results/*.json` +
`models/detector/checkpoints/best_model/metrics.json`，彙整成單一
`experiments/results/dashboard_data.json`（每個實驗各自的 schema 在重構時保持相容，
所以不管檔案是舊版還是新版腳本產生的都能解析；新鮮度直接看每張卡片上的時間戳記）。

把這個檔案拖進 [防禦管線觀測站](https://claude.ai/code/artifact/9192560f-7775-46d2-bfc2-4f7ce03b6919)
（一個發布好的 Artifact 頁面）的「載入最新結果」，就能看到 L2 偵測器 test/ood 對比、
Experience 1–5 的 ASR/FPR/消融長條圖、以及 run_evaluation 的整合結果——資料只讀進瀏覽器
的 `localStorage`，不會上傳到任何地方，每次重新執行實驗後重新拖一次檔案即可更新。

## API

```bash
uvicorn api.main:app --reload     # 需先完成步驟 1；互動文件在 http://127.0.0.1:8000/docs
```

| 端點 | 用途 |
|------|------|
| `POST /rag/query` | 單一查詢；`defense_enabled` 切換防禦。回傳 traces、檢索來源、被 context scan 丟掉的文件 |
| `POST /documents/upload` | 上傳 .txt/.md/.pdf/.docx 進知識庫；預設先用 L1/L2 掃描每個 chunk，`reject_flagged=true` 時不收被標記的 chunk。檔案存於 `data/uploads/`（刻意不放 `data/raw/documents/`，避免重建時混入 L2 訓練資料）；重跑步驟 1 會重建索引、上傳內容需重新上傳 |
| `GET /documents` | 列出知識庫來源與 chunk 數 |
| `POST /experiment/ab` | A/B 比較：同一查詢跑多個變體（`enabled_layers: null` = 無防禦 baseline；其他為層子集 + `overrides`，例如 `{"output_format": "json"}`），回傳各變體結果與相對 baseline 的 `latency_delta_ms` |
| `POST /evaluation/run`、`GET /evaluation/status/{name}` | 背景評估（含良性樣本、選用外部資料集與間接注入情境） |

## 設定檔

| 檔案 | 用途 |
|------|------|
| `configs/default.yaml` | RAG / LLM / 五層防禦參數（`DefensePipeline.from_config` 讀取；L4 `output_format`、L5 `audit_model` / `audit_prompt` 等）、W&B |
| `configs/defense.yaml` | L2 偵測器訓練超參數（各層執行參數都在 default.yaml） |
| `configs/attack.yaml` | 攻擊型別、資料集清單、每型別模板/參數 |
| `configs/blacklist.txt` | L1 正則黑名單（一行一條，`#` 為註解） |

## 測試

```bash
python -m pytest -q
```

（`pytest` 不需 API 金鑰；會呼叫 LLM 的僅有 `scripts/Experience_*` 與 `run_evaluation.py`。）
