# 系統架構與流程總覽

本文件取代舊的 60 個 `*_summary.txt`（已封存於 `docs/legacy_summaries_20260922.zip`，內容停留在 2026-09-22，未涵蓋之後的 context scan、JSON 輸出模式、PoisoningRetriever 等）。
逐行細節以程式碼內的 docstring / 註解為準；本文件只描述**模組分工、資料流與評估設計**，修改架構時請一併更新。

---

## 1. 一次查詢的資料流

```
使用者查詢 q
  │
  ├─ L1 InputSanitizer.sanitize(q)        Unicode 正規化、去 HTML、長度上限、正則黑名單 (configs/blacklist.txt)
  │     命中 → 擋下 (block_layer="L1")
  ├─ L2 InjectionDetector.detect(q)       mBERT 二元分類，p(injection) ≥ threshold → 擋下
  │
  ├─ Retrieval   retriever.retrieve(q)    FAISS top-k (default.yaml rag.top_k = 5)
  │
  ├─ L1 ContextScan  每份文件跑 CONTEXT_BLACKLIST_DEFAULT（只收結構特徵，乾淨語料 0 命中）
  ├─ L2 ContextScan  320 字滑動視窗 (stride 160)，文件分數 = 視窗最大值 ≥ context_threshold
  │     命中 → 只丟棄該文件（記在 DefenseResult.dropped_docs），查詢本身不擋
  │
  ├─ L3 PromptSegmenter   [SYSTEM]/[CONTEXT]/[USER]/[END] 標籤 + 信任邊界規則系統提示
  ├─ L4 ControlledGenerator  temperature 0、max_tokens、(選用) JSON {answer, sources, refused}、(選用) Llama Guard
  │     JSON 解析失敗重試 1 次後仍失敗 → 擋下 (format_violation)
  └─ L5 OutputValidator   通用正則（洩漏系統提示、承認執行覆寫指令、越獄口吻）
                          + 格式合規（空輸出、過長、≥2 個 L3 標籤 = 版面外洩、JSON 缺欄位）
                          + 獨立模型稽核 (audit_model: gpt-4o，生成用 gpt-4o-mini)
```

- 協調者：`src/defense/pipeline.py` 的 `DefensePipeline`。`from_config()` 讀 `configs/default.yaml`；`enabled_layers` 子集供消融實驗使用。
- `traces` 每層一筆 `LayerTrace`（含延遲）。`Retrieval` 與 `L4_ControlledGeneration` 是 baseline 也要做的步驟，其餘加總為 **defense-only latency**。
- L4 在 `enabled_layers` 中**不是開關**：生成一定會執行；差異只在 `output_format=json` 或 Llama Guard 開啟時出現。
- 上傳文件 (`POST /documents/upload`) 在**寫入知識庫之前**跑同一套 L1/L2 context scan（`api/routers/documents.py: scan_chunks`），被標記的 chunk 預設不入庫。這是 L2「檢查使用者上傳檔案」的入口；查詢時的 context scan 則是第二道防線（處理已存在或經其他管道進入的文件）。

## 2. 模組地圖

| 路徑 | 職責 |
|------|------|
| `src/rag/` | 文件載入/切塊 (512/64)、FAISS 向量庫、`Retriever`、無防禦的 `RAGPipeline`（baseline） |
| `src/defense/` | 五層防禦 + `pipeline.py` |
| `src/attack/` | 攻擊樣本產生器（direct / jailbreak / prompt_leaking / multilingual_obfuscation；`generate_samples()` 統一入口）、`indirect_injection.py`（payload 清單與毒化）、`indirect_scenarios.py`（觸發查詢、`PoisoningRetriever`）、`dataset_loader.py`（自建繁中集、AdvBench/JailbreakBench）、`attack_runner.py`（執行 + LLM judge） |
| `src/evaluation/` | `metrics.py`（ASR/FPR/F1/延遲、Wilson 信賴區間）、`benign.py`（held-out 良性查詢）、`evaluator.py`（defended vs baseline）、`report_generator.py` |
| `src/utils/` | config 載入、logging、W&B (`tracking.py`，無有效金鑰時自動略過) |
| `models/detector/train.py` | L2 微調（group-aware split、OOD 評估寫入 `metrics.json`） |
| `scripts/` | 資料建置、訓練、離線評估 (`eval_context_scan.py`)、實驗 (`Experience_*`)、整合評估 (`run_evaluation.py`)、dashboard 彙整 |
| `api/` | FastAPI：查詢、上傳、A/B 比較、背景評估 |

## 3. L2 偵測器：資料與保留集設計

`scripts/build_detector_dataset.py`（純本地）產生 `data/processed/attack_sets/`：

| 檔案 | 內容 | 是否進訓練 |
|------|------|:---:|
| `benign.json` / `all_attacks.json` | 查詢層級樣本（模板 + 增強）＋ **知識庫文件視窗**（乾淨 = 0；插入 `DOC_POISON_TRAIN_PAYLOADS` = 1） | ✅ |
| `ood_eval.json` | 38 筆人工手寫、非模板的釋義查詢 | ❌ |
| `ood_doc_eval.json` | `DOC_HOLDOUT` 三份文件的視窗，乾淨 vs 插入實驗 payload | ❌ |

切分以 `group_id` 為單位（同一基礎句的所有增強變體、同一份文件的所有視窗共用一個 group），避免近似重複跨 train/test 洩漏。

**洩漏防護（皆有測試）**

| 規則 | 測試 |
|------|------|
| `MARKER_PAYLOADS` / `SUCCESS_MARKERS`（實驗 1–3 用）不出現在任何訓練資料 | `tests/test_detector/test_build_dataset.py` |
| `DOC_HOLDOUT` 文件不進訓練 | 同上 |
| held-out 良性查詢（含 44 個觸發查詢）不在訓練資料 | 同上 `test_heldout_benign_queries_never_reach_training` |
| L5 預設正則**不含**實驗 payload 的標記字串（否則等於防禦端拿到答案） | `tests/test_defense/test_layer5_format.py` |
| L1 context 黑名單在乾淨知識庫上 0 命中 | `tests/test_defense/test_context_scan.py` |

> `build_detector_dataset.py` 有改動後**必須重訓**。舊 checkpoint（2026-09-14）是在加入文件視窗之前訓練的，對乾淨文件誤丟 98.5%；重訓後為 4.5%（見 ExperienceNotes.txt）。判斷方式：`best_model/metrics.json` 應有 `ood_doc` 欄位。

## 4. 評估設計

| 名詞 | 定義 |
|------|------|
| ASR | 攻擊成功數 / 試驗數。實驗 1–3 以回應中出現 `SUCCESS_MARKERS` 判定；`run_evaluation` / 實驗 5 用 LLM judge |
| FPR | 被擋下的良性查詢 / 良性查詢數；**一律用 held-out 良性查詢**（`src/evaluation/benign.py`，93 筆：kb 44 / general 19 / edge 22 / multilingual 8） |
| Clean drop | context scan 丟掉的**未毒化**文件 / 檢索到的文件數（ground truth 來自 chunk 的 `metadata["poisoned"]`） |
| Useful | 未被擋且至少保留 1 份文件的查詢比例 —— 用來揭露「把 context 全丟掉」換來的假 0% ASR |
| 95% CI | Wilson score interval（`metrics.wilson_interval`），小樣本、比率接近 0/100% 時仍有意義 |

**試驗規模**：實驗 1–3 = 毒化 seed（預設 42/43/44，每個 seed 毒化不同的 30% chunk）× 觸發查詢（預設 extended 44 個，涵蓋全部 13 份文件）= 132 次 / 配置。temperature 0 下重複同一查詢幾乎沒有新資訊，所以增加的是查詢與毒化方式，而非重複次數。

**兩種間接注入情境**

| | 實驗 1–3 | `run_evaluation` / 實驗 5 |
|---|---|---|
| 毒化方式 | 知識庫中隨機 30% chunk 被毒化後建索引 | `PoisoningRetriever` 對每次檢索的第 1 份文件注入 |
| 量的是 | 檢索是否撈到 + 防禦是否擋住 | 只量防禦（payload 保證進入 context） |

## 5. 實驗一覽

| 腳本 | 問題 | 輸出 |
|------|------|------|
| `Experience_Init/run_veryBaseAttack.py` | 無防禦下直接注入 ASR（冒煙測試） | `baseline_quick.json` |
| `Experience_1/run_indirect_baseline.py` | 無防禦下間接注入 ASR（乾淨 vs 毒化） | `indirect_baseline.json` |
| `Experience_2/run_indirect_defense.py` | 完整管線 vs baseline，含 clean drop | `indirect_defense_comparison.json` |
| `Experience_3/run_ablation.py` | 9 種層組合的消融 | `ablation_results.json` |
| `Experience_4/run_fpr_test.py` | 完整管線的 FPR 與 context 誤丟 | `fpr_test.json` |
| `Experience_5/run_chinese_attack_set.py` | 自建繁中直接 / 間接注入集 | `chinese_attack_set_eval.json` |
| `eval_context_scan.py` | L1/L2 文件掃描離線評估 + threshold 掃描（不需 API） | `context_scan_eval.json` |
| `run_evaluation.py` | 整合評估（使用者輸入 + 間接 + 良性 + 延遲） | `full_evaluation_*` |

## 6. 已知限制

- L2 的查詢層級泛化：加入文件視窗後，手寫釋義攻擊 (`ood`) 的 recall 下降（訓練資料中有大量「討論攻擊」的安全文件被標為良性）。查詢端另有 L1/L3/L5 與模型本身的 RLHF 抵抗，且實測直接注入在 gpt-4o-mini 上 baseline ASR 已接近 0%。
- `run_evaluation` 報告中的 `detector_f1` 用的是與訓練同模板家族的樣本（in-distribution），泛化請引用 `metrics.json` 的 `ood` / `ood_doc`。
- 實驗 payload 只有 5 種 (`MARKER_PAYLOADS`)，攻擊目標皆為「輸出標記字串」；資料外洩、連結注入等目標只出現在訓練 payload 與繁中集。
- 延遲：L2 context scan 在 GPU 上約 90 ms / top-5，CPU 上約 1.9 s；L5 gpt-4o 稽核約 1.3 s 是主要成本。
