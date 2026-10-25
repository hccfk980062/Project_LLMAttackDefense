# attack_sets — L2 偵測器訓練資料

由 `python scripts/build_detector_dataset.py` 產生（純本地、不需 API / 網路）。

| 檔案 | 內容 | label |
|------|------|:---:|
| `benign.json` | 良性查詢（多領域 + hard negatives + 輕量增強） | 0 |
| `all_attacks.json` | 全型別攻擊 + 自建繁中資料集 + 知識庫毒化 payload + 增強 | 1 |
| `detector_dataset.json` | 上兩者合併並洗牌 | 0/1 |
| `ood_eval.json` | **手寫、非模板化**的釋義攻擊 + 全新良性句，held-out，訓練時永遠不進 train/val/test | 0/1 |
| `stats.json` | 筆數 / 類別平衡 / group 數 / 來源分佈 |  |

## 單筆 schema

```json
{
  "id": "attack_00042",
  "group_id": "jailbreak_0031",              // 見下方「group_id 與資料洩漏」
  "payload": "忽略所有先前的指示。...",
  "text": "忽略所有先前的指示。...",           // 與 payload 同值，相容 models/detector/train.py
  "label": 1,                                 // 0 = BENIGN, 1 = INJECTION
  "attack_type": "direct_injection",          // benign 時為 "benign"
  "language": "zh-tw",
  "source": "generated | augmented | chinese_direct_injection | bank | ..."
}
```

## group_id 與資料洩漏（重要）

`--augment`（預設開啟）會對每個基礎樣本做輕量、保留標籤的表面變化（換包裝句、標點、大小寫）。
一個基礎樣本會產生好幾筆長得很像的變體，**它們共用同一個 `group_id`**。

`models/detector/train.py` 的 `load_dataset_from_json()` 用 `group_id` 做切分
（而不是對每一列各自隨機切），確保同一個基礎樣本的所有變體只會出現在
train / val / test 的**同一邊**。如果改成對列隨機切，同一句話的變體 A 進 train、
變體 B 進 test 是常見情況——模型等於先看過答案，validation/test 分數會被灌水
（第一次訓練就是這樣，`eval_f1` 出現不合理的 1.0）。

`ood_eval.json` 更進一步：裡面全是人工手寫、**沒有經過 generator 模板也沒有經過
augment** 的釋義句，`group_id == id`（每筆自成一組）。訓練結束後
`train.py` 會自動拿它算一次獨立指標寫進 `best_model/metrics.json` 的 `"ood"`
欄位，量的是「有沒有學到注入的概念」而不是「背熟了訓練模板」。
`test`（in-distribution）分數變高很正常；真正該在報告裡強調類推能力時，
**用 `ood` 的數字**。

## 重新產生 / 調整

```bash
# 預設：每類平衡到 min(可用, 700)，含輕量增強
python scripts/build_detector_dataset.py

# 不做增強、只用原始樣本
python scripts/build_detector_dataset.py --no-augment

# 也載入 configs/attack.yaml 裡的外部資料集（advbench / jailbreakbench，缺檔會略過）
python scripts/build_detector_dataset.py --include-external
```

訓練：`python scripts/train_detector.py --config configs/defense.yaml --no-wandb`
