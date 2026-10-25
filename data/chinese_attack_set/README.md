# 繁體中文 Prompt Injection 攻擊資料集

本資料集為本研究自行建立，針對繁體中文場景設計，填補現有英文資料集的空白。

## 資料集統計

| 檔案 | 類型 | 筆數 | 語言 |
|------|------|------|------|
| `direct_injection.json` | 直接注入 | 10 | 繁中、混合 |
| `indirect_injection.json` | 間接注入 | 5 | 繁中、英文 |

## 攻擊類別

### 直接注入 (Direct Injection)

| 類別 | 說明 |
|------|------|
| `instruction_override` | 直接覆蓋系統指令 |
| `role_override` | 角色置換攻擊 |
| `behavior_modification` | 修改模型行為 |
| `prompt_leaking` | 提取系統提示 |
| `jailbreak` | DAN/越獄攻擊 |
| `social_engineering` | 社交工程手法 |
| `multilingual_obfuscation` | 多語言混淆 |
| `roleplay_bypass` | 角色扮演繞過 |
| `developer_mode_spoof` | 偽裝開發者模式 |
| `translation_injection` | 翻譯任務注入 |

### 間接注入 (Indirect Injection)

| 類別 | 說明 |
|------|------|
| `knowledge_base_poisoning` | 知識庫文件污染 |
| `output_manipulation` | 輸出內容操控 |
| `html_comment_injection` | HTML 注解夾帶指令 |
| `system_override_tag` | 假系統覆蓋標籤 |
| `output_suffix_injection` | 輸出後綴強制注入 |

## 資料格式

```json
{
  "id": "zh_direct_XXXX",
  "attack_type": "direct_injection | indirect_injection",
  "language": "zh-tw | en | mixed",
  "payload": "攻擊載荷（直接注入）",
  "document_content": "惡意文件內容（間接注入）",
  "malicious_goal": "攻擊意圖描述",
  "category": "攻擊子類別",
  "notes": "備注"
}
```

## 本資料集的使用位置

| 使用者 | 方式 |
|--------|------|
| `scripts/build_detector_dataset.py` | 併入 L2 偵測器的惡意訓練樣本（`load_chinese_attack_set()`） |
| `scripts/Experience_5/run_chinese_attack_set.py` | 直接注入 10 筆跑 baseline vs 完整 DefensePipeline，輸出 per-category ASR |
| `scripts/run_evaluation.py --attack-set chinese` | 用本資料集取代程式產生的攻擊樣本做整合評估 |
| `src/attack/dataset_loader.py` | `load_attack_set_file()` / `load_chinese_attack_set()` 把 JSON 轉成 `AttackSample`；間接注入的 `document_content` 會映射為 `payload` 並標 `metadata.is_document=True` |

## 授權與使用

本資料集僅供學術研究與安全防禦研究使用，請勿用於惡意用途。
