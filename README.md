# AI 自強化防火牆系統

大學畢業專題 — 架構規劃與技術分工

以多層式防禦取代傳統 WAF（移除 ModSecurity），結合統計模型（LightGBM）與深度學習模型
（TextCNN）辨識惡意流量，並透過高互動蜜罐收集真實攻擊/誤判樣本，形成「偵測 → 蜜罐驗證 →
增量訓練」的自強化閉環。

詳細設計請見 [`AI自強化防火牆系統設計文件.pdf`](./AI自強化防火牆系統設計文件.pdf)。
nftables的L3/L4防禦經討論後刪除，未來若開發提早完成或L3/L4攻擊影響過大再考慮加入。

## 系統架構

| 流程層 | 模組 / 元件 | 說明 |
|---|---|---|
| 訓練資料層（離線） | CICIDS2017、CSE-CIC-IDS2018（正常流量部分） | 擷取正常流量作為初始正樣本，解決正常/惡意樣本不平衡問題 |
~~| 第一層　邊緣防禦 | nftables（L3/L4 靜態規則） | 依 IP、Port、TCP Flags 封鎖 SYN Flood、Port Scan 等 L3/L4 攻擊 |~~
| 第二層　統計防禦 | Redis + LightGBM | 統計同 IP 請求頻率、URL 變異度等 window-based 特徵，識別 HTTP 洪水（CC Attack） |
| 第三層　語意防禦 | TextCNN（PyTorch） | 對通過前兩層的 HTTP Payload 做字元/詞嵌入，辨識 SQLi、XSS、RCE 等應用層攻擊 |
| 分流引流模組 | Reverse Proxy（Flow Router） | 依模型判定分流：正常流量 → 真實伺服器 + 鏡像蜜罐；惡意流量 → 蜜罐 |
| 主動防禦 | 高互動蜜罐（Docker + Crontab） | 搭建真實漏洞環境，定時自動還原 |
| 主動防禦閉環 | Falco / Auditd | 監控蜜罐內異常行為，判定模型漏報（FN）/誤報（FP）/正確（TP），並將結果寫入中央資料庫 |
| 自強化微調排程 | APScheduler / Cron | 累積錯誤樣本後（或每週），結合舊資料 + 新樣本對 LightGBM、TextCNN 進行增量訓練，避免災難性遺忘 |

### 蜜罐反饋判定矩陣（自強化核心）

蜜罐內嵌 Suricata IDS，依「前端模型標記」與「蜜罐行為審計結果」的組合自動決定反饋動作：

| 前端模型標記 | 蜜罐內部行為審計 | 結論 | 動態反饋動作 |
|---|---|---|---|
| 正常 (Normal) | 觸發惡意特徵 / 系統提權 | 漏報 (FN) | 標記為「惡意」，加入黑名單並回傳訓練集 |
| 惡意 (Malicious) | 完成正常業務流程且無異常 | 誤報 (FP) | 標記為「正常」，回傳訓練集調整模型權重 |
| 惡意 (Malicious) | 執行掃描 / 漏洞利用 | 正確 (TP) | 確認模型判斷正確，維持現有權重 |

## 技術堆疊

| 模組 | 語言 | 關鍵技術 / 套件 |
|---|---|---|
| 訓練資料整合（CICIDS2017 / CSE-CIC-IDS2018） | Python | pandas, scikit-learn, imbalanced-learn |
~~| L3/L4 邊緣防禦 | C / Python | nftables, python-nftables |~~
| L7 統計防禦（HTTP 洪水） | Python | LightGBM, Redis, scikit-learn, pandas |
| L7 語意防禦（應用層攻擊） | Python | TextCNN, PyTorch, NumPy |
| 流量分流代理 | Go / Python | Scapy, Nginx (Lua), FastAPI |
| 主動防禦 / 蜜罐 | Shell / Python / YAML | Falco, Auditd, Docker, Docker-compose, Crontab |
| 模型微調排程 | Python | APScheduler, SQLAlchemy, SQLite/MySQL |

## 三人實作分工

**同學 A｜網路流量控制與基礎防禦**
- ~~nftables 規則腳本撰寫（L3/L4 防禦）~~
- 開發分流代理伺服器（Reverse Proxy），依模型結果路由流量
- 實作流量鏡像複製（Scapy），引流至蜜罐
- ~~整合 CICIDS2017 資料集之 L3/L4 特徵擷取~~

**同學 B｜機器學習與特徵工程**
- 建立 Redis window-based 統計計數機制
- 訓練並部署 LightGBM 模型（含 CICIDS2017 / CSE-CIC-IDS2018 正常流量訓練）
- 設計增量訓練腳本（keep_current_trees），接收蜜罐反饋資料進行微調
- 建置中央資料庫（SQLite/MySQL）暫存誤報/漏報樣本

**同學 C｜深度學習與蜜罐反饋閉環**
- 收集 Web 攻擊語句資料集，設計並訓練 TextCNN 模型
- 部署蜜罐並配置 Falco / Auditd，自動產出行為判定報告
- 撰寫蜜罐行為審計腳本，自動識別誤報/漏報並寫入反饋資料庫
- 實作 TextCNN 增量微調機制，完成自強化閉環
- 為蜜罐設定定時自動重啟還原環境

## 專案結構

```
.
├── AI自強化防火牆系統設計文件.pdf   # 完整設計文件（架構、分工、技術堆疊）
├── project/                        # 專題本體（實作程式碼，開發中）
└── 練習或測試/                      # 個人練習與技術驗證用程式碼
    └── c/0/
        ├── 二分模型.py              # numpy 手刻二元分類神經網路練習（breast cancer 資料集）
        └── pyproject.toml
```

## Git 規範

針對專題本體（`project/` 資料夾內的東西）的 git 操作，嚴格遵循
[這部影片](https://www.youtube.com/watch?v=uj8hjLyEBmU) 的規則。
