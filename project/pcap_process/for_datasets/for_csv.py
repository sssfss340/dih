"""
CIC-IDS2018 流量統計特徵 CSV 的共用讀取／欄位比對邏輯。

這裡是 CSV 相關邏輯唯一的來源：
  * LightGBM（model/LightGBM/model.py）訓練／評估時呼叫這裡的 load_csv /
    load_dataset，不再自己重複實作一份。
  * export_custom_features 額外從同一批 CSV 抽出一小部分欄位另存新檔（例如
    HTTP Flood 分析用），欄位用 fuzzy match 對應。

csv/ 底下目前的 3 份 CSV 用的是 GintsEngelen 版 CICFlowMeter 的原始欄位名稱
（例如 "Total Fwd Packet"、"Flow Bytes/s"），不是官方 CIC-IDS2018 精簡過的
欄位名（"Tot Fwd Pkts"）。三份檔案欄位完全一致，所以 load_dataset 用「欄位
名稱交集」對齊即可；fuzzy match 只用在 export_custom_features，讓呼叫端可以
用官方精簡欄位名或口語化名稱描述想要的欄位。
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DIR))  # for_datasets -> pcap_process -> 專案根目錄
CSV_DIR = os.path.join(PROJECT_ROOT, "csv")

LABEL_COL = "Label"
# 這些欄位不是流量統計特徵，訓練前要排除：
#   Flow ID / *IP / *Port(來源) / Timestamp -- 識別資訊，不是特徵
#   id                  -- 單純的列序號。攻擊多半是連續一段時間擷取，id 區間
#                          會跟 Label 高度相關，當特徵會讓模型記「第幾列」
#                          而不是學流量特徵，屬於資料洩漏
#   Attempted Category  -- CICFlowMeter fork 自己算出來的「連線是否只是嘗試
#                          （attempted）」旗標，實測 -1/0/1 幾乎直接對應
#                          Label 有沒有 " - Attempted" 後綴，等於把答案洩漏
#                          給模型
DROP_COLS = {
    "id",
    "Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
    "Src Port", "Source Port", "Timestamp",
    "Attempted Category",
}

# 1. 納入包含 HTTP Flood / 大量請求癱瘓攻擊的目標 CSV 檔案
TARGET_FILES = [
    "Thursday-15-02-2018.csv",
    "Friday-16-02-2018.csv",
    "Wednesday-21-02-2018.csv",
]


def default_csv_paths() -> list[str]:
    """csv/ 底下所有 csv 檔（依檔名排序）。"""
    return sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))


def load_csv(path: str, chunksize: int = 100_000) -> tuple[np.ndarray, list[str], list[str]]:
    """讀單一份 CSV，回傳 (X, y_labels, feature_names)。

    用 pandas 分批讀取＋向量化轉型，而不是逐列跑 Python for-loop 呼叫
    float()：csv/ 底下單一檔案就有 500 萬～700 萬列 x 91 欄，逐格呼叫 float()
    在這個資料量下太慢（3 份合計近 2000 萬列）。DROP_COLS 在 usecols 階段就
    排除掉，不會被讀進記憶體。欄位名稱前後空白會被清掉。

    X 是 float32 的二維陣列（LightGBM 用不到 float64 的精度，改用 float32
    讓近 2000 萬列 x 80 幾欄的資料集省一半記憶體）；inf/-inf 與無法轉成數字
    的值一律換成 nan，LightGBM 原生把 nan 當缺值處理。
    """
    raw_header = pd.read_csv(path, nrows=0).columns.tolist()
    stripped = {raw: raw.strip() for raw in raw_header}
    header = list(stripped.values())
    if LABEL_COL not in header:
        raise ValueError(f"{path} 找不到 {LABEL_COL!r} 欄位")

    feat_names = [name for name in header if name != LABEL_COL and name not in DROP_COLS]
    raw_of = {stripped_name: raw for raw, stripped_name in stripped.items()}
    usecols_raw = [raw_of[name] for name in feat_names] + [raw_of[LABEL_COL]]

    X_chunks: list[np.ndarray] = []
    y_rows: list[str] = []
    for chunk in pd.read_csv(path, usecols=usecols_raw, chunksize=chunksize, low_memory=False):
        chunk = chunk.rename(columns=stripped)
        y_rows.extend(chunk[LABEL_COL].astype(str))
        feats = chunk[feat_names].apply(pd.to_numeric, errors="coerce")
        X_chunks.append(feats.to_numpy(dtype=np.float32))

    X = (np.concatenate(X_chunks, axis=0) if X_chunks
         else np.empty((0, len(feat_names)), dtype=np.float32))
    X[~np.isfinite(X)] = np.nan
    return X, y_rows, feat_names


def load_dataset(csv_paths: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """讀多份 CSV 並合併成一份資料集。

    各檔案欄位順序不一定完全相同，所以用欄位「名稱」對齊，取所有檔案共有的
    欄位（依第一份檔案的欄位順序排列）。
    """
    if not csv_paths:
        raise ValueError("csv_paths 不可為空")

    parts = [load_csv(p) for p in csv_paths]

    common = set(parts[0][2])
    for _, _, names in parts[1:]:
        common &= set(names)
    if not common:
        raise ValueError("多份 CSV 沒有共同的特徵欄位")
    feature_names = [n for n in parts[0][2] if n in common]

    X_list, y_list = [], []
    for X, y, names in parts:
        col_of = {name: i for i, name in enumerate(names)}
        reorder = [col_of[n] for n in feature_names]
        X_list.append(X[:, reorder])
        y_list.extend(y)

    return np.concatenate(X_list, axis=0), y_list, feature_names


def export_custom_features(requested_columns, output_filename="http_flood_extracted.csv"):
    combined_data = []

    for file_name in TARGET_FILES:
        file_path = os.path.join(CSV_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"找不到檔案：{file_path}，跳過處理。")
            continue

        print(f"正在讀取：{file_path} ...")
        try:
            # 讀取首列以抓取實際欄位名稱並清理前後空白
            preview = pd.read_csv(file_path, nrows=1)
            preview.columns = preview.columns.str.strip()
            all_cols = list(preview.columns)

            matched_cols = []
            for req in requested_columns:
                req_clean = req.lower().replace(" ", "").replace("_", "")

                # 模糊搜尋：只要原始欄位去掉空格後包含目標關鍵字，或目標關鍵字包含原始欄位
                found_col = None
                for col in all_cols:
                    col_clean = col.lower().replace(" ", "").replace("_", "")

                    # 比對關鍵字（包含 Port, Protocol, Label 或 Fwd/Bwd 封包）
                    if req_clean in col_clean or col_clean in req_clean:
                        found_col = col
                        break
                    # 特殊處理 Fwd / Bwd 封包相關欄位
                    elif "fwd" in req_clean and "fwd" in col_clean and ("pkt" in col_clean or "packet" in col_clean):
                        found_col = col
                        break
                    elif "bwd" in req_clean and "bwd" in col_clean and ("pkt" in col_clean or "packet" in col_clean):
                        found_col = col
                        break

                if found_col and found_col not in matched_cols:
                    matched_cols.append(found_col)
                else:
                    if not found_col:
                        print(f"警告：無法在 {file_name} 中找到與 '{req}' 對應的欄位")

            if not matched_cols:
                print(f"檔案 {file_name} 無可提取的欄位，跳過。")
                continue

            print(f"成功對應到的實際欄位：{matched_cols}")

            # 分批讀取指定欄位以確保記憶體穩定
            chunks = []
            for chunk in pd.read_csv(file_path, usecols=matched_cols, chunksize=100000, low_memory=False):
                chunk.columns = chunk.columns.str.strip()
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)
            combined_data.append(df)
            print(f"成功提取 {len(df)} 筆資料")

        except Exception as e:
            print(f"讀取 {file_name} 失敗：{e}")

    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        final_df.to_csv(output_filename, index=False)
        print("=" * 60)
        print(f"處理完成！提取資料已輸出至：{output_filename}")
        print(f"總筆數：{len(final_df)} 筆")
    else:
        print("未能提取任何數據。")


if __name__ == "__main__":
    USER_COLUMNS = ['Dst Port', 'Protocol', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Label']
    export_custom_features(requested_columns=USER_COLUMNS, output_filename="http_flood_extracted.csv")
