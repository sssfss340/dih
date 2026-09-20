import os
import pandas as pd

# 1. 納入包含 HTTP Flood / 大量請求癱瘓攻擊的目標 CSV 檔案
TARGET_FILES = [
    "Thursday-15-02-2018.csv",
    "Friday-16-02-2018.csv",
    "Wednesday-21-02-2018.csv"
]

def export_custom_features(requested_columns, output_filename="http_flood_extracted.csv"):
    combined_data = []

    for file_name in TARGET_FILES:
        if not os.path.exists(file_name):
            print(f"⚠️ 找不到檔案：{file_name}，跳過處理。")
            continue

        print(f"🔍 正在讀取：{file_name} ...")
        try:
            # 讀取首列以抓取實際欄位名稱並清理前後空白
            preview = pd.read_csv(file_name, nrows=1)
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
                        print(f"  ⚠️ 警告：無法在 {file_name} 中找到與 '{req}' 對應的欄位")

            if not matched_cols:
                print(f"❌ 檔案 {file_name} 無可提取的欄位，跳過。")
                continue

            print(f"🎯 成功對應到的實際欄位：{matched_cols}")

            # 分批讀取指定欄位以確保記憶體穩定
            chunks = []
            for chunk in pd.read_csv(file_name, usecols=matched_cols, chunksize=100000, low_memory=False):
                chunk.columns = chunk.columns.str.strip()
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)
            combined_data.append(df)
            print(f"✅ 成功提取 {len(df)} 筆資料")

        except Exception as e:
            print(f"❌ 讀取 {file_name} 失敗：{e}")

    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        final_df.to_csv(output_filename, index=False)
        print("=" * 60)
        print(f"🎉 處理完成！提取資料已輸出至：{output_filename}")
        print(f"📊 總筆數：{len(final_df)} 筆")
    else:
        print("⚠️ 未能提取任何數據。")

if __name__ == "__main__":
    USER_COLUMNS = ['Dst Port', 'Protocol', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Label']
    export_custom_features(requested_columns=USER_COLUMNS, output_filename="http_flood_extracted.csv")