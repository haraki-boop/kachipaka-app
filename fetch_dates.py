import os
import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
import sys

CSV_FILE = "cleaned_keiba_data.csv"

def get_date_from_netkeiba(race_id: str) -> str:
    """netkeibaから正確な年月日（例: 2024-01-06）を取得"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 403:
            print(f"\n🚨【アクセス拒否】IPがブロックされました。回線を切り替えてください。")
            sys.exit()
        elif res.status_code != 200:
            return None
            
        res.encoding = 'euc-jp'
        soup = BeautifulSoup(res.text, "html.parser")
        
        # 開催日の記述を抽出
        smalltxt = soup.find("p", class_="smalltxt")
        if smalltxt:
            text = smalltxt.text
            match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
            if match:
                y, m, d = match.groups()
                return f"{y}-{int(m):02d}-{int(d):02d}"
        return None
    except Exception as e:
        print(f"  通信エラー (race_id: {race_id}): {e}")
        return None

def is_valid_date(val) -> bool:
    """値が正しい日付形式（YYYY-MM-DD や YYYY年MM月DD日）か判定"""
    if pd.isna(val):
        return False
    val_str = str(val).strip()
    if val_str in ['', '不明', 'None', 'nan', 'null', '-', '0', '0000-00-00']:
        return False
    # 数字4桁が含まれているか確認
    return bool(re.search(r'\d{4}', val_str))

def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌ エラー: {CSV_FILE} が見つかりません。")
        return

    print(f"📂 {CSV_FILE} を解析中...")
    df = pd.read_csv(CSV_FILE, dtype=str)

    # 日付列の特定（無ければ 'date' 列を新規作成）
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', '日付', 'date_str', '開催日']:
            date_col = col
            break
            
    if not date_col:
        date_col = 'date'
        df[date_col] = None
        print("💡 日付列が存在しないため 'date' 列を新規作成しました。")

    # 日付が無効な行を特定
    invalid_mask = ~df[date_col].apply(is_valid_date)
    target_race_ids = df[invalid_mask]['race_id'].dropna().unique()

    print(f"📊 検証結果: 総データ {len(df):,} 行 / 日付不明のレース: {len(target_race_ids):,} 件")

    if len(target_race_ids) == 0:
        print("🎉 すべてのレースに正常な日付が割り振られています。")
        return

    print(f"\n==================== 不明日付の補正処理を開始 ({len(target_race_ids)}件) ====================")
    
    race_date_map = {}
    processed = 0

    for race_id in target_race_ids:
        race_id_str = str(race_id).strip()
        print(f"[{processed + 1}/{len(target_race_ids)}] 取得中... race_id: {race_id_str}")
        
        date_val = get_date_from_netkeiba(race_id_str)
        
        if date_val:
            race_date_map[race_id_str] = date_val
            print(f"  └ 成功: {date_val}")
        else:
            print(f"  └ 失敗")

        processed += 1

        # 10件ごと、または最後にCSVに反映して即時保存
        if processed % 10 == 0 or processed == len(target_race_ids):
            for r_id, d_str in race_date_map.items():
                df.loc[df['race_id'] == r_id, date_col] = d_str
            df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
            print(f"💾 進捗保存完了（{processed}/{len(target_race_ids)} レース完了）\n")

        time.sleep(5.0)  # 安全用5秒待機

    print("全データの日付補正が完了しました！")

if __name__ == "__main__":
    main()