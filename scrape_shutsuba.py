import os
import re
import time
import random
import requests
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

FUTURE_CSV = "future_races.csv"
ML_TARGET_CSV = "ml_target_data.csv"

def get_target_dates():
    today = datetime.now()
    dates = []
    for i in range(7):
        d = today + timedelta(days=i)
        if d.weekday() in [5, 6]:
            dates.append(d.strftime("%Y%m%d"))
    return sorted(list(set(dates)))

def clean_text(text):
    if not text: return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s\u3000]+', '', s)

# 👑やG1表記を完全に除去し、純粋なレース名だけにする関数
def clean_race_name(race_name):
    if not race_name: return ""
    s = str(race_name)
    # G1, Ｇ１, (G1), （G1）, 👑 などの表記をすべて消去
    s = re.sub(r'\(?G[1-3１-３I-V]+\)?|（?G[1-3１-３I-V]+）?|👑|G[1-3１-３]', '', s, flags=re.IGNORECASE)
    return s.strip()

def fetch_with_retry(session, url, max_retries=3):
    for i in range(max_retries):
        try:
            res = session.get(url, timeout=15)
            if res.status_code == 200:
                res.encoding = 'utf-8'
                return res
            else:
                print(f"    [警告] 通信エラー(コード: {res.status_code}) - 3秒待機して再試行します...")
        except Exception as e:
            print(f"    [エラー] {e} - 3秒待機して再試行します...")
        time.sleep(3)
    return None

def scrape_shutsuba():
    target_dates = get_target_dates()
    print(f"🏇 取得対象日: {target_dates}")
    all_race_ids = []
    id_to_date = {}

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://race.netkeiba.com/",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
    })

    for date_str in target_dates:
        urls_to_check = [
            f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}",
            f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
        ]
        
        for url in urls_to_check:
            res = fetch_with_retry(session, url)
            if res:
                found_ids = re.findall(r'race_id=["\']?(\d{12})["\']?', res.text)
                for rid in found_ids:
                    if rid not in all_race_ids:
                        all_race_ids.append(rid)
                        id_to_date[rid] = date_str
            time.sleep(1)

    if not all_race_ids:
        print("❌ 対象日のレースIDが見つかりませんでした。")
        return

    all_race_ids.sort()
    print(f"\n🎉 合計 {len(all_race_ids)} レースが見つかりました！出馬表を取得中...")
    
    race_data_list = []
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    
    for i, race_id in enumerate(all_race_ids):
        print(f"[{i+1}/{len(all_race_ids)}] 取得中: {race_id}")
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        res = fetch_with_retry(session, url)
        if not res:
            print(f"  └ ❌ 取得失敗（スキップします）: {race_id}")
            continue

        try:
            soup = BeautifulSoup(res.text, "html.parser")
            
            date_str = id_to_date.get(race_id, "")
            display_date = f"{datetime.strptime(date_str, '%Y%m%d').month}月{datetime.strptime(date_str, '%Y%m%d').day}日({weekdays[datetime.strptime(date_str, '%Y%m%d').weekday()]})" if date_str else "不明"
                
            raw_race_name = ""
            rn_elem = soup.find(class_="RaceName") or soup.find(class_="RaceList_Item02")
            if rn_elem:
                raw_race_name = clean_text(rn_elem.text)
            
            # ここで余計なグレード表記や👑を削除！
            final_race_name = clean_race_name(raw_race_name)
            
            rows = soup.select("tr.HorseList")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 7:
                    waku = clean_text(cols[0].text)
                    umaban = clean_text(cols[1].text)
                    horse_td = cols[3]
                    horse_name = clean_text(horse_td.find("a").text) if horse_td.find("a") else clean_text(horse_td.text)
                    sex_age = clean_text(cols[4].text)
                    kinryo = clean_text(cols[5].text)
                    jockey_td = cols[6]
                    jockey = clean_text(jockey_td.find("a").text) if jockey_td.find("a") else clean_text(jockey_td.text)
                    
                    odds_str = ""
                    if len(cols) >= 10:
                        odds_str = clean_text(cols[9].text)
                    
                    if horse_name and umaban.isdigit():
                        race_data_list.append({
                            "race_id": race_id,
                            "date": display_date,
                            "race_name": final_race_name,
                            "枠番": waku,
                            "馬番": umaban,
                            "馬名": horse_name,
                            "sex_code": sex_age[0] if sex_age else "",
                            "age": sex_age[1:] if len(sex_age) > 1 else "",
                            "斤量": kinryo,
                            "騎手": jockey,
                            "オッズ": odds_str
                        })
            time.sleep(random.uniform(0.5, 1.0))
            
        except Exception as e:
            print(f"  └ 解析エラー: {e}")

    if race_data_list:
        df_future = pd.DataFrame(race_data_list)
        df_future['オッズ'] = pd.to_numeric(df_future['オッズ'], errors='coerce')
        df_future['人気'] = df_future.groupby('race_id')['オッズ'].rank(method='min').astype('Int64')

        # 🎯 過去データからタイム指数などの特徴量を結合
        if os.path.exists(ML_TARGET_CSV):
            print(f"\n🔗 過去データ({ML_TARGET_CSV})からタイム指数などの特徴量を結合しています...")
            try:
                df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='utf-8-sig')
            except:
                df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='cp932')
            
            if '馬名' in df_past.columns:
                df_past['馬名_clean'] = df_past['馬名'].apply(clean_horse_name)
                df_future['馬名_clean'] = df_future['馬名'].apply(clean_horse_name)
                
                # 各馬の最新レースのみ残す
                if 'date' in df_past.columns:
                    df_past = df_past.sort_values('date')
                df_past_latest = df_past.drop_duplicates(subset='馬名_clean', keep='last')
                
                # 結合不要な重複カラムを除外
                cols_to_drop = ['race_id', 'date', 'race_name', '枠番', '馬番', '馬名', 'sex_code', 'age', '斤量', '騎手', 'オッズ', '人気', '単勝', '着順']
                cols_to_keep = [c for c in df_past_latest.columns if c not in cols_to_drop]
                df_past_latest = df_past_latest[cols_to_keep]
                
                # 出馬表に合体
                df_future = pd.merge(df_future, df_past_latest, on='馬名_clean', how='left')
                df_future.drop(columns=['馬名_clean'], inplace=True)
                print("✅ 特徴量の結合が完了しました。")

        df_future.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
        print(f"\n✅ {len(race_data_list)}頭のデータを保存完了！ ({FUTURE_CSV})")
    else:
        print("\n❌ データが1件も取得できませんでした。")

if __name__ == "__main__":
    scrape_shutsuba()