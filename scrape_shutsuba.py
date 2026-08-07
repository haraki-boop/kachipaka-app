import os
import re
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

FUTURE_CSV = "future_races.csv"

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

def detect_grade(race_name, soup):
    grade_prefix = ""
    grade_icon = soup.find(class_=re.compile(r'Icon_GradeType\d+'))
    if grade_icon:
        cls_str = " ".join(grade_icon.get('class', []))
        if 'GradeType1' in cls_str: grade_prefix = "👑 G1 "
        elif 'GradeType2' in cls_str: grade_prefix = "👑 G2 "
        elif 'GradeType3' in cls_str: grade_prefix = "👑 G3 "
            
    if not grade_prefix:
        if re.search(r'G1|Ｇ１|ＧＩ|\(G1\)', race_name, re.IGNORECASE): grade_prefix = "👑 G1 "
        elif re.search(r'G2|Ｇ２|ＧⅡ|\(G2\)', race_name, re.IGNORECASE): grade_prefix = "👑 G2 "
        elif re.search(r'G3|Ｇ３|ＧⅢ|\(G3\)', race_name, re.IGNORECASE): grade_prefix = "👑 G3 "

    clean_name = re.sub(r'\(?G[123１２３ＩⅡⅢ]\)?', '', race_name).strip()
    return f"{grade_prefix}{clean_name}".strip()

def fetch_with_retry(session, url, max_retries=3):
    """通信が切断されたりブロックされた場合に、自動で再試行する機能"""
    for i in range(max_retries):
        try:
            res = session.get(url, timeout=15)
            if res.status_code == 200:
                return res
            else:
                print(f"    [警告] 通信エラー(コード: {res.status_code}) - 3秒待機して再試行します...")
        except Exception as e:
            print(f"    [エラー] {e} - 3秒待機して再試行します...")
        time.sleep(3) # エラー時は長めに待機
    return None

def scrape_shutsuba():
    target_dates = get_target_dates()
    print(f"🏇 取得対象日: {target_dates}")
    all_race_ids = []
    id_to_date = {}

    # セッションを作成（同じブラウザからのアクセスだと認識させ、ブロックを防ぐ）
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://race.netkeiba.com/",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
    })

    # 1. レースIDの収集（2つの取得ルートを使って取りこぼしを完全防止）
    for date_str in target_dates:
        urls_to_check = [
            f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}",
            f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
        ]
        
        for url in urls_to_check:
            res = fetch_with_retry(session, url)
            if res:
                # バイトデータから強制的にEUC-JPとしてデコード（文字化けの根本解決）
                html_text = res.content.decode('euc-jp', errors='replace')
                found_ids = re.findall(r'race_id=["\']?(\d{12})["\']?', html_text)
                for rid in found_ids:
                    if rid not in all_race_ids:
                        all_race_ids.append(rid)
                        id_to_date[rid] = date_str
            time.sleep(1)

    if not all_race_ids:
        print("❌ 対象日のレースIDが見つかりませんでした。")
        return

    all_race_ids.sort() # レースを順番通りに整列
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
            # バイトデータから強制的にEUC-JPとしてデコード
            html_text = res.content.decode('euc-jp', errors='replace')
            soup = BeautifulSoup(html_text, "html.parser")
            
            date_str = id_to_date.get(race_id, "")
            display_date = f"{datetime.strptime(date_str, '%Y%m%d').month}月{datetime.strptime(date_str, '%Y%m%d').day}日({weekdays[datetime.strptime(date_str, '%Y%m%d').weekday()]})" if date_str else "不明"
                
            raw_race_name = ""
            rn_elem = soup.find(class_="RaceName") or soup.find(class_="RaceList_Item02")
            if rn_elem:
                raw_race_name = clean_text(rn_elem.text)
            
            final_race_name = detect_grade(raw_race_name, soup)
            
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
                        })
            # ブロック回避のため、待機時間を「1秒〜1.5秒」に増加
            time.sleep(random.uniform(1.0, 1.5))
            
        except Exception as e:
            print(f"  └ 解析エラー: {e}")

    if race_data_list:
        df_future = pd.DataFrame(race_data_list)
        # utf-8-sig (BOM付きUTF-8)で保存し、Windows環境での文字化けを防ぐ
        df_future.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
        print(f"\n✅ {len(race_data_list)}頭のデータを保存完了！ ({FUTURE_CSV})")
    else:
        print("\n❌ データが1件も取得できませんでした。")

if __name__ == "__main__":
    scrape_shutsuba()