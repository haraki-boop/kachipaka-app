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
    return text.strip().replace('\ufffd', '') # 不正な記号を除去

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

def scrape_shutsuba():
    target_dates = get_target_dates()
    print(f"🏇 取得対象日: {target_dates}")
    all_race_ids = []
    id_to_date = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://race.netkeiba.com/",
    }

    for date_str in target_dates:
        url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.encoding = 'euc-jp' # 標準のeuc-jpに戻す
            found_ids = re.findall(r'race_id=(\d{12})', res.text)
            for rid in found_ids:
                if rid not in all_race_ids:
                    all_race_ids.append(rid)
                    id_to_date[rid] = date_str
            time.sleep(1)
        except Exception as e:
            print(f"一覧取得エラー: {e}")

    if not all_race_ids:
        print("❌ 対象日のレースIDが見つかりませんでした。")
        return

    print(f"\n🎉 {len(all_race_ids)} レースの出馬表を取得中...")
    race_data_list = []
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    
    for i, race_id in enumerate(all_race_ids):
        print(f"[{i+1}/{len(all_race_ids)}] 取得中: {race_id}")
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.encoding = 'euc-jp'
            soup = BeautifulSoup(res.text, "html.parser")
            
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
            time.sleep(random.uniform(0.3, 0.6))
        except Exception as e:
            print(f"  └ エラー: {e}")

    if race_data_list:
        df_future = pd.DataFrame(race_data_list)
        # utf-8-sigで保存する
        df_future.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
        print(f"\n✅ 保存完了！ ({FUTURE_CSV})")

if __name__ == "__main__":
    scrape_shutsuba()