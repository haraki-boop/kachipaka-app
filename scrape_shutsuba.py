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
    """直近の土日の日付(YYYYMMDD)を取得"""
    today = datetime.now()
    dates = []
    for i in range(7):
        d = today + timedelta(days=i)
        if d.weekday() in [5, 6]:
            dates.append(d.strftime("%Y%m%d"))
    return sorted(list(set(dates)))

def scrape_shutsuba():
    target_dates = get_target_dates()
    print(f"🏇 取得対象日: {target_dates}")
    all_race_ids = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://race.netkeiba.com/",
    }

    # 1. レースIDを取得（JavaScript用の裏エンドポイント race_list_sub.html を直接取得）
    for date_str in target_dates:
        url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.encoding = 'euc-jp'
            
            # レースID (12桁数字) を一括抽出
            found_ids = re.findall(r'race_id=(\d{12})', res.text)
            date_ids = []
            for rid in found_ids:
                if rid not in all_race_ids:
                    all_race_ids.append(rid)
                    date_ids.append(rid)
            print(f"  └ {date_str}: {len(date_ids)} 件のレースIDを検出")
            time.sleep(1)
        except Exception as e:
            print(f"日付 {date_str} の一覧取得でエラー: {e}")

    if not all_race_ids:
        print("❌ 対象日のレースIDが見つかりませんでした。")
        return

    print(f"\n🎉 合計 {len(all_race_ids)} レースが見つかりました！出馬表（全馬データ）を取得します...")

    # 2. 各レースの出馬表を取得
    race_data_list = []
    for i, race_id in enumerate(all_race_ids):
        print(f"[{i+1}/{len(all_race_ids)}] 取得中: {race_id}")
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.encoding = 'euc-jp'
            soup = BeautifulSoup(res.text, "html.parser")
            
            year = race_id[:4]
            rows = soup.select("tr.HorseList")
            
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 7:
                    waku = cols[0].text.strip()
                    umaban = cols[1].text.strip()
                    
                    horse_td = cols[3]
                    horse_name = horse_td.find("a").text.strip() if horse_td.find("a") else horse_td.text.strip()
                    
                    sex_age = cols[4].text.strip()
                    kinryo = cols[5].text.strip()
                    
                    jockey_td = cols[6]
                    jockey = jockey_td.find("a").text.strip() if jockey_td.find("a") else jockey_td.text.strip()
                    
                    if horse_name and umaban.isdigit():
                        race_data_list.append({
                            "race_id": race_id,
                            "date": f"{year}-00-00",
                            "枠番": waku,
                            "馬番": umaban,
                            "馬名": horse_name,
                            "sex_code": sex_age[0] if sex_age else "",
                            "age": sex_age[1:] if len(sex_age) > 1 else "",
                            "斤量": kinryo,
                            "騎手": jockey,
                        })
            time.sleep(random.uniform(0.3, 0.8))
            
        except Exception as e:
            print(f"  └ エラー: {e}")

    if race_data_list:
        df_future = pd.DataFrame(race_data_list)
        df_future.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
        print(f"\n✅ 全 {len(race_data_list)} 頭分のデータを {FUTURE_CSV} に保存完了しました！")
    else:
        print("❌ 出馬表の解析に失敗しました。")

if __name__ == "__main__":
    scrape_shutsuba()