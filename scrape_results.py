import os
import re
import time
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

FUTURE_CSV = "future_races.csv"

def get_target_dates():
    today = datetime.now()
    dates = []
    # 土日・祝日の変則開催を取りこぼさないよう、本日から直近8日間をすべてリストアップ
    # （レースが開催されていない日は、後続の処理で自動的にスキップされます）
    for i in range(8):
        d = today + timedelta(days=i)
        dates.append(d.strftime("%Y%m%d"))
    return sorted(list(set(dates)))

def clean_text(text):
    if not text: return ""
    return re.sub(r'[\s\u3000]+', '', str(text)).strip()

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s\u3000]+', '', s)

def clean_race_name(race_name):
    if not race_name: return ""
    s = str(race_name)
    s = re.sub(r'\(?G[1-3１-３I-V]+\)?|（?G[1-3１-３I-V]+）?|👑|G[1-3１-３]', '', s, flags=re.IGNORECASE)
    return s.strip()

def setup_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36')
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(5)
    return driver

def scrape_shutsuba():
    target_dates = get_target_dates()
    print(f"🏇 取得対象日（候補）: {target_dates}")
    all_race_ids = []
    id_to_date = {}

    print("🌐 ブラウザ（Selenium）をバックグラウンドで起動中...")
    try:
        driver = setup_driver()
    except Exception as e:
        print(f"❌ Seleniumの起動に失敗しました。\n詳細: {e}")
        return

    try:
        # --- 1. レースIDの取得 ---
        for date_str in target_dates:
            urls_to_check = [
                f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}",
                f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
            ]
            for url in urls_to_check:
                try:
                    driver.get(url)
                    time.sleep(2)
                    html = driver.page_source
                    found_ids = re.findall(r'race_id=["\']?(\d{12})["\']?', html)
                    for rid in found_ids:
                        if rid not in all_race_ids:
                            all_race_ids.append(rid)
                            id_to_date[rid] = date_str
                except Exception:
                    pass

        if not all_race_ids:
            print("❌ 対象期間内にレースIDが見つかりませんでした。")
            return

        all_race_ids.sort()
        print(f"\n🎉 合計 {len(all_race_ids)} レースが見つかりました！データを取得中...")
        
        race_data_list = []
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        
        # --- 2. 出馬表の取得 ---
        for i, race_id in enumerate(all_race_ids):
            print(f"[{i+1}/{len(all_race_ids)}] 取得中: {race_id}")
            place_code = int(str(race_id)[4:6])
            domain = "race.netkeiba.com" if place_code <= 10 else "nar.netkeiba.com"
            shutuba_url = f"https://{domain}/race/shutuba.html?race_id={race_id}"
            
            try:
                driver.get(shutuba_url)
                time.sleep(3) # オッズ読み込み待ち
                
                soup = BeautifulSoup(driver.page_source, "html.parser")
                
                date_str = id_to_date.get(race_id, "")
                dt_obj = datetime.strptime(date_str, '%Y%m%d') if date_str else None
                display_date = f"{dt_obj.month}月{dt_obj.day}日({weekdays[dt_obj.weekday()]})" if dt_obj else "不明"
                    
                raw_race_name = ""
                rn_elem = soup.find(class_="RaceName") or soup.find(class_="RaceList_Item02")
                if rn_elem: raw_race_name = clean_text(rn_elem.text)
                final_race_name = clean_race_name(raw_race_name)

                temp_horse_list = []

                for tr in soup.find_all("tr", class_=re.compile("HorseList")):
                    tds = tr.find_all("td")
                    if len(tds) < 8: continue
                    
                    try:
                        waku = clean_text(tds[0].text)
                        umaban = clean_text(tds[1].text)
                        if not umaban.isdigit(): continue

                        horse_td = tds[3]
                        horse_name = clean_text(horse_td.find("a").text) if horse_td.find("a") else clean_text(horse_td.text)
                        sex_age = clean_text(tds[4].text)
                        kinryo = clean_text(tds[5].text)
                        jockey_td = tds[6]
                        jockey = clean_text(jockey_td.find("a").text) if jockey_td.find("a") else clean_text(jockey_td.text)
                        trainer_td = tds[7]
                        trainer = clean_text(trainer_td.find("a").text) if trainer_td.find("a") else clean_text(trainer_td.text)
                        
                        o_val, p_val = None, None
                        
                        for td in tds:
                            cls_str = " ".join(td.get('class', [])).lower()
                            txt = clean_text(td.text)
                            
                            if ("odds" in cls_str or "txt_r" in cls_str) and re.search(r'\d+\.\d+', txt):
                                m = re.search(r'(\d+\.\d+)', txt)
                                if m: o_val = float(m.group(1))
                                
                            if "pop" in cls_str or "ninki" in cls_str:
                                m = re.search(r'(\d+)', txt)
                                if m: p_val = int(m.group(1))

                        if o_val is None and len(tds) > 9:
                            m = re.search(r'(\d+\.\d+)', clean_text(tds[9].text))
                            if m: o_val = float(m.group(1))
                        if p_val is None and len(tds) > 10:
                            m = re.search(r'^(\d+)$', clean_text(tds[10].text))
                            if m: p_val = int(m.group(1))

                        temp_horse_list.append({
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
                            "調教師": trainer,
                            "オッズ": o_val,
                            "人気": p_val
                        })
                    except Exception:
                        continue
                
                has_missing_pop = any(h["オッズ"] is not None and h["人気"] is None for h in temp_horse_list)
                if has_missing_pop:
                    valid_odds = [h for h in temp_horse_list if h["オッズ"] is not None]
                    valid_odds.sort(key=lambda x: x["オッズ"])
                    for rank, horse in enumerate(valid_odds, 1):
                        if horse["人気"] is None:
                            horse["人気"] = rank

                race_data_list.extend(temp_horse_list)
                
            except Exception as e:
                print(f"  └ 解析エラー: {e}")

    finally:
        driver.quit()

    # --- 3. CSVへの純粋な保存処理（余計なマージはしない） ---
    if race_data_list:
        df_future = pd.DataFrame(race_data_list)
        df_future['オッズ'] = pd.to_numeric(df_future['オッズ'], errors='coerce')
        df_future['人気'] = pd.to_numeric(df_future['人気'], errors='coerce').astype('Int64')
        df_future.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
        print(f"\n✅ {len(race_data_list)}頭分の純粋な出馬表データを {FUTURE_CSV} に保存完了！")
    else:
        print("\n❌ データが1件も取得できませんでした。")

if __name__ == "__main__":
    scrape_shutsuba()