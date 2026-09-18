import os
import re
import time
import random
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

MASTER_DB_CSV = "keiba_database.csv"
TARGET_START_DATE = "2026-08-08"

def clean_and_prepare_db():
    print(f"🧹 {MASTER_DB_CSV} をクリーンアップします（手動編集による文字化けを防止）...")
    df = pd.read_csv(MASTER_DB_CSV, low_memory=False, encoding='utf-8-sig')
    
    # 日付パース
    date_clean = df['date'].astype(str).str.replace('年', '-').str.replace('月', '-').str.replace('日', '').str.replace('/', '-')
    df['date_parsed'] = pd.to_datetime(date_clean, errors='coerce')
    
    # 2026/08/08以降の行を削除（前回の中途半端なデータをリセット）
    cutoff = pd.to_datetime(TARGET_START_DATE)
    initial_len = len(df)
    df_clean = df[df['date_parsed'] < cutoff].copy()
    df_clean.drop(columns=['date_parsed'], inplace=True)
    
    removed_count = initial_len - len(df_clean)
    df_clean.to_csv(MASTER_DB_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ {removed_count}件のテスト/不完全データを削除し、初期化しました。")
    
    return list(df_clean.columns)

def get_target_dates():
    start_date = datetime.strptime(TARGET_START_DATE.replace("-", ""), "%Y%m%d")
    end_date = datetime.now()
    dates = []
    curr = start_date
    while curr <= end_date:
        if curr.weekday() in [5, 6]: # 土日のみ
            dates.append(curr.strftime("%Y%m%d"))
        curr += timedelta(days=1)
    return dates

def clean_text(text):
    if not text: return ""
    s = f"{text}".replace('\n', '').replace('\r', '').replace('\t', '')
    return re.sub(r'[\s\u3000]+', '', s).strip()

def setup_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36')
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(20)
    driver.implicitly_wait(5)
    return driver

def fetch_race_results():
    if not os.path.exists(MASTER_DB_CSV):
        print(f"❌ {MASTER_DB_CSV} が見つかりません。")
        return

    # 1. データベースのクリーンアップとマスターカラムの取得
    master_columns = clean_and_prepare_db()
    
    # 2. 対象日の算出
    target_dates = get_target_dates()
    print(f"📥 取得対象日: {len(target_dates)}日分（{TARGET_START_DATE} 以降）")
    
    if not target_dates:
        print("🎉 対象日がありません。")
        return

    for date_str in target_dates:
        print(f"\n========================================")
        print(f"🏇 開催日: {date_str} のデータ取得を開始します")
        print(f"========================================")
        
        driver = setup_driver()
        race_ids_for_date = []
        
        try:
            urls = [
                f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}",
                f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
            ]
            for url in urls:
                try:
                    driver.get(url)
                    time.sleep(random.uniform(0.8, 1.0))
                    found_ids = re.findall(r'race_id=["\']?(\d{12})["\']?', driver.page_source)
                    for rid in found_ids:
                        if rid not in race_ids_for_date:
                            race_ids_for_date.append(rid)
                except TimeoutException:
                    pass
                except Exception:
                    pass

            if not race_ids_for_date:
                print(f"⚠️ {date_str} のレースは見つかりませんでした。スキップします。")
                driver.quit()
                continue

            race_ids_for_date.sort()
            print(f"🎯 {len(race_ids_for_date)} レースの結果を取得中...")

            results_list = []
            for i, race_id in enumerate(race_ids_for_date):
                print(f"  [{i+1}/{len(race_ids_for_date)}] {race_id}")
                result_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
                
                for attempt in range(2):
                    try:
                        driver.get(result_url)
                        time.sleep(random.uniform(0.8, 1.2))
                        soup = BeautifulSoup(driver.page_source, "html.parser")
                        break 
                    except TimeoutException:
                        if attempt == 0: time.sleep(2)
                        else: soup = None
                    except Exception:
                        soup = None
                        break
                
                if not soup: continue
                
                # 必須4項目
                race_date_str = ""
                dt_elem = soup.find("dd", class_="Active") or soup.find(class_="RaceData02")
                if dt_elem:
                    m = re.search(r'(\d+年\d+月\d+日|\d+/\d+/\d+|\d+月\d+日)', dt_elem.text)
                    if m: race_date_str = m.group(1)

                surface, distance, condition = "芝", "", "良"
                c_elem = soup.find("div", class_="RaceData01")
                if c_elem:
                    info_text = clean_text(c_elem.text)
                    dist_m = re.search(r'(\d+)m', info_text)
                    if dist_m: distance = dist_m.group(1)
                    if "ダ" in info_text or "ダート" in info_text: surface = "ダート"
                    elif "障" in info_text: surface = "障害"
                    if "稍" in info_text: condition = "稍重"
                    elif "重" in info_text: condition = "重"
                    elif "不" in info_text: condition = "不良"

                table = soup.find("table", id="All_Result_Table") or soup.find("table", class_=re.compile("ResultTable"))
                if not table: continue
                
                for row in table.find_all("tr"):
                    tds = row.find_all("td")
                    if len(tds) < 13: continue
                    try:
                        rank = clean_text(tds[0].text)
                        if not rank.isdigit() and rank not in ['取', '除', '中', '失']: continue
                            
                        waku, umaban = clean_text(tds[1].text), clean_text(tds[2].text)
                        h_a = tds[3].find("a")
                        horse_name = clean_text(h_a.text) if h_a else clean_text(tds[3].text)
                        sex_age, kinryo = clean_text(tds[4].text), clean_text(tds[5].text)
                        j_a = tds[6].find("a")
                        jockey = clean_text(j_a.text) if j_a else clean_text(tds[6].text)
                        time_str, margin = clean_text(tds[7].text), clean_text(tds[8].text)
                        
                        passing, last3f, odds, pop, weight_info = "", "", "", "", ""
                        trainer, owner, prize = "", "", ""

                        # 調教師・馬主を確実にとる（リンクから抽出）
                        a_tags = row.find_all("a")
                        for a in a_tags:
                            href = a.get('href', '')
                            if '/trainer/' in href and not trainer:
                                trainer = clean_text(a.text)
                            elif '/owner/' in href and not owner:
                                owner = clean_text(a.text)

                        for td in tds[9:]:
                            txt = clean_text(td.text)
                            if re.match(r'^\d+-\d+', txt): passing = txt
                            elif re.match(r'^\d{2}\.\d$', txt): last3f = txt
                            elif re.match(r'^\d+\.\d$', txt) and not odds: odds = txt
                            elif txt.isdigit() and len(txt) <= 2 and not pop: pop = txt
                            elif re.match(r'^\d{3}\([+-]?\d+\)$', txt): weight_info = txt
                            # 賞金パターン (例: 1,165.2 または 467.2 など、数字とカンマ・ピリオドのみ)
                            elif ("万円" in txt or "." in txt or "," in txt) and not prize:
                                if re.search(r'\d', txt) and not re.search(r'[a-zA-Z]', txt):
                                    prize = txt.replace('万円', '').replace(',', '')

                        row_dict = {col: "" for col in master_columns}
                        row_dict["着順"] = rank
                        row_dict["枠番"] = waku
                        row_dict["馬番"] = umaban
                        row_dict["馬名"] = horse_name
                        row_dict["性齢"] = sex_age
                        row_dict["斤量"] = kinryo
                        row_dict["騎手"] = jockey
                        row_dict["タイム"] = time_str
                        row_dict["着差"] = margin
                        row_dict["通過"] = passing
                        row_dict["上り"] = last3f
                        row_dict["単勝"] = odds
                        row_dict["人気"] = pop
                        row_dict["馬体重"] = weight_info
                        row_dict["調教師"] = trainer
                        row_dict["馬主"] = owner
                        row_dict["賞金(万円)"] = prize
                        row_dict["race_id"] = str(race_id)
                        row_dict["date"] = race_date_str
                        row_dict["surface"] = surface
                        row_dict["distance"] = distance
                        row_dict["condition"] = condition

                        results_list.append(row_dict)
                    except Exception:
                        pass
            
            if results_list:
                df_new = pd.DataFrame(results_list)
                df_new = df_new.reindex(columns=master_columns)
                df_new['race_id'] = df_new['race_id'].astype(str)
                
                try:
                    df_past = pd.read_csv(MASTER_DB_CSV, low_memory=False, encoding='utf-8-sig')
                except:
                    df_past = pd.read_csv(MASTER_DB_CSV, low_memory=False, encoding='cp932')
                    
                df_past['race_id'] = df_past['race_id'].astype(str).str.replace('.0', '', regex=False)
                existing_ids = set(df_past['race_id'])
                df_append = df_new[~df_new['race_id'].isin(existing_ids)].copy()
                
                if not df_append.empty:
                    df_combined = pd.concat([df_past, df_append], ignore_index=True)
                    df_combined.to_csv(MASTER_DB_CSV, index=False, encoding='utf-8-sig')
                    print(f"💾 {date_str} のデータ（{len(df_append)}行）をマスターに追記セーブしました！")
                else:
                    print(f"ℹ️ {date_str} の追加データはありませんでした（取得済み）。")
            
        finally:
            driver.quit()
            
    print("\n✅ 全ての不足データの取得が完了しました。")

if __name__ == "__main__":
    fetch_race_results()