import os
import re
import time
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

ML_TARGET_CSV = "ml_target_data.csv"

def get_past_weekend_dates():
    # 🌟 NEW: 手動で取得したい日付をここに直接指定します（YYYYMMDD形式）
    # 取得したい日を追加したり減らしたりできます。
    manual_dates = ["20260822", "20260823"]
    
    if manual_dates:
        return sorted(list(set(manual_dates)))

    # もし manual_dates = [] と空に設定した場合は、自動で直近3日間の土日だけを取得します
    today = datetime.now()
    dates = []
    for i in range(0, 3):
        d = today - timedelta(days=i)
        if d.weekday() in [5, 6]:
            dates.append(d.strftime("%Y%m%d"))
    return sorted(list(set(dates)))

def clean_text(text):
    if not text: return ""
    return re.sub(r'[\s\u3000]+', '', str(text)).strip()

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

def fetch_race_results():
    target_dates = get_past_weekend_dates()
    print(f"📥 取得対象日: {target_dates}")
    
    if not target_dates:
        print("ℹ️ 取得対象の日付がありません。")
        return

    print("🌐 ブラウザ（Selenium）をバックグラウンドで起動中...")
    try:
        driver = setup_driver()
    except Exception as e:
        print(f"❌ Seleniumの起動に失敗しました: {e}")
        return

    all_race_ids = []
    try:
        for date_str in target_dates:
            urls = [
                f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}",
                f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
            ]
            for url in urls:
                try:
                    driver.get(url)
                    time.sleep(1.5)
                    found_ids = re.findall(r'race_id=["\']?(\d{12})["\']?', driver.page_source)
                    for rid in found_ids:
                        if rid not in all_race_ids:
                            all_race_ids.append(rid)
                except Exception:
                    pass

        if not all_race_ids:
            print("❌ 対象日のレースIDが見つかりませんでした。")
            return

        all_race_ids.sort()
        print(f"\n🎉 合計 {len(all_race_ids)} レースの結果を取得開始します...")

        results_list = []
        
        for i, race_id in enumerate(all_race_ids):
            print(f"[{i+1}/{len(all_race_ids)}] 結果取得中: {race_id}")
            result_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
            
            try:
                driver.get(result_url)
                time.sleep(2.0)
                
                soup = BeautifulSoup(driver.page_source, "html.parser")
                
                # 日付 (YYYY-MM-DD または YYYY年MM月DD日)
                date_str = ""
                dt_elem = soup.find("dd", class_="Active") or soup.find(class_="RaceData02")
                if dt_elem:
                    m = re.search(r'(\d+年\d+月\d+日|\d+/\d+/\d+|\d+月\d+日)', dt_elem.text)
                    if m:
                        raw_d = m.group(1)
                        if "年" not in raw_d and "/" not in raw_d:
                            year = target_dates[0][:4]
                            raw_d = f"{year}年{raw_d}"
                        date_str = raw_d

                course_info = ""
                c_elem = soup.find("div", class_="RaceData01")
                if c_elem: course_info = clean_text(c_elem.text)
                
                dist_m = re.search(r'(\d+)m', course_info)
                distance = dist_m.group(1) if dist_m else ""
                surface = "芝" if "芝" in course_info else ("ダート" if ("ダ" in course_info or "ダート" in course_info) else "障害")
                
                condition = "良"
                if "稍" in course_info: condition = "稍重"
                elif "重" in course_info: condition = "重"
                elif "不" in course_info: condition = "不良"

                race_name = ""
                rn_elem = soup.find(class_="RaceName")
                if rn_elem: race_name = clean_text(rn_elem.text)

                table = soup.find("table", id="All_Result_Table") or soup.find("table", class_=re.compile("ResultTable"))
                if not table: continue
                
                rows = table.find_all("tr")
                for row in rows:
                    tds = row.find_all("td")
                    if len(tds) < 10: continue
                    
                    try:
                        rank = clean_text(tds[0].text)
                        if not rank.isdigit(): continue
                            
                        waku = clean_text(tds[1].text)
                        umaban = clean_text(tds[2].text)
                        
                        h_a = tds[3].find("a")
                        horse_name = clean_text(h_a.text) if h_a else clean_text(tds[3].text)
                        
                        sex_age = clean_text(tds[4].text)
                        kinryo = clean_text(tds[5].text)
                        
                        j_a = tds[6].find("a")
                        jockey = clean_text(j_a.text) if j_a else clean_text(tds[6].text)
                        
                        time_str = clean_text(tds[7].text)
                        margin = clean_text(tds[8].text) if len(tds) > 8 else ""
                        
                        passing, last3f, odds, pop, weight_info = "", "", "", "", ""
                        for td in tds[9:]:
                            txt = clean_text(td.text)
                            if re.search(r'^\d+-\d+', txt): passing = txt
                            elif re.search(r'^\d{2}\.\d$', txt): last3f = txt
                            elif re.search(r'^\d+\.\d$', txt) and not odds: odds = txt
                            elif txt.isdigit() and len(txt) <= 2 and not pop: pop = txt
                            elif "(" in txt and ")" in txt: weight_info = txt

                        results_list.append({
                            "race_id": str(race_id),
                            "date": date_str,
                            "race_name": race_name,
                            "着順": int(rank) if rank.isdigit() else rank,
                            "枠番": int(waku) if waku.isdigit() else waku,
                            "馬番": int(umaban) if umaban.isdigit() else umaban,
                            "馬名": horse_name,
                            "性齢": sex_age,
                            "斤量": kinryo,
                            "騎手": jockey,
                            "タイム": time_str,
                            "着差": margin,
                            "通過": passing,
                            "上がり": last3f,
                            "単勝": odds,
                            "人気": pop,
                            "馬体重": weight_info,
                            "distance": distance,
                            "surface": surface,
                            "condition": condition
                        })
                    except Exception:
                        pass
            except Exception as e:
                print(f"  └ {race_id} 取得エラー: {e}")

    finally:
        driver.quit()

    if results_list:
        df_new = pd.DataFrame(results_list)
        print(f"\n✅ {len(df_new)} 件（{len(all_race_ids)} レース分）の結果を取得しました。")
        
        if os.path.exists(ML_TARGET_CSV):
            try:
                df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='utf-8-sig')
            except:
                df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='cp932')
            
            # 数値型・文字列型を厳密に統一
            df_past['race_id_clean'] = df_past['race_id'].astype(str).str.replace('.0', '', regex=False)
            df_past['umaban_clean'] = pd.to_numeric(df_past['馬番'], errors='coerce').fillna(0).astype(int).astype(str)
            df_past['uid'] = df_past['race_id_clean'] + "_" + df_past['umaban_clean']

            df_new['race_id_clean'] = df_new['race_id'].astype(str).str.replace('.0', '', regex=False)
            df_new['umaban_clean'] = pd.to_numeric(df_new['馬番'], errors='coerce').fillna(0).astype(int).astype(str)
            df_new['uid'] = df_new['race_id_clean'] + "_" + df_new['umaban_clean']
            
            # 過去データに含まれない新規レコードのみ抽出
            df_append = df_new[~df_new['uid'].isin(df_past['uid'])].copy()
            
            # 一時列削除
            df_past.drop(columns=['race_id_clean', 'umaban_clean', 'uid'], errors='ignore', inplace=True)
            df_append.drop(columns=['race_id_clean', 'umaban_clean', 'uid'], errors='ignore', inplace=True)

            if not df_append.empty:
                df_combined = pd.concat([df_past, df_append], ignore_index=True)
                df_combined.to_csv(ML_TARGET_CSV, index=False, encoding='utf-8-sig')
                print(f"🎉 {len(df_append)} 件（{len(df_append['race_id'].unique())}レース分）の新規結果を {ML_TARGET_CSV} に書き込みました！")
            else:
                print(f"ℹ️ すでに最新データが保存されています。")
        else:
            df_new.to_csv(ML_TARGET_CSV, index=False, encoding='utf-8-sig')
            print(f"🎉 新規ファイル {ML_TARGET_CSV} を作成し保存しました。")
    else:
        print("❌ レース結果データが1件も取得できませんでした。")

if __name__ == "__main__":
    fetch_race_results()