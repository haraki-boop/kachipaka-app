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
ML_TARGET_CSV = "ml_target_data.csv"

def get_target_dates():
    today = datetime.now()
    dates = []
    # 今週の土日を取得
    for i in range(7):
        d = today + timedelta(days=i)
        if d.weekday() in [5, 6]:
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
    """Seleniumの初期設定（画面を表示しない裏側起動）"""
    options = Options()
    options.add_argument('--headless=new')  # バックグラウンドで動かす設定
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # 一般的なChromeブラウザに偽装
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36')
    
    # ドライバーを起動（最近のSeleniumは自動でドライバーを準備してくれます）
    driver = webdriver.Chrome(options=options)
    # 暗黙の待機時間を設定
    driver.implicitly_wait(5)
    return driver

def scrape_shutsuba():
    target_dates = get_target_dates()
    print(f"🏇 取得対象日: {target_dates}")
    all_race_ids = []
    id_to_date = {}

    print("🌐 ブラウザ（Selenium）をバックグラウンドで起動中...")
    try:
        driver = setup_driver()
    except Exception as e:
        print(f"❌ Seleniumの起動に失敗しました。Google Chromeがインストールされているか確認してください。\n詳細: {e}")
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
                    time.sleep(2) # 念のため描画待ち
                    html = driver.page_source
                    found_ids = re.findall(r'race_id=["\']?(\d{12})["\']?', html)
                    for rid in found_ids:
                        if rid not in all_race_ids:
                            all_race_ids.append(rid)
                            id_to_date[rid] = date_str
                except Exception:
                    pass

        if not all_race_ids:
            print("❌ 対象日のレースIDが見つかりませんでした。")
            return

        all_race_ids.sort()
        print(f"\n🎉 合計 {len(all_race_ids)} レースが見つかりました！データを取得中...")
        
        race_data_list = []
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        
        # --- 2. 出馬表の取得（JavaScriptがロードされるのを待つ） ---
        for i, race_id in enumerate(all_race_ids):
            print(f"[{i+1}/{len(all_race_ids)}] 取得中: {race_id}")
            place_code = int(str(race_id)[4:6])
            domain = "race.netkeiba.com" if place_code <= 10 else "nar.netkeiba.com"
            shutuba_url = f"https://{domain}/race/shutuba.html?race_id={race_id}"
            
            try:
                driver.get(shutuba_url)
                
                # ★超重要★
                # 出馬表のHTMLが開いた後、JavaScriptが裏でオッズを読み込んで表示させるのを3秒待つ
                time.sleep(3) 
                
                # 完全に描画されたHTML（オッズ表示済み）を取得
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")
                
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
                        
                        o_val, p_val = None, None
                        
                        # SeleniumによりJavaScriptが展開済みの表から、直接オッズを探す
                        for td in tds:
                            cls_str = " ".join(td.get('class', [])).lower()
                            txt = clean_text(td.text)
                            
                            if ("odds" in cls_str or "txt_r" in cls_str) and re.search(r'\d+\.\d+', txt):
                                m = re.search(r'(\d+\.\d+)', txt)
                                if m: o_val = float(m.group(1))
                                
                            if "pop" in cls_str or "ninki" in cls_str:
                                m = re.search(r'(\d+)', txt)
                                if m: p_val = int(m.group(1))

                        # フォールバック（インデックス決め打ち）
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
                            "オッズ": o_val,
                            "人気": p_val
                        })
                    except Exception:
                        continue
                
                # --- 人気の補完（オッズはあるのに人気が空欄の場合、並べ替えて自作） ---
                has_missing_pop = any(h["オッズ"] is not None and h["人気"] is None for h in temp_horse_list)
                if has_missing_pop:
                    # オッズ昇順に並べて人気を付与
                    valid_odds = [h for h in temp_horse_list if h["オッズ"] is not None]
                    valid_odds.sort(key=lambda x: x["オッズ"])
                    for rank, horse in enumerate(valid_odds, 1):
                        if horse["人気"] is None:
                            horse["人気"] = rank

                race_data_list.extend(temp_horse_list)
                
            except Exception as e:
                print(f"  └ 解析エラー: {e}")

    finally:
        # 処理が終わったら確実にブラウザを閉じる
        driver.quit()

    # --- 3. CSVへの保存処理 ---
    if race_data_list:
        df_future = pd.DataFrame(race_data_list)
        df_future['オッズ'] = pd.to_numeric(df_future['オッズ'], errors='coerce')
        df_future['人気'] = pd.to_numeric(df_future['人気'], errors='coerce').astype('Int64')
        
        if os.path.exists(ML_TARGET_CSV):
            try:
                df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='utf-8-sig')
            except Exception:
                df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='cp932')
            
            if '馬名' in df_past.columns:
                df_past['馬名_clean'] = df_past['馬名'].apply(clean_horse_name)
                df_future['馬名_clean'] = df_future['馬名'].apply(clean_horse_name)
                
                if 'date' in df_past.columns:
                    df_past = df_past.sort_values('date')
                df_past_latest = df_past.drop_duplicates(subset='馬名_clean', keep='last')
                
                cols_to_drop = ['race_id', 'date', 'race_name', '枠番', '馬番', '馬名', 'sex_code', 'age', '斤量', '騎手', 'オッズ', '人気', '単勝', '着順']
                cols_to_keep = [c for c in df_past_latest.columns if c not in cols_to_drop]
                df_past_latest = df_past_latest[cols_to_keep]
                
                df_future = pd.merge(df_future, df_past_latest, on='馬名_clean', how='left')
                df_future.drop(columns=['馬名_clean'], inplace=True)

        df_future.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
        print(f"\n✅ {len(race_data_list)}頭のデータを保存完了！ ({FUTURE_CSV})")
    else:
        print("\n❌ データが1件も取得できませんでした。")


if __name__ == "__main__":
    scrape_shutsuba()