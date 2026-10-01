import os
import re
import time
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SAVE_CSV = "keiba_database2.csv"

def setup_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-extensions')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(5)
    return driver

def login_netkeiba(driver):
    login_id = "a.b_c12345@icloud.com"  # 🚨ご自身のアカウントに変更してください
    password = "Uma12345"
    
    print("🔑 netkeibaにスーパープレミアムアカウントでログインしています...")
    try:
        driver.get("https://regist.netkeiba.com/account/?pid=login")
        wait = WebDriverWait(driver, 10)
        id_elem = wait.until(EC.presence_of_element_located((By.NAME, "login_id")))
        id_elem.send_keys(login_id)
        pw_elem = driver.find_element(By.NAME, "pswd")
        pw_elem.send_keys(password)
        driver.execute_script("arguments[0].form.submit();", pw_elem)
        time.sleep(3)
        print("🔓 ログイン成功！")
    except Exception as e:
        print(f"⚠️ ログイン失敗。詳細: {e}")

def get_race_ids_for_date(driver, date_str):
    url = f"https://db.netkeiba.com/race/list/{date_str}/"
    driver.get(url)
    time.sleep(1.5)
    html = driver.page_source
    found_ids = re.findall(r'/race/(\d{12})/?', html)
    
    jra_ids = []
    for rid in set(found_ids):
        place_code = int(str(rid)[4:6])
        if place_code <= 10:
            jra_ids.append(rid)
    return jra_ids

def get_premium_details(driver, race_id):
    premium_data = {}
    
    def init_horse(u):
        if u not in premium_data:
            premium_data[u] = {
                "厩舎コメント": "", "調教タイム": "", "調教短評": "", "調教評価": "", 
                "脚質": "", "調子偏差値": "", 
                "同条件_馬場指数": "", "同距離_馬場指数": "", "他距離_馬場指数": ""
            }

    # 📌 1. 馬柱(5走)ページ (脚質)
    try:
        driver.get(f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}")
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for tr in soup.find_all("tr", class_=re.compile("HorseList")):
            tds = tr.find_all("td")
            if len(tds) < 2: continue
            umaban = re.sub(r'\D', '', tds[1].text.strip())
            if not umaban: continue
            init_horse(umaban)
            
            kyaku_span = tr.find("span", class_="kyakusitu")
            if kyaku_span:
                txt = kyaku_span.text.strip()
                if txt in ["逃", "先", "差", "追"]:
                    premium_data[umaban]["脚質"] = txt
    except Exception:
        pass

    # 📌 2. コメントページ
    try:
        driver.get(f"https://race.netkeiba.com/race/comment.html?race_id={race_id}")
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for item in soup.find_all(class_=re.compile("HorseList")):
            umaban_elem = item.find(class_=re.compile("Umaban"))
            if umaban_elem:
                umaban = re.sub(r'\D', '', umaban_elem.text.strip())
                if not umaban: continue
                comment_elem = item.find(class_=re.compile("Comment"))
                if comment_elem:
                    init_horse(umaban)
                    premium_data[umaban]["厩舎コメント"] = comment_elem.text.strip().replace('\n', ' ')
    except Exception:
        pass

    # 📌 3. 調教ページ
    try:
        driver.get(f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}")
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for tr in soup.find_all("tr", class_=re.compile("HorseList")):
            umaban_elem = tr.find("td", class_=re.compile("Umaban"))
            time_elem = tr.find("td", class_=re.compile("Time"))
            eval_elem = tr.find("td", class_=re.compile("Hyoka")) or tr.find("td", class_=re.compile("Eval"))
            
            if umaban_elem:
                umaban = re.sub(r'\D', '', umaban_elem.text.strip())
                if not umaban: continue
                init_horse(umaban)
                
                raw_time = time_elem.get_text(separator=' ').strip() if time_elem else ""
                raw_time = re.sub(r'\s+', ' ', raw_time)
                m = re.match(r'^([^\d\-\.]*)([\d\.\(\)\-\s]+)(.*)', raw_time)
                if m:
                    premium_data[umaban]["調教タイム"] = f"{m.group(1).strip()} {m.group(2).strip()}".strip()
                    premium_data[umaban]["調教短評"] = m.group(3).strip()
                else:
                    premium_data[umaban]["調教短評"] = raw_time
                premium_data[umaban]["調教評価"] = eval_elem.text.strip().replace('\n', ' ') if eval_elem else ""
    except Exception:
        pass
        
    # 📌 4. 調子偏差値
    try:
        driver.get(f"https://race.sp.netkeiba.com/barometer/score.html?race_id={race_id}")
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for tr in soup.find_all("tr", class_=re.compile("HorseList")):
            umaban = ""
            if tr.has_attr("id") and "select_" in tr["id"]:
                umaban = tr["id"].split("_")[-1]
            if not umaban:
                for td in tr.find_all("td"):
                    txt = td.text.strip()
                    if txt.isdigit() and 1 <= int(txt) <= 18:
                        umaban = txt
                        break
            if umaban:
                init_horse(umaban)
                val_span = tr.find("span", class_="Value_Num")
                if val_span:
                    premium_data[umaban]["調子偏差値"] = val_span.text.strip()
    except Exception:
        pass

    # 📌 5. 持ちタイムページ (同条件だけでも確保し、取れれば他距離も拾う)
    try:
        driver.get(f"https://race.netkeiba.com/race/holding_time.html?race_id={race_id}")
        time.sleep(1.5)
        
        tabs = driver.find_elements(By.CSS_SELECTOR, "ul[class*='Tab_'] li")
        num_tabs = len(tabs)
        
        for i in range(num_tabs):
            try:
                current_tabs = driver.find_elements(By.CSS_SELECTOR, "ul[class*='Tab_'] li")
                if i >= len(current_tabs): break
                tab = current_tabs[i]
                
                if "NoData" in tab.get_attribute("class"):
                    continue
                    
                span_elem = tab.find_element(By.TAG_NAME, "span")
                try:
                    span_elem.click()
                except:
                    driver.execute_script("arguments[0].click();", span_elem)
                
                time.sleep(1.5)
                soup = BeautifulSoup(driver.page_source, "html.parser")
                
                for tr in soup.find_all("tr", class_=re.compile("HorseList")):
                    tds = tr.find_all("td")
                    if len(tds) < 10: continue
                    
                    umaban = re.sub(r'\D', '', tds[1].text.strip())
                    if not umaban: continue
                    init_horse(umaban)
                    
                    index_td = tr.find("td", class_=re.compile("index"))
                    if index_td:
                        val = index_td.text.strip()
                        if val:
                            if i == 0:
                                premium_data[umaban]["同条件_馬場指数"] = val
                            elif i == 1:
                                premium_data[umaban]["同距離_馬場指数"] = val
                            elif i >= 2:
                                premium_data[umaban]["他距離_馬場指数"] = val
            except Exception:
                continue
    except Exception:
        pass
        
    return premium_data

def parse_race_result(driver, race_id, date_str):
    premium_dict = get_premium_details(driver, race_id)
    
    url = f"https://db.netkeiba.com/race/{race_id}/"
    driver.get(url)
    time.sleep(2)
    
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    race_name = "不明"
    surface_val = ""
    distance_val = ""
    condition_val = ""
    date_val = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
    
    race_name_elem = soup.find("dl", class_=re.compile("racedata"))
    if race_name_elem:
        h1 = race_name_elem.find("h1")
        if h1: race_name = h1.text.strip()
        span = race_name_elem.find("span")
        if span:
            span_text = span.text.strip().replace('\xa0', ' ')
            m1 = re.search(r'(芝|ダ|障)?[^\d]*(\d+)m', span_text)
            if m1:
                surface_val = m1.group(1) or "不明"
                distance_val = m1.group(2)
            m2 = re.search(r'(?:芝|ダ|障|ダート)\s*:\s*(良|稍重|重|不良)', span_text)
            if m2:
                condition_val = m2.group(1)
    
    table = soup.find("table", class_="race_table_01")
    if not table: return []

    horse_list = []
    headers = [th.text.strip().replace('\n', '') for th in table.find_all("th")]
    
    def get_col(row_data, target_names):
        for name in target_names:
            for i, h in enumerate(headers):
                if name in h: return row_data[i] if i < len(row_data) else ""
        return ""

    rows = table.find_all("tr")[1:]
    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 8: continue
        try:
            row_data = [td.text.strip().replace('\n', '') for td in tds]
            umaban = get_col(row_data, ["馬番"])
            if not umaban and len(row_data) > 2: umaban = row_data[2]
            umaban_clean = re.sub(r'\D', '', umaban) if umaban else ""
            
            p_data = premium_dict.get(umaban_clean, {
                "厩舎コメント": "", "調教タイム": "", "調教短評": "", "調教評価": "", 
                "脚質": "", "調子偏差値": "", 
                "同条件_馬場指数": "", "同距離_馬場指数": "", "他距離_馬場指数": ""
            })
            
            horse_list.append({
                "race_id": race_id,
                "date": date_val,
                "surface": surface_val,
                "distance": distance_val,
                "condition": condition_val,
                "race_name": race_name,
                "着順": get_col(row_data, ["着順"]),
                "枠番": get_col(row_data, ["枠番", "枠"]),
                "馬番": umaban,
                "馬名": get_col(row_data, ["馬名"]),
                "性齢": get_col(row_data, ["性齢"]),
                "斤量": get_col(row_data, ["斤量"]),
                "騎手": get_col(row_data, ["騎手"]),
                "タイム": get_col(row_data, ["タイム"]),
                "着差": get_col(row_data, ["着差"]),
                "単勝オッズ": get_col(row_data, ["単勝", "単勝オッズ"]),
                "人気": get_col(row_data, ["人気"]),
                "馬体重": get_col(row_data, ["馬体重"]),
                "脚質": p_data["脚質"],
                "通過": get_col(row_data, ["通過"]),
                "上り": get_col(row_data, ["上り"]),
                "タイム指数": get_col(row_data, ["タイム指数", "ﾀｲﾑ指数"]),
                "調子偏差値": p_data["調子偏差値"],
                "同条件_馬場指数": p_data["同条件_馬場指数"],
                "同距離_馬場指数": p_data["同距離_馬場指数"],
                "他距離_馬場指数": p_data["他距離_馬場指数"],
                "調教師": get_col(row_data, ["調教師"]),
                "馬主": get_col(row_data, ["馬主"]),
                "賞金(万円)": get_col(row_data, ["賞金(万円)", "賞金"]),
                "調教タイム": p_data["調教タイム"],
                "調教短評": p_data["調教短評"],
                "調教評価": p_data["調教評価"],
                "厩舎コメント": p_data["厩舎コメント"]
            })
        except Exception:
            pass
            
    return horse_list

def scrape_historical_data():
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now() # 現在の日付まで
    
    print(f"📅 【本番スクレイピング】取得期間: {start_date.strftime('%Y/%m/%d')} 〜 現在")
    
    # 📌 既存CSVがあれば読み込み、取得済みレースをスキップ（レジューム機能）
    existing_race_ids = set()
    if os.path.exists(SAVE_CSV):
        try:
            df_existing = pd.read_csv(SAVE_CSV, encoding='utf-8-sig', dtype={'race_id': str})
            if 'race_id' in df_existing.columns:
                existing_race_ids = set(df_existing['race_id'].astype(str).tolist())
                print(f"📂 既存のCSVから {len(existing_race_ids)} 件の取得済みレースを読み込みました。スキップして続きから開始します。")
        except Exception as e:
            print(f"⚠️ 既存CSVの読み込みに失敗しました: {e}")

    driver = setup_driver()
    login_netkeiba(driver)
    
    temp_date = start_date
    week_race_data = []
    
    try:
        while temp_date <= end_date:
            if temp_date.weekday() in [5, 6]:
                date_str = temp_date.strftime("%Y%m%d")
                race_ids = get_race_ids_for_date(driver, date_str)
                if race_ids:
                    new_race_ids = [rid for rid in race_ids if str(rid) not in existing_race_ids]
                    if new_race_ids:
                        print(f"  🏇 {date_str}: 新規 {len(new_race_ids)}レース取得中...")
                        for i, rid in enumerate(new_race_ids):
                            
                            if i > 0 and i % 30 == 0:
                                print("  🔄 メモリ解放のためブラウザを再起動します...")
                                try: driver.quit()
                                except: pass
                                driver = setup_driver()
                                login_netkeiba(driver)
                                
                            if i % 10 == 0 and i > 0: print(f"    ... {i}件完了")
                            
                            retry_count = 0
                            success = False
                            while not success and retry_count < 3:
                                try:
                                    data = parse_race_result(driver, rid, date_str)
                                    week_race_data.extend(data)
                                    existing_race_ids.add(str(rid))
                                    success = True
                                except Exception as e:
                                    retry_count += 1
                                    print(f"    ⚠️ エラー検知(ID:{rid})。ブラウザを再起動してリトライします({retry_count}/3)...")
                                    try: driver.quit()
                                    except: pass
                                    driver = setup_driver()
                                    login_netkeiba(driver)
            
            # 📌 日曜日が終わったタイミング（または最終日）でCSVに追記保存（オートセーブ）
            if temp_date.weekday() == 6 or temp_date.date() == end_date.date():
                if week_race_data:
                    df_week = pd.DataFrame(week_race_data)
                    write_header = not os.path.exists(SAVE_CSV) # ファイルがなければヘッダーを付ける
                    df_week.to_csv(SAVE_CSV, index=False, encoding='utf-8-sig', mode='a', header=write_header)
                    print(f"💾 {temp_date.strftime('%Y/%m/%d')} までのデータを {SAVE_CSV} に追記保存しました！(今週 {len(week_race_data)} 頭分)")
                    week_race_data = [] # 次の週のためにリストを空にする
                    
            temp_date += timedelta(days=1)
            
    finally:
        print("🛑 処理を終了します。ブラウザを閉じます。")
        try: driver.quit()
        except: pass

if __name__ == "__main__":
    scrape_historical_data()