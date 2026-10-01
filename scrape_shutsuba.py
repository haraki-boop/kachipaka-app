import os
import re
import time
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

FUTURE_CSV = "future_races.csv"

def get_target_dates():
    today = datetime.now()
    dates = []
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
        print("🔓 ログイン成功！プレミアム情報を取得可能です。")
    except Exception as e:
        print(f"⚠️ ログイン失敗。詳細: {e}")

def get_premium_details(driver, race_id):
    premium_data = {}
    
    def init_horse(name):
        if name not in premium_data:
            premium_data[name] = {
                "厩舎コメント": "", "調教タイム": "", "調教短評": "", "調教評価": "", "調子偏差値": ""
            }

    # 📌 2. コメントページ（馬名で取得）
    try:
        driver.get(f"https://race.netkeiba.com/race/comment.html?race_id={race_id}")
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for item in soup.find_all(class_=re.compile("HorseList")):
            a_tag = item.find("a", href=re.compile(r"/horse/"))
            if not a_tag: continue
            h_name = clean_horse_name(a_tag.text)
            comment_elem = item.find(class_=re.compile("Comment"))
            if comment_elem:
                init_horse(h_name)
                premium_data[h_name]["厩舎コメント"] = comment_elem.text.strip().replace('\n', ' ')
    except Exception:
        pass

    # 📌 3. 調教ページ（馬名で取得）
    try:
        driver.get(f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}")
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for tr in soup.find_all("tr", class_=re.compile("HorseList")):
            a_tag = tr.find("a", href=re.compile(r"/horse/"))
            if not a_tag: continue
            h_name = clean_horse_name(a_tag.text)
            init_horse(h_name)
            
            time_elem = tr.find("td", class_=re.compile("Time"))
            eval_elem = tr.find("td", class_=re.compile("Hyoka")) or tr.find("td", class_=re.compile("Eval"))
            
            raw_time = time_elem.get_text(separator=' ').strip() if time_elem else ""
            raw_time = re.sub(r'\s+', ' ', raw_time)
            m = re.match(r'^([^\d\-\.]*)([\d\.\(\)\-\s]+)(.*)', raw_time)
            if m:
                premium_data[h_name]["調教タイム"] = f"{m.group(1).strip()} {m.group(2).strip()}".strip()
                premium_data[h_name]["調教短評"] = m.group(3).strip()
            else:
                premium_data[h_name]["調教短評"] = raw_time
            premium_data[h_name]["調教評価"] = eval_elem.text.strip().replace('\n', ' ') if eval_elem else ""
    except Exception:
        pass
        
    # 📌 4. 調子偏差値（馬名で取得）
    try:
        driver.get(f"https://race.sp.netkeiba.com/barometer/score.html?race_id={race_id}")
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for tr in soup.find_all("tr", class_=re.compile("HorseList")):
            name_elem = tr.find(class_=re.compile("Horse_Name")) or tr.find("a", href=re.compile(r"/horse/"))
            if not name_elem: continue
            h_name = clean_horse_name(name_elem.text)
            init_horse(h_name)
            
            val_span = tr.find("span", class_="Value_Num")
            if val_span:
                premium_data[h_name]["調子偏差値"] = val_span.text.strip()
    except Exception:
        pass

    return premium_data

def scrape_shutsuba():
    target_dates = get_target_dates()
    print(f"🏇 取得対象日（候補）: {target_dates}")
    all_race_ids = []
    id_to_date = {}

    print("🌐 ブラウザ（Selenium）をバックグラウンドで起動中...")
    try:
        driver = setup_driver()
        login_netkeiba(driver)
    except Exception as e:
        print(f"❌ Seleniumの起動またはログインに失敗しました。\n詳細: {e}")
        return

    try:
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
        print(f"\n🎉 合計 {len(all_race_ids)} レースが見つかりました！プレミアムデータを含めて取得中...")
        
        race_data_list = []
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        
        for i, race_id in enumerate(all_race_ids):
            print(f"[{i+1}/{len(all_race_ids)}] 取得中: {race_id}")
            
            p_data_dict = get_premium_details(driver, race_id)
            
            place_code = int(str(race_id)[4:6])
            domain = "race.netkeiba.com" if place_code <= 10 else "nar.netkeiba.com"
            shutuba_url = f"https://{domain}/race/shutuba.html?race_id={race_id}"
            
            try:
                driver.get(shutuba_url)
                time.sleep(3)
                
                soup = BeautifulSoup(driver.page_source, "html.parser")
                
                date_str = id_to_date.get(race_id, "")
                dt_obj = datetime.strptime(date_str, '%Y%m%d') if date_str else None
                display_date = f"{dt_obj.month}月{dt_obj.day}日({weekdays[dt_obj.weekday()]})" if dt_obj else "不明"
                    
                raw_race_name = ""
                rn_elem = soup.find(class_="RaceName") or soup.find(class_="RaceList_Item02")
                if rn_elem: raw_race_name = clean_text(rn_elem.text)
                final_race_name = clean_race_name(raw_race_name)

                surface_val = "不明"
                distance_val = "1600"
                race_data01 = soup.find(class_="RaceData01")
                if race_data01:
                    race_data_text = race_data01.text
                    m = re.search(r'(芝|ダ|障)[^\d]*(\d+)m', race_data_text)
                    if m:
                        surface_val = m.group(1)
                        distance_val = m.group(2)

                table = soup.find("table", class_=re.compile("RaceTable01|race_table_01"))
                if not table: continue
                
                headers = [clean_text(th.text) for th in table.find_all("th")]
                
                def get_idx(keywords):
                    for kw in keywords:
                        for i_h, h in enumerate(headers):
                            if kw in h: return i_h
                    return -1

                # 💡 列の並びが特別登録時と確定時で変わっても柔軟に対応する設計
                idx_waku = get_idx(["枠"])
                idx_umaban = get_idx(["馬番"])
                idx_name = get_idx(["馬名"])
                idx_sexage = get_idx(["性齢"])
                idx_kinryo = get_idx(["斤量"])
                idx_jockey = get_idx(["騎手"])
                idx_trainer = get_idx(["厩舎", "調教師"])
                idx_weight = get_idx(["馬体重"])

                temp_horse_list = []

                for tr in table.find_all("tr", class_=re.compile("HorseList")):
                    tds = tr.find_all("td")
                    if not tds: continue
                    
                    try:
                        horse_name = ""
                        if idx_name >= 0 and idx_name < len(tds):
                            a_tag = tds[idx_name].find("a")
                            horse_name = clean_horse_name(a_tag.text) if a_tag else clean_horse_name(tds[idx_name].text)
                        if not horse_name: continue
                        
                        waku = clean_text(tds[idx_waku].text) if idx_waku >= 0 and idx_waku < len(tds) else ""
                        umaban = clean_text(tds[idx_umaban].text) if idx_umaban >= 0 and idx_umaban < len(tds) else ""
                        sex_age = clean_text(tds[idx_sexage].text) if idx_sexage >= 0 and idx_sexage < len(tds) else ""
                        kinryo = clean_text(tds[idx_kinryo].text) if idx_kinryo >= 0 and idx_kinryo < len(tds) else ""
                        jockey = clean_text(tds[idx_jockey].text) if idx_jockey >= 0 and idx_jockey < len(tds) else ""
                        trainer = clean_text(tds[idx_trainer].text) if idx_trainer >= 0 and idx_trainer < len(tds) else ""
                        weight_str = clean_text(tds[idx_weight].text) if idx_weight >= 0 and idx_weight < len(tds) else ""
                        
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

                        p_data = p_data_dict.get(horse_name, {
                            "厩舎コメント": "", "調教タイム": "", "調教短評": "", "調教評価": "", "調子偏差値": ""
                        })

                        temp_horse_list.append({
                            "race_id": race_id,
                            "date": display_date,
                            "race_name": final_race_name,
                            "surface": surface_val,
                            "distance": distance_val,
                            "枠番": waku,
                            "馬番": umaban,
                            "馬名": horse_name,
                            "sex_code": sex_age[0] if sex_age else "",
                            "age": sex_age[1:] if len(sex_age) > 1 else "",
                            "斤量": kinryo,
                            "騎手": jockey,
                            "調教師": trainer,
                            "馬体重": weight_str,
                            "オッズ": o_val,
                            "人気": p_val,
                            "調子偏差値": p_data["調子偏差値"],
                            "調教タイム": p_data["調教タイム"],
                            "調教短評": p_data["調教短評"],
                            "調教評価": p_data["調教評価"],
                            "厩舎コメント": p_data["厩舎コメント"]
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

    if race_data_list:
        df_future = pd.DataFrame(race_data_list)
        df_future['オッズ'] = pd.to_numeric(df_future['オッズ'], errors='coerce')
        df_future['人気'] = pd.to_numeric(df_future['人気'], errors='coerce').astype('Int64')
        df_future.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
        print(f"\n✅ {len(race_data_list)}頭分のプレミアム出馬表データを {FUTURE_CSV} に保存完了！")
    else:
        print("\n❌ データが1件も取得できませんでした。")

if __name__ == "__main__":
    scrape_shutsuba()