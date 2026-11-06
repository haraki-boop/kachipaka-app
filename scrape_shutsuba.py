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
    # 🎯 今日から向こう7日（または8日）分をまとめて対象にする
    # （開催がない平日などは自動的に0件でスキップされるため、祝日・変則日程もこれで全網羅）
    today = datetime.now()
    return [(today + timedelta(days=i)).strftime("%Y%m%d") for i in range(8)]

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
    print("🔑 netkeibaにログインしています...")
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

def get_premium_details(driver, race_id):
    premium_data = {}
    
    def init_horse(name):
        if name not in premium_data:
            premium_data[name] = {
                "調教タイム": "", "調教評価": "", "推測脚質": "",
                "父": "", "母父": "", "前走着順": "", "前走頭数": "", "間隔": "",
                "前走通過順": "", "前走上がり3F": "", "前走上がり順位": "",
                "前走斤量": "", "前々走斤量": ""
            }

    place_code = int(str(race_id)[4:6])
    domain = "race.netkeiba.com" if place_code <= 10 else "nar.netkeiba.com"

    # 📌 1. 競馬新聞ページ (前走・前々走の斤量取得を追加)
    try:
        driver.get(f"https://{domain}/race/newspaper.html?race_id={race_id}")
        time.sleep(1.2)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        for dl in soup.find_all("dl", class_=re.compile("HorseList")):
            name_dt = dl.find("dt", class_=re.compile("HorseName"))
            if not name_dt: continue
            a_tag = name_dt.find("a", href=re.compile(r"/horse/"))
            if not a_tag: continue
            h_name = clean_horse_name(a_tag.text)
            if not h_name: continue
            init_horse(h_name)

            full_text = dl.get_text(" ", strip=True)
            m_kan = re.search(r'(中\d+週|連闘|\d+カ?月半?休養|\d+ヵ?月半?休養|\d+年\d+カ?月休養)', full_text)
            if m_kan and not premium_data[h_name]["間隔"]:
                premium_data[h_name]["間隔"] = m_kan.group(1)

            past_wrapper = dl.find("dd", class_=re.compile("Past_Wrapper"))
            if past_wrapper:
                past_boxes = past_wrapper.find_all("div", class_=re.compile("PastBox"))
                
                # 前走データ (past_boxes[0])
                if len(past_boxes) >= 1:
                    first_past = past_boxes[0]
                    span_tou = first_past.find("span", class_="Data05")
                    if span_tou and "頭" in span_tou.text:
                        premium_data[h_name]["前走頭数"] = re.sub(r'\D', '', span_tou.text)
                    else:
                        m_tou = re.search(r'(\d+)頭', first_past.get_text(" ", strip=True))
                        if m_tou and not premium_data[h_name]["前走頭数"]:
                            premium_data[h_name]["前走頭数"] = m_tou.group(1)
                    
                    # 前走斤量の取得
                    span_kinryo = first_past.find("span", class_="Data15")
                    if span_kinryo:
                        premium_data[h_name]["前走斤量"] = span_kinryo.get_text(strip=True)
                
                # 前々走データ (past_boxes[1])
                if len(past_boxes) >= 2:
                    second_past = past_boxes[1]
                    # 前々走斤量の取得
                    span_kinryo2 = second_past.find("span", class_="Data15")
                    if span_kinryo2:
                        premium_data[h_name]["前々走斤量"] = span_kinryo2.get_text(strip=True)
    except Exception as e:
        pass

    # 📌 2. 調教ページ
    try:
        driver.get(f"https://{domain}/race/oikiri.html?race_id={race_id}")
        time.sleep(1.2)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        current_horse = None
        for tr in soup.find_all("tr"):
            name_td = tr.find(class_=re.compile("Horse_Info|Horse_Name|HorseList_Item"))
            if name_td:
                a_tag = name_td.find("a", href=re.compile(r"/horse/"))
                if a_tag:
                    h_name = clean_horse_name(a_tag.text)
                    if h_name:
                        current_horse = h_name
                        init_horse(current_horse)

            if current_horse:
                time_td = tr.find("td", class_=re.compile("TrainingTimeData"))
                if time_td:
                    time_td_clone = BeautifulSoup(str(time_td), "html.parser")
                    for comment_div in time_td_clone.find_all(class_=re.compile("Comment_Cell")):
                        comment_div.decompose()
                    raw_time_text = time_td_clone.get_text(" ", strip=True)
                    lap_times = re.findall(r'\b\d{1,3}\.\d\s*\(\d{1,3}\.\d\)|\b\d{1,3}\.\d\b', raw_time_text)
                    if lap_times and not premium_data[current_horse]["調教タイム"]:
                        premium_data[current_horse]["調教タイム"] = " ".join(lap_times)
                
                eval_elem = tr.find(class_=re.compile("Rank_")) or tr.find("td", class_=re.compile("Hyoka|Eval"))
                if eval_elem:
                    e_text = eval_elem.get_text(strip=True).replace('\n', ' ')
                    e_text_clean = re.sub(r'[^A-ZS]', '', e_text)
                    if e_text_clean and not premium_data[current_horse]["調教評価"]:
                        premium_data[current_horse]["調教評価"] = e_text_clean
    except Exception as e:
        pass

    # 📌 3. 前走成績の取得
    try:
        driver.get(f"https://{domain}/race/shutuba_past.html?race_id={race_id}")
        time.sleep(1.2)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        for tr in soup.find_all("tr", class_=re.compile("HorseList")):
            a_tag = tr.find("a", href=re.compile(r"/horse/"))
            if not a_tag: continue
            h_name = clean_horse_name(a_tag.text)
            init_horse(h_name)
            
            kyakusitu_elem = tr.find("span", class_="kyakusitu")
            if kyakusitu_elem:
                premium_data[h_name]["推測脚質"] = clean_text(kyakusitu_elem.text)

            past1 = tr.find("td", class_=re.compile(r"Past1|Past01|Past_1")) or tr.find(class_="Data_Item")
            if not past1:
                for td in tr.find_all("td"):
                    if td.find(class_="Data02"):
                        past1 = td
                        break

            if past1:
                d1 = past1.find(class_="Data01")
                if d1:
                    num_span = d1.find("span", class_="Num")
                    if num_span:
                        premium_data[h_name]["前走着順"] = num_span.get_text(strip=True) + "着"

                d6 = past1.find(class_="Data06")
                if d6:
                    d6_text = d6.get_text(" ", strip=True)
                    m_up3f = re.search(r'\((\d{2}\.\d)\)', d6_text)
                    if m_up3f: premium_data[h_name]["前走上がり3F"] = m_up3f.group(1)
                    
                    m_rank = re.search(r'\(.*?(\d+)位?\)', d6_text)
                    if m_rank:
                        premium_data[h_name]["前走上がり順位"] = m_rank.group(1)
                    else:
                        maru_dict = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5', '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10'}
                        for k, v in maru_dict.items():
                            if k in d6_text:
                                premium_data[h_name]["前走上がり順位"] = v
                                break

                    m_pass = re.search(r'(\d+-\d+(?:-\d+)*)', d6_text)
                    if m_pass: premium_data[h_name]["前走通過順"] = m_pass.group(1)
    except Exception as e:
        pass

    # 📌 4. 血統データの取得
    try:
        driver.get(f"https://{domain}/race/bias.html?race_id={race_id}")
        time.sleep(1.0)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for tr in soup.find_all("tr", class_=re.compile("HorseList")):
            a_tag = tr.find("a", href=re.compile(r"/horse/"))
            if not a_tag: continue
            h_name = clean_horse_name(a_tag.text)
            init_horse(h_name)
            
            blood_cells = tr.find_all("td", class_="Blood_Cell")
            if len(blood_cells) >= 2:
                premium_data[h_name]["父"] = clean_text(blood_cells[0].get_text(strip=True))
                premium_data[h_name]["母父"] = clean_text(blood_cells[1].get_text(strip=True))
    except Exception:
        pass

    return premium_data

def scrape_shutsuba():
    target_dates = get_target_dates()
    print(f"🏇 取得対象日（候補）: {target_dates}")
    all_race_ids = []
    id_to_date = {}

    print("🌐 ブラウザ（Selenium）を起動中...")
    try:
        driver = setup_driver()
        login_netkeiba(driver)
    except Exception as e:
        print(f"❌ 起動またはログイン失敗: {e}")
        return

    try:
        for date_str in target_dates:
            urls = [
                f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}",
                f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
            ]
            for url in urls:
                try:
                    driver.get(url)
                    time.sleep(1.5)
                    found_ids = re.findall(r'race_id=["\']?(\d{12})["\']?', driver.page_source)
                    for rid in found_ids:
                        if rid not in all_race_ids:
                            all_race_ids.append(rid)
                            id_to_date[rid] = date_str
                except Exception:
                    pass

        if not all_race_ids:
            return

        all_race_ids.sort()
        print(f"\n🎉 合計 {len(all_race_ids)} レースを取得中...")
        
        race_data_list = []
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        
        for i, race_id in enumerate(all_race_ids):
            print(f"[{i+1}/{len(all_race_ids)}] {race_id}")
            p_data_dict = get_premium_details(driver, race_id)
            place_code = int(str(race_id)[4:6])
            domain = "race.netkeiba.com" if place_code <= 10 else "nar.netkeiba.com"
            shutuba_url = f"https://{domain}/race/shutuba.html?race_id={race_id}"
            
            try:
                driver.get(shutuba_url)
                time.sleep(1.5)
                soup = BeautifulSoup(driver.page_source, "html.parser")
                
                date_str = id_to_date.get(race_id, "")
                dt_obj = datetime.strptime(date_str, '%Y%m%d') if date_str else None
                display_date = f"{dt_obj.month}月{dt_obj.day}日({weekdays[dt_obj.weekday()]})" if dt_obj else "不明"
                    
                raw_race_name = ""
                rn_elem = soup.find(class_="RaceName") or soup.find(class_="RaceList_Item02")
                if rn_elem: raw_race_name = clean_text(rn_elem.text)
                final_race_name = clean_race_name(raw_race_name)

                surface_val, distance_val = "不明", "1600"
                race_data01 = soup.find(class_="RaceData01")
                if race_data01:
                    m = re.search(r'(芝|ダ|障)[^\d]*(\d+)m', race_data01.text)
                    if m:
                        surface_val, distance_val = m.group(1), m.group(2)

                table = soup.find("table", class_=re.compile("RaceTable01|race_table_01"))
                if not table: continue
                headers = [clean_text(th.text) for th in table.find_all("th")]
                
                def get_idx(keywords):
                    for kw in keywords:
                        for i_h, h in enumerate(headers):
                            if kw in h: return i_h
                    return -1

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
                            "調教タイム": "", "調教評価": "", "推測脚質": "",
                            "父": "", "母父": "", "前走着順": "", "前走頭数": "", "間隔": "",
                            "前走通過順": "", "前走上がり3F": "", "前走上がり順位": "",
                            "前走斤量": "", "前々走斤量": ""
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
                            "推測脚質": p_data["推測脚質"],
                            "父": p_data.get("父", ""),
                            "母父": p_data.get("母父", ""),
                            "前走着順": p_data.get("前走着順", ""),
                            "前走頭数": p_data.get("前走頭数", ""),
                            "間隔": p_data.get("間隔", ""),
                            "前走通過順": p_data.get("前走通過順", ""),
                            "前走上がり3F": p_data.get("前走上がり3F", ""),
                            "前走上がり順位": p_data.get("前走上がり順位", ""),
                            "前走斤量": p_data.get("前走斤量", ""),
                            "前々走斤量": p_data.get("前々走斤量", ""),
                            "調教タイム": p_data["調教タイム"],
                            "調教評価": p_data["調教評価"]
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
            except Exception:
                pass
    finally:
        driver.quit()

    if race_data_list:
        df_future = pd.DataFrame(race_data_list)
        df_future['オッズ'] = pd.to_numeric(df_future['オッズ'], errors='coerce')
        df_future['人気'] = pd.to_numeric(df_future['人気'], errors='coerce').astype('Int64')
        try:
            df_future.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
            print(f"\n✅ {len(race_data_list)}頭分のデータを保存完了！")
        except PermissionError:
            print("\n❌ エクセル等を閉じて再実行してください。")

if __name__ == "__main__":
    scrape_shutsuba()