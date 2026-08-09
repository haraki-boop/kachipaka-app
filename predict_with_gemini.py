import os
import re
import time
import random
import requests
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import subprocess
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from google import genai
from google.genai import types

st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="🐴", layout="wide")

col_img, col_text = st.columns([1, 10])
with col_img:
    try:
        st.image("image_61b676.png", width=70)
    except Exception:
        st.write("🐴")
with col_text:
    st.title("AI予想 勝ちぱかくん")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        GEMINI_API_KEY = None

HISTORY_CSV = "prediction_history.csv"
FUTURE_CSV = "future_races.csv"
ML_TARGET_CSV = "ml_target_data.csv"

# ==========================================
# 1. データとAIモデルの読み込み
# ==========================================
@st.cache_resource
def load_model():
    if os.path.exists("keiba_ai_model.pkl") and os.path.getsize("keiba_ai_model.pkl") > 0:
        try:
            return joblib.load("keiba_ai_model.pkl")
        except Exception:
            return None
    return None

@st.cache_data
def load_past_data():
    if os.path.exists(ML_TARGET_CSV) and os.path.getsize(ML_TARGET_CSV) > 0:
        try:
            return pd.read_csv(ML_TARGET_CSV, low_memory=False, dtype={'race_id': str}, encoding='utf-8-sig')
        except Exception:
            try:
                return pd.read_csv(ML_TARGET_CSV, low_memory=False, dtype={'race_id': str}, encoding='cp932')
            except Exception:
                pass
    return pd.DataFrame()

def load_future_data():
    if os.path.exists(FUTURE_CSV) and os.path.getsize(FUTURE_CSV) > 0:
        try:
            df = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding='utf-8-sig')
        except Exception:
            try:
                df = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding='cp932')
            except Exception:
                return pd.DataFrame()
                
        PLACE_MAP_REV = {
            "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
        }
        if 'race_id' in df.columns:
            df['place_code'] = df['race_id'].str[4:6]
            df['place_name'] = df['place_code'].map(PLACE_MAP_REV).fillna("不明")
            df['r_num'] = df['race_id'].str[10:12].astype(int)
        
        if 'race_name' not in df.columns: df['race_name'] = ""
        df['day_label'] = df['date'] if 'date' in df.columns else "不明"
        return df
    return pd.DataFrame()

def load_history_data():
    if os.path.exists(HISTORY_CSV) and os.path.getsize(HISTORY_CSV) > 0:
        try:
            df = pd.read_csv(HISTORY_CSV, dtype={'race_id': str, 'honmei_umaban': str, 'partners': str}, encoding='utf-8-sig')
        except Exception:
            try:
                df = pd.read_csv(HISTORY_CSV, dtype={'race_id': str, 'honmei_umaban': str, 'partners': str}, encoding='cp932')
            except Exception:
                return pd.DataFrame()
        if 'partners' not in df.columns:
            df['partners'] = ""
        for col in ['pay_tansho', 'pay_umaren', 'pay_wide', 'pay_sanrenpuku', 'pay_sanrentan']:
            if col not in df.columns:
                df[col] = 0
        return df
    return pd.DataFrame(columns=['date', 'race_id', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay', 'pay_tansho', 'pay_umaren', 'pay_wide', 'pay_sanrenpuku', 'pay_sanrentan'])

model_data = load_model()
df_past = load_past_data()
df_future = load_future_data()
df_history = load_history_data()

# ==========================================
# 2. スクレイパー関数
# ==========================================
def get_target_dates():
    today = datetime.now()
    return sorted(list(set([(today + timedelta(days=i)).strftime("%Y%m%d") for i in range(7) if (today + timedelta(days=i)).weekday() in [5, 6]])))

def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[\r\n\t]+', ' ', text)).strip() if text else ""

def run_scraper(p_text, p_bar):
    target_dates = get_target_dates()
    all_race_ids, id_to_date = [], {}
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://race.netkeiba.com/"
    })

    p_text.text(f"📅 検索対象日: {target_dates}")
    for date_str in target_dates:
        url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
        try:
            res = session.get(url, timeout=10)
            if res.status_code == 200:
                found_ids = re.findall(r'race_id=["\']?(\d{12})["\']?', res.text)
                for rid in found_ids:
                    if rid not in all_race_ids:
                        all_race_ids.append(rid)
                        id_to_date[rid] = date_str
        except: pass
        time.sleep(1)

    if not all_race_ids: return False

    race_data_list = []
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    total = len(all_race_ids)

    for i, race_id in enumerate(all_race_ids):
        p_text.text(f"📥 出馬表とオッズを取得中... ({i+1}/{total} レース)")
        p_bar.progress((i + 1) / total)
        
        odds_dict = {}
        try:
            odds_res = session.get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init", timeout=5)
            if odds_res.status_code == 200:
                odds_json = odds_res.json()
                if 'data' in odds_json and 'odds' in odds_json['data'] and '1' in odds_json['data']['odds']:
                    for umaban, vals in odds_json['data']['odds']['1'].items(): odds_dict[str(int(umaban))] = float(vals[0])
        except: pass
        if not odds_dict:
            try:
                odds_res = session.get(f"https://race.netkeiba.com/api/api_get_nar_odds.html?race_id={race_id}&type=1&action=init", timeout=5)
                if odds_res.status_code == 200:
                    odds_json = odds_res.json()
                    if 'data' in odds_json and 'odds' in odds_json['data'] and '1' in odds_json['data']['odds']:
                        for umaban, vals in odds_json['data']['odds']['1'].items():
                            odds_dict[str(int(umaban))] = float(vals[0])
            except: pass

        try:
            res = session.get(f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}", timeout=10)
            if res.status_code == 200:
                res.encoding = 'utf-8'
                soup = BeautifulSoup(res.text, "html.parser")
                d_str = id_to_date.get(race_id, "")
                display_date = f"{datetime.strptime(d_str, '%Y%m%d').month}月{datetime.strptime(d_str, '%Y%m%d').day}日({weekdays[datetime.strptime(d_str, '%Y%m%d').weekday()]})" if d_str else "不明"
                
                r_name_elem = soup.find(class_="RaceName") or soup.find(class_="RaceList_Item02")
                r_name = clean_text(r_name_elem.find("span", class_="RaceName_main").text) if r_name_elem and r_name_elem.find("span", class_="RaceName_main") else (clean_text(r_name_elem.text) if r_name_elem else "")

                surface, distance, condition = "不明", np.nan, "不明"
                race_data01 = soup.find(class_="RaceData01")
                if race_data01:
                    rd_text = race_data01.text
                    if "芝" in rd_text: surface = "芝"
                    elif "ダ" in rd_text: surface = "ダート"
                    elif "障" in rd_text: surface = "障害"
                    dist_match = re.search(r'(\d+)m', rd_text)
                    if dist_match: distance = float(dist_match.group(1))
                    if "良" in rd_text: condition = "良"
                    elif "稍" in rd_text: condition = "稍重"
                    elif "重" in rd_text: condition = "重"
                    elif "不良" in rd_text: condition = "不良"

                for row in soup.select("tr.HorseList"):
                    cols = row.find_all("td")
                    if len(cols) >= 7:
                        nm = clean_text(cols[3].find("a").text) if cols[3].find("a") else clean_text(cols[3].text)
                        ub = clean_text(cols[1].text)
                        if nm and ub.isdigit():
                            sa = clean_text(cols[4].text)
                            pop_val = np.nan
                            odds_val = odds_dict.get(str(int(ub)), np.nan)
                            if pd.isna(odds_val):
                                odds_elem = row.find(class_=re.compile(r'Popular_Txt|Odds'))
                                if odds_elem:
                                    try: odds_val = float(clean_text(odds_elem.text))
                                    except: pass
                            pop_elem = row.find(class_=re.compile(r'Popular_Num'))
                            if pop_elem:
                                try: pop_val = float(clean_text(pop_elem.text))
                                except: pass

                            race_data_list.append({
                                "race_id": str(race_id), "date": display_date, "race_name": r_name,
                                "枠番": clean_text(cols[0].text), "馬番": ub, "馬名": nm,
                                "sex_code": sa[0] if sa else "", "age": sa[1:] if len(sa)>1 else "",
                                "斤量": clean_text(cols[5].text),
                                "騎手": clean_text(cols[6].find("a").text) if cols[6].find("a") else clean_text(cols[6].text),
                                "単勝": odds_val, "人気": pop_val,
                                "surface": surface, "distance": distance, "condition": condition
                            })
            time.sleep(random.uniform(0.3, 0.7))
        except: pass

    if race_data_list:
        pd.DataFrame(race_data_list).to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
        return True
    return False

# ==========================================
# サイドバー UI
# ==========================================
st.sidebar.header("🔄 画面の更新")
if st.sidebar.button("🔄 最新の情報にリロード", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🏁 実戦結果の検証")
if st.sidebar.button("🏆 終了したレースの配当を取得", use_container_width=True):
    with st.spinner("🏁 実際の着順と全券種の配当をリアルタイム検索中..."):
        if not df_history.empty:
            updated = False
            headers = {"User-Agent": "Mozilla/5.0"}
            
            for idx, row in df_history.iterrows():
                try:
                    axis = str(int(row['honmei_umaban']))
                    partners = [str(int(p.strip())) for p in str(row.get('partners', '')).split(',') if p.strip().isdigit()]
                    payout_found = False
                    
                    payouts = {'単勝': 0, '馬連': 0, 'ワイド': 0, '三連複': 0, '三連単': 0}
                    
                    for url in [f"https://race.netkeiba.com/race/result.html?race_id={str(row['race_id'])}", f"https://db.netkeiba.com/race/{str(row['race_id'])}/"]:
                        res = requests.get(url, headers=headers, timeout=5)
                        res.encoding = 'EUC-JP' if 'db.netkeiba' in url else 'utf-8'
                        soup = BeautifulSoup(res.text, "html.parser")
                        
                        for tr in soup.find_all("tr"):
                            th = tr.find("th")
                            if not th: continue
                            kind = th.text.strip().replace("3連", "三連")
                            if kind not in payouts: continue
                            
                            tds = tr.find_all("td")
                            if len(tds) < 2: continue
                            
                            payout_found = True
                            
                            for br in tds[0].find_all('br'): br.replace_with(' _SPLIT_ ')
                            for ul in tds[0].find_all('ul'): ul.insert_after(' _SPLIT_ ')
                            for div in tds[0].find_all('div'): div.insert_after(' _SPLIT_ ')
                            for br in tds[1].find_all('br'): br.replace_with(' _SPLIT_ ')
                            for ul in tds[1].find_all('ul'): ul.insert_after(' _SPLIT_ ')
                            for div in tds[1].find_all('div'): div.insert_after(' _SPLIT_ ')
                            
                            w_items = tds[0].get_text(separator=" ").split('_SPLIT_')
                            a_items = tds[1].get_text(separator=" ").split('_SPLIT_')
                            
                            for w_str, a_str in zip(w_items, a_items):
                                amt_str = re.sub(r'\D', '', a_str)
                                if not amt_str.isdigit(): continue
                                amt = int(amt_str)
                                
                                w_str_clean = re.sub(r'\D', ' ', w_str)
                                w_nums = [str(int(n)) for n in w_str_clean.split() if n.isdigit()]
                                
                                if not w_nums: continue

                                if kind == "単勝" and len(w_nums) >= 1 and w_nums[0] == axis:
                                    payouts['単勝'] += amt
                                elif kind == "馬連" and len(w_nums) >= 2:
                                    if axis in w_nums[:2] and any(p in w_nums[:2] for p in partners[:3]):
                                        payouts['馬連'] += amt
                                elif kind == "ワイド" and len(w_nums) >= 2:
                                    if axis in w_nums[:2] and any(p in w_nums[:2] for p in partners[:3]):
                                        payouts['ワイド'] += amt
                                elif kind == "三連複" and len(w_nums) >= 3:
                                    if axis in w_nums[:3] and len(set(w_nums[:3]).intersection(set(partners))) >= 2:
                                        payouts['三連複'] += amt
                                elif kind == "三連単" and len(w_nums) >= 3:
                                    if w_nums[0] == axis and w_nums[1] in partners and w_nums[2] in partners:
                                        payouts['三連単'] += amt
                        
                        if payout_found:
                            df_history.at[idx, 'pay_tansho'] = payouts['単勝']
                            df_history.at[idx, 'pay_umaren'] = payouts['馬連']
                            df_history.at[idx, 'pay_wide'] = payouts['ワイド']
                            df_history.at[idx, 'pay_sanrenpuku'] = payouts['三連複']
                            df_history.at[idx, 'pay_sanrentan'] = payouts['三連単']
                            df_history.at[idx, 'result_pay'] = sum(payouts.values())
                            updated = True
                            break
                except Exception as e: 
                    pass
                time.sleep(1)
            
            if updated: df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig', errors='replace')
        st.cache_data.clear()
        st.success("✅ 実戦結果を最新化しました！")
        time.sleep(1.5)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🗑️ 履歴の完全リセット")
if st.sidebar.button("💥 ゴミ予想履歴を完全消去", type="primary", use_container_width=True):
    try:
        if os.path.exists(HISTORY_CSV): os.remove(HISTORY_CSV)
        st.cache_data.clear()
        st.sidebar.success("✅ 履歴データを消去しました！")
        time.sleep(1.5)
        st.rerun()
    except: pass


# ==========================================
# 3. AIスコア計算 ＋ 予想出力
# ==========================================
def calculate_race_scores(race_id_target, target_df):
    if target_df.empty or model_data is None: return None

    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty: return None

    features = model_data.get('features', [])
    model = model_data.get('model')

    if not df_past.empty and '馬名' in df_past.columns:
        df_past['is_win'] = (pd.to_numeric(df_past['着順'], errors='coerce') == 1).astype(int)
        agg_dict = {'is_win': ['sum', 'count']}
        if 'my_time_idx' in df_past.columns: agg_dict['my_time_idx'] = 'mean'
        if 'my_last3f_idx' in df_past.columns: agg_dict['my_last3f_idx'] = 'mean'
        if 'my_pace_idx' in df_past.columns: agg_dict['my_pace_idx'] = 'mean'
        if 'my_start_idx' in df_past.columns: agg_dict['my_start_idx'] = 'mean'
        
        horse_stats = df_past.groupby('馬名').agg(agg_dict).reset_index()
        
        new_cols = ['馬名', 'total_wins', 'total_runs']
        if 'my_time_idx' in df_past.columns: new_cols.append('my_time_idx')
        if 'my_last3f_idx' in df_past.columns: new_cols.append('my_last3f_idx')
        if 'my_pace_idx' in df_past.columns: new_cols.append('my_pace_idx')
        if 'my_start_idx' in df_past.columns: new_cols.append('my_start_idx')
        
        horse_stats.columns = new_cols
        race_df = pd.merge(race_df, horse_stats, on='馬名', how='left')

    if not df_past.empty and '騎手' in df_past.columns:
        j_stats = df_past.groupby('騎手')['is_win'].mean().reset_index()
        j_stats.rename(columns={'is_win': 'jockey_win_power'}, inplace=True)
        race_df = pd.merge(race_df, j_stats, on='騎手', how='left')
    else:
        race_df['jockey_win_power'] = 0.0

    for f in features:
        if f not in race_df.columns:
            race_df[f] = 50.0 if 'idx' in f else 0.0
    if 'my_time_idx' in race_df.columns: race_df['my_time_idx'] = race_df['my_time_idx'].fillna(50.0)
    if 'my_last3f_idx' in race_df.columns: race_df['my_last3f_idx'] = race_df['my_last3f_idx'].fillna(50.0)
    if 'my_pace_idx' in race_df.columns: race_df['my_pace_idx'] = race_df['my_pace_idx'].fillna(50.0)
    if 'my_start_idx' in race_df.columns: race_df['my_start_idx'] = race_df['my_start_idx'].fillna(50.0)
    if 'jockey_win_power' in race_df.columns: race_df['jockey_win_power'] = race_df['jockey_win_power'].fillna(0.0)

    if 'sex_code' in race_df.columns and race_df['sex_code'].dtype == object:
        race_df['sex_code'] = race_df['sex_code'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
        
    if 'le_surf' in model_data and 'surface' in race_df.columns:
        le_surf = model_data['le_surf']
        race_df['surface_code'] = race_df['surface'].map(lambda s: le_surf.transform([s])[0] if s in le_surf.classes_ else le_surf.transform(['不明'])[0])
    if 'le_cond' in model_data and 'condition' in race_df.columns:
        le_cond = model_data['le_cond']
        race_df['condition_code'] = race_df['condition'].map(lambda c: le_cond.transform([c])[0] if c in le_cond.classes_ else le_cond.transform(['不明'])[0])

    X = race_df[features].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
        else:
            prob = model.predict(X)
    except Exception: return None

    s = prob.sum()
    race_df['win_prob'] = prob / s if s > 0 else 1.0 / len(race_df)
    
    rs = 40 + (race_df['win_prob'] * 400)
    race_df['score'] = np.clip(rs, 30, 150).round().astype(int)

    if '人気' in race_df.columns:
        race_df['人気_sort'] = pd.to_numeric(race_df['人気'], errors='coerce').fillna(999)
        return race_df.sort_values(by=['score', '人気_sort'], ascending=[False, True]).reset_index(drop=True)
    
    return race_df.sort_values(by='score', ascending=False).reset_index(drop=True)

def get_all_markers():
    markers = {}
    if df_future.empty: return markers
    for rid in df_future['race_id'].unique():
        sdf = calculate_race_scores(rid, df_future)
        if sdf is not None and len(sdf) >= 5:
            sc = sdf['score'].tolist()
            
            top_diff = sc[0] - sc[1]
            top3_diff = sc[0] - sc[2]
            top5_diff = sc[0] - sc[4]

            # 期待値の計算と【🔥狙い目】フラグの判定
            is_neraime = False
            if '単勝' in sdf.columns and 'win_prob' in sdf.columns:
                for _, r in sdf.iterrows():
                    odds = pd.to_numeric(r.get('単勝', 0), errors='coerce')
                    if pd.isna(odds): odds = 0
                    ev = r['win_prob'] * odds
                    # AIスコアが上位水準(50以上)で、期待値が1.5以上、かつオッズ15倍以上の穴馬がいれば狙い目
                    if r['score'] >= 50 and ev >= 1.5 and odds >= 15.0:
                        is_neraime = True
                        break

            if sc[0] >= 110 and (top_diff >= 18 or top3_diff >= 30):
                race_type = "【堅】"
            elif sc[0] < 95 or (top_diff <= 7 and top5_diff <= 25):
                race_type = "【穴】"
            else:
                race_type = "【普】"

            if is_neraime:
                race_type = "【🔥狙い目】" + race_type

            if sc[0] >= 115 and top_diff >= 15:
                mark = "★"
            elif top5_diff <= 10:
                mark = "◎"
            else:
                mark = ""
            
            markers[rid] = f"{race_type} {mark}".strip()
    return markers
markers = get_all_markers()

tab_forecast, tab_dashboard = st.tabs(["🏇 レース予想", "📈 実戦成績"])

with tab_forecast:
    if df_future.empty:
        st.warning("⚠️ 出馬表データが存在しません。BOT（auto_pipeline_bot.py）を実行してデータを読み込ませてください。")
    else:
        date_options = sorted(df_future['day_label'].unique())
        selected_date = st.radio("開催日", date_options, horizontal=True, label_visibility="collapsed")
        
        day_df = df_future[df_future['day_label'] == selected_date]
        places = day_df['place_name'].unique()
        
        place_tabs = st.tabs([f"🏇 {p}" for p in places])
        for p_idx, place in enumerate(places):
            with place_tabs[p_idx]:
                place_df = day_df[day_df['place_name'] == place]
                races = sorted(place_df['r_num'].unique())
                
                cols = st.columns(6)
                for i, r in enumerate(races):
                    col = cols[i % 6]
                    race_rows = place_df[place_df['r_num'] == r]
                    rid = race_rows['race_id'].iloc[0]
                    rname = str(race_rows['race_name'].iloc[0]).strip() if 'race_name' in race_rows.columns else ""
                    mark = markers.get(rid, "")
                    
                    label = f"{r}R {mark}".strip()
                    if rname: label = f"{r}R {rname[:5]}… {mark}".strip()
                        
                    # 狙い目レースはボタンの色を変えてアピール
                    btn_type = "primary" if ("【堅】" in mark or "★" in mark or "🔥" in mark) else "secondary"
                    if col.button(label, key=f"btn_{rid}", use_container_width=True, type=btn_type):
                        st.session_state['selected_race_id'] = rid

    if st.session_state['selected_race_id'] and not df_future.empty and 'race_id' in df_future.columns:
        st.markdown("---")
        target_id = st.session_state['selected_race_id']
        target_race_info = df_future[df_future['race_id'] == target_id].iloc[0]
        rname = target_race_info.get('race_name', "")
        race_display_name = f"{target_race_info['place_name']} {target_race_info['r_num']}R 【{rname}】" if rname else f"{target_race_info['place_name']} {target_race_info['r_num']}R"
            
        st.subheader(f"🚀 {race_display_name}")
        
        if st.button("🧠 勝ちぱかくんに最終予想させる！", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()
                
            scored_df = calculate_race_scores(target_id, df_future)
            if scored_df is None or len(scored_df) < 6:
                st.error("出走頭数が少ない、またはデータが不足しているため予想をスキップします。")
                st.stop()

            # オッズと期待値(EV)を計算
            scored_df['odds_num'] = pd.to_numeric(scored_df['単勝'], errors='coerce').fillna(0)
            scored_df['ev'] = scored_df['win_prob'] * scored_df['odds_num']

            # 本命はAIスコア1位
            honmei_row = scored_df.iloc[0]
            honmei_umaban = int(honmei_row['馬番'])
            honmei_name = honmei_row['馬名']
            
            # 相手馬5頭の抽出ロジック（人気保護 ＋ 高期待値穴馬 ＋ AIスコア）
            other_df = scored_df.iloc[1:].copy()
            other_df['人気_num'] = pd.to_numeric(other_df['人気'], errors='coerce').fillna(999)
            
            # 1. ヒモ抜け防止で1〜3番人気を必ず抽出
            pop_df = other_df[other_df['人気_num'] <= 3.0]
            
            # 2. 期待値の高い穴馬(オッズ15倍以上でEV最上位)を抽出
            sleeper_df = other_df[(~other_df['馬番'].isin(pop_df['馬番'])) & (other_df['odds_num'] >= 15.0)].sort_values('ev', ascending=False)
            top_sleeper = sleeper_df.head(1) if not sleeper_df.empty and sleeper_df.iloc[0]['ev'] >= 1.0 else pd.DataFrame()
            
            # 3. 残りをAIスコア上位から抽出
            exclude_umbans = pd.concat([pop_df['馬番'], top_sleeper['馬番']]) if not top_sleeper.empty else pop_df['馬番']
            ai_df = other_df[~other_df['馬番'].isin(exclude_umbans)].sort_values('score', ascending=False)
            
            # 4. 全て合体させて上位5頭に絞り、再度AIスコア順（実力順）に並べ直す
            partners_df = pd.concat([pop_df, top_sleeper, ai_df]).head(5).sort_values('score', ascending=False)
            partners_list = partners_df['馬番'].astype(int).tolist()
            partners_str = ",".join(map(str, partners_list))
            
            if df_history.empty or str(target_id) not in df_history['race_id'].astype(str).values:
                new_record = pd.DataFrame([{'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'race_id': str(target_id), 'race_name': race_display_name, 'honmei_umaban': honmei_umaban, 'partners': partners_str, 'honmei_name': honmei_name, 'result_pay': "", 'pay_tansho': 0, 'pay_umaren': 0, 'pay_wide': 0, 'pay_sanrenpuku': 0, 'pay_sanrentan': 0}])
                df_history = pd.concat([df_history, new_record], ignore_index=True)
                df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')

            table_summary = []
            for _, row in scored_df.iterrows():
                odds = row['odds_num']
                pop = row.get('人気', '-') if pd.notna(row.get('人気')) else '-'
                j_power = row.get('jockey_win_power', 0.0)
                j_info = "過去データなし(要検索)" if j_power == 0.0 else f"勝率{j_power*100:.1f}%"
                ev = row['ev']
                
                table_summary.append(
                    f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | "
                    f"騎手:{row.get('騎手', '不明')}({j_info}) | "
                    f"オッズ:{odds}倍 ({pop}人気) | "
                    f"純粋勝率:{row['win_prob']*100:.1f}% | 🎯期待値:{ev:.2f} | "
                    f"AIスコア:{row['score']:3d}"
                )

            system_instruction = f"""
あなたはプロの競馬分析AI「勝ちぱかくん」です。以下の絶対ルールに従い、指定された買い目の解説を行ってください。

【絶対遵守事項】
1. 挨拶や前置きは一切不要。結果のみを出力すること。
2. 【買い目の完全固定】: 今回のレースの買い目（◎と相手5頭）はシステム側で確定しています。提供された「確定済みの買い目」を必ずそのまま使用してください。自分で勝手に馬番を変えたり、印を削ったりすることは絶対に禁止です。
3. 【解説の方向性とアピール】: 
   - 軸馬(◎)は「AIスコアが最も高い実力馬」として解説。
   - 相手馬には「ヒモ抜け防止の人気馬」と「期待値の高い穴馬（妙味馬）」が含まれています。
   - ※オッズが高く期待値（EV）が良い馬が相手にいる場合、表の評価欄に「🔥 激走警戒」や「🎯 妙味抜群」と記載し、本文でも「💥 勝ちぱかくんの「特注」爆発狙い馬！」として熱くアピールしてください。
4. 過激すぎる煽り文句（バク穴、全財産など）は使用しない。ただし「狙い目」「妙味」といったプロ視点での熱い推奨は歓迎します。
5. 【推奨買い目のフォーマット】: （指定された軸と相手をそのまま当てはめること）
   - 単勝: ◎ (1点)
   - 馬連: ◎ － ◯, ▲, △ (相手の先頭3頭)
   - ワイド: ◎ － ◯, ▲, △ (相手の先頭3頭)
   - 三連複: ◎ 1頭軸流し － 相手5頭 (10点)
   - 三連単 (フォーメーション): ◎ 1着固定流し → 相手5頭 (20点)
   - 三連単 (マルチ): ◎・◯ 2頭軸マルチ － 相手残り4頭 (24点)

出力フォーマット：
---
### 📊 1. 出走馬 期待値＆データ一覧
（※表の行は絶対に【提供したデータの並び順そのまま（AIスコア順）】とし、馬番順に並び替えないこと。コードブロック「```」で囲まないこと。期待値が高い穴馬の評価には🔥や🎯をつけること）
| 馬番 | 馬名 | 騎手 | 推定オッズ | 純粋勝率 | 🎯期待値 | AIスコア | 評価 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|

### 🌪️ 2. レース波乱度と展開分析
* **【レース判定】:** 「堅実決着」または「波乱（混戦）」とその理由
* **【展開・バリュー評価】:** （冷静な分析）

### 💥 3. 勝ちぱかくんの「特注」爆発狙い馬！（※該当馬がいる場合のみ）
* ここに、AIスコアと期待値が高い穴馬へのアピール文を熱く記載してください。

### 🎯 4. 勝ちぱかくんの印と詳細見解
* **◎（本命）:** [確定軸馬]（馬名） - （抜擢理由）
* **◯（対抗）:** [確定相手1頭目]（馬名） - （理由）
* **▲（単穴）:** [確定相手2頭目]（馬名） - （理由）
* **△（連下）:** [確定相手3頭目]、[確定相手4頭目]
* **☆（押さえ）:** [確定相手5頭目]

### 💡 5. 戦略的・推奨買い目
* **単勝:** ◎ (1点)
* **馬連:** ◎ － ◯, ▲, △
* **ワイド:** ◎ － ◯, ▲, △
* **三連複:** ◎ 1頭軸流し － ◯, ▲, △(2頭), ☆ (10点)
* **三連単 (フォーメーション):** ◎ 1着固定流し → ◯, ▲, △(2頭), ☆ (20点)
* **三連単 (マルチ):** ◎・◯ 2頭軸マルチ － ▲, △(2頭), ☆ (24点)
---
"""
            prompt = f"対象レース: {race_display_name}\n\n"
            prompt += f"【確定済みの買い目（絶対にこれに従うこと）】\n"
            prompt += f"軸馬(◎): {honmei_umaban:02d}番\n"
            prompt += f"相手馬(◯▲△☆): {partners_str}\n\n"
            prompt += f"出走馬データ:\n{chr(10).join(table_summary)}"

            with st.spinner("AIがレース波乱度を分析し、Gemini(3.6 Flash)が最適戦略を構築中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                try:
                    response = client.models.generate_content(
                        model='gemini-3.6-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.0,
                            tools=[{"googleSearch": {}}]
                        )
                    )
                    res_text = response.text if response.text else (response.candidates[0].content.parts[0].text if response.candidates else "")
                    if res_text: st.markdown(res_text)
                    else: st.warning("⚠️ 回答を取得できませんでした。")
                    st.success("📝 実戦履歴に記録しました！")
                except Exception as e: st.error(f"【APIエラー】: {e}")

with tab_dashboard:
    st.subheader("📈 実戦成績ダッシュボード")
    if df_history.empty: st.info("まだ予想履歴がありません。")
    else:
        df_history['datetime'] = pd.to_datetime(df_history['date'], errors='coerce')
        df_history['year_month'] = df_history['datetime'].dt.strftime('%Y年%m月')
        df_history['just_date'] = df_history['datetime'].dt.strftime('%Y-%m-%d')
        
        tab_total, tab_month, tab_day = st.tabs(["🏆 総合成績", "📅 月別成績", "📆 日別成績"])
        
        def render_dashboard_for_df(raw_df, title_prefix):
            total_races = len(raw_df)
            if total_races == 0:
                st.info(f"この期間のレースはありません。")
                return
            
            finished_df = raw_df[pd.to_numeric(raw_df['result_pay'], errors='coerce').notna()]
            total = len(finished_df)
            
            if total == 0:
                st.markdown(f"**{title_prefix} 判明レース**: 0 件 （結果待ち: {total_races} 件）")
            else:
                hits = len(finished_df[finished_df['result_pay'].astype(float) > 0])
                returns = finished_df['result_pay'].astype(float).sum()
                
                invested_total = 0
                inv_tansho = 0
                inv_umaren = 0
                inv_wide = 0
                inv_sanrenpuku = 0
                inv_sanrentan = 0
                
                for _, r in finished_df.iterrows():
                    p_list = [x for x in str(r.get('partners', '')).split(',') if x.strip().isdigit()]
                    p_len = len(p_list)
                    if p_len >= 3:
                        inv_tansho += 100
                        inv_umaren += 300  
                        inv_wide += 300    
                        inv_sanrenpuku += int(p_len * (p_len - 1) / 2) * 100 
                        inv_sanrentan += int(p_len * (p_len - 1)) * 100      
                
                invested_total = inv_tansho + inv_umaren + inv_wide + inv_sanrenpuku + inv_sanrentan
                roi_total = (returns / invested_total) * 100 if invested_total > 0 else 0.0
                
                st.markdown(f"**{title_prefix} 判明レース**: {total} 件 （結果待ち: {total_races - total} 件）")
                col1, col2, col3 = st.columns(3)
                col1.metric("🎯 的中率", f"{(hits/total)*100:.1f}%", f"{hits} / {total} 的中")
                col2.metric("💰 回収率", f"{roi_total:.1f}%", delta_color="normal" if returns >= invested_total else "inverse")
                col3.metric("💴 収支", f"{int(returns - invested_total):,} 円")
                
                st.markdown("<br><h5>🎫 券種別の詳細データ</h5>", unsafe_allow_html=True)
                ticket_cols = st.columns(5)
                
                def make_ticket_card(col, name, hits_val, returns_val, inv_val):
                    roi_val = (returns_val / inv_val) * 100 if inv_val > 0 else 0
                    profit_val = int(returns_val - inv_val)
                    color = "red" if profit_val < 0 else "green"
                    sign = "+" if profit_val > 0 else ""
                    col.markdown(f'''
                    <div style="border:1px solid #ddd; padding:10px; border-radius:5px; text-align:center; background-color:#fafafa;">
                        <div style="font-weight:bold; font-size:1.1em; margin-bottom:5px;">{name}</div>
                        <div style="font-size:0.85em; color:#555;">投資: {int(inv_val):,}円</div>
                        <div style="font-size:0.85em; color:#555;">的中率: {(hits_val/total)*100:.1f}%</div>
                        <div style="font-size:0.85em; color:#555;">回収率: {roi_val:.1f}%</div>
                        <div style="font-weight:bold; color:{color}; margin-top:5px;">{sign}{profit_val:,} 円</div>
                    </div>
                    ''', unsafe_allow_html=True)

                if 'pay_tansho' in finished_df.columns:
                    h_t = len(finished_df[finished_df['pay_tansho'].astype(float) > 0])
                    r_t = finished_df['pay_tansho'].astype(float).sum()
                    make_ticket_card(ticket_cols[0], "単勝", h_t, r_t, inv_tansho)
                    
                    h_u = len(finished_df[finished_df['pay_umaren'].astype(float) > 0])
                    r_u = finished_df['pay_umaren'].astype(float).sum()
                    make_ticket_card(ticket_cols[1], "馬連", h_u, r_u, inv_umaren)
                    
                    h_w = len(finished_df[finished_df['pay_wide'].astype(float) > 0])
                    r_w = finished_df['pay_wide'].astype(float).sum()
                    make_ticket_card(ticket_cols[2], "ワイド", h_w, r_w, inv_wide)
                    
                    h_3f = len(finished_df[finished_df['pay_sanrenpuku'].astype(float) > 0])
                    r_3f = finished_df['pay_sanrenpuku'].astype(float).sum()
                    make_ticket_card(ticket_cols[3], "三連複", h_3f, r_3f, inv_sanrenpuku)
                    
                    h_3t = len(finished_df[finished_df['pay_sanrentan'].astype(float) > 0])
                    r_3t = finished_df['pay_sanrentan'].astype(float).sum()
                    make_ticket_card(ticket_cols[4], "三連単", h_3t, r_3t, inv_sanrentan)

                st.markdown("<br>", unsafe_allow_html=True)
                st.caption(f"※ 投資金額・回収率は「三連単◎1着固定流し(20点)」や「馬連・ワイド上位3点」などを想定した最適化実点数で計算されています。")
                
            st.dataframe(raw_df[['date', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay']].sort_values(by='date', ascending=False), use_container_width=True)

        with tab_total:
            render_dashboard_for_df(df_history, "総合")
            
        with tab_month:
            months = sorted(df_history['year_month'].dropna().unique(), reverse=True)
            if months:
                selected_month = st.selectbox("表示する月を選択", months)
                month_df = df_history[df_history['year_month'] == selected_month]
                render_dashboard_for_df(month_df, f"{selected_month} の")
                
        with tab_day:
            days = sorted(df_history['just_date'].dropna().unique(), reverse=True)
            if days:
                selected_day = st.selectbox("表示する日付を選択", days)
                day_df = df_history[df_history['just_date'] == selected_day]
                render_dashboard_for_df(day_df, f"{selected_day} の")