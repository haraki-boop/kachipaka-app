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
        return df
    return pd.DataFrame(columns=['date', 'race_id', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay'])

model_data = load_model()
df_past = load_past_data()
df_future = load_future_data()
df_history = load_history_data()

# ==========================================
# 2. スクレイパー関数 (BOT実行用)
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
                    total_payout = 0
                    
                    for url in [f"https://race.netkeiba.com/race/result.html?race_id={str(row['race_id'])}", f"https://db.netkeiba.com/race/{str(row['race_id'])}/"]:
                        res = requests.get(url, headers=headers, timeout=5)
                        res.encoding = 'EUC-JP' if 'db.netkeiba' in url else 'utf-8'
                        soup = BeautifulSoup(res.text, "html.parser")
                        
                        for tr in soup.find_all("tr"):
                            th = tr.find("th")
                            if not th: continue
                            kind = th.text.strip()
                            if kind not in ["単勝", "馬連", "ワイド", "三連複", "三連単"]: continue
                            
                            tds = tr.find_all("td")
                            if len(tds) < 2: continue
                            
                            payout_found = True
                            
                            w_html = str(tds[0]).replace('<br/>', 'SPLIT').replace('<br>', 'SPLIT').replace('</div>', 'SPLIT')
                            a_html = str(tds[1]).replace('<br/>', 'SPLIT').replace('<br>', 'SPLIT').replace('</div>', 'SPLIT')
                            
                            w_soup = BeautifulSoup(w_html, "html.parser")
                            a_soup = BeautifulSoup(a_html, "html.parser")
                            
                            w_items = w_soup.get_text().split('SPLIT')
                            a_items = a_soup.get_text().split('SPLIT')
                            
                            for w_str, a_str in zip(w_items, a_items):
                                amt_str = re.sub(r'\D', '', a_str)
                                if not amt_str.isdigit(): continue
                                amt = int(amt_str)
                                
                                w_str_clean = re.sub(r'\D', ' ', w_str)
                                w_nums = [str(int(n)) for n in w_str_clean.split() if n.isdigit()]
                                
                                if not w_nums: continue

                                if kind == "単勝" and len(w_nums) >= 1 and w_nums[0] == axis:
                                    total_payout += amt
                                elif kind in ["馬連", "ワイド"] and len(w_nums) >= 2:
                                    if axis in w_nums[:2] and any(p in w_nums[:2] for p in partners):
                                        total_payout += amt
                                elif kind in ["三連複", "三連単"] and len(w_nums) >= 3:
                                    if axis in w_nums[:3] and len(set(w_nums[:3]).intersection(set(partners))) >= 2:
                                        total_payout += amt
                        
                        if payout_found:
                            df_history.at[idx, 'result_pay'] = total_payout
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
# 3. AIスコア計算 ＋ 予想出力 (実力ベース)
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
        if 'my_time_idx' in df_past.columns: new_cols.append('horse_avg_time_idx')
        if 'my_last3f_idx' in df_past.columns: new_cols.append('horse_avg_last3f_idx')
        if 'my_pace_idx' in df_past.columns: new_cols.append('horse_avg_pace_idx')
        if 'my_start_idx' in df_past.columns: new_cols.append('horse_avg_start_idx')
        
        horse_stats.columns = new_cols
        horse_stats['horse_win_rate'] = np.where(horse_stats['total_runs'] > 0, horse_stats['total_wins'] / horse_stats['total_runs'], 0.0)
        race_df = pd.merge(race_df, horse_stats, on='馬名', how='left')

    if not df_past.empty and '騎手' in df_past.columns:
        j_stats = df_past.groupby('騎手')['is_win'].mean().reset_index()
        j_stats.rename(columns={'is_win': 'jockey_win_power'}, inplace=True)
        race_df = pd.merge(race_df, j_stats, on='騎手', how='left')
    else:
        race_df['jockey_win_power'] = 0.0

    for f in features:
        if f not in race_df.columns: race_df[f] = 0.0

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
        preds = model.predict(X)
    except Exception: return None

    inv_preds = 1.0 / np.clip(preds, 1.0, 18.0)
    race_df['win_prob'] = inv_preds / inv_preds.sum()
    
    mp = inv_preds.mean()
    rs = 100 + ((inv_preds - mp) / mp) * 35 if mp > 0 else 100
    race_df['score'] = np.clip(rs, 50, 120).round().astype(int)

    return race_df.sort_values(by='score', ascending=False).reset_index(drop=True)

def get_all_markers():
    markers = {}
    if df_future.empty: return markers
    for rid in df_future['race_id'].unique():
        sdf = calculate_race_scores(rid, df_future)
        if sdf is not None and len(sdf) >= 5:
            sc = sdf['score'].tolist()
            if sc[0] >= 105 and (sc[0] - sc[2]) >= 10: race_type = "【堅】"
            elif (sc[0] - sc[4]) <= 7: race_type = "【穴】"
            else: race_type = "【普】"

            if sc[0] >= 108 and (sc[0] - sc[1]) >= 4: mark = "★"
            elif (sc[0] - sc[4]) <= 5: mark = "◎"
            else: mark = ""
            
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
                        
                    btn_type = "primary" if "【堅】" in mark or "★" in mark else "secondary"
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

            honmei_row = scored_df.iloc[0]
            honmei_umaban = int(honmei_row['馬番'])
            honmei_name = honmei_row['馬名']
            partners_str = ",".join(map(str, scored_df.iloc[1:6]['馬番'].astype(int).tolist()))
            
            if df_history.empty or str(target_id) not in df_history['race_id'].astype(str).values:
                new_record = pd.DataFrame([{'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'race_id': str(target_id), 'race_name': race_display_name, 'honmei_umaban': honmei_umaban, 'partners': partners_str, 'honmei_name': honmei_name, 'result_pay': ""}])
                df_history = pd.concat([df_history, new_record], ignore_index=True)
                df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')

            table_summary = []
            for _, row in scored_df.iterrows():
                odds = row.get('単勝', 0) if pd.notna(row.get('単勝')) else 0.0
                pop = row.get('人気', '-') if pd.notna(row.get('人気')) else '-'
                j_power = row.get('jockey_win_power', 0.0)
                j_info = "過去データなし(要検索)" if j_power == 0.0 else f"勝率{j_power*100:.1f}%"
                
                ev = (row['win_prob'] * odds) if odds > 0 else 0.0
                
                table_summary.append(
                    f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | "
                    f"騎手:{row.get('騎手', '不明')}({j_info}) | "
                    f"オッズ:{odds}倍 ({pop}人気) | "
                    f"純粋勝率:{row['win_prob']*100:.1f}% | 🎯期待値:{ev:.2f} | "
                    f"AIスコア:{row['score']:3d}"
                )

            system_instruction = f"""
あなたはプロの競馬分析AI「勝ちぱかくん」です。以下の絶対ルールに従い、レースの性質を読み切り最適な馬券戦略を遂行してください。

【絶対遵守事項】
1. 挨拶や前置きは一切不要。
2. 【目標】: 過度な大穴狙いは避け、「的中率25%・回収率115%」の黄金ラインを安定して狙う戦略をとること。
3. 【印の基本原則】: 印（◎本命、◯対抗、▲単穴）は、「AIスコア（純粋な実力値）」をベースにしつつも、実際の「オッズ」や「人気」を”適度に加味”して総合的に判断すること。過剰人気馬は評価を下げ、実力があるのにオッズが高い（妙味がある）馬を高く評価する。
4. 【期待値馬の扱い（△・☆）】: AIスコアが上位でなくても、【🎯期待値】が1.0を超えている美味しい馬がいれば、それらを連下（△）や特注穴馬（☆）として確実に紐（相手）に抑えること。
5. 【紐抜け防止】: AIスコアが120（最高評価）に達している馬は非常に能力が高いため、もし本命（◎）にしなくても必ず相手（紐）に含めること。
6. いかなる戦略でも、トリガミ（的中してもマイナス）は完全に排除した買い目を構築する。
7. 【最重要: 買い目の指定】
   - 三連複: 1着候補の馬（◎）ではなく、「3着以内を死守しそうな堅実な馬」または「3着に滑り込みそうな期待値の高い穴馬（☆や△など）」をあえて1頭軸に指定し、4〜5頭へ流す（6点〜10点）フォーメーションをベースとすること。
   - ワイド: ◎から☆（穴馬）や期待値の高い馬への流しを併用し、三連複が紐抜けした時の「保険（資金回収クッション）」として機能させること。

出力フォーマット：
---
### 📊 1. 出走馬 期待値＆データ一覧
（Markdownテーブルで出力）

### 🌪️ 2. レース波乱度と展開分析
* **【レース判定】:** 「堅実決着」または「波乱（混戦）」とその理由
* **【展開・バリュー評価】:** （AIスコアによる実力評価と、実際のオッズや期待値を加味した妙味の指摘）

### 🎯 3. 勝ちぱかくんの印と詳細見解
* **◎（本命）:** 〇番（馬名） - （抜擢理由）
* **◯（対抗）:** 〇番（馬名）
* **▲（単穴）:** 〇番（馬名）
* **△（連下）:** 〇番（馬名）、〇番（馬名）
* **☆（穴馬）:** 〇番（馬名）

### 💡 4. 戦略的・推奨買い目（トリガミ完全排除）
* **【戦略】:** （堅いので少点数で絞る、混戦なので手広く買う等）
* **単勝:** ◎ (1点)
* **馬連:** ◎ － ◯, ▲, △, ☆ (方針に合わせて点数を調整)
* **ワイド (保険):** ◎ － ☆など期待値上位 (連敗防止のクッションとして活用)
* **三連複 (本線):** [3着狙いの軸馬] 1頭軸流し － [相手4〜5頭]
* **三連単:** 1着:◎ → 2・3着:◯, ▲, △, ☆ (方針に合わせて点数を調整)
---
"""
            prompt = f"対象レース: {race_display_name}\n\n出走馬データ:\n{chr(10).join(table_summary)}"

            with st.spinner("AIがレース波乱度を分析し、Gemini(3.6 Flash)が最適戦略を構築中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                try:
                    response = client.models.generate_content(
                        model='gemini-3.6-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.7,
                            tools=[{"googleSearch": {}}]
                        )
                    )
                    res_text = response.text if response.text else (response.candidates[0].content.parts[0].text if response.candidates else "")
                    if res_text: st.markdown(res_text)
                    else: st.warning("⚠️ 回答を取得できませんでした。")
                    st.success("📝 実戦履歴に記録しました！")
                except Exception as e: st.error(f"【APIエラー】: {e}")

with tab_dashboard:
    st.subheader("📈 実戦成績（単・連・ワイ・3複・3単 総合ベース）")
    if df_history.empty: st.info("まだ予想履歴がありません。")
    else:
        finished_races = df_history[pd.to_numeric(df_history['result_pay'], errors='coerce').notna()]
        total = len(finished_races)
        st.markdown(f"**結果判明レース**: {total} 件 （結果待ち: {len(df_history) - total} 件）")
        if total > 0:
            hits = len(finished_races[finished_races['result_pay'].astype(float) > 0])
            returns = finished_races['result_pay'].astype(float).sum()
            invested = total * 5000
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 レース的中率", f"{(hits/total)*100:.1f}%", f"{hits} / {total} 的中")
            col2.metric("💰 総合回収率", f"{(returns/invested)*100:.1f}%", delta_color="normal" if returns >= invested else "inverse")
            col3.metric("💴 累計収支", f"{int(returns - invested):,} 円")
        st.dataframe(df_history[['date', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay']].sort_values(by='date', ascending=False), use_container_width=True)