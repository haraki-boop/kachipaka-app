import os
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from google import genai
from google.genai import types

st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="image_61b676.png", layout="wide")

col_img, col_text = st.columns([1, 10])
with col_img:
    try:
        st.image("image_61b676.png", width=70)
    except:
        pass
with col_text:
    st.title("AI予想 勝ちぱかくん")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
HISTORY_CSV = "prediction_history.csv"

@st.cache_resource
def load_model():
    if os.path.exists("keiba_ai_model.pkl"):
        return joblib.load("keiba_ai_model.pkl")
    return None

@st.cache_data
def load_past_data():
    if os.path.exists("cleaned_keiba_data.csv"):
        df = pd.read_csv("cleaned_keiba_data.csv", low_memory=False, dtype={'race_id': str})
        if 'time_seconds' not in df.columns and 'タイム' in df.columns:
            def ts(t):
                if pd.isna(t): return np.nan
                p = str(t).strip().split(':')
                if len(p) == 2: return float(p[0]) * 60 + float(p[1])
                try: return float(p[0])
                except: return np.nan
            df['time_seconds'] = df['タイム'].apply(ts)
        return df
    return pd.DataFrame()

@st.cache_data
def load_future_data():
    if os.path.exists("future_races.csv"):
        df = pd.read_csv("future_races.csv", dtype={'race_id': str})
        PLACE_MAP_REV = {
            "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
        }
        df['place_code'] = df['race_id'].str[4:6]
        df['place_name'] = df['place_code'].map(PLACE_MAP_REV).fillna("不明")
        df['r_num'] = df['race_id'].str[10:12].astype(int)
        
        if 'race_name' not in df.columns:
            df['race_name'] = ""
        df['day_label'] = df['date'] if 'date' in df.columns else "不明"
        return df
    return pd.DataFrame()

def load_history_data():
    if os.path.exists(HISTORY_CSV):
        return pd.read_csv(HISTORY_CSV, dtype={'race_id': str, 'honmei_umaban': str})
    return pd.DataFrame(columns=['date', 'race_id', 'race_name', 'honmei_umaban', 'honmei_name', 'result_pay'])

model = load_model()
df_past = load_past_data()
df_future = load_future_data()
df_history = load_history_data()

def update_race_results():
    if df_history.empty: return
    updated = False
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for idx, row in df_history.iterrows():
        if pd.isna(row['result_pay']) or str(row['result_pay']).strip() == "":
            rid = str(row['race_id'])
            url = f"https://db.netkeiba.com/race/{rid}/"
            try:
                res = requests.get(url, headers=headers, timeout=5)
                res.encoding = 'euc-jp'
                soup = BeautifulSoup(res.text, "html.parser")
                tables = soup.find_all("table", summary="払い戻し")
                if tables:
                    for table in tables:
                        for tr in table.find_all("tr"):
                            th = tr.find("th")
                            if th and th.text.strip() == "単勝":
                                tds = tr.find_all("td")
                                winners = [w.strip() for w in list(tds[0].stripped_strings)]
                                amounts = [a.replace(',', '').strip() for a in list(tds[1].stripped_strings)]
                                pred_num = str(int(row['honmei_umaban']))
                                if pred_num in winners:
                                    win_idx = winners.index(pred_num)
                                    df_history.at[idx, 'result_pay'] = int(amounts[win_idx])
                                else:
                                    df_history.at[idx, 'result_pay'] = 0
                                updated = True
            except:
                pass
            time.sleep(1)
    if updated:
        df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')

st.sidebar.header("🏁 実戦結果の更新")
if st.sidebar.button("🏆 終了したレースの結果を取得", use_container_width=True):
    with st.spinner("🏁 レース結果を照合中..."):
        update_race_results()
        st.cache_data.clear()
        st.success("✅ 成績を最新化しました！")
        time.sleep(1)
        st.rerun()

def calculate_race_scores(race_id_target, target_df):
    race_df = target_df[target_df['race_id'] == str(race_id_target)].copy()
    if race_df.empty or df_past.empty or model is None: return None
    features = ['枠番', '馬番', '斤量', 'time_seconds', 'sex_code', 'age', 'horse_weight', 'weight_change']
    for f in features:
        if f not in race_df.columns: race_df[f] = 0
    overall_mean = df_past['time_seconds'].dropna().mean() if 'time_seconds' in df_past.columns else 90.0
    X = race_df[features].copy()
    X['time_seconds'] = X['time_seconds'].fillna(overall_mean)
    X['単勝'] = 10.0
    X['人気'] = 5.0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    prob = model.predict_proba(X)[:, 1]
    s = prob.sum()
    race_df['win_prob'] = prob / s if s > 0 else 1.0 / len(race_df)
    mp = race_df['win_prob'].mean()
    rs = 100 + ((race_df['win_prob'] - mp) / mp) * 35 if mp > 0 else 100
    race_df['score'] = np.clip(rs, 50, 120).round().astype(int)
    return race_df.sort_values(by='score', ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def get_all_markers():
    markers = {}
    if df_future.empty: return markers
    for rid in df_future['race_id'].unique():
        sdf = calculate_race_scores(rid, df_future)
        if sdf is not None and len(sdf) >= 5:
            sc = sdf['score'].tolist()
            if sc[0] >= 108 and (sc[0] - sc[1]) >= 4:
                markers[rid] = "★"
            elif (sc[0] - sc[4]) <= 5:
                markers[rid] = "◎"
            else:
                markers[rid] = ""
    return markers
markers = get_all_markers()

tab_forecast, tab_dashboard = st.tabs(["🏇 レース予想", "📈 実戦成績"])

with tab_forecast:
    if df_future.empty:
        st.warning("⚠️ 出馬表データが存在しません。ローカルで scrape_shutsuba.py を実行してください。")
    else:
        date_options = sorted(df_future['day_label'].unique())
        selected_date = st.radio("開催日", date_options, horizontal=True, label_visibility="collapsed")
        
        day_df = df_future[df_future['day_label'] == selected_date]
        places = day_df['place_name'].unique()
        
        # 会場をタブでコンパクトに切り替え
        place_tabs = st.tabs([f"🏇 {p}" for p in places])
        
        for p_idx, place in enumerate(places):
            with place_tabs[p_idx]:
                place_df = day_df[day_df['place_name'] == place]
                races = sorted(place_df['r_num'].unique())
                
                # 横並びグリッド配置（コンパクト表示）
                cols = st.columns(6)
                for i, r in enumerate(races):
                    col = cols[i % 6]
                    race_rows = place_df[place_df['r_num'] == r]
                    rid = race_rows['race_id'].iloc[0]
                    rname = str(race_rows['race_name'].iloc[0]).strip() if 'race_name' in race_rows.columns else ""
                    mark = markers.get(rid, "")
                    
                    label = f"{r}R {mark}".strip()
                    if rname:
                        # 省スペース用に短縮ラベル
                        short_name = rname if len(rname) <= 8 else rname[:7] + "…"
                        label = f"{r}R {short_name} {mark}".strip()
                        
                    btn_type = "primary" if "★" in mark else "secondary"
                    if col.button(label, key=f"btn_{rid}", use_container_width=True, type=btn_type):
                        st.session_state['selected_race_id'] = rid

    if st.session_state['selected_race_id']:
        st.markdown("---")
        target_id = st.session_state['selected_race_id']
        target_race_info = df_future[df_future['race_id'] == target_id].iloc[0]
        
        rname = target_race_info.get('race_name', "")
        race_display_name = f"{target_race_info['place_name']} {target_race_info['r_num']}R 【{rname}】" if rname else f"{target_race_info['place_name']} {target_race_info['r_num']}R"
            
        st.subheader(f"🚀 {race_display_name}")
        
        if st.button("🧠 勝ちぱかくんに最終予想させる！", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】GEMINI_API_KEY が見つかりません。")
                st.stop()
                
            scored_df = calculate_race_scores(target_id, df_future)
            honmei_row = scored_df.iloc[0]
            honmei_umaban = int(honmei_row['馬番'])
            honmei_name = honmei_row['馬名']
            
            if df_history.empty or target_id not in df_history['race_id'].values:
                new_record = pd.DataFrame([{
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'race_id': target_id,
                    'race_name': race_display_name,
                    'honmei_umaban': honmei_umaban,
                    'honmei_name': honmei_name,
                    'result_pay': ""
                }])
                df_history = pd.concat([df_history, new_record], ignore_index=True)
                df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
            
            table_summary = []
            for _, row in scored_df.iterrows():
                table_summary.append(
                    f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | 騎手:{row.get('騎手', '不明')} | "
                    f"score:{row['score']:3d} | AI勝率予想:{row['win_prob']*100:.1f}% | 斤量:{row.get('斤量', 0)}kg"
                )
            prompt_data = "\n".join(table_summary)

            system_instruction = f"""
あなたは競馬分析AI「勝ちぱかくん」です。提供されたデータ（score）を絶対の基準とし、シンプルに印と買い目のみを出力してください。
1. Web検索を使用し、{race_display_name}の最新情報を確認してください。
2. 印ルール: scoreの最高値に【◎】、2位に【◯】、3位に【▲】、4位に【△】、5位に【☆】
3. 長文解説は一切不要。結果のみを出力。

### 1. 【全頭 データ一覧】
| 馬番 | 馬名 | 騎手 | score | AI勝率予想 |
### 2. 【勝ちぱかくんの印】
◎：〇番（馬名）など
### 3. 【推奨買い目】
* 単勝: ◎
* 複勝: ◎、◯
* 馬連: ◎ - ◯, ▲, △, ☆
"""
            prompt = f"以下のレース（ID: {target_id}、{race_display_name}）の出走馬データです。\n{prompt_data}"

            with st.spinner("解析とWeb検索を実行中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            tools=[{"google_search": {}}],
                            temperature=0.7
                        )
                    )
                    st.markdown(response.text)
                    st.success("📝 このレースの予想（本命馬）を実戦履歴に記録しました！")
                except Exception as e:
                    st.error(f"【APIエラー】: {e}")

with tab_dashboard:
    st.subheader("📈 実戦成績（単勝ベース）")
    if df_history.empty:
        st.info("まだ予想履歴がありません。")
    else:
        finished_races = df_history[pd.to_numeric(df_history['result_pay'], errors='coerce').notna()]
        total_races = len(finished_races)
        waiting_races = len(df_history) - total_races
        st.markdown(f"**結果判明レース**: {total_races} 件 （結果待ち: {waiting_races} 件）")
        
        if total_races > 0:
            hits = len(finished_races[finished_races['result_pay'].astype(float) > 0])
            returns = finished_races['result_pay'].astype(float).sum()
            invested = total_races * 100
            hit_rate = (hits / total_races) * 100
            recovery_rate = (returns / invested) * 100
            profit = returns - invested
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 実戦 的中率", f"{hit_rate:.1f}%", f"{hits} / {total_races} 的中")
            delta_color = "normal" if recovery_rate >= 100 else "inverse"
            col2.metric("💰 実戦 回収率", f"{recovery_rate:.1f}%", f"{recovery_rate - 100:.1f}%", delta_color=delta_color)
            col3.metric("💴 累計収支 (1R100円)", f"{int(profit):,} 円")
            
        st.markdown("---")
        st.dataframe(df_history[['date', 'race_name', 'honmei_umaban', 'honmei_name', 'result_pay']].sort_values(by='date', ascending=False), use_container_width=True)