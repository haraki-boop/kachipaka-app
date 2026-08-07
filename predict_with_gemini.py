import os
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import subprocess
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from google import genai
from google.genai import types

# --------------------------------------------------
# ページ初期設定
# --------------------------------------------------
st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="image_61b676.png", layout="wide")

col_img, col_text = st.columns([1, 10])
with col_img:
    try:
        st.image("image_61b676.png", width=80)
    except:
        pass
with col_text:
    st.title("AI予想 勝ちぱかくん")

st.caption("AI基礎スコア判定 ＋ Geminiリアルタイム検索 ＋ 実戦成績ダッシュボード")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

HISTORY_CSV = "prediction_history.csv"

# --------------------------------------------------
# モデル＆データ読み込み
# --------------------------------------------------
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
        df['kai'] = df['race_id'].str[6:8].astype(int)
        df['day'] = df['race_id'].str[8:10].astype(int)
        df['day_label'] = "第" + df['kai'].astype(str) + "回 " + df['day'].astype(str) + "日目"
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

# --------------------------------------------------
# 実戦結果の更新ロジック（レース終了後に結果を取得）
# --------------------------------------------------
def update_race_results():
    if df_history.empty:
        return
        
    updated = False
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    for idx, row in df_history.iterrows():
        # まだ結果が出ていない（NaNまたは空）レースのみ取得
        if pd.isna(row['result_pay']) or str(row['result_pay']).strip() == "":
            rid = str(row['race_id'])
            url = f"https://db.netkeiba.com/race/{rid}/"
            try:
                res = requests.get(url, headers=headers, timeout=5)
                res.encoding = 'euc-jp'
                soup = BeautifulSoup(res.text, "html.parser")
                
                # 単勝の払い戻しを取得
                tables = soup.find_all("table", summary="払い戻し")
                if tables:
                    for table in tables:
                        for tr in table.find_all("tr"):
                            th = tr.find("th")
                            if th and th.text.strip() == "単勝":
                                tds = tr.find_all("td")
                                winners = [w.strip() for w in list(tds[0].stripped_strings)]
                                amounts = [a.replace(',', '').strip() for a in list(tds[1].stripped_strings)]
                                
                                # ◎の馬番が勝っていれば配当を記録、外れていれば0を記録
                                pred_num = str(int(row['honmei_umaban']))
                                if pred_num in winners:
                                    win_idx = winners.index(pred_num)
                                    df_history.at[idx, 'result_pay'] = int(amounts[win_idx])
                                else:
                                    df_history.at[idx, 'result_pay'] = 0
                                updated = True
            except Exception:
                pass
            time.sleep(1)
            
    if updated:
        df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')

# --------------------------------------------------
# 🔄 サイドバー：データ管理＆結果更新
# --------------------------------------------------
st.sidebar.header("⚙️ データ管理")
if st.sidebar.button("🔄 今週末の出馬表を取得", use_container_width=True):
    with st.spinner("🔄 データを取得中..."):
        subprocess.run(["python", "scrape_shutsuba.py"], capture_output=True, text=True)
        st.cache_data.clear()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🏁 実戦結果の更新")
if st.sidebar.button("🏆 終了したレースの結果を取得", use_container_width=True):
    with st.spinner("🏁 レース結果を照合中..."):
        update_race_results()
        st.cache_data.clear()
        st.success("✅ 成績を最新化しました！")
        time.sleep(1)
        st.rerun()
st.sidebar.info("※予想を保存したレースが終わった後に押すと、自動で結果と配当がダッシュボードに反映されます。")

# --------------------------------------------------
# AIスコア計算ロジック
# --------------------------------------------------
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

    raw_prob = model.predict_proba(X)[:, 1]
    prob = raw_prob
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

# --------------------------------------------------
# 画面レイアウト（タブ切り替え）
# --------------------------------------------------
tab_forecast, tab_dashboard = st.tabs(["🏇 今週のレース予想", "📈 実戦成績ダッシュボード"])

with tab_forecast:
    if df_future.empty:
        st.warning("⚠️ 予定されているレースデータがありません。左の「🔄 今週末の出馬表を取得」を押してください。")
    else:
        st.markdown("### 📅 開催日を選択")
        date_options = sorted(df_future['day_label'].unique())
        selected_date = st.radio("", date_options, horizontal=True, label_visibility="collapsed")
        
        st.markdown("`★` = 鉄板・高確率レース（勝負） / `◎` = 大混戦・波乱レース（穴狙い）")
        st.markdown("---")
        
        day_df = df_future[df_future['day_label'] == selected_date]
        places = day_df['place_name'].unique()
        cols = st.columns(len(places))
        
        for j, place in enumerate(places):
            with cols[j]:
                st.markdown(f"#### 🏇 {place}")
                place_df = day_df[day_df['place_name'] == place]
                races = sorted(place_df['r_num'].unique())
                
                for r in races:
                    rid = place_df[place_df['r_num'] == r]['race_id'].iloc[0]
                    mark = markers.get(rid, "")
                    btn_label = f"{r}R {mark}" if mark else f"{r}R"
                    btn_type = "primary" if "★" in mark else "secondary"
                    
                    if st.button(btn_label, key=f"btn_{rid}", use_container_width=True, type=btn_type):
                        st.session_state['selected_race_id'] = rid

    st.markdown("---")

    if st.session_state['selected_race_id']:
        target_id = st.session_state['selected_race_id']
        target_race_info = df_future[df_future['race_id'] == target_id].iloc[0]
        race_display_name = f"{target_race_info['place_name']} {target_race_info['r_num']}R"
        
        st.subheader(f"🚀 {race_display_name} のAI予想")
        
        if st.button("🧠 勝ちぱかくんに最終予想させる！", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】GEMINI_API_KEY が見つかりません。")
                st.stop()
                
            scored_df = calculate_race_scores(target_id, df_future)
            
            # --- 実戦記録を保存（◎本命馬を記憶） ---
            honmei_row = scored_df.iloc[0]
            honmei_umaban = int(honmei_row['馬番'])
            honmei_name = honmei_row['馬名']
            
            # 既に記録済みでなければ履歴CSVに追記
            if df_history.empty or target_id not in df_history['race_id'].values:
                new_record = pd.DataFrame([{
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'race_id': target_id,
                    'race_name': race_display_name,
                    'honmei_umaban': honmei_umaban,
                    'honmei_name': honmei_name,
                    'result_pay': "" # レース終了後に更新
                }])
                df_history = pd.concat([df_history, new_record], ignore_index=True)
                df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
            
            # --- 予想出力 ---
            table_summary = []
            for _, row in scored_df.iterrows():
                table_summary.append(
                    f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | 騎手:{row.get('騎手', '不明')} | "
                    f"score:{row['score']:3d} | AI勝率予想:{row['win_prob']*100:.1f}% | 斤量:{row.get('斤量', 0)}kg"
                )
            prompt_data = "\n".join(table_summary)

            system_instruction = f"""
あなたは競馬分析AI「勝ちぱかくん」です。提供されたデータ（score）を絶対の基準とし、シンプルに印と買い目のみを出力してください。

【厳守事項】
1. Web検索を使用し、{race_display_name}の最新情報を確認してください。
2. **印の打刻ルール**: 提供された「score」の最高値に【◎】、2位に【◯】、3位に【▲】、4位に【△】、5位に【☆】と厳格に打ってください。
3. **長文の解説は【一切不要】です。** 結果のみを出力してください。

【出力フォーマット】
### 1. 【全頭 データ一覧】
| 馬番 | 馬名 | 騎手 | score | AI勝率予想 |
### 2. 【勝ちぱかくんの印】
◎：〇番（馬名）など
### 3. 【推奨買い目】
* 単勝: ◎
* 複勝: ◎、◯
* 馬連: ◎ - ◯, ▲, △, ☆
* 馬単: ◎ → ◯, ▲, △, ☆
* ワイド: ◎ - ◯, ▲, △, ☆
* 三連複: ◎ - ◯, ▲, △, ☆
* 三連単: 1着 ◎ → 2着 ◯, ▲ → 3着 ◯, ▲, △, ☆
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
                    st.success("📝 このレースの予想（本命馬）を実戦履歴に記録しました！レース終了後にサイドバーの更新ボタンを押してください。")
                except Exception as e:
                    st.error(f"【APIエラー】: {e}")

# ==================================================
# タブ2：実戦成績ダッシュボード
# ==================================================
with tab_dashboard:
    st.subheader("📈 あなたと勝ちぱかくんの実戦成績（単勝ベース）")
    
    if df_history.empty:
        st.info("まだ予想履歴がありません。「今週のレース予想」から予想を実行すると、ここに記録が蓄積されます。")
    else:
        # 結果が出ているレースだけを集計対象にする
        finished_races = df_history[pd.to_numeric(df_history['result_pay'], errors='coerce').notna()]
        
        total_races = len(finished_races)
        waiting_races = len(df_history) - total_races
        
        st.markdown(f"**結果判明レース**: {total_races} 件 （結果待ち: {waiting_races} 件）")
        
        if total_races > 0:
            hits = len(finished_races[finished_races['result_pay'].astype(float) > 0])
            returns = finished_races['result_pay'].astype(float).sum()
            invested = total_races * 100 # 1R 100円計算
            
            hit_rate = (hits / total_races) * 100
            recovery_rate = (returns / invested) * 100
            profit = returns - invested
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 実戦 的中率", f"{hit_rate:.1f}%", f"{hits} / {total_races} 的中")
            
            delta_color = "normal" if recovery_rate >= 100 else "inverse"
            col2.metric("💰 実戦 回収率", f"{recovery_rate:.1f}%", f"{recovery_rate - 100:.1f}%", delta_color=delta_color)
            
            col3.metric("💴 累計収支 (1R100円)", f"{int(profit):,} 円")
            
        st.markdown("---")
        st.markdown("#### 📝 直近の予想履歴")
        st.dataframe(df_history[['date', 'race_name', 'honmei_umaban', 'honmei_name', 'result_pay']].sort_values(by='date', ascending=False), use_container_width=True)
