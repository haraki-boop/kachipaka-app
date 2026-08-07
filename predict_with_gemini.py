import os
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import subprocess
from google import genai
from google.genai import types

# --------------------------------------------------
# ページ初期設定
# --------------------------------------------------
st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="🦙", layout="wide")

st.title("🦙 AI予想 勝ちぱかくん (実戦モード)")
st.caption("AI基礎スコア判定 ＋ Geminiリアルタイム検索 ＋ バックテスト成績")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

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

@st.cache_data
def load_payout_data():
    if os.path.exists("payout_data.csv"):
        return pd.read_csv("payout_data.csv", dtype={'race_id': str})
    return pd.DataFrame()

model = load_model()
df_past = load_past_data()
df_future = load_future_data()
df_payout = load_payout_data()

# --------------------------------------------------
# 🔄 最新データ更新ボタン（サイドバー）
# --------------------------------------------------
st.sidebar.header("⚙️ データ管理")
if st.sidebar.button("🔄 今週末の出馬表を取得", use_container_width=True):
    with st.spinner("🔄 JRA/netkeiba等からデータを取得中..."):
        try:
            result = subprocess.run(["python", "scrape_shutsuba.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.cache_data.clear()
                st.success("✅ 取得完了！")
                time.sleep(1)
                st.rerun()
            else:
                st.sidebar.error("取得失敗。")
        except Exception as e:
            st.sidebar.error(f"エラー: {e}")

if st.sidebar.button("📊 払戻金・成績データを最新化", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("💡 収集スクリプトが動いている間でも、「成績データを最新化」を押せば、そこまで集まったデータで回収率が計算されます。")

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
    
    # 高速化のためバックテスト時は簡易平均を使用
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
# バックテスト計算ロジック
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def calculate_backtest():
    if df_past.empty or df_payout.empty or model is None: return None
    
    # 簡易的に全過去データのAIスコアを算出して◎(1位)を取得
    X = df_past[['枠番', '馬番', '斤量', 'time_seconds', 'sex_code', 'age', 'horse_weight', 'weight_change']].copy()
    X['単勝'] = 10.0
    X['人気'] = 5.0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    df_past_copy = df_past.copy()
    df_past_copy['raw_prob'] = model.predict_proba(X)[:, 1]
    
    # 各レースの1位（本命◎）を抽出
    idx = df_past_copy.groupby('race_id')['raw_prob'].idxmax()
    top_horses = df_past_copy.loc[idx, ['race_id', '馬番', '馬名']]
    
    # 払戻金データと結合
    merged = pd.merge(top_horses, df_payout, on='race_id', how='inner')
    
    total_races = len(merged)
    if total_races == 0: return None
    
    tansho_hits = 0
    tansho_returns = 0
    
    for _, row in merged.iterrows():
        try:
            pred_num = str(int(row['馬番']))
            # 単勝の当たり馬番と配当を取得
            actual_nums = [str(int(n)) for n in str(row['tansho_num']).split('/') if n.strip().isdigit()]
            if pred_num in actual_nums:
                tansho_hits += 1
                pays = str(row['tansho_pay']).split('/')
                tansho_returns += int(pays[0].replace(',', '').strip())
        except:
            continue
            
    invested = total_races * 100 # 1レース100円投資とした場合
    recovery_rate = (tansho_returns / invested) * 100 if invested > 0 else 0
    hit_rate = (tansho_hits / total_races) * 100 if total_races > 0 else 0
    
    return {
        'total_races': total_races,
        'tansho_hits': tansho_hits,
        'invested': invested,
        'returns': tansho_returns,
        'recovery_rate': recovery_rate,
        'hit_rate': hit_rate
    }

backtest_results = calculate_backtest()

# --------------------------------------------------
# 画面レイアウト（タブ切り替え）
# --------------------------------------------------
tab_forecast, tab_dashboard = st.tabs(["🏇 今週のレース予想", "📈 AI成績ダッシュボード"])

# ==================================================
# タブ1：今週の予想（メインUI）
# ==================================================
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
            table_summary = []
            for _, row in scored_df.iterrows():
                table_summary.append(
                    f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | 騎手:{row.get('騎手', '不明')} | "
                    f"score:{row['score']:3d} | AI勝率予想:{row['win_prob']*100:.1f}% | 斤量:{row.get('斤量', 0)}kg"
                )
            prompt_data = "\n".join(table_summary)

            system_instruction = f"""
あなたは競馬分析AI「勝ちぱかくん」です。提供されたデータ（score）を絶対の基準とし、シンプルかつ機械的に印と買い目のみを出力してください。

【厳守事項・最優先ルール】
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

            with st.spinner("🦙 過去データ解析と最新情報のWeb検索を実行中..."):
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
                except Exception as e:
                    st.error(f"【APIエラー】: {e}")


# ==================================================
# タブ2：AI成績ダッシュボード
# ==================================================
with tab_dashboard:
    st.subheader("📊 勝ちぱかくん 過去成績（単勝・本命◎）")
    
    if df_payout.empty:
        st.info("🔄 現在、過去の払戻金データを収集中です。裏側で動いているターミナルの処理が終わるまでお待ちください。")
    elif backtest_results is None:
        st.warning("⚠️ 成績の計算に必要なデータが不足しています。")
    else:
        st.markdown(f"**集計対象レース数**: {backtest_results['total_races']:,} レース（データ収集中...）")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 単勝 的中率", f"{backtest_results['hit_rate']:.1f}%")
        
        # 回収率が100%超えなら赤字（ポジティブ）、下回れば青字などの装飾
        ret_rate = backtest_results['recovery_rate']
        delta_color = "normal" if ret_rate >= 100 else "inverse"
        col2.metric("💰 単勝 回収率", f"{ret_rate:.1f}%", f"{ret_rate - 100:.1f}%", delta_color=delta_color)
        
        col3.metric("💴 累計収支 (1R100円)", f"{backtest_results['returns'] - backtest_results['invested']:,} 円")
        
        st.markdown("---")
        st.markdown("""
        *※現在は「単勝（◎の1点買い）」の基本成績を表示しています。*
        *払戻金データの収集が完了次第、馬連や三連複などの買い目ごとの詳細な回収率分析も追加可能です！*
        """)
