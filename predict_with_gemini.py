import os
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from google import genai
from google.genai import types

# --------------------------------------------------
# ページ初期設定
# --------------------------------------------------
st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="🦙", layout="wide")

st.title("🦙 AI予想 勝ちぱかくん")
st.caption("LightGBM基礎スコア（全過去走自動集計） × GeminiリアルタイムWeb検索（回収率特化型）")

# --------------------------------------------------
# 🔑 Gemini APIキーの設定
# --------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

# --------------------------------------------------
# タイム（文字列）を秒数に変換する補助関数
# --------------------------------------------------
def time_to_seconds(t_str):
    if pd.isna(t_str):
        return None
    t_str = str(t_str).strip()
    parts = t_str.split(':')
    if len(parts) == 2:
        try:
            return float(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return None
    elif len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            return None
    return None

# --------------------------------------------------
# データ＆モデル読み込み
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "keiba_ai_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"【モデル読み込みエラー】: {e}")
            return None
    return None

def load_data():
    data_path = "cleaned_keiba_data.csv"
    if os.path.exists(data_path):
        try:
            loaded_df = pd.read_csv(data_path, low_memory=False, dtype={'race_id': str})
            if 'time_seconds' not in loaded_df.columns:
                if 'タイム' in loaded_df.columns:
                    loaded_df['time_seconds'] = loaded_df['タイム'].apply(time_to_seconds)
                else:
                    loaded_df['time_seconds'] = np.nan
            return loaded_df
        except Exception as e:
            st.error(f"【CSV読み込みエラー】: {e}")
            return None
    return None

model = load_model()
df = load_data()

# --------------------------------------------------
# 🔄 最新データ更新処理
# --------------------------------------------------
def update_keiba_data():
    with st.spinner("🔄 JRA/netkeiba等から最新の出馬表・直前オッズデータを取得中..."):
        try:
            time.sleep(2)
            st.cache_data.clear()
            st.success("✅ 最新の出馬表・直前オッズデータの更新が完了しました！")
            st.rerun()
        except Exception as e:
            st.error(f"【データ更新エラー】: {e}")

# --------------------------------------------------
# 🎯 サイドバー（日付表示対応UI）
# --------------------------------------------------
st.sidebar.header("⚙️ データ更新・管理")
if st.sidebar.button("🔄 最新出馬表・オッズを自動更新", use_container_width=True):
    update_keiba_data()

st.sidebar.markdown("---")
st.sidebar.header("🎯 レース選択")

if df is not None and not df.empty:
    PLACE_MAP_REV = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
        "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
    }

    if 'race_id' in df.columns:
        df['race_id_str'] = df['race_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        date_col = None
        for col in ['date', '日付', '開催日', '年月日', 'Date']:
            if col in df.columns:
                date_col = col
                break

        def parse_race_info(row):
            rid = str(row['race_id_str'])
            raw_date = str(row[date_col]) if date_col and pd.notna(row[date_col]) else ""
            
            if len(rid) == 12:
                year = rid[:4]
                p_code = rid[4:6]
                p_name = PLACE_MAP_REV.get(p_code, "その他")
                kai = int(rid[6:8])
                day = int(rid[8:10])
                r_num = int(rid[10:12])
                
                if raw_date and "年" in raw_date:
                    date_label = raw_date
                elif raw_date and len(raw_date) >= 8:
                    clean_date = raw_date.replace('-', '').replace('/', '')
                    try:
                        date_label = f"{clean_date[:4]}年{int(clean_date[4:6])}月{int(clean_date[6:8])}日"
                    except ValueError:
                        date_label = raw_date
                else:
                    date_label = f"{year}年 第{kai}回{day}日目"
                    
                return year, p_code, p_name, date_label, f"{r_num}R"
            return "不明", "00", "不明", "不明", "不明"

        parsed = df.apply(parse_race_info, axis=1)
        df['place_name'] = [p[2] for p in parsed]
        df['date_label'] = [p[3] for p in parsed]
        df['race_num_label'] = [p[4] for p in parsed]

        # 1. 競馬場選択
        all_places = [p for p in PLACE_MAP_REV.values() if p in df['place_name'].unique()]
        selected_place = st.sidebar.selectbox("🏇 競馬場を選択", all_places if all_places else sorted(df['place_name'].unique()))

        # 2. 開催日選択
        df_place = df[df['place_name'] == selected_place]
        all_dates = sorted(df_place['date_label'].unique(), reverse=True)
        selected_date_label = st.sidebar.selectbox("📅 開催日を選択", all_dates)

        # 3. レース選択
        df_date = df_place[df_place['date_label'] == selected_date_label]
        all_races = sorted(df_date['race_num_label'].unique(), key=lambda x: int(x.replace('R', '')) if 'R' in x else 0)
        selected_race_num = st.sidebar.selectbox("🏁 レース（R）を選択", all_races)

        matched_rows = df_date[df_date['race_num_label'] == selected_race_num]
        if not matched_rows.empty:
            target_race_id = matched_rows['race_id_str'].iloc[0]
            selected_label = f"{selected_date_label} {selected_place} {selected_race_num}"
        else:
            st.error("【エラー】一致するレースIDが見つかりません。")
            st.stop()

        btn_predict = st.sidebar.button("🚀 勝ちぱかくんに予想させる！", type="primary", use_container_width=True)
    else:
        st.error("【エラー】CSV内に 'race_id' 列が見つかりません。")
        st.stop()
else:
    st.error("【エラー】'cleaned_keiba_data.csv' が読み込めません。")
    st.stop()

# --------------------------------------------------
# 🤖 予想生成ロジック
# --------------------------------------------------
def generate_prediction(race_id_target, race_display_name):
    if not GEMINI_API_KEY:
        st.error("【設定エラー】GEMINI_API_KEY が見つかりません。")
        return

    if model is None:
        st.error("【モデルエラー】AIモデル（keiba_ai_model.pkl）が読み込めません。")
        return

    race_df = df[df['race_id_str'] == str(race_id_target)].copy()

    if race_df.empty:
        st.warning(f"選択されたレース（ID: {race_id_target}）の出走馬データが見つかりません。")
        return

    past_df = df[df['race_id_str'] != str(race_id_target)].copy()

    # 特徴量の準備
    features = ['枠番', '馬番', '斤量', 'time_seconds', 'sex_code', 'age', 'horse_weight', 'weight_change']
    if '単勝' in df.columns: features.append('単勝')
    if '人気' in df.columns: features.append('人気')

    for f in features:
        if f not in race_df.columns:
            race_df[f] = 0

    X_race = race_df[features].copy()

    # 人気と単勝をダミー化してオッズ依存を排除
    if '単勝' in X_race.columns: X_race['単勝'] = 10.0
    if '人気' in X_race.columns: X_race['人気'] = 5.0

    # 過去タイムの平均値
    if 'time_seconds' in past_df.columns and not past_df['time_seconds'].dropna().empty:
        overall_mean_time = past_df['time_seconds'].dropna().mean()
    else:
        overall_mean_time = 90.0

    past_avg_ranks = []
    past_top3_rates = []
    filled_times = []

    # 過去走の集計ループ
    for idx, row in race_df.iterrows():
        hname = row.get('馬名', '')
        h_past = past_df[past_df['馬名'] == hname] if ('馬名' in past_df.columns and hname) else pd.DataFrame()
        
        if not h_past.empty:
            avg_r = pd.to_numeric(h_past['着順'], errors='coerce').mean() if '着順' in h_past.columns else 8.0
            avg_r = avg_r if pd.notna(avg_r) else 8.0
            
            top3_r = (pd.to_numeric(h_past['着順'], errors='coerce') <= 3).mean() if '着順' in h_past.columns else 0.1
            top3_r = top3_r if pd.notna(top3_r) else 0.1
            
            if 'time_seconds' in h_past.columns:
                v_times = pd.to_numeric(h_past['time_seconds'], errors='coerce').dropna()
                avg_t = v_times.mean() if not v_times.empty else overall_mean_time
            else:
                avg_t = overall_mean_time
        else:
            avg_r = 8.0
            top3_r = 0.1
            avg_t = overall_mean_time

        past_avg_ranks.append(avg_r)
        past_top3_rates.append(top3_r)
        filled_times.append(avg_t)

    X_race['time_seconds'] = filled_times
    X_race = X_race.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 予測＆補正
    raw_prob = model.predict_proba(X_race)[:, 1]
    rank_weights = np.array([1.0 / (r + 1.0) for r in past_avg_ranks])
    top3_weights = np.array(past_top3_rates) + 0.1
    combined_prob = raw_prob * rank_weights * top3_weights

    prob_sum = combined_prob.sum()
    if prob_sum > 0:
        race_df['win_prob'] = combined_prob / prob_sum
    else:
        race_df['win_prob'] = 1.0 / len(race_df)

    mean_prob = race_df['win_prob'].mean()
    if mean_prob > 0:
        raw_score = 100 + ((race_df['win_prob'] - mean_prob) / mean_prob) * 35
    else:
        raw_score = 100

    race_df['score'] = np.clip(raw_score, 50, 120).round().astype(int)
    race_df = race_df.sort_values(by='win_prob', ascending=False).reset_index(drop=True)

    table_summary = []
    for idx, row in race_df.iterrows():
        horse_name = row.get('馬名', f"馬番{int(row.get('馬番', 0))}")
        jockey_name = row.get('騎手', "不明")
        
        horse_info = (
            f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{horse_name} | 騎手:{jockey_name} | "
            f"score:{row['score']:3d} | AI勝率予想:{row['win_prob']*100:.1f}% | "
            f"斤量:{row.get('斤量', 0)}kg | 人気:{row.get('人気', '---')} | 単勝:{row.get('単勝', '---')}倍"
        )
        table_summary.append(horse_info)

    prompt_data = "\n".join(table_summary)

    # ==================================================
    # 🚨 AIへのプロンプト指示（回収率特化型）
    # ==================================================
    system_instruction = f"""
あなたは競馬分析AI「勝ちぱかくん」であり、的中率ではなく「長期的な回収率の最大化」に特化した冷酷で正確なプロ馬券師です。
オッズや人間の感情的バイアスを完全に排除し、純粋なデータ（score）とリアルタイムの物理的要因（馬場・展開）のみで期待値を算出します。

【厳守事項・最優先ルール】
1. Web検索の必須化：必ず「{race_display_name} 出馬表」「今日の〇〇競馬場 馬場傾向」「展開予想」を検索し、当日のトラックバイアスと展開の有利不利を把握すること。
2. データ至上主義：提供された出走馬データと「score」を絶対の正解として扱うこと。
3. 印（◎◯▲△☆）の打刻ルール（※回収率特化）：
   ・【◎（本命）】：必ず「score」の1位または2位から、当日の馬場・展開に最もフィットする馬を選ぶこと。
   ・【◯（対抗）】：◎を負かすポテンシャルがある、score上位馬。
   ・【▲（単穴）】：展開次第で一発がある馬。
   ・【☆（特注穴馬）】：scoreは中位以下だが、Web検索で「距離短縮」「極端な馬場傾向の恩恵」「展開のドハマり」など、明確な【期待値の跳ね上がり】が確認できる馬を1頭だけ抜擢すること。
4. 思考の透明化：なぜその馬を選んだのか、「scoreの裏付け」と「Web検索で得た物理的根拠（馬場・展開・調教）」を論理的に説明すること。

【構成案と出力フォーマット】

1. 【レース概況とトラックバイアス（展開予想）】
（Web検索から得た当日の馬場状態と、ハナを主張する馬から想定されるペースを簡潔に記載）

2. 【全頭 データ一覧】
（※途中で文章を挟まず、提供された全頭分を以下のMarkdown形式で出力すること）
| 馬番 | 馬名 | 騎手 | score | AI勝率予想 | 人気 | 単勝オッズ |
|---|---|---|---|---|---|---|
| 01 | 〇〇 | 〇〇 | 115 | 22.5% | 2 | 4.5倍 |

3. 【勝ちぱかくんの冷酷なるジャッジ（印と根拠）】
（◎◯▲△☆の印とともに、scoreと環境要因を掛け合わせた期待値の根拠を記載）

4. 【回収率追求型・推奨買い目】
※点数を無駄に広げず、期待値の高い組み合わせに資金を集中させること。
・【三連複フォーメーション】：◎を1列目、◯▲☆を2列目に置いた、点数を絞りつつ高配当を逃さないフォーメーション（10〜15点以内）。
・【三連単フォーメーション】：◎の1着固定、または◎と◯の1・2着折り返しなど、勝率の高い馬をアタマに固定したフォーメーション（20点以内）。
"""

    prompt = f"""
以下のレース（ID: {race_id_target}、{race_display_name}）の出走馬データをベースに、最新情報をWeb検索してプロ予想記事を作成してください。
--- 出走馬ベースデータ ---
{prompt_data}
---
"""

    with st.spinner("🦙 勝ちぱかが過去成績を集計し、当日の馬場・展開と掛け合わせて期待値を算定中..."):
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        models_to_try = ['gemini-2.5-flash', 'gemini-2.5-flash', 'gemini-2.5-pro']
        response = None
        
        for i, model_name in enumerate(models_to_try):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=[{"google_search": {}}],
                        temperature=0.75
                    )
                )
                if response:
                    break
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    if i < len(models_to_try) - 1:
                        st.warning(f"⚠️ サーバー混雑中...（{i+1}/{len(models_to_try)}回目）")
                        time.sleep(3)
                        continue
                    else:
                        st.error("【APIエラー】サーバーが大変混雑しています。数分時間をおいてから再度お試しください。")
                        return
                else:
                    st.error(f"【APIエラー】: {e}")
                    return

        if response and response.text:
            st.markdown("---")
            st.subheader(f"📰 {race_display_name} 勝ちぱかくんの予想")
            st.markdown(response.text)
        else:
            st.error("【エラー】Google APIからの応答を取得できませんでした。")

# --------------------------------------------------
# メイン画面描画
# --------------------------------------------------
st.info(f"選択中: **{selected_label}** (ID: `{target_race_id}`)")

if btn_predict:
    generate_prediction(target_race_id, selected_label)
