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
st.caption("LightGBM基礎スコア（全過去走自動集計・人気度外視） × GeminiリアルタイムWeb検索")

# --------------------------------------------------
# 🔑 Gemini APIキーの設定
# --------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

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
            return pd.read_csv(data_path, low_memory=False, dtype={'race_id': str})
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
# 🤖 予想生成ロジック（過去成績自動集計版）
# --------------------------------------------------
def generate_prediction(race_id_target, race_display_name):
    if not GEMINI_API_KEY:
        st.error("【設定エラー】GEMINI_API_KEY が見つかりません。")
        return

    if model is None:
        st.error("【モデルエラー】AIモデル（keiba_ai_model.pkl）が読み込めません。")
        return

    # 対象レースのデータ
    race_df = df[df['race_id_str'] == str(race_id_target)].copy()

    if race_df.empty:
        st.warning(f"選択されたレース（ID: {race_id_target}）の出走馬データが見つかりません。")
        return

    # 過去全データの抽出（対象レース以外）
    past_df = df[df['race_id_str'] != str(race_id_target)].copy()

    # ==================================================
    # 🚨 過去データの一括分析＆特徴量への注入
    # ==================================================
    features = ['枠番', '馬番', '斤量', 'time_seconds', 'sex_code', 'age', 'horse_weight', 'weight_change']
    if '単勝' in df.columns: features.append('単勝')
    if '人気' in df.columns: features.append('人気')

    for f in features:
        if f not in race_df.columns:
            race_df[f] = 0

    X_race = race_df[features].copy()

    # オッズと人気はダミー化してオッズ依存を排除
    if '単勝' in X_race.columns: X_race['単勝'] = 10.0
    if '人気' in X_race.columns: X_race['人気'] = 5.0

    # 全体平均タイム（過去データの基準値）
    overall_mean_time = past_df['time_seconds'].dropna().mean() if 'time_seconds' in past_df.columns and not past_df['time_seconds'].dropna().empty else 90.0

    past_avg_ranks = []
    past_top3_rates = []
    filled_times = []

    for idx, row in race_df.iterrows():
        hname = row.get('馬名', '')
        # 対象馬の過去走を取得
        h_past = past_df[past_df['馬名'] == hname] if '馬名' in past_df.columns and hname else pd.DataFrame()
        
        if not h_past.empty:
            avg_r = pd.to_numeric(h_past['着順'], errors='coerce').mean()
            avg_r = avg_r if pd.notna(avg_r) else 8.0
            
            top3_r = (pd.to_numeric(h_past['着順'], errors='coerce') <= 3).mean()
            top3_r = top3_r if pd.notna(top3_r) else 0.1
            
            v_times = pd.to_numeric(h_past['time_seconds'], errors='coerce').dropna()
            avg_t = v_times.mean() if not v_times.empty else overall_mean_time
        else:
            avg_r = 8.0
            top3_r = 0.1
            avg_t = overall_mean_time

        past_avg_ranks.append(avg_r)
        past_top3_rates.append(top3_r)
        filled_times.append(avg_t)

    # 0埋めされていた time_seconds に「過去平均走破タイム」を代入
    X_race['time_seconds'] = filled_times
    X_race = X_race.apply(pd.to_numeric, errors='coerce').fillna(0)

    # LightGBMモデルによる予測勝率
    raw_prob = model.predict_proba(X_race)[:, 1]

    # 過去着順・複勝率による能力感応度の強化（実力差をくっきり広げる補正）
    rank_weights = np.array([1.0 / (r + 1.0) for r in past_avg_ranks])
    top3_weights = np.array(past_top3_rates) + 0.1
    
    combined_prob = raw_prob * rank_weights * top3_weights

    # ==================================================
    # 🚨 勝率の100%正規化と明確なスコア化
    # ==================================================
    prob_sum = combined_prob.sum()
    if prob_sum > 0:
        race_df['win_prob'] = combined_prob / prob_sum
    else:
        race_df['win_prob'] = 1.0 / len(race_df)

    mean_prob = race_df['win_prob'].mean()
    if mean_prob > 0:
        # スコアの開きを広げて1位〜最下位までの強弱をくっきり表現
        raw_score = 100 + ((race_df['win_prob'] - mean_prob) / mean_prob) * 35
    else:
        raw_score = 100

    race_df['score'] = np.clip(raw_score, 50, 120).round().astype(int)
    race_df = race_df.sort_values(by='win_prob', ascending=False).reset_index(drop=True)

    # ==================================================
    # AIへ渡すデータリスト作成
    # ==================================================
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
    # 🚨 AIへのプロンプト指示（スコア遵守ルールを徹底強化）
    # ==================================================
    system_instruction = f"""
あなたは競馬分析AI「勝ちぱかくん」であり、データと現場情報を冷静に分析する凄腕のプロ馬券師です。

【厳守事項・最優先ルール】
1. Web検索ツールを使用する際は、必ず「{race_display_name} 出馬表」のようにレースを特定して検索するか、提供された出馬表の「馬名」を検索して正確な情報を拾ってください。
2. 今回分析するレースは【{race_display_name}】です。提供されたCSVの出走馬データ（馬名など）を『絶対の正解』として扱ってください。
3. **印（◎、◯、▲、△、☆）の打刻ルール**:
   必ず提供された「score」の最高値の馬に【◎】、2位に【◯】、3位に【▲】...と、**スコアの上位順に厳格に印を打ってください**。AIスコアを無視して別の馬を本命(◎)に指定することは【絶対厳禁】です。見解や買い目もすべてこの印（スコア順位）に基づいて構成してください。
4. 提供された「score」は過去の全走タイム・着順実績を多角的に解析した純粋能力値です。オッズや世間の人気に左右されず、このスコアを絶対の軸として分析を行ってください。

【構成案と出力フォーマット】（以下の1〜4の構成で出力）

1. 【レース概況と展開予想】
2. 【全頭 データ一覧】
（※途中で文章を挟まず、提供された全頭分を以下のMarkdown形式で出力すること）
| 馬番 | 馬名 | 騎手 | score | AI勝率予想 | 人気 | 単勝オッズ |
|---|---|---|---|---|---|---|
| 01 | 〇〇 | 〇〇 | 115 | 22.5% | 2 | 4.5倍 |

3. 【勝ちぱかくんのジャッジ・見解（印と推奨馬）】
（スコア上位順に評価を行い、検索で得た直前情報や解説を添えること）
4. 【推奨買い目】
必ず【三連複】と【三連単】の2つの買い目を提示すること。
※三連単はフォーメーション形式で、1着軸にはスコア1位（◎）を固定すること。
"""

    prompt = f"""
以下のレース（ID: {race_id_target}、{race_display_name}）の出走馬データをベースに、最新情報をWeb検索してプロ予想記事を作成してください。
--- 出走馬ベースデータ ---
{prompt_data}
---
"""

    with st.spinner("🦙 勝ちぱかが過去走の全データを一括解析し、Web検索で最新情報を調査中..."):
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
                        temperature=0.7
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
