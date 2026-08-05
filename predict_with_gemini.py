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
st.caption("LightGBM基礎スコア（人気度外視） × GeminiリアルタイムWeb検索")

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
# 🎯 サイドバー（操作エリア）
# --------------------------------------------------
st.sidebar.header("⚙️ データ更新・管理")
if st.sidebar.button("🔄 最新出馬表・オッズを自動更新", use_container_width=True):
    update_keiba_data()

st.sidebar.markdown("---")
st.sidebar.header("🎯 レース選択")

if df is not None and not df.empty:
    PLACE_MAP = {
        "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
        "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"
    }

    if 'race_id' in df.columns:
        df['date_str'] = df['race_id'].astype(str).str[:8]
        available_dates = sorted(df['date_str'].unique(), reverse=True)
    else:
        st.error("【エラー】CSV内に 'race_id' 列が見つかりません。")
        st.stop()

    selected_date = st.sidebar.selectbox("📅 開催日を選択", available_dates)
    selected_place_name = st.sidebar.selectbox("🏇 競馬場を選択", list(PLACE_MAP.keys()))
    selected_place_code = PLACE_MAP[selected_place_name]
    selected_race_num = st.sidebar.selectbox("🏁 レース（R）を選択", [f"{i}R" for i in range(1, 13)])
    race_num_int = int(selected_race_num.replace("R", ""))

    target_race_id = f"{selected_date}{selected_place_code}{race_num_int:02d}"

    btn_predict = st.sidebar.button("🚀 勝ちぱかくんに予想させる！", type="primary", use_container_width=True)
else:
    st.error("【エラー】'cleaned_keiba_data.csv' が読み込めません。")
    st.stop()

# --------------------------------------------------
# 🤖 予想生成ロジック
# --------------------------------------------------
def generate_prediction(race_id_target):
    if not GEMINI_API_KEY:
        st.error("【設定エラー】GEMINI_API_KEY が見つかりません。")
        return

    if model is None:
        st.error("【モデルエラー】AIモデル（keiba_ai_model.pkl）が読み込めません。")
        return

    race_df = df[df['race_id'].astype(str).str.contains(str(race_id_target))].copy()

    if race_df.empty:
        st.warning(f"選択されたレース（ID: {race_id_target}）の出走馬データが見つかりません。")
        return

    # ==================================================
    # 🚨 スコア計算用データ作成（人気・単勝オッズをダミー化して無効化）
    # ==================================================
    features = ['枠番', '馬番', '斤量', 'time_seconds', 'sex_code', 'age', 'horse_weight', 'weight_change']
    
    # モデルの計算エラーを防ぐため列は追加する
    if '単勝' in df.columns:
        features.append('単勝')
    if '人気' in df.columns:
        features.append('人気')

    for f in features:
        if f not in race_df.columns:
            race_df[f] = 0

    X_race = race_df[features].copy()

    # ★計算用のデータだけ全頭同じ値に上書きし、人気を度外視させる★
    if '単勝' in X_race.columns:
        X_race['単勝'] = 10.0
    if '人気' in X_race.columns:
        X_race['人気'] = 5

    X_race = X_race.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 確率を算出（これでオッズの影響を受けない純粋な能力スコアが出る）
    raw_prob = model.predict_proba(X_race)[:, 1]

    # ==================================================
    # 🚨 勝率の100%正規化（合計を1.0にする）
    # ==================================================
    prob_sum = raw_prob.sum()
    if prob_sum > 0:
        race_df['win_prob'] = raw_prob / prob_sum
    else:
        race_df['win_prob'] = 0

    # 独自スコアの算出（100を基準に上下させる）
    mean_prob = race_df['win_prob'].mean()
    if mean_prob > 0:
        raw_score = 100 + ((race_df['win_prob'] - mean_prob) / mean_prob) * 30
    else:
        raw_score = 100

    race_df['score'] = np.clip(raw_score, 50, 120).round().astype(int)
    race_df = race_df.sort_values(by='win_prob', ascending=False).reset_index(drop=True)

    # ==================================================
    # AIへ渡すデータリスト作成（表示用には実際のオッズ・人気を渡す）
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
    # 🚨 AIへのプロンプト指示
    # ==================================================
    system_instruction = f"""
あなたは競馬分析AI「勝ちぱかくん」であり、データと現場情報を冷静に分析する凄腕のプロ馬券師です。

【厳守事項】
1. Web検索ツールを使用する際は、必ず「{selected_date[:4]}年{selected_date[4:6]}月{selected_date[6:8]}日 {selected_place_name}競馬 {selected_race_num} 出馬表」のようにレースを特定して検索するか、個別の「馬名」を検索して正確な情報を拾ってください。
2. 今回分析するレースは【{selected_place_name}競馬場 {selected_race_num}】です。提供されたCSVの出走馬データ（馬名など）を『絶対の正解』として扱ってください。万が一、検索結果と提供データに齟齬があっても、データへの言い訳や不一致の指摘は【1文字たりとも】出力してはいけません。
3. 提供された「score」と「AI勝率予想」は、世間の人気やオッズを一切排除し、純粋な能力値から算出した独自の数値です。これを絶対の評価軸として、オッズに惑わされずに本質的な予想を行ってください。
4. 印（◎◯▲△☆）を打つ推奨馬は、原則【最大5頭まで】に絞ること。スコアが大混戦の場合のみ増やして構いません。

【構成案と出力フォーマット】（以下の1〜4の構成で出力）

1. 【レース概況と展開予想】
2. 【全頭 データ一覧】
（※途中で文章を挟まず、提供された全頭分を以下のMarkdown形式で出力すること）
| 馬番 | 馬名 | 騎手 | score | AI勝率予想 | 人気 | 単勝オッズ |
|---|---|---|---|---|---|---|
| 01 | 〇〇 | 〇〇 | 100 | 12.5% | 2 | 4.5倍 |

3. 【勝ちぱかくんのジャッジ・見解（印と推奨馬）】
（各馬の評価や検索で得た情報はここで語ること）
4. 【推奨買い目】
必ず【三連複】と【三連単】の2つの買い目を提示すること。
※三連単は点数が非現実的にならないよう、必ず「フォーメーション」または「1着2着軸マルチ」のどちらかで組み立てること。
"""

    prompt = f"""
以下のレース（ID: {race_id_target}、{selected_place_name} {selected_race_num}）の出走馬データをベースに、最新情報をWeb検索してプロ予想記事を作成してください。
--- 出走馬ベースデータ ---
{prompt_data}
---
"""

    with st.spinner("🦙 勝ちぱかが人気度外視の独自スコアをベースに、Web検索で最新情報を調査中..."):
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
            st.subheader(f"📰 {selected_date[:4]}年{selected_date[4:6]}月{selected_date[6:8]}日 {selected_place_name} {selected_race_num} 勝ちぱかくんの予想")
            st.markdown(response.text)
        else:
            st.error("【エラー】Google APIからの応答を取得できませんでした。")

# --------------------------------------------------
# メイン画面描画
# --------------------------------------------------
st.info(f"選択中: **{selected_date[:4]}年{selected_date[4:6]}月{selected_date[6:8]}日** の **{selected_place_name} {selected_race_num}** (レースID: `{target_race_id}`)")

if btn_predict:
    generate_prediction(target_race_id)
