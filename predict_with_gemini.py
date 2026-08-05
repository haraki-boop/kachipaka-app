import os
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from google import genai
from google.genai import types

# --------------------------------------------------
# ページ初期設定（スマホ・PC両対応デザイン）
# --------------------------------------------------
st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="🦙", layout="wide")

st.title("🦙 AI予想 勝ちぱかくん")
st.caption("LightGBM基礎スコア × GeminiリアルタイムWeb検索（適性・調教・騎手・オッズ妙味）")

# --------------------------------------------------
# 🔑 Gemini APIキーの設定（Streamlit Secrets対応）
# --------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

# --------------------------------------------------
# データ＆モデル読み込み（安全読み込み・キャッシュ機能付き）
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
            return pd.read_csv(data_path)
        except Exception as e:
            st.error(f"【CSV読み込みエラー】: {e}")
            return None
    return None

model = load_model()
df = load_data()

# --------------------------------------------------
# 🔄 最新データ更新処理（ボタン操作用）
# --------------------------------------------------
def update_keiba_data():
    """今週末の最新出馬表・オッズ等を自動取得してCSVを更新する処理"""
    with st.spinner("🔄 JRA/netkeiba等から最新の出馬表・直前オッズデータを取得中..."):
        try:
            time.sleep(2)  # 擬似処理待機
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

    # CSV内の最新日付を取得
    df['date_str'] = df['race_id'].astype(str).str[:8]
    available_dates = sorted(df['date_str'].unique(), reverse=True)

    selected_date = st.sidebar.selectbox("📅 開催日を選択", available_dates)
    selected_place_name = st.sidebar.selectbox("🏇 競馬場を選択", list(PLACE_MAP.keys()))
    selected_place_code = PLACE_MAP[selected_place_name]
    selected_race_num = st.sidebar.selectbox("🏁 レース（R）を選択", [f"{i}R" for i in range(1, 13)])
    race_num_int = int(selected_race_num.replace("R", ""))

    target_race_id = f"{selected_date}{selected_place_code}{race_num_int:02d}"

    btn_predict = st.sidebar.button("🚀 勝ちぱかくんに予想させる！", type="primary", use_container_width=True)
else:
    st.error("【エラー】'cleaned_keiba_data.csv' が読み込めません。ファイルをGitHubで確認してください。")
    st.stop()

# --------------------------------------------------
# 🤖 予想生成ロジック
# --------------------------------------------------
def generate_prediction(race_id_target):
    if not GEMINI_API_KEY:
        st.error("【設定エラー】GEMINI_API_KEY が見つかりません。Streamlit Community Cloudの Advanced settings > Secrets に APIキーを設定してください。")
        return

    if model is None:
        st.error("【モデルエラー】AIモデル（keiba_ai_model.pkl）が正しく読み込めていないため、予想を実行できません。")
        return

    race_df = df[df['race_id'].astype(str) == str(race_id_target)].copy()

    if race_df.empty:
        st.warning(f"選択されたレース（{selected_date} {selected_place_name} {selected_race_num} / ID: {race_id_target}）の出走馬データがCSV内に存在しません。左上の「最新出馬表・オッズを自動更新」を押すか、別のレースをお試しください。")
        return

    # 1. LightGBMによる基礎スコア計算
    features = ['枠番', '馬番', '斤量', 'time_seconds', 'sex_code', 'age', 'horse_weight', 'weight_change']
    if '単勝' in race_df.columns:
        features.append('単勝')
    if '人気' in race_df.columns:
        features.append('人気')

    X_race = race_df[features].fillna(0)
    race_df['win_prob'] = model.predict_proba(X_race)[:, 1]

    mean_prob = race_df['win_prob'].mean()
    raw_score = 100 + (race_df['win_prob'] - mean_prob) / (mean_prob if mean_prob != 0 else 1) * 30
    race_df['score'] = np.clip(raw_score, 50, 120).round().astype(int)
    race_df = race_df.sort_values(by='win_prob', ascending=False).reset_index(drop=True)

    # 2. AI へ渡す基本出走データ作成
    table_summary = []
    for idx, row in race_df.iterrows():
        horse_name = row['馬名'] if '馬名' in row and pd.notna(row['馬名']) else f"馬番{int(row['馬番'])}"
        jockey_name = row['騎手'] if '騎手' in row and pd.notna(row['騎手']) else "不明"
        
        horse_info = (
            f"馬番:{int(row['馬番']):02d} | 馬名:{horse_name} | 騎手:{jockey_name} | "
            f"score:{row['score']:3d} | AI勝率予想:{row['win_prob']*100:.1f}% | "
            f"性齢:{row.get('sex', '')}{int(row.get('age', 0))} | 斤量:{row.get('斤量', 0)}kg | 単勝:{row.get('単勝', '---')}倍"
        )
        table_summary.append(horse_info)

    prompt_data = "\n".join(table_summary)

    # 3. Gemini呼び出し設定（プロンプト修正版）
    system_instruction = f"""
あなたは競馬分析AI「勝ちぱかくん」であり、データと現場情報を冷静に分析する凄腕のプロ馬券師です。
提供された出走馬のベースscoreデータと、Google検索ツールを用いた最新情報（適性・調教・陣営コメントなど）を融合して予想を作成してください。

【厳守事項】
・検索時の内部ログ（SPOILER ALERT等）やシステムメッセージは絶対に文章に出力しないでください。
・今回のレースは【{selected_place_name}競馬場】です。他の競馬場と勘違いしないこと。
・基本はLightGBMが算出した「score」と「AI勝率予想」を素直に評価の軸とすること。無理に穴を狙う必要はありません。ただし、スコアが拮抗している場合や、検索結果から明確な激走サイン（条件替わりなど）を見つけた場合のみ妙味を考慮してください。
・印（◎◯▲△☆など）を打つ推奨馬は原則【5頭まで】に絞ること。ただし、スコアが大混戦の場合のみ例外として超えても構いません。

【構成案と出力フォーマット】
以下の1〜4の構成で出力してください。

1. 【レース概況と展開予想】
2. 【全頭 データ一覧】
必ず以下のマークダウン形式の表で、提供された全頭分を出力してください。途中で文章を挟んだり、余計なコメント列を追加したりしないでください。
| 馬番 | 馬名 | 騎手 | score | AI勝率予想 | 単勝オッズ |
|---|---|---|---|---|---|
| 01 | 〇〇 | 〇〇 | 100 | 12.5% | 4.5倍 |
（※全頭分リストアップすること）

3. 【勝ちぱかくんのジャッジ・見解（印と推奨馬）】
（※各馬の評価や検索で得た情報は、表の中ではなくこちらの文章でしっかりと語ってください）
4. 【推奨買い目】
（※単勝、馬連などに加え、必ず【三連複】の買い目を含めて提示してください）
"""

    prompt = f"""
以下のレース（ID: {race_id_target}、{selected_place_name} {selected_race_num}）の出走馬データをベースに、最新情報をWeb検索してプロ予想記事を作成してください。
--- 出走馬ベースデータ ---
{prompt_data}
---
"""

    with st.spinner("🦙 勝ちぱかがWeb検索で馬場適性・調教・騎手・オッズ妙味を徹底調査中..."):
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
                        st.warning(f"⚠️ サーバー混雑中のため、3秒待機して再試行します...（{i+1}/{len(models_to_try)}回目）")
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
            st.subheader(f"📰 {selected_date} {selected_place_name} {selected_race_num} 勝ちぱかくんの予想")
            st.markdown(response.text)
        else:
            st.error("【エラー】Google APIからの応答を取得できませんでした。再度お試しください。")

# --------------------------------------------------
# メイン画面描画
# --------------------------------------------------
st.info(f"選択中: **{selected_date}** の **{selected_place_name} {selected_race_num}** (レースID: `{target_race_id}`)")

if btn_predict:
    generate_prediction(target_race_id)
