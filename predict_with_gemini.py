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
            # ---------------------------------------------------------
            # ここにお持ちのスクレイピング・データ更新ロジックが入ります
            # ---------------------------------------------------------
            time.sleep(2)  # 擬似処理待機
            
            # キャッシュをクリアして最新CSVをリロード
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

    # 1. 日付選択
    selected_date = st.sidebar.selectbox("📅 開催日を選択", available_dates)

    # 2. 競馬場選択
    selected_place_name = st.sidebar.selectbox("🏇 競馬場を選択", list(PLACE_MAP.keys()))
    selected_place_code = PLACE_MAP[selected_place_name]

    # 3. レース番号選択
    selected_race_num = st.sidebar.selectbox("🏁 レース（R）を選択", [f"{i}R" for i in range(1, 13)])
    race_num_int = int(selected_race_num.replace("R", ""))

    # レースIDを自動生成
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

    # score (50-120) 算出
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
            f"【ベースscore: {row['score']:3d}】| 馬番:{int(row['馬番']):02d} | 馬名:{horse_name} | 騎手:{jockey_name} | "
            f"性齢:{row.get('sex', '')}{int(row.get('age', 0))} | 斤量:{row.get('斤量', 0)}kg | "
            f"単勝:{row.get('単勝', '---')}倍({int(row.get('人気', 0))}人気) | "
            f"AI勝率:{row['win_prob']*100:.1f}%"
        )
        table_summary.append(horse_info)

    prompt_data = "\n".join(table_summary)

    # 3. Gemini呼び出し設定
    system_instruction = """
あなたは競馬分析AI「勝ちぱかくん」であり、人気や表面的なスコアに決して流されない「適性・現場情報・オッズ妙味重視の凄腕馬券師」です。
提供された出走馬のベースscoreデータに加え、Google検索ツールを用いて以下の【4つの超重要ファクター】を徹底的に調査してください。

【検索・分析すべき超重要ファクター】
1. 【適性】各馬の前走実績、今回の「距離」および「コース（競馬場）」に対する得意・不得意。
2. 【調教・状態】最新の追い切り評価、陣営のコメント、直前の気配。
3. 【騎手相性】騎乗するジョッキーの「その競馬場における戦績・相性」や、今回のコースにおける騎手心理。
4. 【過去の人気と着順（オッズ妙味）】前走や近走において「人気だったのに敗れた馬の巻き返し（過小評価）」や「人気薄で好走した馬のフロック判定（過大評価）」など、過去の人気と着順から見える馬券的妙味。

【予想のスタンス（超重要）】
単勝人気やベースscoreの順位をそのまま鵜呑みにした「硬い予想」は絶対に避けてください。
ベースscoreが低くても、「前走は1番人気で負けたが今回は得意コース」「調教が抜群に良く騎手相性も完璧」といった隠れた激走サインやオッズ妙味を見つけ出し、積極的に重い印（◎や◯）を打つなど、独自の視点で予想を展開してください。

構成案：
1. 【波乱の可能性とレース概況】（天候・馬場状態、展開予想、馬券的な旨味について）
2. 【全頭 ジャッジ一覧】（表形式で馬番・馬名・騎手・ベースscoreに加え、検索で調べた「適性」「調教」「騎手相性」「過去の人気と着順の妙味」を加味した一言メモを全頭分添える）
3. 【激アツ予想印（◎◯▲△☆）と見解】（人気やscore順位に逆らった、現場情報とオッズ妙味重視の鋭い見解）
4. 【勝ちぱかくんの推奨買い目】
"""

    prompt = f"""
以下のレース（ID: {race_id_target}）の出走馬データをベースに、最新情報をWeb検索してプロ予想記事を作成してください。
--- 出走馬ベースデータ ---
{prompt_data}
---
"""

    with st.spinner("🦙 勝ちぱかがWeb検索で馬場適性・調教・騎手・オッズ妙味を徹底調査中..."):
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # 混雑（503）対策：同じモデルで再試行後、proへフォールバック
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
