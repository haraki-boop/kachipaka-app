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
# 2. サイドバー UI (配当取得・リセット)
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
# 3. AIスコア計算
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
        
        horse_stats = df_past.groupby('馬名').agg(agg_dict).reset_index()
        new_cols = ['馬名', 'total_wins', 'total_runs']
        if 'my_time_idx' in df_past.columns: new_cols.append('my_time_idx')
        if 'my_last3f_idx' in df_past.columns: new_cols.append('my_last3f_idx')
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
    if 'jockey_win_power' in race_df.columns: race_df['jockey_win_power'] = race_df['jockey_win_power'].fillna(0.0)

    if 'sex_code' in race_df.columns and race_df['sex_code'].dtype == object:
        race_df['sex_code'] = race_df['sex_code'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
        
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

# ==========================================
# 4. マーカー（バッジ）の判定 (新馬対応＆厳格化)
# ==========================================
def get_all_markers():
    markers = {}
    if df_future.empty: return markers
    for rid in df_future['race_id'].unique():
        sdf = calculate_race_scores(rid, df_future)
        if sdf is not None and len(sdf) >= 5:
            rname = str(sdf['race_name'].iloc[0]) if 'race_name' in sdf.columns else ""
            
            # 【新馬戦回避ロジック】新馬戦は過去データがないため別枠扱い
            if "新馬" in rname:
                markers[rid] = "【🐣新馬・データなし】"
                continue

            sc = sdf['score'].tolist()
            top_diff = sc[0] - sc[1]
            top3_diff = sc[0] - sc[2]

            # 【超・厳格化ロジック】真の狙い目（オッズ異常）のみを抽出
            is_neraime = False
            if '人気' in sdf.columns and len(sdf) >= 2:
                pop1 = pd.to_numeric(sdf.iloc[0].get('人気', 999), errors='coerce')
                pop2 = pd.to_numeric(sdf.iloc[1].get('人気', 999), errors='coerce')
                
                # エラー値(999)や欠損を除外した上で判定
                if not pd.isna(pop1) and pop1 != 999 and pop1 >= 4:
                    is_neraime = True  # AI1位なのに4番人気以降（激ウマ本命）
                elif not pd.isna(pop2) and pop2 != 999 and pop2 >= 6:
                    is_neraime = True  # AI2位なのに6番人気以降（特大穴対抗）

            if sc[0] >= 110 and top3_diff >= 30: race_type = "【堅】"
            elif sc[0] < 95 or top_diff <= 7: race_type = "【波乱】"
            else: race_type = "【普】"

            if is_neraime: race_type = "【🔥狙い目】" + race_type

            mark = "★" if sc[0] >= 115 and top_diff >= 15 else ("◎" if top3_diff >= 20 else "")
            markers[rid] = f"{race_type} {mark}".strip()
    return markers
markers = get_all_markers()

# ==========================================
# 5. メインUI (タブ構成)
# ==========================================
tab_forecast, tab_dashboard = st.tabs(["🏇 レース予想", "📈 実戦成績"])

with tab_forecast:
    if df_future.empty:
        st.warning("⚠️ 出馬表データが存在しません。BOTを実行してデータを読み込ませてください。")
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
                        
                    btn_type = "primary" if ("🔥" in mark or "★" in mark) else "secondary"
                    if col.button(label, key=f"btn_{rid}", use_container_width=True, type=btn_type):
                        st.session_state['selected_race_id'] = rid

    # --- 予想実行セクション ---
    if st.session_state['selected_race_id'] and not df_future.empty:
        st.markdown("---")
        target_id = st.session_state['selected_race_id']
        target_race_info = df_future[df_future['race_id'] == target_id].iloc[0]
        rname = target_race_info.get('race_name', "")
        is_newcomer = "新馬" in str(rname)
        
        race_display_name = f"{target_race_info['place_name']} {target_race_info['r_num']}R 【{rname}】"
        st.subheader(f"🚀 {race_display_name}")
        
        scored_df = calculate_race_scores(target_id, df_future)
        
        if scored_df is not None and len(scored_df) > 0:
            # 💡 【完全分業 UI】Python側で綺麗なデータフレームを描画する
            st.markdown("##### 📊 出走馬 データ＆ベースAI評価一覧")
            disp_df = scored_df.copy()
            disp_df['馬番'] = disp_df['馬番'].apply(lambda x: f"{int(x):02d}")
            disp_df['単勝オッズ'] = disp_df['単勝'].apply(lambda x: f"{x}倍" if pd.notnull(x) and x>0 else "-")
            disp_df['人気'] = disp_df['人気'].apply(lambda x: f"{int(x)}人気" if pd.notnull(x) and x!=999 else "-")
            disp_df['AI勝率'] = disp_df['win_prob'].apply(lambda x: f"{x*100:.1f}%")
            disp_df['期待値'] = disp_df.apply(lambda r: f"{(r['win_prob'] * r['単勝']):.2f}" if pd.notnull(r['単勝']) and r['単勝']>0 else "-", axis=1)
            disp_df['騎手(勝率)'] = disp_df.apply(lambda r: f"{r.get('騎手','')} ({r.get('jockey_win_power',0)*100:.1f}%)", axis=1)
            
            show_cols = ['馬番', '馬名', '騎手(勝率)', '単勝オッズ', '人気', 'AI勝率', '期待値', 'score']
            if is_newcomer:
                st.info("🐣 新馬戦のため、過去データに基づくAIスコアは参考値（または無効）です。")
                show_cols = ['馬番', '馬名', '騎手(勝率)', '単勝オッズ', '人気'] # スコア等を除外表示
                
            st.dataframe(disp_df[show_cols], use_container_width=True, hide_index=True)
        
        if st.button("🧠 勝ちぱかくんに最終予想させる！ (検索・分析を実行)", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()
                
            if scored_df is None or len(scored_df) < 6:
                st.error("出走頭数が少ない、またはデータが不足しているため予想をスキップします。")
                st.stop()

            # データ送信用テキスト化
            table_summary = []
            for idx, row in scored_df.iterrows():
                table_summary.append(
                    f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | "
                    f"騎手:{row.get('騎手', '不明')} | オッズ:{row.get('単勝', 0)}倍 ({row.get('人気', 999)}人気) | "
                    f"ベースAIスコア:{row.get('score', 0)}"
                )

            # --- プロンプト構築（裁量権付与） ---
            system_instruction = """
あなたはプロの競馬分析AI「勝ちぱかくん」です。以下の絶対ルールに従い、プロフェッショナルな解説と最終的な買い目を構築してください。

【絶対遵守事項（ハイブリッド予想ルール）】
1. あなたには「Pythonが算出したベースAIスコア」が提供されますが、これらは「一次審査」に過ぎません。
2. あなたは【Web検索ツール】を使用して、直前の調教タイム、血統、陣営コメント、馬場状態などを必ず検索してください。
3. 【裁量権の行使】: 検索で得た最新情報や、オッズから算出した「期待値（妙味）」を考慮し、ベースAIスコアの順位にとらわれず、あなたがプロとして「最終的な印（◎◯▲△☆）」を決定・変更して構いません。
4. 【期待値による足切り】: オッズが低すぎて期待値がマイナスだと判断した場合は、無理に買い目を提示せず「このレースは妙味がないため『見（ケン）』を推奨」と宣言してください。
5. 表（Markdownのテーブル）はシステム側で描画済みのため、あなたは**絶対に表を出力しないでください**。文章と買い目のみを出力すること。

【出力フォーマット】
---
### 🌪️ レース波乱度と展開分析
* （検索結果や血統・馬場を元にした展開予想）

### 💥 勝ちぱかくんの最終ジャッジ（印と根拠）
* **◎（本命）:** 〇〇番（馬名） - （AIスコアや検索結果を加味した抜擢理由）
* **◯（対抗）:** 〇〇番（馬名） - （見解）
* **▲（単穴）:** 〇〇番（馬名） - （見解）
* **△（連下）:** 〇〇番、〇〇番
* **☆（穴馬）:** 〇〇番 - （オッズ妙味や展開での激走期待）

### 💡 戦略的・推奨買い目
* （オッズや自信度に応じた、最も儲かる券種と資金配分を自由に提案してください。ガミる買い目は避けること）
---
"""
            prompt = f"対象レース: {race_display_name}\n\n"
            if is_newcomer:
                prompt += "【⚠️重要指示】このレースは「新馬戦」のため過去データが存在せず、ベースAIスコアは無効（全て同点等）です。AIスコアは完全に無視し、Web検索を用いて『血統』『追い切り（調教）タイム』『厩舎コメント』を調査し、あなた自身の推理で予想を組み立ててください。\n\n"
            else:
                prompt += "【指示】提供されたベースAIスコアを基準にしつつ、オッズの妙味や検索による最新情報（枠の有利不利、血統適性など）を加味して、印を最終決定してください。\n\n"
                
            prompt += f"出走馬データ（ベース情報）:\n{chr(10).join(table_summary)}"

            with st.spinner("AIが最新情報を検索し、最適戦略と最終買い目を構築中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                try:
                    response = client.models.generate_content(
                        model='gemini-3.6-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.3,
                            tools=[{"googleSearch": {}}]
                        )
                    )
                    res_text = response.text if response.text else (response.candidates[0].content.parts[0].text if response.candidates else "")
                    
                    if res_text:
                        st.markdown(res_text)
                        # --- 履歴保存処理 ---
                        # Geminiの出力結果から、正規表現で◎の馬番と相手を抽出して保存するロジック
                        honmei_match = re.search(r'◎.*?[）:]\s*(\d+)番', res_text)
                        h_umaban = int(honmei_match.group(1)) if honmei_match else int(scored_df.iloc[0]['馬番'])
                        
                        all_nums = re.findall(r'(\d+)番', res_text)
                        partners_str = ",".join(list(dict.fromkeys([n for n in all_nums if int(n) != h_umaban]))[:5])
                        
                        if df_history.empty or str(target_id) not in df_history['race_id'].astype(str).values:
                            new_record = pd.DataFrame([{'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'race_id': str(target_id), 'race_name': race_display_name, 'honmei_umaban': h_umaban, 'partners': partners_str, 'honmei_name': "履歴参照", 'result_pay': "", 'pay_tansho': 0, 'pay_umaren': 0, 'pay_wide': 0, 'pay_sanrenpuku': 0, 'pay_sanrentan': 0}])
                            df_history = pd.concat([df_history, new_record], ignore_index=True)
                            df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
                        st.success("📝 実戦履歴に記録しました！(印の最終結果を反映)")
                    else:
                        st.warning("⚠️ 回答を取得できませんでした。")
                except Exception as e:
                    st.error(f"【APIエラー】: {e}")

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