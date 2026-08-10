import os
import re
import time
import requests
import pandas as pd
import numpy as np
import joblib
import unicodedata
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime
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

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s\u3000]+', '', s)

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
        for enc in ['utf-8-sig', 'cp932', 'utf-8', 'shift_jis']:
            try:
                df = pd.read_csv(ML_TARGET_CSV, low_memory=False, dtype={'race_id': str}, encoding=enc)
                if 'date' in df.columns:
                    df = df.sort_values(by='date')
                if '馬名' in df.columns:
                    df['馬名_clean'] = df['馬名'].astype(str).apply(clean_horse_name)
                return df
            except Exception:
                continue
    return pd.DataFrame()

def load_future_data():
    if os.path.exists(FUTURE_CSV) and os.path.getsize(FUTURE_CSV) > 0:
        for enc in ['utf-8-sig', 'cp932', 'utf-8']:
            try:
                df = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding=enc)
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
            except Exception:
                continue
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
# 2. サイドバー UI & 過去検証モード
# ==========================================
st.sidebar.header("🔄 画面の更新")
if st.sidebar.button("🔄 最新の情報にリロード", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🔬 AI精度検証モード")
test_mode = st.sidebar.checkbox("過去データ(直近)でAIをテストする")

def get_display_data():
    if test_mode and not df_past.empty:
        dates = df_past['date'].dropna().unique()
        if len(dates) > 0:
            latest_date = sorted(dates)[-1]
            test_df = df_past[df_past['date'] == latest_date].copy()
            
            PLACE_MAP_REV = {
                "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
                "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
            }
            if 'race_id' in test_df.columns:
                test_df['place_code'] = test_df['race_id'].astype(str).str[4:6]
                test_df['place_name'] = test_df['place_code'].map(PLACE_MAP_REV).fillna("不明")
                test_df['r_num'] = test_df['race_id'].astype(str).str[10:12].astype(int)
            test_df['day_label'] = f"【過去検証】{latest_date}"
            if 'race_name' not in test_df.columns:
                test_df['race_name'] = "過去レース検証"
            if '単勝' not in test_df.columns:
                test_df['単勝'] = test_df.get('オッズ', 0)
            return test_df
    return df_future

df_display = get_display_data()

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
                except: pass
                time.sleep(1)
            
            if updated: df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig', errors='replace')
        st.cache_data.clear()
        st.success("✅ 実戦結果を最新化しました！")
        time.sleep(1.5)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🗑️ 履歴の完全リセット")
if st.sidebar.button("💥 予想履歴を消去", type="primary", use_container_width=True):
    try:
        if os.path.exists(HISTORY_CSV): os.remove(HISTORY_CSV)
        st.cache_data.clear()
        st.sidebar.success("✅ 履歴データを消去しました！")
        time.sleep(1.5)
        st.rerun()
    except: pass

# ==========================================
# 3. AIスコア算出 (第1の脳：純粋AIスコア / 第2の脳：期待値)
# ==========================================
def calculate_race_scores(race_id_target, target_df):
    if target_df.empty or model_data is None: return None
    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty: return None

    features = model_data.get('features', [])
    model = model_data.get('model')
    
    for f in features:
        if f not in race_df.columns:
            race_df[f] = np.nan
            
    X = race_df[features].copy()
    
    if 'sex_code' in X.columns:
        X['sex_code'] = X['sex_code'].replace({'牡': 0, '牝': 1, 'セ': 2})
        
    if 'surface' in X.columns:
        X['surface'] = X['surface'].replace({'芝': 0, 'ダート': 1, '障害': 2})
        
    if 'condition' in X.columns:
        X['condition'] = X['condition'].replace({'良': 0, '稍重': 1, '稍': 1, '重': 2, '不良': 3})

    if 'weather' in X.columns:
        X['weather'] = X['weather'].replace({'晴': 0, '曇': 1, '雨': 2, '雪': 3, '小雨': 2, '小雪': 3})
        
    X = X.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    try:
        if hasattr(model, "predict_proba"):
            # 純粋なAIの予測確率をそのまま使用する（余計な補正はしない）
            prob = model.predict_proba(X)[:, 1]
        else:
            prob = model.predict(X)
    except Exception:
        return None

    s = prob.sum()
    race_df['win_prob'] = prob / s if s > 0 else 1.0 / len(race_df)
    
    min_prob = race_df['win_prob'].min()
    max_prob = race_df['win_prob'].max()
    
    def scale_score(p):
        if max_prob == min_prob: return 100
        return 50 + ((p - min_prob) / (max_prob - min_prob)) * 100
        
    race_df['score_brain1'] = race_df['win_prob'].apply(scale_score).round().astype(int)

    def calc_ev(r):
        raw_odds = r.get('単勝') if pd.notna(r.get('単勝')) else r.get('オッズ', 0)
        odds = pd.to_numeric(raw_odds, errors='coerce')
        if pd.isna(odds) or odds <= 0: return 0.0
        return r['win_prob'] * odds

    race_df['ev_brain2'] = race_df.apply(calc_ev, axis=1)
    
    if '人気' not in race_df.columns or race_df['人気'].isna().all():
        raw_odds_col = race_df['単勝'] if '単勝' in race_df.columns else race_df.get('オッズ', pd.Series())
        numeric_odds = pd.to_numeric(raw_odds_col, errors='coerce')
        race_df['人気'] = numeric_odds.rank(method='min').fillna(999)

    race_df['人気_sort'] = pd.to_numeric(race_df['人気'], errors='coerce').fillna(999)
    return race_df.sort_values(by=['score_brain1', '人気_sort'], ascending=[False, True]).reset_index(drop=True)

def get_mark(idx, ev, odds, win_prob):
    if idx == 0: return "◎ 本命"
    if idx == 1: return "◯ 対抗"
    if idx == 2: return "▲ 単穴"
    if idx in [3, 4]: return "△ 連下"
    
    # 勝率のハードルを元に戻し、シンプルに期待値やオッズで穴馬を拾う
    if win_prob <= 0.07: 
        return "消し"
        
    if ev >= 0.7 or odds >= 15.0: return "☆ 穴馬"
    return "消し"

# ==========================================
# 4. マーカー判定
# ==========================================
def get_all_markers():
    markers = {}
    if df_display.empty: return markers
    for rid in df_display['race_id'].unique():
        sdf = calculate_race_scores(rid, df_display)
        if sdf is not None and len(sdf) >= 5:
            rname = str(sdf['race_name'].iloc[0]) if 'race_name' in sdf.columns else ""
            if "新馬" in rname:
                markers[rid] = "【🐣新馬】"
                continue
            
            top3 = sdf.head(3)
            has_value = False
            for _, row in top3.iterrows():
                pop = pd.to_numeric(row.get('人気', 0), errors='coerce')
                if pd.notna(pop) and pop >= 4:
                    has_value = True
                    break
            
            if has_value:
                markers[rid] = "【🔥妙味あり】"
            else:
                markers[rid] = "【普】"
    return markers

markers = get_all_markers()

# ==========================================
# 5. メインUI
# ==========================================
tab_forecast, tab_dashboard = st.tabs(["🏇 レース予想", "📈 実戦成績"])

with tab_forecast:
    if df_display.empty:
        st.warning("⚠️ 出馬表データが存在しません。")
    else:
        date_options = sorted(df_display['day_label'].unique())
        selected_date = st.radio("開催日", date_options, horizontal=True, label_visibility="collapsed")
        day_df = df_display[df_display['day_label'] == selected_date]
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
                    btn_type = "primary" if "🔥" in mark else "secondary"
                    if col.button(label, key=f"btn_{rid}", use_container_width=True, type=btn_type):
                        st.session_state['selected_race_id'] = rid

    if st.session_state['selected_race_id'] and not df_display.empty:
        st.markdown("---")
        target_id = str(st.session_state['selected_race_id'])
        
        target_rows = df_display[df_display['race_id'].astype(str) == target_id]
        if target_rows.empty:
            st.session_state['selected_race_id'] = None
            st.rerun()
            
        target_race_info = target_rows.iloc[0]
        
        rname = target_race_info.get('race_name', "")
        is_newcomer = "新馬" in str(rname)
        race_display_name = f"{target_race_info['place_name']} {target_race_info['r_num']}R 【{rname}】"
        st.subheader(f"🚀 {race_display_name}")
        
        scored_df = calculate_race_scores(target_id, df_display)
        
        if scored_df is not None:
            st.markdown("### 📊 1. 出走馬 期待値＆データ一覧")
            disp_df = scored_df.copy().sort_values(by=['score_brain1', 'win_prob'], ascending=[False, False]).reset_index(drop=True)
            
            marks_list = []
            for i, r in disp_df.iterrows():
                ev_val = float(r.get('ev_brain2', 0))
                raw_o = r.get('単勝') if pd.notna(r.get('単勝')) else r.get('オッズ', 0)
                odds_val = float(pd.to_numeric(raw_o, errors='coerce')) if pd.notna(raw_o) else 0.0
                win_prob_val = float(r.get('win_prob', 0))
                marks_list.append(get_mark(i, ev_val, odds_val, win_prob_val))
                
            disp_df['馬番'] = disp_df['馬番'].apply(lambda x: f"{int(x):02d}")
            disp_df['AIスコア'] = disp_df['score_brain1'].astype(int)
            disp_df['勝率'] = disp_df['win_prob'].apply(lambda x: f"{x*100:.1f}%")
            
            def format_odds(row):
                raw_o = row.get('単勝') if pd.notna(row.get('単勝')) else row.get('オッズ', 0)
                try:
                    v = float(raw_o)
                    return f"{v:.1f}倍" if v > 0 else "-"
                except: return "-"
                
            disp_df['予想オッズ'] = disp_df.apply(format_odds, axis=1)
            
            def format_ev(val):
                try:
                    v = float(val)
                    return f"{v:.2f}" if v > 0 else "-"
                except: return "-"
                
            disp_df['期待値'] = disp_df['ev_brain2'].apply(format_ev)
            disp_df['騎手'] = disp_df.get('騎手', '-')
            disp_df['評価'] = marks_list
            
            show_cols = ['馬番', '馬名', '騎手', 'AIスコア', '勝率', '予想オッズ', '期待値', '評価']
            if is_newcomer:
                st.info("🐣 新馬戦のため、過去データが存在せずベースAIスコアは参考値です。Geminiの自力予想に委ねます。")
                show_cols = ['馬番', '馬名', '騎手', '予想オッズ']
            
            st.dataframe(disp_df[show_cols], use_container_width=True, hide_index=True)

        if st.button("🧠 Geminiで最終適正化＆買い目生成", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()
            if scored_df is None or len(scored_df) < 6:
                st.error("出走頭数が少ない、またはデータが不足しているため予想をスキップします。")
                st.stop()

            table_summary = []
            prompt_df = scored_df.copy().sort_values(by=['score_brain1', 'win_prob'], ascending=[False, False]).reset_index(drop=True)
            for idx, row in prompt_df.iterrows():
                ev_val = float(row.get('ev_brain2', 0))
                raw_o = row.get('単勝') if pd.notna(row.get('単勝')) else row.get('オッズ', 0)
                odds_val = float(pd.to_numeric(raw_o, errors='coerce')) if pd.notna(raw_o) else 0.0
                win_prob_val = float(row.get('win_prob', 0))
                mark = get_mark(idx, ev_val, odds_val, win_prob_val)
                
                table_summary.append(
                    f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | "
                    f"オッズ:{odds_val}倍 ({row.get('人気', 999)}人気) | "
                    f"純粋スコア:{row.get('score_brain1', 0)} | 期待値:{ev_val:.2f} | システム評価:{mark}"
                )

            system_instruction = f"""
あなたはプロの競馬分析AI「勝ちぱかくん」の最終意思決定者（Gemini脳）です。

【⚠️検索ツールの絶対ルール】
全出走馬の名前を1頭ずつ個別に検索して調べる行為はフリーズの原因になるため絶対に禁止します。
必ず日付とレース名を含めて、1〜2回の一括検索のみで情報を取得してください。

【Geminiが分析すべき4大チェック項目】
1. 今日の天候と馬場状態
2. 当日のトラックバイアス（イン有利か外差し有利か）
3. 直前の調教評価と陣営コメント
4. 展開予想（ペース予想）

【あなたのミッションと絶対ルール】
- 既にシステム側で付与された「システム評価（◎◯▲△☆）」をベースに、天候や調教などを加味してプロとして最終的な印と買い目を決断してください。
- Markdownの表はシステム側で描画済みのため、あなたは**絶対に表を出力しないでください**。解説文と買い目のみを出力すること。

【出力フォーマット】
---
### 🌪️ レース展開とリアルタイム情報の統合
* （プロの分析）

### 💥 勝ちぱかくんの最終ジャッジ（印と根拠）
* **◎（本命）:** 〇〇番（馬名） - （抜擢理由）
* **◯（対抗）:** 〇〇番（馬名） - （見解）
* **▲（単穴）:** 〇〇番（馬名） - （見解）
* **△（連下）:** 〇〇番、〇〇番
* **☆（穴馬）:** 〇〇番 - （期待値馬など）

### 💡 戦略的・推奨全買い目
必ず以下の全券種の買い目を具体的に提示してください。
* **単勝:** ◎
* **馬連:** ◎ - ◯▲△☆ (流し)
* **ワイド:** ◎ - ◯▲△☆ (流し)
* **三連複:** ◎◯▲△☆の印から選んだ4〜5頭のボックス
* **三連単:** ◎ 1着固定 - ◯▲△☆ (フォーメーション等)
---
"""
            prompt = f"対象レース: {selected_date} {race_display_name}\n\n"
            if is_newcomer:
                prompt += "【⚠️重要指示】このレースは「新馬戦」のため過去データが存在せずスコアは無効です。スコアは無視し、Web検索で『血統適性』『追い切り（調教）タイム』を日付指定で一括調査し、あなた自身の推理で予想を組み立ててください。\n\n"
            else:
                prompt += "【指示】システムが算出した「システム評価」とデータに基づき、特に期待値の高い「中穴馬」を拾い上げるプロンプト補助に従って、最終的な印を決定してください。\n\n"
                
            prompt += f"出走馬データ（第1・第2の脳 出力結果）:\n{chr(10).join(table_summary)}"

            with st.spinner("AIが本日付の一括検索で調教情報を取得し、全買い目を生成中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                res_text = ""
                
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=0.3,
                                tools=[{"googleSearch": {}}]
                            )
                        )
                        res_text = response.text if response.text else (response.candidates[0].content.parts[0].text if response.candidates else "")
                        if res_text:
                            break
                    except Exception as e:
                        if "503" in str(e) or "UNAVAILABLE" in str(e):
                            if attempt < 2:
                                time.sleep(3)
                                continue
                        st.error(f"【APIエラー】: {e}")
                        break
                    
                if res_text:
                    st.markdown(res_text)
                    
                    honmei_match = re.search(r'◎.*?[）:]\s*(\d+)番', res_text)
                    h_umaban = int(honmei_match.group(1)) if honmei_match else int(scored_df.iloc[0]['馬番'])
                    all_nums = re.findall(r'(\d+)番', res_text)
                    partners_str = ",".join(list(dict.fromkeys([n for n in all_nums if int(n) != h_umaban]))[:5])
                    
                    if df_history.empty or str(target_id) not in df_history['race_id'].astype(str).values:
                        new_record = pd.DataFrame([{'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'race_id': str(target_id), 'race_name': race_display_name, 'honmei_umaban': h_umaban, 'partners': partners_str, 'honmei_name': "履歴参照", 'result_pay': "", 'pay_tansho': 0, 'pay_umaren': 0, 'pay_wide': 0, 'pay_sanrenpuku': 0, 'pay_sanrentan': 0}])
                        df_history = pd.concat([df_history, new_record], ignore_index=True)
                        df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
                    st.success("📝 実戦履歴に記録しました！")
                else:
                    st.warning("⚠️ 回答を取得できませんでした。")

# ==========================================
# 6. ダッシュボード
# ==========================================
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
                        inv_sanrenpuku += 1000 
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
                    make_ticket_card(ticket_cols[0], "単勝", len(finished_df[finished_df['pay_tansho'].astype(float) > 0]), finished_df['pay_tansho'].astype(float).sum(), inv_tansho)
                    make_ticket_card(ticket_cols[1], "馬連", len(finished_df[finished_df['pay_umaren'].astype(float) > 0]), finished_df['pay_umaren'].astype(float).sum(), inv_umaren)
                    make_ticket_card(ticket_cols[2], "ワイド", len(finished_df[finished_df['pay_wide'].astype(float) > 0]), finished_df['pay_wide'].astype(float).sum(), inv_wide)
                    make_ticket_card(ticket_cols[3], "三連複", len(finished_df[finished_df['pay_sanrenpuku'].astype(float) > 0]), finished_df['pay_sanrenpuku'].astype(float).sum(), inv_sanrenpuku)
                    make_ticket_card(ticket_cols[4], "三連単", len(finished_df[finished_df['pay_sanrentan'].astype(float) > 0]), finished_df['pay_sanrentan'].astype(float).sum(), inv_sanrentan)

                st.markdown("<br>", unsafe_allow_html=True)
                st.caption(f"※ 投資金額・回収率は「三連単◎1着固定流し」「三連複4〜5頭ボックス」などを想定した最適化実点数で計算されています。")
                
            st.dataframe(raw_df[['date', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay']].sort_values(by='date', ascending=False), use_container_width=True)

        with tab_total:
            render_dashboard_for_df(df_history, "総合")
        with tab_month:
            months = sorted(df_history['year_month'].dropna().unique(), reverse=True)
            if months:
                selected_month = st.selectbox("表示する月を選択", months)
                render_dashboard_for_df(df_history[df_history['year_month'] == selected_month], f"{selected_month} の")
        with tab_day:
            days = sorted(df_history['just_date'].dropna().unique(), reverse=True)
            if days:
                selected_day = st.selectbox("表示する日付を選択", days)
                render_dashboard_for_df(df_history[df_history['just_date'] == selected_day], f"{selected_day} の")