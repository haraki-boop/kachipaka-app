import os
import re
import time
import random
import requests
import pandas as pd
import numpy as np
import joblib
import streamlit as st
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
ENHANCED_DB_CSV = "enhanced_keiba_data.csv"

# ==========================================
# 1. データとAIモデルの読み込み
# ==========================================
@st.cache_resource
def load_model():
    if os.path.exists("keiba_ai_model.pkl"):
        return joblib.load("keiba_ai_model.pkl")
    return None

@st.cache_data
def load_past_data():
    target_csv = ENHANCED_DB_CSV if os.path.exists(ENHANCED_DB_CSV) else "cleaned_keiba_data.csv"
    if os.path.exists(target_csv):
        try:
            df = pd.read_csv(target_csv, low_memory=False, dtype={'race_id': str}, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(target_csv, low_memory=False, dtype={'race_id': str}, encoding='cp932')
            
        if 'time_seconds' not in df.columns and 'タイム' in df.columns:
            def ts(t):
                if pd.isna(t): return np.nan
                p = str(t).strip().split(':')
                if len(p) == 2: return float(p[0]) * 60 + float(p[1])
                try: return float(p[0])
                except Exception: return np.nan
            df['time_seconds'] = df['タイム'].apply(ts)
            
        if '上り' in df.columns:
            df['last_3f'] = pd.to_numeric(df['上り'], errors='coerce')
        elif 'last_3f' not in df.columns:
            df['last_3f'] = np.nan
            
        return df
    return pd.DataFrame()

def load_future_data():
    if os.path.exists(FUTURE_CSV):
        try:
            df = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding='cp932')
            
        PLACE_MAP_REV = {
            "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
        }
        df['place_code'] = df['race_id'].str[4:6]
        df['place_name'] = df['place_code'].map(PLACE_MAP_REV).fillna("不明")
        df['r_num'] = df['race_id'].str[10:12].astype(int)
        
        if 'race_name' not in df.columns: df['race_name'] = ""
        df['day_label'] = df['date'] if 'date' in df.columns else "不明"
        return df
    return pd.DataFrame()

def load_history_data():
    if os.path.exists(HISTORY_CSV):
        try:
            df = pd.read_csv(HISTORY_CSV, dtype={'race_id': str, 'honmei_umaban': str, 'partners': str}, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(HISTORY_CSV, dtype={'race_id': str, 'honmei_umaban': str, 'partners': str}, encoding='cp932')
        
        if 'partners' not in df.columns:
            df['partners'] = ""
        return df
    return pd.DataFrame(columns=['date', 'race_id', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay'])

model_data = load_model()
df_past = load_past_data()
df_future = load_future_data()
df_history = load_history_data()

# ==========================================
# 2. 出馬表自動取得機能
# ==========================================
def get_target_dates():
    today = datetime.now()
    return sorted(list(set([(today + timedelta(days=i)).strftime("%Y%m%d") for i in range(7) if (today + timedelta(days=i)).weekday() in [5, 6]])))

def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[\r\n\t]+', ' ', text)).strip() if text else ""

def run_scraper(p_text, p_bar):
    target_dates = get_target_dates()
    all_race_ids, id_to_date = [], {}
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://race.netkeiba.com/"})

    p_text.text("📅 開催日を確認中...")
    for date_str in target_dates:
        for url in [f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}", f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"]:
            try:
                res = session.get(url, timeout=10)
                if res.status_code == 200:
                    res.encoding = 'utf-8'
                    found_ids = re.findall(r'race_id=["\']?(\d{12})["\']?', res.text)
                    for rid in found_ids:
                        if rid not in all_race_ids:
                            all_race_ids.append(rid)
                            id_to_date[rid] = date_str
            except Exception: pass
            time.sleep(1)

    all_race_ids.sort()
    if not all_race_ids: return False

    race_data_list = []
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    total = len(all_race_ids)

    for i, race_id in enumerate(all_race_ids):
        p_text.text(f"📥 出馬表を取得中... ({i+1}/{total} レース)")
        p_bar.progress((i + 1) / total)
        try:
            res = session.get(f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}", timeout=10)
            if res.status_code == 200:
                res.encoding = 'utf-8'
                soup = BeautifulSoup(res.text, "html.parser")
                d_str = id_to_date.get(race_id, "")
                display_date = f"{datetime.strptime(d_str, '%Y%m%d').month}月{datetime.strptime(d_str, '%Y%m%d').day}日({weekdays[datetime.strptime(d_str, '%Y%m%d').weekday()]})" if d_str else "不明"
                
                r_name_elem = soup.find(class_="RaceName") or soup.find(class_="RaceList_Item02")
                if r_name_elem:
                    main_name_span = r_name_elem.find("span", class_="RaceName_main")
                    r_name = clean_text(main_name_span.text) if main_name_span else clean_text(r_name_elem.text)
                else:
                    r_name = ""

                for row in soup.select("tr.HorseList"):
                    cols = row.find_all("td")
                    if len(cols) >= 7:
                        nm = clean_text(cols[3].find("a").text) if cols[3].find("a") else clean_text(cols[3].text)
                        ub = clean_text(cols[1].text)
                        if nm and ub.isdigit():
                            sa = clean_text(cols[4].text)
                            odds_val, pop_val = np.nan, np.nan
                            odds_elem = row.find(class_=re.compile(r'Popular_Txt|Odds'))
                            if odds_elem:
                                try: odds_val = float(clean_text(odds_elem.text))
                                except Exception: pass
                            pop_elem = row.find(class_=re.compile(r'Popular_Num'))
                            if pop_elem:
                                try: pop_val = float(clean_text(pop_elem.text))
                                except Exception: pass

                            race_data_list.append({
                                "race_id": str(race_id), "date": display_date, "race_name": r_name,
                                "枠番": clean_text(cols[0].text), "馬番": ub, "馬名": nm,
                                "sex_code": sa[0] if sa else "", "age": sa[1:] if len(sa)>1 else "",
                                "斤量": clean_text(cols[5].text),
                                "騎手": clean_text(cols[6].find("a").text) if cols[6].find("a") else clean_text(cols[6].text),
                                "単勝": odds_val, "人気": pop_val
                            })
            time.sleep(random.uniform(0.5, 1.0))
        except Exception: pass

    if race_data_list:
        pd.DataFrame(race_data_list).to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
        return True
    return False

# ==========================================
# サイドバー UI
# ==========================================
st.sidebar.header("🔄 画面の更新")
if st.sidebar.button("🔄 最新の情報にリロード", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📥 今週の出馬表取得")
if st.sidebar.button("🌐 出馬表を自動取得する", use_container_width=True):
    with st.spinner("出馬表データを取得しています（約1分）..."):
        p_text = st.sidebar.empty()
        p_bar = st.sidebar.progress(0)
        if run_scraper(p_text, p_bar):
            st.cache_data.clear()
            st.sidebar.success(f"✅ 出馬表を取得しました！（更新: {datetime.now().strftime('%H:%M:%S')}）")
            time.sleep(1.5)
            st.rerun()
        else:
            st.sidebar.error("❌ データの取得に失敗しました。")

st.sidebar.markdown("---")
st.sidebar.header("🏁 実戦結果の検証")
if st.sidebar.button("🏆 終了したレースの配当を取得", use_container_width=True):
    with st.spinner("🏁 実際の着順と全券種の配当を計算中..."):
        if not df_history.empty:
            updated = False
            headers = {"User-Agent": "Mozilla/5.0"}
            for idx, row in df_history.iterrows():
                if pd.isna(row['result_pay']) or str(row['result_pay']).strip() == "":
                    try:
                        res = requests.get(f"https://db.netkeiba.com/race/{str(row['race_id'])}/", headers=headers, timeout=5)
                        res.encoding = 'utf-8'
                        soup = BeautifulSoup(res.text, "html.parser")
                        
                        axis = str(int(row['honmei_umaban']))
                        partners = [str(int(p.strip())) for p in str(row.get('partners', '')).split(',') if p.strip().isdigit()]
                        
                        rt = soup.find("table", class_="race_table_01")
                        r1, r2, r3 = None, None, None
                        if rt:
                            rows = rt.find_all("tr")[1:]
                            ranks = [r.find_all("td")[2].text.strip() for r in rows if r.find_all("td")[0].text.strip().isdigit()]
                            if len(ranks) >= 3:
                                r1, r2, r3 = ranks[0], ranks[1], ranks[2]

                        if not r1: continue
                        top3 = {r1, r2, r3}
                        top2 = {r1, r2}
                        
                        hit_tansho = (axis == r1)
                        hit_umaren = (axis in top2) and any(p in top2 for p in partners)
                        hit_sanrenpuku = (axis in top3) and (len([p for p in partners if p in top3]) >= 2)
                        
                        total_payout = 0
                        for table in soup.find_all("table", summary="払い戻し"):
                            for tr in table.find_all("tr"):
                                th = tr.find("th")
                                if not th: continue
                                kind = th.text.strip()
                                tds = tr.find_all("td")
                                if len(tds) < 2: continue
                                
                                win_texts = tds[0].get_text(separator=" ").split()
                                amt_texts = tds[1].get_text(separator=" ").replace(',', '').replace('円', '').split()
                                
                                if kind == "単勝" and hit_tansho:
                                    for w, a in zip(win_texts, amt_texts):
                                        if w == axis: total_payout += int(a)
                                elif kind == "馬連" and hit_umaren:
                                    for w, a in zip(win_texts, amt_texts):
                                        w_set = set(w.split('-'))
                                        if axis in w_set and any(p in w_set for p in partners): total_payout += int(a)
                                elif kind == "ワイド":
                                    for w, a in zip(win_texts, amt_texts):
                                        w_set = set(w.split('-'))
                                        if axis in w_set and any(p in w_set for p in partners):
                                            if w_set.issubset(top3): total_payout += int(a)
                                elif kind == "三連複" and hit_sanrenpuku:
                                    for w, a in zip(win_texts, amt_texts):
                                        w_set = set(w.split('-'))
                                        if axis in w_set and len(w_set.intersection(set(partners))) >= 2: total_payout += int(a)

                        df_history.at[idx, 'result_pay'] = total_payout
                        updated = True
                    except Exception: pass
                    time.sleep(1)
            if updated: df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig', errors='replace')
        st.cache_data.clear()
        st.success("✅ 実戦結果（全券種）を最新化しました！")
        time.sleep(1.5)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🗑️ 履歴の完全リセット")
if st.sidebar.button("💥 ゴミ予想履歴を完全消去", type="primary", use_container_width=True):
    try:
        if os.path.exists(HISTORY_CSV):
            os.remove(HISTORY_CSV)
        st.cache_data.clear()
        st.sidebar.success("✅ 履歴データを物理的に完全消去しました！")
        time.sleep(1.5)
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"消去エラー: {e}")

# ==========================================
# 3. 🧠 本格予想ロジック
# ==========================================
def calculate_race_scores(race_id_target, target_df):
    if target_df.empty or model_data is None:
        return None

    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty:
        return None

    features = ['枠番', '馬番', '斤量', 'time_seconds', 'sex_code', 'age', 'horse_weight', 'weight_change', '単勝', '人気']
    if isinstance(model_data, dict):
        model = model_data.get('model')
        if 'features' in model_data: features = model_data['features']
    else:
        model = model_data

    if not df_past.empty and '馬名' in df_past.columns:
        agg_dict = {}
        if 'time_seconds' in df_past.columns: agg_dict['time_seconds'] = 'mean'
        if 'last_3f' in df_past.columns: agg_dict['last_3f'] = 'mean'
        if 'horse_weight' in df_past.columns: agg_dict['horse_weight'] = 'mean'
        if 'weight_change' in df_past.columns: agg_dict['weight_change'] = 'mean'

        if agg_dict:
            horse_stats = df_past.groupby('馬名').agg(agg_dict).reset_index()
            race_df = pd.merge(race_df, horse_stats, on='馬名', how='left')

    for f in features:
        if f not in race_df.columns: race_df[f] = np.nan

    if 'sex_code' in race_df.columns and race_df['sex_code'].dtype == object:
        race_df['sex_code'] = race_df['sex_code'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)

    X = race_df[features].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    if '単勝' in X.columns: X['単勝'] = X['単勝'].fillna(10.0)
    if '人気' in X.columns: X['人気'] = X['人気'].fillna(5.0)
    X = X.fillna(0)

    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
        else:
            prob = model.predict(X)
    except Exception:
        return None

    s = prob.sum()
    race_df['win_prob'] = prob / s if s > 0 else 1.0 / len(race_df)
    mp = race_df['win_prob'].mean()
    rs = 100 + ((race_df['win_prob'] - mp) / mp) * 35 if mp > 0 else 100
    race_df['score'] = np.clip(rs, 50, 120).round().astype(int)

    return race_df.sort_values(by='score', ascending=False).reset_index(drop=True)

def get_all_markers():
    markers = {}
    if df_future.empty: return markers
    for rid in df_future['race_id'].unique():
        sdf = calculate_race_scores(rid, df_future)
        if sdf is not None and len(sdf) >= 6:
            sc = sdf['score'].tolist()
            if sc[0] >= 108 and (sc[0] - sc[1]) >= 4: markers[rid] = "★"
            elif (sc[0] - sc[4]) <= 5: markers[rid] = "◎"
            else: markers[rid] = ""
    return markers
markers = get_all_markers()

tab_forecast, tab_dashboard = st.tabs(["🏇 レース予想", "📈 実戦成績"])

with tab_forecast:
    if df_future.empty:
        st.warning("⚠️ 出馬表データが存在しません。左のメニューから「出馬表を自動取得する」を押してください。")
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
                    if rname:
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
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()
                
            scored_df = calculate_race_scores(target_id, df_future)
            if scored_df is None or len(scored_df) < 6:
                st.error("出走頭数が少ない、またはデータが不足しているため予想をスキップします。")
                st.stop()

            honmei_row = scored_df.iloc[0]
            partners_df = scored_df.iloc[1:6]
            
            honmei_umaban = int(honmei_row['馬番'])
            honmei_name = honmei_row['馬名']
            partner_nums = partners_df['馬番'].astype(int).tolist()
            partners_str = ",".join(map(str, partner_nums))
            
            if df_history.empty or str(target_id) not in df_history['race_id'].astype(str).values:
                new_record = pd.DataFrame([{
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'race_id': str(target_id),
                    'race_name': race_display_name,
                    'honmei_umaban': honmei_umaban,
                    'partners': partners_str,
                    'honmei_name': honmei_name,
                    'result_pay': ""
                }])
                df_history = pd.concat([df_history, new_record], ignore_index=True)
                df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')

            table_summary = []
            for _, row in scored_df.iterrows():
                last_3f_val = f"{row['last_3f']:.1f}" if pd.notna(row.get('last_3f')) else "データなし"
                table_summary.append(
                    f"枠:{row.get('枠番','')} | 馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | 騎手:{row.get('騎手', '不明')} | "
                    f"斤量:{row.get('斤量','')} | AI_score:{row['score']:3d} | 勝率予測:{row['win_prob']*100:.1f}% | 過去上がり3F:{last_3f_val}秒"
                )
            prompt_data = "\n".join(table_summary)

            system_instruction = f"""
あなたはプロの競馬分析AI「勝ちぱかくん」です。

【絶対遵守事項】
1. 前置き、挨拶、謝罪、日付やレース名に関する言い訳、検索に関するコメントは一切禁止します。直ちに下記のフォーマット通り出力してください。
2. 提供された【出走馬データ＆AI_score】を絶対的な分析根拠としてください。
3. 印（◎◯▲△☆）と見解、推奨買い目（三連単を含む）を明確に記述してください。

出力フォーマット：
---
### 📊 1. 出走馬データ＆AI分析一覧
（全出走馬の【枠 / 馬番 / 馬名 / 騎手 / AIスコア / 予想勝率】をきれいなMarkdownテーブルで出力）

### 🌪️ 2. レース展開＆馬場分析
* **馬場・トラックバイアス:** （コース特性と想定馬場状態）
* **展開予想:** （想定ペースと有利な脚質）

### 🎯 3. 勝ちぱかくんの印と詳細見解
* **◎（本命）:** 〇番（馬名） - （AIスコア・データ面の根拠）
* **◯（対抗）:** 〇番（馬名） - （評価理由）
* **▲（単穴）:** 〇番（馬名） - （逆転要素）
* **△（連下）:** 〇番（馬名）、〇番（馬名） - （ヒモ候補）
* **☆（穴馬）:** 〇番（馬名） - （高配当要素）

### 💡 4. 戦略的・推奨買い目
* **単勝:** ◎ (1点)
* **馬連:** ◎ － ◯, ▲, △, ☆ (4〜5点)
* **ワイド:** ◎ － ◯, ▲, ☆ (3点)
* **三連複 (軸1頭流し):** ◎ － ◯, ▲, △, ☆ (10点)
* **三連単 (1着固定流し / 軸1頭マルチ):** 1着:◎ → 2・3着:◯, ▲, △, ☆ (12〜60点)
* **【勝ちぱかポイント】:** (資金配分や高配当狙いのポイント)
---
"""
            prompt = f"対象レース: {race_display_name}\n\n出走馬データ:\n{prompt_data}"

            with st.spinner("出走馬データとAIスコアを多角分析中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.7
                        )
                    )
                    
                    res_text = response.text if response.text else ""
                    if not res_text and response.candidates:
                        for part in response.candidates[0].content.parts:
                            if part.text:
                                res_text += part.text
                                
                    if res_text:
                        st.markdown(res_text)
                    else:
                        st.warning("⚠️ Geminiからの回答テキストを取得できませんでした。")
                        
                    st.success("📝 このレースの予想を実戦履歴に記録しました！")
                except Exception as e:
                    st.error(f"【APIエラー】: {e}")

with tab_dashboard:
    st.subheader("📈 実戦成績（単・連・ワイ・3複・3単 総合ベース）")
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
            invested = total_races * 2100
            hit_rate = (hits / total_races) * 100
            recovery_rate = (returns / invested) * 100
            profit = returns - invested
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 レース的中率 (いずれか的中)", f"{hit_rate:.1f}%", f"{hits} / {total_races} 的中")
            delta_color = "normal" if recovery_rate >= 100 else "inverse"
            col2.metric("💰 総合回収率", f"{recovery_rate:.1f}%", f"{recovery_rate - 100:.1f}%", delta_color=delta_color)
            col3.metric("💴 累計収支 (1点100円計算)", f"{int(profit):,} 円")
            
        st.markdown("---")
        st.dataframe(df_history[['date', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay']].sort_values(by='date', ascending=False), use_container_width=True)