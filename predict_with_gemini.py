import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import unicodedata
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Literal

# ==========================================
# 🎨 アプリの基本設定 & スタイル定義
# ==========================================
st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="logo.png", layout="wide")

st.markdown("""
<style>
    /* 🎯 背景グラデーション */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%);
    }
    
    .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }
    
    .section-header { font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin: 1.2rem 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; }
    .kachi-table { width: 100%; border-collapse: collapse; background-color: #ffffff; white-space: nowrap; font-size: 17px; }
    .kachi-table th { padding: 12px 10px; text-align: center; background: #f8f9fa; border-bottom: 2px solid #dee2e6; font-size: 17px; font-weight: bold; }
    .kachi-table td { padding: 12px 10px; text-align: center; border-bottom: 1px solid #f1f3f5; font-size: 17px; }
    .badge-mark { color: #fff; padding: 6px 12px; border-radius: 6px; font-weight: bold; display: inline-block; min-width: 60px; font-size: 16px; }
    .badge-honmei { background: #e74c3c; } .badge-taikou { background: #3498db; }
    .badge-tana { background: #2ecc71; } .badge-renka { background: #f39c12; }
    .badge-ana { background: #9b59b6; } .badge-keshi { background: #e0e0e0; color: #7f8c8d; }
    
    .sense-card { background-color: #ffffff; border-left: 6px solid #8e44ad; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .sense-title { font-size: 1.2rem; font-weight: bold; color: #8e44ad; }
    .ticket-badge { font-size: 1.1rem; font-weight: bold; color: #d35400; background: #fef5e7; padding: 4px 10px; border-radius: 4px; display: inline-block; }
    
    .gemini-view-card { background-color: #f0f8ff; border-left: 5px solid #3498db; padding: 16px 20px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .gemini-view-title { color: #2980b9; font-size: 1.2rem; font-weight: bold; margin-bottom: 8px; }
    .gemini-view-text { font-size: 16px; color: #333; line-height: 1.6; margin: 0; }
    
    /* 🚨 カスタムボタンデザイン（青系・グラデーション） */
    div.stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #1f618d 100%);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FUTURE_CSV = "future_races.csv"
CACHE_FILE = "app_cache.pkl"

# ==========================================
# 📊 Pydanticによる出力スキーマの定義
# ==========================================
class HorseEvaluation(BaseModel):
    馬名: str = Field(description="評価対象の馬名")
    最終印: Literal["◎", "◯", "▲", "△", "☆", "消"] = Field(
        description="割り振る印。◎・◯・▲・△は各1頭ずつ必須。☆は最大1頭（少頭数時は不使用）。それ以外の全馬には必ず「消」を割り振ること。"
    )
    短評: str = Field(
        description="調教ラップ、血統、前走内容に基づく短評。Python上位から【消】にした馬は脱落理由を明記。"
    )

class RaceAuditResult(BaseModel):
    summary: str = Field(
        description="◎の選定根拠、および選定外から抜擢した穴馬や落とした人気馬の理由（300文字以内）"
    )
    evaluations: List[HorseEvaluation] = Field(
        description="出走全頭分の評価リスト（頭数欠落なし）"
    )

# ==========================================
# 🧠 システムプロンプト（system_instruction）
# ==========================================
system_instruction = """
# 役割定義
あなたは競走馬の状態鑑定・客観データ分析に特化した「監査役トラックマン」です。
提供されたデータを元に、Pythonモデルが算出した初期の買い目を監査・アップデートし、最終的な印を決定してください。

# 前提状況
統計・確率モデル（Python）が算出した「Python上位6頭」の中に、通常は約9割の確率で1着馬が存在します。
しかし、初期印の序列が崩れて下位評価が勝つケースや、3着にPython選定外（7番手以下）の伏兵が飛び込むケースがあります。
※ただし【新馬戦】においてはPythonモデルも過去データがないため、Pythonの初期順位は完全に無視し、あなたの独自鑑定でゼロベースで買い目を構築してください。

# あなたの任務
- 通常戦: Python上位6頭の中から最も信頼できる「軸・相手（◎・◯）」を確定させ、全出走馬の中から激走気配を持つ「ヒモ・穴馬（▲・△・☆）」を選別してください。
- 新馬戦: Python順位を一切無視し、全頭の中から【調教加速ラップ】と【血統適性】を基準に最高評価の軸・相手を選定してください。

# 評価の絶対ルール（計算・定性コメントの禁止）
- 枠順成績、コースバイアス、斤量差はPythonモデル側ですべて計算・織り込み済みです。これらに言及したり、計算して評価を変更することは【厳禁】です。
- 記者の主観コメントや厩舎コメントは考慮せず、提示された「客観データ（数字・血統・判定タグ）」のみで判断してください。

# 客観データの監査基準
1. 調教タイム・加速ラップ（現在の状態）:
   - 全体時計の絶対値だけで判断しない（朝イチの内目を回れば時計は出るため）。
   - コース別（坂路・ウッド）の全体時計の差は比較せず、「終い1Fの伸び脚（前Fとの引き算）」のみに集中すること。
   - ラスト2Fからラスト1Fへの差分が「-0.2秒以下」の加速ラップを高く評価する。
   - 【絶対失速ルール】: 終い1Fが「13.0秒以上」を要している馬は、全体時計にかかわらず自動的に「状態不安（失速）」として評価を大幅に割り引くこと。
   - 【例外ルール（データ欠損時）】: 地方馬、海外帰りなどで調教タイムが存在しない（データなし）馬は、減点も加点も行わず「判定保留」とし、前走内容等の客観データのみでフラットに評価すること。データがないことだけを理由に【消】にすることは厳禁。

2. 【新馬戦の特別監査（完全Gemini委譲）】:
   - 前走データがない新馬戦では、前走監査を自動スキップする。
   - Pythonの初期評価・区分は完全に無視し、以下の2点のみで全出走馬をフラットに評価する：
     ① 調教の加速ラップ（終いが鋭く伸びて仕上がっている馬を最上位とする）
     ② 血統適性（父・母父が該当コース・距離・馬場質に合致しているか）

3. 通常戦の前走内容（通過順 × 上がり3F）:
   - 前走着順の数字だけで判断しない。
   - 【着順詐欺（高評価・穴馬候補）】:
     「4コーナー通過順が10番手以降（または後方30%以内）」かつ「上がり3F上位（判定タグあり、または鋭い脚）」で敗れている馬は、展開不向きによる着順詐欺として着順不問で高く評価し、▲・△・☆の最優先候補とする。
   - 【前残り恵まれ（過信禁物・消し候補）】:
     前走好走していても、「4コーナー3番手以内」かつ「上がりが平凡」で単に展開に恵まれただけの馬は過信せず、調教が並以下なら【消】の筆頭候補とする。

4. 休養明け（中10週以上・長期ブランク・前走日付不詳）の扱い:
   - 休養明けの馬は前走データの「賞味期限切れ」とみなし、前走内容の評価ウエイトをゼロに近づけ、【今回の調教状態】を最大ウエイトとする。調教で鋭い加速ラップが出ていれば高評価、調教失速馬は「太め残り・息不足」として即座に【消】とする。

# 印の割り振りルール
出走全頭に対して、必ず以下のいずれかの印を1つずつ割り振ってください。印の数には以下の厳格な制限を設けます。

【通常戦（新馬戦以外）】
- 【◎】（必ず1頭のみ）:「Python上位6頭」の中から選定。調教の加速力・前走内容から最も1着突き抜けが期待できる軸馬。
- 【◯】（必ず1頭のみ）:「Python上位6頭」の中から選定。2着以内に粘る相手本線の実力馬。
- 【▲】（必ず1頭のみ）:「全出走馬」の中から選定。2〜3着に食い込む単穴。
- 【△】（必ず1頭のみ）:「全出走馬」の中から選定。連下として押さえるべき巻き返し馬。
- 【☆】（最大1頭・少頭数免除）:「全出走馬」の中から選定。調教の加速ラップや前走内容から一撃を秘める特注穴馬。
  ※【少頭数免除ルール】: 出走頭数が「9頭以下」のレースでは、点数を絞るため【☆】は打たず、印を最大4頭（◎・◯・▲・△）に制限してください。
- 【消】（印を打った馬以外の残り全頭）:
  ※「Python上位6頭」から漏れて【消】となった馬については、短評に必ず明確な脱落理由（終い失速、前走展開恵まれ等）を記述してください。

【新馬戦の場合】
- 上位6頭の縛りは完全撤廃。全頭フラットに【調教加速ラップ】と【血統】のみで◎・◯・▲・△・(☆)の最大5頭を選定せよ。
- なお、選定されなかった残りの全馬には、通常戦と同様に必ず【消】の印を割り当て、出走全頭分のリストを構築してください。

※上位全滅時の消去法ルール:
万が一Python上位6頭が全員調教失速等の不安を抱えている場合でも、通常戦のルール通り上位6頭の中から最も減点材料の少ない2頭を消去法で◎・◯に据え、短評に懸念を明記すること。
"""

# ==========================================
# 🔧 前処理ヘルパー関数
# ==========================================
def parse_training_lap_stats(t_str):
    if not t_str or pd.isna(t_str) or str(t_str).strip() in ['-', 'nan', 'NaN', '']:
        return "調教データなし"
    
    s = str(t_str).strip()
    m_floats = re.findall(r'\d+\.?\d*', s)
    if not m_floats:
        return "調教データなし"
    
    final_1f = float(m_floats[-1])
    
    two_f_total = None
    for val_str in reversed(m_floats[:-1]):
        v = float(val_str)
        if 21.0 <= v <= 30.0:
            two_f_total = v
            break
            
    if two_f_total and two_f_total > final_1f:
        penultimate_1f = round(two_f_total - final_1f, 1)
        diff = round(final_1f - penultimate_1f, 1)
        if diff <= -0.2:
            status = f"加速({diff:+.1f}s)"
        elif diff <= 0.1:
            status = f"維持({diff:+.1f}s)"
        else:
            status = f"失速({diff:+.1f}s)"
        
        warn = " [⚠️終い失速・要警戒]" if final_1f >= 13.0 else ""
        return f"終い:{final_1f:.1f}s (前F:{penultimate_1f:.1f}s, 差:{diff:+.1f}s, {status}){warn}"
    else:
        warn = " [⚠️終い失速・要警戒]" if final_1f >= 13.0 else ""
        return f"終い:{final_1f:.1f}s{warn}"

def parse_past_race_stats(order_str, passage_str, agari_str, agari_rank=None, past_tousuu=None, interval_str="", is_shinba=False, surface="芝"):
    """
    スクレイパーが取得した前走上がり順位・前走頭数・間隔を元に、着順詐欺フラグや休養明けフラグを生成する
    """
    if is_shinba or pd.isna(order_str) or str(order_str).strip() in ['-', 'nan', 'NaN', '']:
        return "前走データなし ⚠️【休養明け/初出走：前走不問・調教状態を最重視】"

    pass_clean = str(passage_str).strip()
    corner_last = None
    if pass_clean and pass_clean not in ['-', 'nan', 'NaN', '']:
        corners = re.findall(r'\d+', pass_clean)
        if corners:
            corner_last = int(corners[-1])

    # 順位と頭数のキャスト処理
    try:
        rank_val = int(float(str(agari_rank)))
        rank_text = f" (上がり{rank_val}位)"
    except (ValueError, TypeError):
        rank_val = None
        rank_text = ""
        
    try:
        tousuu_val = int(float(str(past_tousuu)))
        tousuu_text = f"{tousuu_val}頭立て / "
    except (ValueError, TypeError):
        tousuu_val = 16
        tousuu_text = ""

    tag = ""
    is_rear = (corner_last is not None) and (corner_last >= 10 or corner_last >= tousuu_val * 0.7)
    is_top_agari = (rank_val is not None) and (rank_val <= 3)

    is_front = (corner_last is not None) and (corner_last <= 3)
    is_slow_agari = (rank_val is not None) and (rank_val >= 7)

    if is_rear and is_top_agari:
        tag = f" 🔥【着順詐欺判定：後方追込×上がり上位】"
    elif is_front and is_slow_agari:
        tag = f" ⚠️【前残り警戒：前受先行×上がり平凡】"

    interval_info = f"[{interval_str}] " if interval_str and str(interval_str) not in ['nan', 'NaN', ''] else ""
    return f"{interval_info}着順:{order_str} ({tousuu_text}通過:{passage_str} / 上がり3F:{agari_str}{rank_text}){tag}"

def clean_name(text):
    if pd.isna(text): return ""
    s = unicodedata.normalize('NFKC', str(text))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip().upper()

def get_badge_class(mark):
    if pd.isna(mark): return "badge-keshi"
    if "◎" in mark: return "badge-honmei"
    elif "◯" in mark: return "badge-taikou"
    elif "▲" in mark: return "badge-tana"
    elif "△" in mark: return "badge-renka"
    elif "☆" in mark: return "badge-ana"
    return "badge-keshi"

def normalize_kyakushitsu(k_str):
    k = str(k_str).strip()
    if pd.isna(k_str) or k in ['nan', 'NaN', 'None', '', '-']:
        return "不明"
    if "大" in k or "逃" in k: return "逃げ"
    if "先" in k: return "先行"
    if "差" in k: return "差し"
    if "追" in k: return "追込"
    if "マ" in k: return "自在"
    return "不明"

@st.cache_data
def load_future_data():
    if os.path.exists(FUTURE_CSV):
        df_f = pd.read_csv(FUTURE_CSV, dtype={'race_id': str, '枠番': str, '馬番': str}, encoding='utf-8-sig')
        df_f['race_id'] = df_f['race_id'].astype(str).str.zfill(12)
        df_f['place_name'] = df_f['race_id'].str[4:6].map({"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}).fillna("開催場")
        df_f['r_num'] = pd.to_numeric(df_f['race_id'].str[-2:], errors='coerce').fillna(1).astype(int)
        df_f['day_label'] = df_f['date'].astype(str).str.strip() if 'date' in df_f.columns else "当日"
        df_f['馬名_clean'] = df_f['馬名'].astype(str).apply(clean_name)
        
        if '推測脚質' in df_f.columns:
            df_f['推測脚質'] = df_f['推測脚質'].apply(normalize_kyakushitsu)
        else:
            df_f['推測脚質'] = "不明"

        for i, row in df_f.iterrows():
            if pd.isna(row['馬番']) or str(row['馬番']).strip() in ["", "nan", "NaN"]:
                df_f.at[i, '馬番'] = str(90 + i)
        
        return df_f
    return pd.DataFrame()

@st.cache_data
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return joblib.load(CACHE_FILE)
        except Exception:
            return {}
    return {}

df_future = load_future_data()

def render_race_predictions(race_id_target):
    if df_future.empty: return None, None, None, None, None
    
    cache_data = load_cache()
    if not cache_data:
        st.warning("⚠️ キャッシュファイル(app_cache.pkl)が見つかりません。")
        return None, None, None, None, None
    
    df = df_future[df_future['race_id'].astype(str) == str(race_id_target)].copy()
    if df.empty: return None, None, None, None, None

    all_preds = cache_data.get('predictions', {})
    target_key = str(race_id_target).zfill(12)
    predictions = all_preds.get(target_key) or all_preds.get(str(race_id_target))
    
    if not predictions:
        st.error(f"⚠️ レースID '{target_key}' の計算済みデータが見つかりません。推論を実行してください。")
        return None, None, None, None, None

    horse_preds = predictions.get('horses', {})
    
    df['win_prob'] = df['馬名_clean'].map(lambda x: horse_preds.get(x, {}).get('win_prob', 0.1))
    
    df = df.sort_values(by=['win_prob'], ascending=False).reset_index(drop=True)
    
    min_p, max_p = df['win_prob'].min(), df['win_prob'].max()
    if max_p > min_p:
        df['ai_score'] = (50 + (df['win_prob'] - min_p) / (max_p - min_p) * 100).round().astype(int)
    else:
         df['ai_score'] = 100

    mark_list = ['◎', '◯', '▲', '△', '☆', '☆']
    new_marks = []
    for i in range(len(df)):
        new_marks.append(mark_list[i] if i < len(mark_list) else '消')
    df['印'] = new_marks

    pat = predictions.get('pattern', '③ 混戦気配')
    rec_ticket = predictions.get('ticket', '3連複')
    
    if len(df) >= 6:
        h1 = df.iloc[0]
        def fmt_m(val):
            v_str = str(val)
            return "未定" if v_str.startswith("90") or int(float(v_str)) >= 90 else f"{int(float(v_str)):02d}"
        
        aite_str = ", ".join([f"{fmt_m(df.iloc[i]['馬番'])}({df.iloc[i]['馬名']})({df.iloc[i]['印']})" for i in range(1, 6)])
        buy_detail = f"軸: {fmt_m(h1['馬番'])}({h1['馬名']})(◎) -> 相手: {aite_str}"
    else:
        buy_detail = predictions.get('buy_detail', '')
        
    try:
        race_day = int(target_key[8:10])
    except:
        race_day = 1

    return df, pat, rec_ticket, buy_detail, race_day

def generate_base_table(disp_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>枠番</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>推測脚質</th><th>斤量</th><th>騎手</th><th>スコア</th><th>1着率</th><th>Python印</th></tr>"
    for _, r in disp_df.iterrows():
        w_num = str(r.get('枠番', ''))
        display_waku = w_num if w_num and w_num != 'nan' and not w_num.startswith("90") else "-"
        
        m_num = str(r['馬番'])
        display_umaban = "未定" if m_num.startswith("90") or int(float(m_num)) >= 90 else f"{int(float(m_num)):02d}"
        
        win_val = float(r.get('win_prob', 0))
        kinryo_val = float(r.get('斤量', 55.0))
        score_str = f"<b>{int(r.get('ai_score', 100))}</b>"
        win_str = f"<span style='color:#e74c3c; font-weight:bold;'>{win_val*100:.1f}%</span>" if win_val >= 0.25 else f"{win_val*100:.1f}%"
        
        mark = r.get('印', '消')
        
        style_raw = str(r.get('推測脚質', '不明')).strip()
        if style_raw == '不明':
            style_html = "<span style='color:#bdc3c7;'>-</span>"
        elif style_raw == '逃げ':
            style_html = "<span style='color:#e74c3c; font-weight:bold;'>逃げ</span>"
        else:
            style_html = style_raw
            
        jockey_val = str(r.get('騎手', '-')).strip()
        
        html += f"<tr><td><b>{display_waku}</b></td><td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{style_html}</td><td>{kinryo_val:.1f}kg</td><td>{jockey_val}</td><td>{score_str}</td><td>{win_str}</td>"
        html += f"<td><span class='badge-mark {get_badge_class(mark)}'>{mark}</span></td></tr>"
    html += "</table></div>"
    return html

def generate_fusion_table(merged_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>枠番</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>AIｽｺア</th><th>調教評価</th><th>Python印</th><th>Gemini印</th><th style='text-align:left; min-width: 400px;'>Gemini短評 (展開・バイアス物理判定)</th></tr>"
    for _, r in merged_df.iterrows():
        w_num = str(r.get('枠番', ''))
        display_waku = w_num if w_num and w_num != 'nan' and not w_num.startswith("90") else "-"
        
        m_num = str(r.get('馬番', ''))
        display_umaban = "未定" if m_num.startswith("90") or int(float(m_num)) >= 90 else f"{int(float(m_num)):02d}"
        
        tanpyo = r.get('短評', '-')
        chokyo_eval = str(r.get('調教評価', '-'))
            
        html += f"<tr><td><b>{display_waku}</b></td><td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td><b>{int(r.get('ai_score', 100))}</b></td>"
        html += f"<td><span style='font-weight:bold; color:#d35400;'>{chokyo_eval}</span></td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('印', '消'))}'>{r.get('印', '消')}</span></td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('Gemini印', '消'))}'>{r.get('Gemini印', '消')}</span></td>"
        html += f"<td style='text-align:left; font-size:14px; white-space:normal; line-height:1.5; padding: 12px; font-weight:500; color:#2c3e50;'>{tanpyo}</td></tr>"
    html += "</table></div>"
    return html

def clear_all_caches():
    st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()

# ==========================================
# 🎯 画面表示
# ==========================================
st.sidebar.button("🔄 画面リロード", on_click=clear_all_caches, use_container_width=True)

col_logo, col_main = st.columns([1, 4])

with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)

with col_main:
    st.markdown("<div class='section-header' style='margin-top: 0;'>🎯 レース選択</div>", unsafe_allow_html=True)

    if df_future.empty:
        st.warning("出馬表データがありません。scrape_shutsuba.py を実行してください。")
        st.stop()

    dates = sorted(df_future['day_label'].unique())
    sel_date = st.radio("開催日", dates, horizontal=True, label_visibility="collapsed")
    day_df = df_future[df_future['day_label'] == sel_date]

    places = day_df['place_name'].unique()
    place_tabs = st.tabs([f"📍 {p}" for p in places])
    for i, place in enumerate(places):
        with place_tabs[i]:
            place_df = day_df[day_df['place_name'] == place]
            races = sorted(place_df['r_num'].unique())
            for j in range(0, len(races), 6):
                cols = st.columns(6)
                for k in range(6):
                    if j + k < len(races):
                        r = races[j + k]
                        r_df = place_df[place_df['r_num'] == r]
                        r_id = r_df['race_id'].iloc[0]
                        r_name = str(r_df.get('race_name', pd.Series([''])).iloc[0]).strip()
                        if cols[k].button(f"{r}R {r_name}" if r_name and r_name!='nan' else f"{r}R", key=f"btn_{r_id}", use_container_width=True):
                            st.session_state['selected_race_id'] = r_id

if st.session_state['selected_race_id']:
    t_id = str(st.session_state['selected_race_id'])
    t_rows = df_future[df_future['race_id'].astype(str) == t_id]
    if t_rows.empty: st.stop()

    r_info = t_rows.iloc[0]
    st.markdown("---")
    
    display_race_name = r_info.get('race_name', '')
    if pd.isna(display_race_name) or display_race_name == 'nan': display_race_name = ''
    surface_info = r_info.get('surface', '不明')
    distance_info = r_info.get('distance', '')
    
    place_name_str = r_info.get('place_name', '')
    dist_str = str(distance_info)
    dist_num_str = re.sub(r'\D', '', dist_str)
    turn_info = ""
    
    if place_name_str == "新潟" and dist_num_str == "1000":
        turn_info = "直線"
    elif place_name_str in ['中山', '阪神', '京都', '札幌', '函館', '福島', '小倉']:
        turn_info = "右回り"
    elif place_name_str in ['東京', '新潟', '中京']:
        turn_info = "左回り"
    
    turn_display = f" / {turn_info}" if turn_info else ""
    
    res_df, pat, rec_ticket, buy_detail, race_day = render_race_predictions(t_id)
    
    st.markdown(f"<h2>🚀 {r_info['place_name']} {r_info['r_num']}R 【{display_race_name}】 <span style='font-size:18px; color:#7f8c8d;'>({surface_info}{distance_info}m{turn_display} / 開催{race_day}日目)</span></h2>", unsafe_allow_html=True)
    
    if res_df is not None:
        is_mokuyou = any(res_df['馬番'].astype(str).str.startswith('90'))
        if is_mokuyou:
            st.info("ℹ️ 現在は枠順確定前の「特別登録状態」のため、枠番・馬番は未定として表示されています。")
            
        st.markdown(f"""
        <div class='sense-card'>
            <div class='sense-title'>🧠 AI勝負気配判定: {pat}</div>
            <div style='margin-top: 8px;'><b>🎟 推奨買い目:</b> <span class='ticket-badge'>{rec_ticket}</span></div>
            <div style='margin-top: 6px; font-size: 15px; color: #555;'><b>📝 買い目詳細:</b> {buy_detail}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>📊 Pythonベース予想データ (バイアス補正・CSV削除済み)</div>", unsafe_allow_html=True)
        st.markdown(generate_base_table(res_df), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🧠 Geminiに全データを渡し、展開予想と定性評価を実行", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()

            is_shinba = "新馬" in display_race_name or res_df['前走着順'].isna().all()
            total_horses = len(res_df)

            table_summary = []
            for i, (_, r) in enumerate(res_df.iterrows()):
                m_str = str(r['馬番'])
                display_m = "未定" if m_str.startswith("90") else f"{int(float(m_str)):02d}"
                
                if is_shinba:
                    py_group = "【新馬戦：全頭フラット評価】"
                else:
                    py_group = "【Python上位6頭】" if i < 6 else "【Python選定外】"
                    
                win_val = float(r.get('win_prob', 0)) * 100
                
                chokyo_summary = parse_training_lap_stats(r.get('調教タイム', ''))
                past_summary = parse_past_race_stats(
                    order_str=r.get('前走着順', '-'),
                    passage_str=r.get('前走通過順', '-'),
                    agari_str=r.get('前走上がり3F', '-'),
                    agari_rank=r.get('前走上がり順位', None),
                    past_tousuu=r.get('前走頭数', None),
                    interval_str=r.get('間隔', ''),
                    is_shinba=is_shinba,
                    surface=surface_info
                )
                
                info = (f"馬番:{display_m} | 馬名:{r.get('馬名','')} | 区分:{py_group} | 初期評価:{r.get('印','消')} (勝率:{win_val:.1f}%) | "
                        f"血統(父/母父):{r.get('父','-')} / {r.get('母父','-')} | "
                        f"脚質:{r.get('推測脚質','-')} | 騎手:{r.get('騎手','-')} | "
                        f"前走:{past_summary} | 調教評価:{r.get('調教評価','-')} | 調教分析:{chokyo_summary}")
                table_summary.append(info)

            prompt = f"""
以下のレースデータに基づき、指示に従って全出走馬を監査し、最終印（JSON）を出力してください。

【分析対象レース】: {r_info['place_name']} {r_info['r_num']}R ({surface_info}{distance_info}m)
【出走頭数】: {total_horses}頭 {"(少頭数レース: ☆なし・最大4頭推奨)" if total_horses <= 9 else ""}
【レース種別】: {"【新馬戦】(Python評価無視・調教加速ラップと血統重視)" if is_shinba else "通常戦"}

【出走馬データ一覧】
{chr(10).join(table_summary)}
"""

            with st.spinner("🧠 Geminiがメンバー構成から独自の展開ストーリーを紡ぎ出しています..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=prompt, 
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction, 
                            temperature=0.2, 
                            response_mime_type="application/json",
                            response_schema=RaceAuditResult,
                        )
                    )
                    
                    result = RaceAuditResult.model_validate_json(response.text)
                    
                    eval_df = pd.DataFrame([e.model_dump() for e in result.evaluations])
                    eval_df.rename(columns={'最終印': 'Gemini印'}, inplace=True)
                    
                    if not eval_df.empty and '馬名' in eval_df.columns:
                        eval_df['馬名_clean'] = eval_df['馬名'].astype(str).apply(clean_name)
                        eval_df_clean = eval_df[['馬名_clean', 'Gemini印', '短評']].dropna(subset=['馬名_clean'])
                        
                        merged_df = pd.merge(res_df, eval_df_clean, on='馬名_clean', how='left')
                        merged_df['Gemini印'] = merged_df['Gemini印'].fillna('消')
                        merged_df['短評'] = merged_df['短評'].fillna('-')
                        
                        st.markdown(f"""
                        <div class='gemini-view-card'>
                            <div class='gemini-view-title'>📝 トラックマンGeminiの展開・血統総合見解</div>
                            <p class='gemini-view-text'>{result.summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<div class='section-header'>🔥 Python（確率脳） × Gemini（定性・血統脳） 独立評価比較</div>", unsafe_allow_html=True)
                        st.markdown(generate_fusion_table(merged_df), unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ API通信エラー: {e}")