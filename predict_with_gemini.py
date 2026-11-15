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
from typing import List

# ==========================================
# 🎨 アプリの基本設定 & スタイル定義
# ==========================================
st.set_page_config(page_title="AI予想 勝ちぱかくん (純度100%・物理レバレッジ版)", page_icon="logo.png", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%); }
    .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }
    .section-header { font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin: 1.2rem 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; }
    .kachi-table { width: 100%; border-collapse: collapse; background-color: #ffffff; white-space: nowrap; font-size: 16px; }
    .kachi-table th { padding: 12px 10px; text-align: center; background: #f8f9fa; border-bottom: 2px solid #dee2e6; font-size: 16px; font-weight: bold; }
    .kachi-table td { padding: 10px 8px; text-align: center; border-bottom: 1px solid #f1f3f5; font-size: 16px; }
    
    .badge-mark { color: #fff; padding: 5px 10px; border-radius: 6px; font-weight: bold; display: inline-block; min-width: 50px; font-size: 15px; }
    .badge-honmei { background: #e74c3c; } .badge-taikou { background: #3498db; }
    .badge-tana { background: #2ecc71; } .badge-renka { background: #f39c12; }
    .badge-ana { background: #9b59b6; } .badge-chuu { background: #e67e22; } .badge-keshi { background: #e0e0e0; color: #7f8c8d; }
    
    .multiplier-up { color: #e74c3c; font-weight: bold; }
    .multiplier-down { color: #3498db; font-weight: bold; }
    .multiplier-flat { color: #7f8c8d; }

    .sense-card { background-color: #ffffff; border-left: 6px solid #2980b9; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .sense-title { font-size: 1.3rem; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-bottom: 12px; }
    
    .gemini-view-card { background-color: #f0f8ff; border-left: 5px solid #3498db; padding: 16px 20px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .gemini-view-title { color: #2980b9; font-size: 1.2rem; font-weight: bold; margin-bottom: 8px; }
    .gemini-view-text { font-size: 15px; color: #333; line-height: 1.6; margin: 0; }
    
    .shinba-badge { background-color: #e67e22; color: white; padding: 3px 8px; border-radius: 4px; font-size: 13px; font-weight: bold; margin-left: 8px; vertical-align: middle; }
    .age2-badge { background-color: #16a085; color: white; padding: 3px 8px; border-radius: 4px; font-size: 13px; font-weight: bold; margin-left: 8px; vertical-align: middle; }
    .old-badge { background-color: #2980b9; color: white; padding: 3px 8px; border-radius: 4px; font-size: 13px; font-weight: bold; margin-left: 8px; vertical-align: middle; }
    
    div.stButton > button { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); color: white; border: none; border-radius: 6px; padding: 12px 24px; font-weight: bold; font-size: 17px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; }
    div.stButton > button:hover { background: linear-gradient(135deg, #2980b9 0%, #1f618d 100%); box-shadow: 0 6px 8px rgba(0,0,0,0.15); transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FUTURE_CSV = "future_races.csv"
CACHE_FILE = "app_cache.pkl"

# ==========================================
# 📊 Pydanticによる係数出力スキーマ
# ==========================================
class HorseCoefficient(BaseModel):
    馬名: str = Field(description="評価対象の馬名（データと完全一致）")
    補正係数: float = Field(
        ge=0.5, le=1.6,
        description="好走確率評価。通常レースでは確率補正倍率（0.50〜1.60）。新馬戦では絶対好走評価スコア（0.50〜1.60）。"
    )
    短評: str = Field(
        description="なぜその数値を付与したかの物理的理由（調教ラップ、斤量差分など）"
    )

class RaceCoefficientResult(BaseModel):
    summary: str = Field(
        description="レース全体の物理バイアス（展開ペース、クラス特性、斤量傾向）と重要補正馬の統括見解（300文字以内）"
    )
    coefficients: List[HorseCoefficient] = Field(
        description="出走全頭分の補正係数リスト（全頭必須・欠落厳禁）"
    )

# ==========================================
# 🧠 システムプロンプト（クラス別物理バイアスのみの最高純度版）
# ==========================================
base_system_instruction = """
# 役割定義
あなたは統計モデル（Python）が算出したベース勝率に対し、直前の物理データから乗算補正倍率を付与する「精密レバレッジ鑑定AI」です。
印を決める必要はありません。通常レースではレース全体の平均倍率が【1.00前後】に収束するようメリハリをつけ、各馬の好走確率を何倍にすべきか【0.50 〜 1.60】の補正倍率を客観的根拠のみで精密に鑑定してください。

# 厳格な共通鑑定ルール
1. 【インフレ防止（メリハリの徹底）】: 通常レースにおいて全頭に1.1以上のプラス倍率を乱発してはならない。加点馬と同程度の減点馬（0.75〜0.90）を作り、強弱を明確にせよ。
2. 【調教減速馬への加点厳禁】: 調教分析に「減速」と記載がある馬には、いかなる理由があっても1.00を超える倍率を付与してはならない（必ず1.00以下とせよ）。
3. 【主観ノイズの遮断】: 記者コメントや主観的な血統への忖度は完全に排除し、Pythonが算出した「調教ラップ差分」「斤量差分」という物理事実のみを根拠とせよ。
4. 【出力形式と短評】: 出走全頭分を欠落なく出力し、短評には「なぜその倍率にしたのか」を明記せよ。
"""

shinba_addon = """
# 【適用評価モード：新馬戦（過去走なし）】
- Python勝率は完全に無視し、調教ラップの物理時計・加速度のみから、各馬の絶対好走評価値（0.50〜1.60）を単独鑑定せよ。
- 終いの加速が鋭い馬に1.40〜1.60を与え、仕上がり途上や減速調教の馬は0.50〜0.85として明確に差をつけよ。
"""

age2_addon = """
# 【適用評価モード：2歳戦（未勝利・1勝クラス）— 成長・一変重視】
2歳戦は馬の完成度が低く、1戦ごとの「急成長・一変」が結果のすべてを決める。以下の基準で倍率を配分せよ：
- 1.45 〜 1.60（一変・急成長レバレッジ）:
  調教分析に「🔥【極限加速】」を持ち、前走4〜9着（掲示板前後）から調教ラップの加速度が大きく良化している成長馬。または「🔥【着順詐欺判定】」を持ち調教が加速している馬。
- 1.25 〜 1.40（大幅プラス補正）:
  確定タグ（「🔥【極限加速】」「【減量恩恵】」）を持つ馬、または前走4〜8着で調教が前F比-0.5s以上鋭く加速している馬。
- 1.05 〜 1.20（標準プラス補正）: 調教が「加速」または「維持」で順調な馬（※上限1.30まで）。
- 0.95 〜 1.05（中立）: 平均的な内容の馬。
- 0.70 〜 0.90（マイナス補正）: 調教「減速」の馬。
"""

old_horse_addon = """
# 【適用評価モード：3歳以上（古馬戦・条件戦・特別・重賞）— 地力・CW切れ味・着順詐欺重視】
古馬戦はすでに基礎能力が固まっており、「鋭いギアチェンジ（極限加速）」と「前走の不利・着順詐欺」が最も馬券に直結する。以下の基準で倍率を配分せよ：
- 1.50 〜 1.60（特注・極限レバレッジ / 1レース最大0〜2頭まで）:
  調教分析に「🔥【極限加速】」を持ち、さらに「🔥【着順詐欺判定】」「休養明け（中10週以上）の好走歴」のいずれかが噛み合った馬。
- 1.35 〜 1.48（大幅プラス補正）:
  「🔥【極限加速】」「🔥【着順詐欺判定】」「【減量恩恵】」のいずれかの確定タグを持ち、上がり3Fの切れ味が鋭い馬（3〜6番人気帯の伏兵を積極的に拾い上げよ）。
- 1.10 〜 1.30（標準プラス補正）: 調教が「加速」または「維持」で順調な馬（※上限1.30まで）。
- 0.95 〜 1.05（中立）: Pythonのベース勝率通りに評価すべき平均的な馬。
- 0.75 〜 0.90（マイナス補正）: 調教分析で「減速」と判定されている馬。
- 0.50 〜 0.70（大幅マイナス補正）: 調教で明確な大失速（差:+0.6s以上の減速）、または59.0kg以上の酷量・前走3着以下からの急激な重量増（+3.0kg以上）で物理的負荷が限界を超えている馬。
"""

# ==========================================
# 🔧 前処理ヘルパー関数
# ==========================================
def parse_training_lap_stats(t_str):
    if not t_str or pd.isna(t_str) or str(t_str).strip() in ['-', 'nan', 'NaN', '']:
        return "調教データなし"
    
    s = str(t_str).strip()
    m_floats = [float(x) for x in re.findall(r'\d+\.?\d*', s)]
    if not m_floats:
        return "調教データなし"
    
    total_time = m_floats[0]
    final_1f = m_floats[-1]
    
    is_hanro = total_time < 70.0
    course_type = "坂路" if is_hanro else "ウッド/CW"
    
    two_f_total = None
    for v in reversed(m_floats[:-1]):
        if 21.0 <= v <= 30.0:
            two_f_total = v
            break
            
    if two_f_total and two_f_total > final_1f:
        penultimate_1f = round(two_f_total - final_1f, 1)
        diff = round(final_1f - penultimate_1f, 1)
        
        is_gear_change = False
        if is_hanro:
            if final_1f <= 12.1 and diff <= -0.5:
                is_gear_change = True
        else:
            if (final_1f <= 11.5 and diff <= -0.5) or (final_1f <= 11.9 and diff <= -1.8):
                is_gear_change = True
            
        gear_tag = " 🔥【極限加速】" if is_gear_change else ""
        status = f"加速({diff:+.1f}s)" if diff <= -0.2 else (f"維持({diff:+.1f}s)" if diff <= 0.3 else f"減速({diff:+.1f}s)")
        
        return f"[{course_type}] 終い:{final_1f:.1f}s (前F:{penultimate_1f:.1f}s, 差:{diff:+.1f}s, {status}){gear_tag}"
    else:
        return f"[{course_type}] 終い:{final_1f:.1f}s"

def parse_past_race_stats(order_str, passage_str, agari_str, agari_rank=None, past_tousuu=None, interval_str="", is_shinba=False, surface="芝"):
    if is_shinba or pd.isna(order_str) or str(order_str).strip() in ['-', 'nan', 'NaN', '']:
        return "前走データなし ⚠️【休養明け/初出走】"

    pass_clean = str(passage_str).strip()
    corner_last = None
    if pass_clean and pass_clean not in ['-', 'nan', 'NaN', '']:
        corners = re.findall(r'\d+', pass_clean)
        if corners:
            corner_last = int(corners[-1])

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
    is_top_agari = (rank_val is not None) and (1 <= rank_val <= 3)
    is_front = (corner_last is not None) and (corner_last <= 3)
    is_slow_agari = (rank_val is not None) and (rank_val >= 7)

    if is_rear and is_top_agari:
        tag = f" 🔥【着順詐欺判定：後方追込×上がり上位】"
    elif is_front and is_slow_agari:
        tag = f" ⚠️【前残り警戒：前受先行×上がり平凡】"

    interval_info = f"[{interval_str}] " if interval_str and str(interval_str) not in ['nan', 'NaN', ''] else ""
    return f"{interval_info}着順:{order_str} ({tousuu_text}通過:{passage_str} / 上がり3F:{agari_str}{rank_text}){tag}"

def format_weight_analysis(kinryo_val, min_w, max_w, avg_w, has_weight_gap, past1_w=None, past2_w=None, order_str=""):
    tags = []
    is_good_prev = str(order_str).strip() in ['1着', '2着']
    try:
        if past1_w and str(past1_w).strip() not in ["", "nan", "NaN"]:
            diff_p1 = round(kinryo_val - float(past1_w), 1)
            if diff_p1 <= -2.0:
                tags.append(f"【前走比 {diff_p1:+.1f}kg: 減量恩恵】")
            elif diff_p1 >= 3.0 and not is_good_prev:
                tags.append(f"【前走比 {diff_p1:+.1f}kg: 急な重量増に注意】")
            elif diff_p1 >= 3.0 and is_good_prev:
                tags.append(f"[前走比 {diff_p1:+.1f}kg: 好走による別定増]")
            else:
                tags.append(f"[前走比 {diff_p1:+.1f}kg]")
    except Exception:
        pass

    try:
        if past2_w and str(past2_w).strip() not in ["", "nan", "NaN"]:
            diff_p2 = round(kinryo_val - float(past2_w), 1)
            if diff_p2 == 0 and "前走比" in "".join(tags) and ("+" in "".join(tags)):
                tags.append("【好走時(前々走)の斤量に戻る】")
    except Exception:
        pass

    if not has_weight_gap:
        tags.append("(全頭同斤量)")
    else:
        diff_from_max = round(kinryo_val - max_w, 1)
        diff_from_avg = round(kinryo_val - avg_w, 1)
        if kinryo_val >= 59.0:
            tags.append(f"【59kg以上のトップハンデ: 平均比 {diff_from_avg:+.1f}kg】")
        elif kinryo_val == min_w and (max_w - min_w) >= 3.0:
            tags.append(f"【ハンデ戦の軽量馬: トップ比 {diff_from_max:+.1f}kg】")
        else:
            tags.append(f"(平均比 {diff_from_avg:+.1f}kg)")
            
    return f"斤量:{kinryo_val:.1f}kg " + " ".join(tags)

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
    elif "注" in mark: return "badge-chuu"
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
        st.error(f"⚠️ レースID '{target_key}' の計算済みデータが見つかりません。")
        return None, None, None, None, None

    horse_preds = predictions.get('horses', {})
    df['win_prob'] = df['馬名_clean'].map(lambda x: horse_preds.get(x, {}).get('win_prob', 0.1))
    df = df.sort_values(by=['win_prob'], ascending=False).reset_index(drop=True)
    
    min_p, max_p = df['win_prob'].min(), df['win_prob'].max()
    df['ai_score'] = (50 + (df['win_prob'] - min_p) / (max_p - min_p) * 100).round().astype(int) if max_p > min_p else 100

    mark_list = ['◎', '◯', '▲', '△', '☆', '☆']
    df['初期印'] = [mark_list[i] if i < len(mark_list) else '消' for i in range(len(df))]

    pat = predictions.get('pattern', '③ 混戦気配')
    rec_ticket = predictions.get('ticket', '3連複')
    
    def fmt_m(val):
        v_str = str(val)
        return "未定" if v_str.startswith("90") or int(float(v_str)) >= 90 else f"{int(float(v_str)):02d}"

    if len(df) >= 6:
        h1 = df.iloc[0]
        aite_str = ", ".join([f"{fmt_m(df.iloc[i]['馬番'])}({df.iloc[i]['馬名']})({df.iloc[i]['初期印']})" for i in range(1, 6)])
        buy_detail = f"軸: {fmt_m(h1['馬番'])}({h1['馬名']})(◎) -> 相手: {aite_str}"
    else:
        buy_detail = predictions.get('buy_detail', '')
        
    try: race_day = int(target_key[8:10])
    except: race_day = 1

    return df, pat, rec_ticket, buy_detail, race_day

def generate_base_table(disp_df, is_shinba=False):
    html = "<div class='table-container'><table class='kachi-table'>"
    init_prob_header = "Python参考値" if is_shinba else "Python勝率"
    init_mark_header = "初期印(参考外)" if is_shinba else "初期印"
    html += f"<tr><th>枠番</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>人気(前日)</th><th>オッズ</th><th>斤量</th><th>騎手</th><th>{init_prob_header}</th><th>{init_mark_header}</th></tr>"
    for _, r in disp_df.iterrows():
        w_num = str(r.get('枠番', ''))
        display_waku = w_num if w_num and w_num != 'nan' and not w_num.startswith("90") else "-"
        m_num = str(r['馬番'])
        display_umaban = "未定" if m_num.startswith("90") or int(float(m_num)) >= 90 else f"{int(float(m_num)):02d}"
        win_val = float(r.get('win_prob', 0)) * 100
        kinryo_val = float(r.get('斤量', 55.0))
        mark = r.get('初期印', '消')
        
        ninki_disp = int(float(r['人気'])) if pd.notna(r.get('人気')) else "-"
        odds_disp = f"{float(r['オッズ']):.1f}" if pd.notna(r.get('オッズ')) else "-"
        
        prob_disp = f"<span style='color:#7f8c8d;'>({win_val:.1f}%)</span>" if is_shinba else f"<b>{win_val:.1f}%</b>"
        mark_disp = f"<span class='badge-mark badge-keshi'>-</span>" if is_shinba else f"<span class='badge-mark {get_badge_class(mark)}'>{mark}</span>"

        html += f"<tr><td><b>{display_waku}</b></td><td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{ninki_disp}</td><td>{odds_disp}</td><td>{kinryo_val:.1f}kg</td><td>{r.get('騎手', '-')}</td><td>{prob_disp}</td>"
        html += f"<td>{mark_disp}</td></tr>"
    html += "</table></div>"
    return html

def generate_fusion_multiplied_table(merged_df, is_shinba=False):
    html = "<div class='table-container'><table class='kachi-table'>"
    if is_shinba:
        html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>人気(前日)</th><th>Gemini評価値</th><th>最終印</th><th style='text-align:left; min-width: 420px;'>Gemini単独評価根拠 (調教・物理バイアス)</th></tr>"
    else:
        html += "<tr><th>初期印</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>人気(前日)</th><th>元勝率</th><th>Gemini倍率</th><th>補正後勝率</th><th>最終印</th><th style='text-align:left; min-width: 420px;'>補正理由 (物理根拠のみ)</th></tr>"
        
    for _, r in merged_df.iterrows():
        m_num = str(r.get('馬番', ''))
        display_umaban = "未定" if m_num.startswith("90") or int(float(m_num)) >= 90 else f"{int(float(m_num)):02d}"
        
        f_mark = r.get('最終印', '消')
        b_mark = r.get('初期印', '消')
        w_base = float(r.get('win_prob', 0)) * 100
        w_final = float(r.get('final_prob', 0)) * 100
        mult = float(r.get('補正係数', 1.0))
        
        ninki_val = int(float(r['人気'])) if pd.notna(r.get('人気')) else 99
        odds_val = float(r['オッズ']) if pd.notna(r.get('オッズ')) else 99.9
        ninki_disp = f"{ninki_val} ({odds_val:.1f}倍)" if ninki_val != 99 else "-"
        
        if mult >= 1.2:
            mult_str = f"<span class='multiplier-up'>×{mult:.2f} ⬆</span>" if not is_shinba else f"<span class='multiplier-up'>★ {mult:.2f}</span>"
        elif mult <= 0.8:
            mult_str = f"<span class='multiplier-down'>×{mult:.2f} ⬇</span>" if not is_shinba else f"<span class='multiplier-down'>▼ {mult:.2f}</span>"
        else:
            mult_str = f"<span class='multiplier-flat'>×{mult:.2f}</span>" if not is_shinba else f"<span class='multiplier-flat'>- {mult:.2f}</span>"
            
        tanpyo = r.get('短評', '-')
            
        if is_shinba:
            html += f"<tr><td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
            html += f"<td>{ninki_disp}</td>"
            html += f"<td>{mult_str}</td>"
            html += f"<td><span class='badge-mark {get_badge_class(f_mark)}'>{f_mark}</span></td>"
            html += f"<td style='text-align:left; font-size:14px; white-space:normal; line-height:1.5; padding: 10px; color:#2c3e50;'>{tanpyo}</td></tr>"
        else:
            html += f"<tr><td><span class='badge-mark {get_badge_class(b_mark)}'>{b_mark}</span></td>"
            html += f"<td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
            html += f"<td>{ninki_disp}</td>"
            html += f"<td>{w_base:.1f}%</td><td>{mult_str}</td><td><b style='color:#e74c3c; font-size:17px;'>{w_final:.1f}%</b></td>"
            html += f"<td><span class='badge-mark {get_badge_class(f_mark)}'>{f_mark}</span></td>"
            html += f"<td style='text-align:left; font-size:14px; white-space:normal; line-height:1.5; padding: 10px; color:#2c3e50;'>{tanpyo}</td></tr>"
            
    html += "</table></div>"
    return html

def clear_all_caches():
    st.cache_data.clear()
    if hasattr(st, 'cache_resource'): st.cache_resource.clear()

# ==========================================
# 🎯 画面表示
# ==========================================
st.sidebar.button("🔄 画面リロード", on_click=clear_all_caches, use_container_width=True)

col_logo, col_main = st.columns([1, 4])
if os.path.exists("logo.png"):
    with col_logo: st.image("logo.png", use_container_width=True)

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

if st.session_state.get('selected_race_id'):
    t_id = str(st.session_state['selected_race_id'])
    t_rows = df_future[df_future['race_id'].astype(str) == t_id]
    if t_rows.empty: st.stop()

    r_info = t_rows.iloc[0]
    st.markdown("---")
    
    display_race_name = r_info.get('race_name', '')
    if pd.isna(display_race_name) or display_race_name == 'nan': display_race_name = ''
    surface_info = r_info.get('surface', '不明')
    distance_info = r_info.get('distance', '')
    
    res_df, pat, rec_ticket, buy_detail, race_day = render_race_predictions(t_id)
    
    is_shinba = "新馬" in str(display_race_name) or (res_df['前走着順'].isna().all() if res_df is not None and '前走着順' in res_df.columns else False)
    is_age2 = (not is_shinba) and ("2歳" in str(display_race_name) or (res_df['age'].max() == 2 if res_df is not None and 'age' in res_df.columns else False))

    if is_shinba:
        mode_badge_html = "<span class='shinba-badge'>🐴 新馬戦 (Gemini単独評価)</span>"
        active_system_instruction = base_system_instruction + "\n" + shinba_addon
        mode_label = "新馬戦モード"
    elif is_age2:
        mode_badge_html = "<span class='age2-badge'>🌱 2歳戦モード (成長・一変重視)</span>"
        active_system_instruction = base_system_instruction + "\n" + age2_addon
        mode_label = "2歳戦モード"
    else:
        mode_badge_html = "<span class='old-badge'>🔥 3歳以上・古馬戦モード (CW切れ味・着順詐欺重視)</span>"
        active_system_instruction = base_system_instruction + "\n" + old_horse_addon
        mode_label = "3歳以上・古馬戦モード"


    st.markdown(f"<h2>🚀 {r_info['place_name']} {r_info['r_num']}R 【{display_race_name}】 {mode_badge_html} <span style='font-size:18px; color:#7f8c8d;'>({surface_info}{distance_info}m)</span></h2>", unsafe_allow_html=True)
    
    if res_df is not None:
        if is_shinba:
            st.info("💡【新馬戦モード】過去走データが存在しないため、Pythonの初期勝率は無効化されています。")
            st.markdown("<div class='section-header'>📊 1. 出走馬一覧（過去走データなし）</div>", unsafe_allow_html=True)
            st.markdown(generate_base_table(res_df, is_shinba=True), unsafe_allow_html=True)
        else:
            st.markdown("<div class='section-header'>📊 1. Pythonベース地力スコア</div>", unsafe_allow_html=True)
            st.markdown(generate_base_table(res_df, is_shinba=False), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        btn_label = f"⚡ クラス別物理バイアスのみで精密レバレッジ補正を実行"
        if st.button(btn_label, type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】GEMINI_API_KEYが設定されていません。")
                st.stop()

            total_horses = len(res_df)
            res_df['chokyo_summary'] = res_df['調教タイム'].apply(parse_training_lap_stats)

            table_summary = []
            for i, (_, r) in enumerate(res_df.iterrows()):
                m_str = str(r['馬番'])
                display_m = "未定" if m_str.startswith("90") else f"{int(float(m_str)):02d}"
                win_val = float(r.get('win_prob', 0)) * 100
                kinryo = float(r.get('斤量', 55.0))
                jockey_val = str(r.get('騎手', '-'))
                kyakushitsu_val = str(r.get('推測脚質', '-'))
                chokyo_summary = r['chokyo_summary']
                
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
                
                python_prob_str = f"Python勝率:(新馬戦のため無効)" if is_shinba else f"Python勝率:{win_val:.1f}%"
                
                # 純粋な物理情報のみ渡す (血統情報は渡さない)
                info = (f"馬番:{display_m} | 馬名:{r.get('馬名','')} | {python_prob_str} | 斤量:{kinryo}kg | "
                        f"脚質:{kyakushitsu_val} | 騎手:{jockey_val} | "
                        f"前走:{past_summary} | 調教分析:{chokyo_summary}")
                table_summary.append(info)

            prompt = f"""
以下の出走馬データに基づき、指示に従って全出走馬の補正係数（0.50〜1.60）と理由を出力してください。

【対象レース】: {r_info['place_name']} {r_info['r_num']}R 【{display_race_name}】 ({surface_info}{distance_info}m)
【出走頭数】: {total_horses}頭

【出走馬データ一覧】
{chr(10).join(table_summary)}
"""

            with st.spinner(f"🧠 Geminiが【{mode_label}】の物理バイアスのみで精密鑑定中..."):
                client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1beta"})
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=prompt, 
                        config=types.GenerateContentConfig(
                            system_instruction=active_system_instruction, 
                            temperature=0.1, 
                            response_mime_type="application/json",
                            response_schema=RaceCoefficientResult,
                        )
                    )
                    
                    result = RaceCoefficientResult.model_validate_json(response.text)
                    coeff_df = pd.DataFrame([c.model_dump() for c in result.coefficients])
                    
                    if not coeff_df.empty and '馬名' in coeff_df.columns:
                        coeff_df['馬名_clean'] = coeff_df['馬名'].astype(str).apply(clean_name)
                        coeff_df_clean = coeff_df[['馬名_clean', '補正係数', '短評']].dropna(subset=['馬名_clean'])
                        
                        merged = pd.merge(res_df, coeff_df_clean, on='馬名_clean', how='left')
                        merged['補正係数'] = merged['補正係数'].fillna(1.0)
                        merged['短評'] = merged['短評'].fillna('データ保留')
                        
                        # 防衛的プログラミング（上限・下限クリップ）
                        clipped_coeffs = []
                        for _, r in merged.iterrows():
                            mult = float(r.get('補正係数', 1.0))
                            c_tag = str(r.get('chokyo_summary', ''))
                            if not is_shinba and "極限加速" not in c_tag and mult > 1.40: mult = 1.40
                            if "減速" in c_tag and mult > 1.00: mult = 1.00
                            clipped_coeffs.append(mult)
                        merged['補正係数'] = clipped_coeffs

                        if is_shinba:
                            merged['raw_final'] = merged['補正係数']
                            merged['final_prob'] = merged['raw_final'] / merged['raw_final'].sum()
                            merged = merged.sort_values(by='final_prob', ascending=False).reset_index(drop=True)
                        else:
                            merged['raw_final'] = merged['win_prob'] * merged['補正係数']
                            merged['final_prob'] = merged['raw_final'] / merged['raw_final'].sum()
                            merged = merged.sort_values(by='final_prob', ascending=False).reset_index(drop=True)
                        
                        # 印の付与
                        marks = ['◎', '◯', '▲', '△', '☆']
                        final_marks = ['消'] * len(merged)
                        for idx in range(min(len(marks), len(merged))):
                            final_marks[idx] = marks[idx]

                        # 注の付与（極限加速 or 1.50以上）
                        chuu_candidates = []
                        for idx in range(len(marks), len(merged)):
                            row_m = merged.iloc[idx]
                            mult = float(row_m.get('補正係数', 1.0))
                            c_tag = str(row_m.get('chokyo_summary', ''))
                            if ("極限加速" in c_tag) or (mult >= 1.50):
                                chuu_candidates.append((idx, mult))
                        chuu_candidates.sort(key=lambda x: -x[1])
                        for candidate in chuu_candidates[:2]:
                            final_marks[candidate[0]] = '注'
                        merged['最終印'] = final_marks
                        
                        st.markdown(f"""
                        <div class='gemini-view-card'>
                            <div class='gemini-view-title'>📝 トラックマンGeminiのレース見解 ({mode_label})</div>
                            <p class='gemini-view-text'>{result.summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        h_o = merged.iloc[0]
                        h_m = merged.iloc[1]
                        
                        chuu_horses = merged[merged['最終印'] == '注']
                        chuu_html = ""
                        if not chuu_horses.empty:
                            chuu_list_text = [f"<b>【馬番{str(ch.get('馬番',''))} {ch['馬名']}】</b>(倍率 ×{ch['補正係数']:.2f}) : {ch.get('短評', '-')}" for _, ch in chuu_horses.iterrows()]
                            chuu_html = f"""
                            <div style='margin-top: 12px; padding: 10px 14px; background-color: #fff9e6; border-left: 4px solid #f39c12; border-radius: 4px;'>
                                <div style='font-weight: bold; color: #d35400; font-size: 15px;'>🔥【仕上がり超抜の特注伏兵（注）】</div>
                                <div style='font-size: 14px; color: #444; margin-top: 4px; line-height: 1.6;'>{'<br>'.join(chuu_list_text)}</div>
                            </div>
                            """
                            aite_label = "◯, ▲, △, ☆, 注"
                            ticket_count = f"計{6 + len(chuu_horses)*4 - (1 if len(chuu_horses)==2 else 0)}点"
                        else:
                            aite_label = "◯, ▲, △, ☆"
                            ticket_count = "計6点"

                        # 余計な条件分岐バッジを全て削除し、純粋に2パターンの買い目を併記するだけにする
                        st.markdown("<div class='sense-card'>", unsafe_allow_html=True)
                        st.markdown("<div class='sense-title'>🎟 当日の馬場に応じた推奨買い目</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div style='margin-bottom: 12px; font-size: 15px; color: #333;'>
                            当日の実際の天候・馬場に合わせて、以下のどちらかの買い目を選択してください。
                        </div>
                        <ul style='margin-top: 5px; margin-bottom: 10px; font-size: 15px; color: #333;'>
                            <li><span class='strategy-badge-good' style='margin-bottom: 4px;'>☀️【良馬場なら】</span> <b>三連複 ◎1頭軸流し</b> ➔ 相手: {aite_label} ({ticket_count})</li>
                            <li style='margin-top: 10px;'><span class='strategy-badge-wet' style='margin-bottom: 4px;'>🌧【稍重・重・不良なら】</span> <b>◎ {h_o['馬名']}</b> の単勝500円 ＋ 複勝500円（計1000円）</li>
                        </ul>
                        """, unsafe_allow_html=True)
                            
                        st.markdown(f"""
                            <div style='margin-top: 15px; border-top: 1px dashed #ccc; padding-top: 10px;'>
                                <b>【本線単勝・馬連】</b>: ◎ {h_o['馬名']} (推定勝率 {h_o['final_prob']*100:.1f}%) ⇄ ◯ {h_m['馬名']}
                            </div>
                            {chuu_html}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"<div class='section-header'>🔥 2. 数学融合結果（Python地力 × Gemini倍率）</div>", unsafe_allow_html=True)
                        st.markdown(generate_fusion_multiplied_table(merged, is_shinba=is_shinba), unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ API通信・計算エラー: {e}")
