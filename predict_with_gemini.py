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
st.set_page_config(page_title="AI予想 勝ちぱかくん (乗算補正版)", page_icon="logo.png", layout="wide")

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

    .sense-card { background-color: #ffffff; border-left: 6px solid #8e44ad; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .sense-title { font-size: 1.2rem; font-weight: bold; color: #8e44ad; }
    .ticket-badge { font-size: 1.1rem; font-weight: bold; color: #d35400; background: #fef5e7; padding: 4px 10px; border-radius: 4px; display: inline-block; }
    
    .gemini-view-card { background-color: #f0f8ff; border-left: 5px solid #3498db; padding: 16px 20px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .gemini-view-title { color: #2980b9; font-size: 1.2rem; font-weight: bold; margin-bottom: 8px; }
    .gemini-view-text { font-size: 15px; color: #333; line-height: 1.6; margin: 0; }
    
    .shinba-badge { background-color: #e67e22; color: white; padding: 3px 8px; border-radius: 4px; font-size: 13px; font-weight: bold; margin-left: 8px; vertical-align: middle; }

    div.stButton > button { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); color: white; border: none; border-radius: 6px; padding: 10px 24px; font-weight: bold; font-size: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; }
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
        description="好走確率評価。通常レースでは確率補正倍率（0.5〜1.6）。新馬戦ではPython勝率を参照せず調教・血統による絶対好走評価スコア（0.5〜1.6）。"
    )
    短評: str = Field(
        description="なぜその数値を付与したかの物理的・定性的理由（調教ラップ、斤量差分、血統適性、コース脚質適合度など）"
    )

class RaceCoefficientResult(BaseModel):
    summary: str = Field(
        description="レース全体の物理バイアス（展開ペース、馬場、斤量・血統傾向）と重要補正馬の統括見解（300文字以内）"
    )
    coefficients: List[HorseCoefficient] = Field(
        description="出走全頭分の補正係数リスト（全頭必須・欠落厳禁）"
    )

# ==========================================
# 🧠 システムプロンプト（乗算レバレッジ完全仕様＋馬場＋新馬単独ルール）
# ==========================================
system_instruction = """
# 役割定義
あなたは統計モデル（Python）が算出したベース確率に対し、直前定性データから乗算補正倍率を付与する「精密レバレッジ鑑定AI」です。
印を決める必要はありません。各馬の好走確率を何倍にすべきか、客観的根拠から【0.5 〜 1.6】の補正倍率を精密に鑑定してください。
※ただし【新馬戦】と指定されたレースでは、Pythonの過去データ勝率は完全に無意味であるため、調教ラップの物理時計と血統コース適性・厩舎仕上げのみから、各馬の絶対的な「好走評価値（0.5〜1.6）」を単独鑑定せよ。

# 係数付与の基準（0.5 〜 1.6）
- 1.4 〜 1.6（極大プラス補正 / 新馬戦最上位）:
  【好走休養明け×究極仕上がり（激走穴馬特注ルール）】:
  「前走1〜2着」かつ「レース間隔中10週以上の休養明け」で、調教分析に「🔥【極限加速】」またはラスト鋭い加速（坂路12.1秒以下/ウッド11.5秒以下）を記録している馬。
  目標レースに向けて極限まで体調が仕上げられた物理的証明（勝負気配の客観的裏付け）であるため、過去の実績不足やPython初期勝率が低くても積極的に【1.35〜1.50倍】のレバレッジをかけて浮上させよ。
  また【明確な斤量恩恵（ハンデ戦等で他馬より2kg〜4kg以上軽い物理的有利）】や【前走からの大幅斤量減（-2kg〜-4kg）】のある馬。
  また【血統適性の爆発】（新馬戦・ダート替わり・芝1400mの先行シルバーステートなど、コース条件にドンピシャで合致する馬。新馬戦で調教終い加速×コース特注血統の馬はここに属する）。

- 1.1 〜 1.3（微小プラス補正）:
  調教が加速・維持で順調な動きを見せている馬。コース脚質適性に合致する馬。
  または【明確な鞍上強化】により、前走の物理的敗因が今回解消される見込みが立つ馬。

- 1.0（中立・補正なし）:
  可もなく不可もない状態。データ通りの走りが期待できる馬。
  ※【58kgに関する絶対ルール】: 古馬・牡馬における「55.0kg〜58.0kg」は日本の競馬における【完全に標準的な規定重量】である。この範囲の斤量を背負っていること「だけ」を理由にマイナス補正をかけることは【完全厳禁】とする。

- 0.8 〜 0.9（微小マイナス補正）:
  調教タイムの前Fとの落差が物足りない馬。または血統適性に疑問がある馬。

- 0.5 〜 0.7（大幅マイナス補正）:
  調教分析で明確な「大失速（前Fより0.6秒以上の失速、またはバテている記述）」がある明らかな状態不安馬。
  または、小回り・短距離戦で物理的に届かない極端な後方脚質（追込）の馬。
  または【物理的限界を超えた酷量】（ハンデ戦で「59.0kg〜60.0kg超」を背負う、あるいは「前走から+3.0kg以上の急激な重量増」で物理的負荷が限界を超えている馬）。

# 厳格な鑑定ルール
1. 【調教減速馬への加点厳禁】: 調教分析に「減速」と明記されている馬に対し、調教を理由として1.0を超えるプラス補正を付与することは【完全厳禁】とする（現状維持1.0、または0.8〜0.95のマイナス補正とせよ）。
2. 【特大レバレッジ（>1.30）の発動条件】: 1.30倍を超える極大補正は、調教の「🔥【極限加速】」または斤量の「【減量恩恵】」などの確定タグが存在する馬にのみ厳格に限定せよ（新馬戦で調教終い鋭く加速している抜けた仕上がりの馬を含む）。
3. 【記者コメント・主観マークの完全無視】: 新聞記者の主観ラベルは信用せず、Python側で算出された「調教分析（終い時計・前Fとの差分秒数）」という客観的な物理タイムのみを信じよ。
4. 【性齢の規定重量への配慮】: 牡馬（58kg）と牝馬（56kg）の差は性別の規定重量差であり、59kg以上の異常な酷量でない限り58kgを嫌って格下げしてはならない。
5. 【短評に物理根拠を明記】: なぜその係数を付与したのか（特に「好走休養×極限加速」「斤量減量」「血統コース適合」など）を短評に明確に記述せよ。
6. 出走全頭に対して必ず1つずつ係数と短評を出力せよ。

# 🌧 馬場状態の適性補正（追加ルール）
7. 入力情報で指定される【当日の馬場状態想定（良・稍重・重・不良）】を必ず考慮し、馬場バイアス（時計のかかる馬場への血統適性、道悪巧者、荒れ馬場適性など）を係数および統括見解（summary）に反映させよ。
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
        if is_hanro and final_1f <= 12.1 and diff <= -0.5:
            is_gear_change = True
        elif (not is_hanro) and final_1f <= 11.5 and diff <= -0.5:
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
    is_top_agari = (rank_val is not None) and (rank_val <= 3)
    is_front = (corner_last is not None) and (corner_last <= 3)
    is_slow_agari = (rank_val is not None) and (rank_val >= 7)

    if is_rear and is_top_agari:
        tag = f" 🔥【着順詐欺判定：後方追込×上がり上位】"
    elif is_front and is_slow_agari:
        tag = f" ⚠️【前残り警戒：前受先行×上がり平凡】"

    interval_info = f"[{interval_str}] " if interval_str and str(interval_str) not in ['nan', 'NaN', ''] else ""
    return f"{interval_info}着順:{order_str} ({tousuu_text}通過:{passage_str} / 上がり3F:{agari_str}{rank_text}){tag}"

def format_weight_analysis(kinryo_val, min_w, max_w, avg_w, has_weight_gap, past1_w=None, past2_w=None):
    tags = []
    try:
        if past1_w and str(past1_w).strip() not in ["", "nan", "NaN"]:
            diff_p1 = round(kinryo_val - float(past1_w), 1)
            if diff_p1 <= -2.0:
                tags.append(f"【前走比 {diff_p1:+.1f}kg: 減量恩恵】")
            elif diff_p1 >= 3.0:
                tags.append(f"【前走比 {diff_p1:+.1f}kg: 急な重量増に注意】")
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
    html += f"<tr><th>枠番</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>脚質</th><th>斤量</th><th>騎手</th><th>{init_prob_header}</th><th>{init_mark_header}</th></tr>"
    for _, r in disp_df.iterrows():
        w_num = str(r.get('枠番', ''))
        display_waku = w_num if w_num and w_num != 'nan' and not w_num.startswith("90") else "-"
        m_num = str(r['馬番'])
        display_umaban = "未定" if m_num.startswith("90") or int(float(m_num)) >= 90 else f"{int(float(m_num)):02d}"
        win_val = float(r.get('win_prob', 0)) * 100
        kinryo_val = float(r.get('斤量', 55.0))
        mark = r.get('初期印', '消')
        style_raw = str(r.get('推測脚質', '不明')).strip()
        
        prob_disp = f"<span style='color:#7f8c8d;'>({win_val:.1f}%)</span>" if is_shinba else f"<b>{win_val:.1f}%</b>"
        mark_disp = f"<span class='badge-mark badge-keshi'>-</span>" if is_shinba else f"<span class='badge-mark {get_badge_class(mark)}'>{mark}</span>"

        html += f"<tr><td><b>{display_waku}</b></td><td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{style_raw}</td><td>{kinryo_val:.1f}kg</td><td>{r.get('騎手', '-')}</td><td>{prob_disp}</td>"
        html += f"<td>{mark_disp}</td></tr>"
    html += "</table></div>"
    return html

def generate_fusion_multiplied_table(merged_df, is_shinba=False):
    html = "<div class='table-container'><table class='kachi-table'>"
    if is_shinba:
        html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>Gemini評価値</th><th>最終勝率</th><th>最終印</th><th style='text-align:left; min-width: 420px;'>Gemini単独評価根拠 (調教・血統・厩舎バイアス)</th></tr>"
    else:
        html += "<tr><th>初期印</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>元勝率</th><th>Gemini倍率</th><th>補正後勝率</th><th>最終印</th><th style='text-align:left; min-width: 420px;'>補正理由 (物理・定性根拠)</th></tr>"
        
    for _, r in merged_df.iterrows():
        m_num = str(r.get('馬番', ''))
        display_umaban = "未定" if m_num.startswith("90") or int(float(m_num)) >= 90 else f"{int(float(m_num)):02d}"
        
        f_mark = r.get('最終印', '消')
        b_mark = r.get('初期印', '消')
        w_base = float(r.get('win_prob', 0)) * 100
        w_final = float(r.get('final_prob', 0)) * 100
        mult = float(r.get('補正係数', 1.0))
        
        if mult >= 1.2:
            mult_str = f"<span class='multiplier-up'>×{mult:.2f} ⬆</span>" if not is_shinba else f"<span class='multiplier-up'>★ {mult:.2f}</span>"
        elif mult <= 0.8:
            mult_str = f"<span class='multiplier-down'>×{mult:.2f} ⬇</span>" if not is_shinba else f"<span class='multiplier-down'>▼ {mult:.2f}</span>"
        else:
            mult_str = f"<span class='multiplier-flat'>×{mult:.2f}</span>" if not is_shinba else f"<span class='multiplier-flat'>- {mult:.2f}</span>"
            
        tanpyo = r.get('短評', '-')
            
        if is_shinba:
            html += f"<tr><td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
            html += f"<td>{mult_str}</td><td><b style='color:#e74c3c; font-size:17px;'>{w_final:.1f}%</b></td>"
            html += f"<td><span class='badge-mark {get_badge_class(f_mark)}'>{f_mark}</span></td>"
            html += f"<td style='text-align:left; font-size:14px; white-space:normal; line-height:1.5; padding: 10px; color:#2c3e50;'>{tanpyo}</td></tr>"
        else:
            html += f"<tr><td><span class='badge-mark {get_badge_class(b_mark)}'>{b_mark}</span></td>"
            html += f"<td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
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
    
    # 💡 新馬戦の判定（レース名または全頭前走欠落）
    is_shinba = "新馬" in str(display_race_name) or (res_df['前走着順'].isna().all() if res_df is not None and '前走着順' in res_df.columns else False)
    shinba_badge_html = "<span class='shinba-badge'>🐴 新馬戦 (Gemini単独評価対象)</span>" if is_shinba else ""

    st.markdown(f"<h2>🚀 {r_info['place_name']} {r_info['r_num']}R 【{display_race_name}】 {shinba_badge_html} <span style='font-size:18px; color:#7f8c8d;'>({surface_info}{distance_info}m / 開催{race_day}日目)</span></h2>", unsafe_allow_html=True)
    
    if res_df is not None:
        if is_shinba:
            st.info("💡【新馬戦モード】過去走データが存在しないため、Pythonの初期勝率・買い目は無効化されています。下のGemini補正を実行することで、調教時計と血統からGeminiが単独で印（◎〜☆）を直接決定します。")
            st.markdown("<div class='section-header'>📊 1. 出走馬一覧（過去走データなし）</div>", unsafe_allow_html=True)
            st.markdown(generate_base_table(res_df, is_shinba=True), unsafe_allow_html=True)
        else:
            st.markdown("<div class='section-header'>📊 1. Pythonベース地力スコア</div>", unsafe_allow_html=True)
            st.markdown(generate_base_table(res_df, is_shinba=False), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 🌧 馬場状態の選択UI
        st.markdown("##### 🌧 当日の馬場状態想定")
        baba_condition = st.radio(
            "馬場状態",
            ["良", "稍重", "重", "不良"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        btn_label = "⚡ Geminiによる単独定性評価を実行（新馬戦：調教×血統モデル）" if is_shinba else "⚡ Geminiによる動的レバレッジ補正を実行（血統・斤量差分モデル）"
        if st.button(btn_label, type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】GEMINI_API_KEYが設定されていません。")
                st.stop()

            total_horses = len(res_df)

            weights = res_df['斤量'].astype(float)
            min_w = float(weights.min())
            max_w = float(weights.max())
            avg_w = float(weights.mean())
            has_weight_gap = (max_w - min_w) >= 0.5

            res_df['chokyo_summary'] = res_df['調教タイム'].apply(parse_training_lap_stats)
            res_df['weight_info'] = [
                format_weight_analysis(float(r.get('斤量', 55.0)), min_w, max_w, avg_w, has_weight_gap, r.get('前走斤量'), r.get('前々走斤量'))
                for _, r in res_df.iterrows()
            ]

            table_summary = []
            for i, (_, r) in enumerate(res_df.iterrows()):
                m_str = str(r['馬番'])
                display_m = "未定" if m_str.startswith("90") else f"{int(float(m_str)):02d}"
                win_val = float(r.get('win_prob', 0)) * 100
                weight_info = r['weight_info']
                jockey_val = str(r.get('騎手', '-'))
                kyakushitsu_val = str(r.get('推測脚質', '-'))
                sire = str(r.get('父', '不明'))
                bms = str(r.get('母父', '不明'))
                blood_info = f"血統: 父[{sire}] 母父[{bms}]"
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
                info = (f"馬番:{display_m} | 馬名:{r.get('馬名','')} | {python_prob_str} | {weight_info} | "
                        f"脚質:{kyakushitsu_val} | 騎手:{jockey_val} | {blood_info} | "
                        f"前走:{past_summary} | 調教分析:{chokyo_summary}")
                table_summary.append(info)

            # 新馬戦専用プロンプト指示の分岐アドオン
            shinba_prompt_instruction = """
【重要：新馬戦特注評価モード】
このレースは【新馬戦】です。出走全頭に過去走データが一切なく、Python統計モデルの勝率は適当（無効）です。
Pythonの勝率は完全に無視し、あなたの調教分析（終い時計・前F差の物理加速）と血統コース適性、および当日の馬場状態のみから、各馬の絶対好走スコア（0.50〜1.60）を単独鑑定して全頭出力せよ。
最も仕上がりが抜けており血統適性がある馬に1.40〜1.60を与え、仕上がり途上や減速調教の馬は0.50〜0.90として差を明確につけよ。
""" if is_shinba else ""

            prompt = f"""
以下の出走馬データに基づき、指示に従って全出走馬の補正係数（0.5〜1.6）と理由を出力してください。
{shinba_prompt_instruction}
【対象レース】: {r_info['place_name']} {r_info['r_num']}R ({surface_info}{distance_info}m)
【当日の馬場状態想定】: {baba_condition}
【出走頭数】: {total_horses}頭
【斤量条件】: {"【全頭同斤量（斤量差なし）】" if not has_weight_gap else f"【斤量差あり（ハンデ/別定）】最軽量:{min_w:.1f}kg 〜 最重量:{max_w:.1f}kg (最大格差:{max_w-min_w:.1f}kg)"}

【出走馬データ一覧】
{chr(10).join(table_summary)}
"""

            with st.spinner("🧠 Geminiが調教・血統・馬場想定データから各馬を精密鑑定中..."):
                client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1beta"})
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=prompt, 
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction, 
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
                        
                        # 🛡️ Pythonバックエンド防衛的プログラミング
                        clipped_coeffs = []
                        for _, r in merged.iterrows():
                            mult = float(r.get('補正係数', 1.0))
                            c_tag = str(r.get('chokyo_summary', ''))
                            w_tag = str(r.get('weight_info', ''))
                            
                            kinryo_f = float(r.get('斤量', 55.0))
                            p1_w = r.get('前走斤量')
                            is_num_weight_bonus = False
                            if pd.notna(p1_w) and str(p1_w).strip() not in ['', 'nan', 'NaN']:
                                try:
                                    if round(kinryo_f - float(p1_w), 1) <= -2.0:
                                        is_num_weight_bonus = True
                                except Exception:
                                    pass
                            
                            has_special_flag = ("極限加速" in c_tag) or ("減量恩恵" in w_tag) or is_num_weight_bonus
                            is_gensoku = "減速" in c_tag
                            
                            # 新馬戦では調教終い加速の馬も特大評価を許容、通常戦はタグ必須
                            if not is_shinba and not has_special_flag and mult > 1.30:
                                mult = 1.30
                            # ガードレール2: 調教減速馬の不当加点は新馬戦でも完全カット
                            if is_gensoku and mult > 1.00:
                                mult = 1.00
                                
                            clipped_coeffs.append(mult)
                        merged['補正係数'] = clipped_coeffs

                        # 🎯 新馬戦と通常戦で勝率計算・ソートを分岐
                        if is_shinba:
                            # 🐴 新馬戦：Python勝率は完全無視し、Geminiのスコア単体で正規化してソート
                            merged['raw_final'] = merged['補正係数']
                            merged['final_prob'] = merged['raw_final'] / merged['raw_final'].sum()
                            merged = merged.sort_values(by='final_prob', ascending=False).reset_index(drop=True)
                        else:
                            # 🏁 通常戦：Python地力 × Gemini倍率による乗算融合
                            merged['raw_final'] = merged['win_prob'] * merged['補正係数']
                            merged['final_prob'] = merged['raw_final'] / merged['raw_final'].sum()
                            merged = merged.sort_values(by='final_prob', ascending=False).reset_index(drop=True)
                        
                        # ==========================================
                        # 🎯 最終印の決定ロジック（本線5頭 ＋ 圏外極限値・注最大2頭）
                        # ==========================================
                        marks = ['◎', '◯', '▲', '△', '☆']
                        final_marks = ['消'] * len(merged)

                        # 1. 補正後勝率の上位5頭に本線印を付与
                        for idx in range(min(len(marks), len(merged))):
                            final_marks[idx] = marks[idx]

                        # 2. 6番手以降（圏外）から、倍率1.55以上の真の極限仕上がり馬を抽出
                        chuu_candidates = []
                        for idx in range(len(marks), len(merged)):
                            row_m = merged.iloc[idx]
                            mult = float(row_m.get('補正係数', 1.0))
                            if mult >= 1.55:
                                t_str = str(row_m.get('調教タイム', ''))
                                floats = [float(x) for x in re.findall(r'\d+\.?\d*', t_str)]
                                final_1f = floats[-1] if floats else 99.0
                                chuu_candidates.append((idx, mult, final_1f))

                        # 3. 補正倍率（降順）➔ 調教終い時計（速い順）で厳格ソート
                        chuu_candidates.sort(key=lambda x: (-x[1], x[2]))

                        # 4. 最大2頭までを「注」に昇格
                        for candidate in chuu_candidates[:2]:
                            final_marks[candidate[0]] = '注'

                        merged['最終印'] = final_marks
                        
                        st.markdown(f"""
                        <div class='gemini-view-card'>
                            <div class='gemini-view-title'>📝 トラックマンGeminiのレース物理バイアス統括 {"(新馬戦・調教血統単独評価)" if is_shinba else ""}</div>
                            <p class='gemini-view-text'>{result.summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # ==========================================
                        # 🎟 推奨買い目の動的生成
                        # ==========================================
                        h_o = merged.iloc[0]
                        h_m = merged.iloc[1]

                        chuu_horses = merged[merged['最終印'] == '注']

                        chuu_html = ""
                        if not chuu_horses.empty:
                            chuu_list = []
                            medals = ['🥇', '🥈']
                            for idx_c, (_, ch) in enumerate(chuu_horses.iterrows()):
                                m_str = str(ch.get('馬番', ''))
                                umaban = f"{int(float(m_str)):02d}" if m_str and not m_str.startswith('90') else "-"
                                medal = medals[idx_c] if idx_c < len(medals) else "・"
                                score_label = f"スコア {ch['補正係数']:.2f}" if is_shinba else f"倍率 ×{ch['補正係数']:.2f}"
                                chuu_list.append(f"<b>{medal}【馬番{umaban} {ch['馬名']}】</b>({score_label}) : {ch.get('短評', '-')}")
                            
                            chuu_html = f"""
                            <div style='margin-top: 12px; padding: 10px 14px; background-color: #fff9e6; border-left: 4px solid #f39c12; border-radius: 4px;'>
                                <div style='font-weight: bold; color: #d35400; font-size: 15px;'>🔥【仕上がり超抜の特注伏兵（注）】※物理極限値を突破した厳選穴馬</div>
                                <div style='font-size: 14px; color: #444; margin-top: 4px; line-height: 1.6;'>{'<br>'.join(chuu_list)}</div>
                            </div>
                            """
                            aite_label = "◯, ▲, △, ☆, 注"
                            ticket_count_label = f"計{6 + len(chuu_horses)*4 - (1 if len(chuu_horses)==2 else 0)}点（勝負拡大）"
                        else:
                            aite_label = "◯, ▲, △, ☆"
                            ticket_count_label = "計6点（標準）"

                        title_prefix = "🎟 Gemini単独定性評価による新馬戦推奨買い目" if is_shinba else "🎟 乗算補正後の最終推奨買い目"
                        st.markdown(f"""
                        <div class='sense-card'>
                            <div class='sense-title'>{title_prefix}</div>
                            <div style='margin-top: 6px;'><b>【本線単勝・馬連】</b>: ◎ {h_o['馬名']} (推定勝率 {h_o['final_prob']*100:.1f}%) ⇄ ◯ {h_m['馬名']}</div>
                            <div style='margin-top: 4px;'><b>【3連複フォーメーション】</b>: ◎ 1頭軸 ➔ 相手: {aite_label} <b>({ticket_count_label})</b></div>
                            {chuu_html}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        section2_header = "🔥 2. Gemini単独評価結果（調教ラップ×血統適性ソート）" if is_shinba else "🔥 2. 数学融合結果（Python地力 × Gemini倍率）"
                        st.markdown(f"<div class='section-header'>{section2_header}</div>", unsafe_allow_html=True)
                        st.markdown(generate_fusion_multiplied_table(merged, is_shinba=is_shinba), unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ API通信・計算エラー: {e}")