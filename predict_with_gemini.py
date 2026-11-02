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

st.markdown('''
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
    .badge-ana { background: #9b59b6; } .badge-keshi { background: #e0e0e0; color: #7f8c8d; }
    
    .multiplier-up { color: #e74c3c; font-weight: bold; }
    .multiplier-down { color: #3498db; font-weight: bold; }
    .multiplier-flat { color: #7f8c8d; }

    .sense-card { background-color: #ffffff; border-left: 6px solid #8e44ad; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .sense-title { font-size: 1.2rem; font-weight: bold; color: #8e44ad; }
    .ticket-badge { font-size: 1.1rem; font-weight: bold; color: #d35400; background: #fef5e7; padding: 4px 10px; border-radius: 4px; display: inline-block; }
    
    .gemini-view-card { background-color: #f0f8ff; border-left: 5px solid #3498db; padding: 16px 20px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .gemini-view-title { color: #2980b9; font-size: 1.2rem; font-weight: bold; margin-bottom: 8px; }
    .gemini-view-text { font-size: 15px; color: #333; line-height: 1.6; margin: 0; }
    
    div.stButton > button { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); color: white; border: none; border-radius: 6px; padding: 10px 24px; font-weight: bold; font-size: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; }
    div.stButton > button:hover { background: linear-gradient(135deg, #2980b9 0%, #1f618d 100%); box-shadow: 0 6px 8px rgba(0,0,0,0.15); transform: translateY(-2px); }
</style>
''', unsafe_allow_html=True)

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
        description="定性根拠に基づく確率補正倍率。0.5（状態不安・展開逆境）〜 1.0（現状維持）〜 1.6（調教絶好・展開物理ドンピシャ）の連続数値。"
    )
    短評: str = Field(
        description="なぜその補正倍率を付与したかの物理的・定性的理由（調教ラップ、斤量差分、血統適性、コース脚質適合度など）"
    )

class RaceCoefficientResult(BaseModel):
    summary: str = Field(
        description="レース全体の物理バイアス（展開ペース、馬場、斤量・血統傾向）と重要補正馬の統括見解（300文字以内）"
    )
    coefficients: List[HorseCoefficient] = Field(
        description="出走全頭分の補正係数リスト（全頭必須・欠落厳禁）"
    )

# ==========================================
# 🧠 システムプロンプト（乗算レバレッジ完全仕様）
# ==========================================
system_instruction = """
# 役割定義
あなたは統計モデル（Python）が算出したベース確率に対し、直前定性データから乗算補正倍率を付与する「動的レバレッジ鑑定AI」です。
印（◎◯▲等）を決める必要はありません。各馬の好走確率を何倍にすべきか、客観的根拠から【0.5 〜 1.6】の補正倍率を精密に鑑定してください。

# 係数付与の基準（0.5 〜 1.6）
- 1.4 〜 1.6（極大プラス補正）:
  調教で鋭い加速ラップを刻み仕上がり超抜、かつ今回のコース形態と推測脚質が完全に噛み合っている馬。
  または【明確な斤量恩恵（ハンデ戦等で最軽量、他馬より2kg〜4kg以上軽い物理的有利）】がある馬。
  または【前走からの大幅な斤量減（-2kg〜-4kg）】により、前走の着順詐欺（重い斤量でバテただけ）から一変が確定的な馬。好走実績のある斤量に戻った馬も含む。
  または【血統適性の爆発】（新馬戦・ダート替わり・距離延長などで、父・母父の血統キャラクターが今回のコース・馬場条件にドンピシャで合致する馬）。
- 1.1 〜 1.3（微小プラス補正）:
  調教順調、脚質や枠・血統に適性が見込める馬、または手頃な斤量恩恵がある馬。
  または【明確な鞍上強化】により、前走の物理的敗因（折り合い欠き、進路ミス）が今回解消される見込みが立つ馬。
- 1.0（中立・補正なし）:
  可もなく不可もない状態。データ通りの走りが期待できる馬。全頭同斤量のレース、または斤量差・血統適性が平均的な馬。
- 0.8 〜 0.9（微小マイナス補正）:
  調教の時計が物足りない、血統適性に疑問がある、または平均より斤量がやや重い馬。
- 0.5 〜 0.7（大幅マイナス補正）:
  調教で「⚠️急減速」などの明らかな状態不安、小回り・短距離戦で物理的に届かない後方脚質、および【トップハンデ・酷量（最軽量馬と3kg〜5kg以上の格差を背負う馬）】。

# 厳格な鑑定ルール
1. 【斤量物理の重視】: 斤量差は走破タイムとスタミナ消費に直結する物理的負荷である。前走からの大幅減量馬や最軽量馬は物理的恩恵として積極的にプラス補正（1.2〜1.5倍）せよ。
2. 【血統ポテンシャルの評価】: Pythonは血統を単なる記号として扱うが、あなたは「血統の物理的キャラクター（ダート適性、距離適性、道悪適性など）」を熟知している。レース条件と血統が完璧に合致する穴馬はレバレッジをかけて浮上させよ。
3. 【乗り替わりの扱い（二重計上防止）】: ルメールや川田などの名前だけで無条件に加点してはならない（Pythonがすでにベース勝率に織り込み済みのため）。「前走は若手騎手で制御できず敗退 ➔ 今回トップ騎手への乗り替わりで弱点が解消される」という物理的な理由がある場合のみプラス評価せよ。
4. 出走全頭に対して必ず1つずつ係数と短評を出力せよ。
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
        elif diff <= 0.3:
            status = f"維持({diff:+.1f}s)"
        else:
            status = f"減速({diff:+.1f}s)"
        
        warn = " [⚠️終い急減速]" if (diff >= 0.8 and final_1f >= 13.2) else ""
        return f"終い:{final_1f:.1f}s (前F:{penultimate_1f:.1f}s, 差:{diff:+.1f}s, {status}){warn}"
    else:
        warn = " [⚠️終い要確認]" if final_1f >= 13.5 else ""
        return f"終い:{final_1f:.1f}s{warn}"

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
    """
    今回斤量・他馬との格差だけでなく、前走・前々走からの斤量増減（自分自身の重量変化）を算出
    """
    tags = []
    
    # --- 自分自身の斤量変化（前走比・前々走比） ---
    try:
        if past1_w and str(past1_w).strip() != "":
            diff_p1 = round(kinryo_val - float(past1_w), 1)
            if diff_p1 <= -2.0:
                tags.append(f"【🔥前走比 {diff_p1:+.1f}kg: 大幅減量で一変警戒】")
            elif diff_p1 >= 2.0:
                tags.append(f"【⚠️前走比 {diff_p1:+.1f}kg: 大幅増量で失速警戒】")
            elif diff_p1 < 0:
                tags.append(f"[前走比 {diff_p1:+.1f}kg]")
            elif diff_p1 > 0:
                tags.append(f"[前走比 {diff_p1:+.1f}kg]")
    except Exception:
        pass

    try:
        if past2_w and str(past2_w).strip() != "":
            diff_p2 = round(kinryo_val - float(past2_w), 1)
            if diff_p2 == 0 and "前走比" in "".join(tags) and ("+" in "".join(tags)):
                tags.append("【🚀好走時(前々走)の軽斤量へ戻り恩恵】")
    except Exception:
        pass

    # --- 他馬との格差 ---
    if not has_weight_gap:
        tags.append("(全頭同斤量)")
    else:
        diff_from_min = round(kinryo_val - min_w, 1)
        diff_from_max = round(kinryo_val - max_w, 1)
        diff_from_avg = round(kinryo_val - avg_w, 1)
        
        if kinryo_val == min_w:
            tags.append(f"【最軽量恩恵: トップハンデ比 {diff_from_max:+.1f}kg】")
        elif diff_from_avg <= -1.5:
            tags.append(f"【軽量有利: 平均比 {diff_from_avg:+.1f}kg】")
        elif kinryo_val == max_w:
            tags.append(f"【最重量酷量: 最軽量比 {diff_from_min:+.1f}kg】")
        elif diff_from_avg >= 1.5:
            tags.append(f"【重量不利: 平均比 {diff_from_avg:+.1f}kg】")
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

def generate_base_table(disp_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>枠番</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>脚質</th><th>斤量</th><th>騎手</th><th>Python勝率</th><th>初期印</th></tr>"
    for _, r in disp_df.iterrows():
        w_num = str(r.get('枠番', ''))
        display_waku = w_num if w_num and w_num != 'nan' and not w_num.startswith("90") else "-"
        m_num = str(r['馬番'])
        display_umaban = "未定" if m_num.startswith("90") or int(float(m_num)) >= 90 else f"{int(float(m_num)):02d}"
        win_val = float(r.get('win_prob', 0)) * 100
        kinryo_val = float(r.get('斤量', 55.0))
        mark = r.get('初期印', '消')
        style_raw = str(r.get('推測脚質', '不明')).strip()
        
        html += f"<tr><td><b>{display_waku}</b></td><td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{style_raw}</td><td>{kinryo_val:.1f}kg</td><td>{r.get('騎手', '-')}</td><td><b>{win_val:.1f}%</b></td>"
        html += f"<td><span class='badge-mark {get_badge_class(mark)}'>{mark}</span></td></tr>"
    html += "</table></div>"
    return html

def generate_fusion_multiplied_table(merged_df):
    # 🎯 左端に初期印、計算後に確定した最終印を中央右に配置
    html = "<div class='table-container'><table class='kachi-table'>"
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
            mult_str = f"<span class='multiplier-up'>×{mult:.2f} ⬆</span>"
        elif mult <= 0.8:
            mult_str = f"<span class='multiplier-down'>×{mult:.2f} ⬇</span>"
        else:
            mult_str = f"<span class='multiplier-flat'>×{mult:.2f}</span>"
            
        tanpyo = r.get('短評', '-')
            
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
    st.markdown(f"<h2>🚀 {r_info['place_name']} {r_info['r_num']}R 【{display_race_name}】 <span style='font-size:18px; color:#7f8c8d;'>({surface_info}{distance_info}m / 開催{race_day}日目)</span></h2>", unsafe_allow_html=True)
    
    if res_df is not None:
        st.markdown("<div class='section-header'>📊 1. Pythonベース地力スコア</div>", unsafe_allow_html=True)
        st.markdown(generate_base_table(res_df), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⚡ Geminiによる動的レバレッジ補正を実行（血統・斤量差分モデル）", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】GEMINI_API_KEYが設定されていません。")
                st.stop()

            is_shinba = "新馬" in display_race_name or res_df['前走着順'].isna().all()
            total_horses = len(res_df)

            weights = res_df['斤量'].astype(float)
            min_w = float(weights.min())
            max_w = float(weights.max())
            avg_w = float(weights.mean())
            has_weight_gap = (max_w - min_w) >= 0.5

            table_summary = []
            for i, (_, r) in enumerate(res_df.iterrows()):
                m_str = str(r['馬番'])
                display_m = "未定" if m_str.startswith("90") else f"{int(float(m_str)):02d}"
                win_val = float(r.get('win_prob', 0)) * 100
                
                kinryo_f = float(r.get('斤量', 55.0))
                past1_w = r.get('前走斤量', None)
                past2_w = r.get('前々走斤量', None)
                weight_info = format_weight_analysis(kinryo_f, min_w, max_w, avg_w, has_weight_gap, past1_w, past2_w)
                jockey_val = str(r.get('騎手', '-'))
                kyakushitsu_val = str(r.get('推測脚質', '-'))
                
                # 🎯 血統情報の抽出と埋め込み
                sire = str(r.get('父', '不明'))
                bms = str(r.get('母父', '不明'))
                blood_info = f"血統: 父[{sire}] 母父[{bms}]"

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
                
                info = (f"馬番:{display_m} | 馬名:{r.get('馬名','')} | Python勝率:{win_val:.1f}% | {weight_info} | "
                        f"脚質:{kyakushitsu_val} | 騎手:{jockey_val} | {blood_info} | "
                        f"前走:{past_summary} | 調教評価:{r.get('調教評価','-')} | 調教分析:{chokyo_summary}")
                table_summary.append(info)

            prompt = f"""
以下の出走馬データに基づき、指示に従って全出走馬の補正係数（0.5〜1.6）と理由を出力してください。

【対象レース】: {r_info['place_name']} {r_info['r_num']}R ({surface_info}{distance_info}m)
【出走頭数】: {total_horses}頭
【斤量条件】: {"【全頭同斤量（斤量差なし）】" if not has_weight_gap else f"【斤量差あり（ハンデ/別定）】最軽量:{min_w:.1f}kg 〜 最重量:{max_w:.1f}kg (最大格差:{max_w-min_w:.1f}kg)"}

【出走馬データ一覧】
{chr(10).join(table_summary)}
"""

            with st.spinner("🧠 Geminiが調教・展開・血統・斤量差データから各馬の確率補正倍率を精密算出中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
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
                        
                        # 結合して最終確率を数学的に乗算
                        merged = pd.merge(res_df, coeff_df_clean, on='馬名_clean', how='left')
                        merged['補正係数'] = merged['補正係数'].fillna(1.0)
                        merged['短評'] = merged['短評'].fillna('データ保留')
                        
                        # 🎯 数学的な乗算: 最終確率 = Python基準勝率 × 補正係数
                        merged['raw_final'] = merged['win_prob'] * merged['補正係数']
                        # 確率の正規化（合計100%に揃える）
                        merged['final_prob'] = merged['raw_final'] / merged['raw_final'].sum()
                        
                        # 最終確率順にソートして新たな印（◎◯▲△☆）を機械的に付与
                        merged = merged.sort_values(by='final_prob', ascending=False).reset_index(drop=True)
                        
                        marks = ['◎', '◯', '▲', '△', '☆']
                        final_marks = []
                        for idx in range(len(merged)):
                            if idx < len(marks):
                                final_marks.append(marks[idx])
                            else:
                                final_marks.append('消')
                        merged['最終印'] = final_marks
                        
                        st.markdown(f"""
                        <div class='gemini-view-card'>
                            <div class='gemini-view-title'>📝 トラックマンGeminiのレース物理バイアス統括</div>
                            <p class='gemini-view-text'>{result.summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 買い目サマリーの動的更新
                        h_o = merged.iloc[0]
                        h_m = merged.iloc[1]
                        st.markdown(f"""
                        <div class='sense-card'>
                            <div class='sense-title'>🎟 乗算補正後の最終推奨買い目</div>
                            <div style='margin-top: 6px;'><b>【本線単勝・馬連】</b>: ◎ {h_o['馬名']} (新勝率 {h_o['final_prob']*100:.1f}%) ⇄ ◯ {h_m['馬名']}</div>
                            <div style='margin-top: 4px;'><b>【3連複フォーメーション】</b>: ◎ 1頭軸 ➔ 相手: ◯, ▲, △, ☆</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<div class='section-header'>🔥 2. 数学融合結果（Python地力 × Gemini倍率）</div>", unsafe_allow_html=True)
                        st.markdown(generate_fusion_multiplied_table(merged), unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ API通信・計算エラー: {e}")