import os
import re
import joblib
import unicodedata
import numpy as np
import pandas as pd

# ==========================================
# 🎯 「勝負気配」自動察知＆適応購入 検証スクリプト（修正版）
# ==========================================

MODEL_PATHS = ["keiba_ai_model.pkl", "勝ちパカくん.pkl"]
DATA_FILE = "ml_target_data.csv"

TARGET_YEAR = 2026
TARGET_MONTH = 8
TARGET_DAY = 15
EXCLUDE_NEWCOMER = True  # 新馬戦を除外

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip().upper()

def get_dist_cat(d):
    if pd.isna(d): return np.nan
    if d <= 1400: return 'sprint'
    elif d <= 2200: return 'mile_middle'
    else: return 'stayer'

def parse_sex_age(val):
    if pd.isna(val): return 0, 4.0
    val = str(val).strip()
    sex_char = val[0] if len(val) > 0 else '牡'
    sex_code = 0 if sex_char == '牡' else (1 if sex_char == '牝' else 2)
    try: age = float(val[1:])
    except: age = 4.0
    return sex_code, age

def parse_weight(val):
    if pd.isna(val): return np.nan, np.nan
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    if m:
        w = float(m.group(1))
        diff = float(m.group(2)) if m.group(2) else 0.0
        return w, diff
    return np.nan, np.nan

def parse_passing(val):
    if pd.isna(val): return np.nan, np.nan, np.nan
    parts = str(val).split('-')
    try: return float(parts[0]), float(parts[-1]), float(parts[0]) - float(parts[-1])
    except: return np.nan, np.nan, np.nan

def extract_valid_odds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if '-' in s: return np.nan
    m = re.search(r'(\d+\.\d+|\d+)', s)
    if m:
        try: return float(m.group(1))
        except: return np.nan
    return np.nan

def main():
    model_path = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            model_path = p
            break

    if not model_path or not os.path.exists(DATA_FILE):
        print("❌ 必要なモデルファイルまたはデータファイルが見つかりません。")
        return

    print("🔄 モデルとデータを読み込んでいます...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    features = model_data['features']

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['rank_num_target'] = pd.to_numeric(df['着順'], errors='coerce')
    df = df.dropna(subset=['rank_num_target', 'race_id']).copy()

    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    date_filtered = df[
        (df['date_parsed'].dt.year == TARGET_YEAR) & 
        (df['date_parsed'].dt.month == TARGET_MONTH) & 
        (df['date_parsed'].dt.day == TARGET_DAY)
    ].copy()

    if date_filtered.empty:
        print(f"⚠️ {TARGET_YEAR}年{TARGET_MONTH}月{TARGET_DAY}日 のデータが見つかりませんでした。")
        return

    if EXCLUDE_NEWCOMER:
        if 'race_name' in date_filtered.columns:
            date_filtered = date_filtered[~date_filtered['race_name'].astype(str).str.contains("新馬", na=False)]

    target_races = date_filtered['race_id'].unique()

    df['馬名_clean'] = df['馬名'].astype(str).apply(clean_horse_name)
    df['distance_num'] = pd.to_numeric(df.get('distance'), errors='coerce')
    df['dist_cat'] = df['distance_num'].apply(get_dist_cat)
    df['is_top3_past'] = (df['rank_num_target'] <= 3).astype(int)

    sex_age = df['性齢'].apply(parse_sex_age)
    df['sex_code'] = [s[0] for s in sex_age]
    df['age'] = [s[1] for s in sex_age]

    df['kinryo_num'] = pd.to_numeric(df.get('斤量'), errors='coerce')
    df['wakuban_num'] = pd.to_numeric(df.get('枠番'), errors='coerce')
    df['umaban_num'] = pd.to_numeric(df.get('馬番'), errors='coerce')

    weights_parsed = df.get('馬体重', pd.Series()).apply(parse_weight)
    df['body_weight'] = [p[0] for p in weights_parsed]
    df['body_weight_diff'] = [p[1] for p in weights_parsed]
    df['kinryo_body_ratio'] = df['kinryo_num'] / df['body_weight'].fillna(470)

    passing_src = df['通過'] if '通過' in df.columns else df.get('Through', pd.Series())
    passing = passing_src.apply(parse_passing)
    df['first_corner_pos'] = [p[0] for p in passing]
    df['last_corner_pos'] = [p[1] for p in passing]
    df['pos_gain'] = [p[2] for p in passing]

    df['単勝_num'] = df['単勝'].apply(extract_valid_odds)

    df['eff_rank_avg'] = df.groupby('馬名_clean')['rank_num_target'].apply(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    df['eff_top5_rate'] = df.groupby('馬名_clean')['rank_num_target'].apply(
        lambda x: (x.shift(1) <= 5).astype(float).rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    df['eff_top3_rate'] = df.groupby('馬名_clean')['is_top3_past'].apply(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    target_cols = ['my_time_idx', 'my_last3f_idx', 'my_pace_idx', 'my_start_idx']
    for col in target_cols:
        if col in df.columns:
            num_col = pd.to_numeric(df[col], errors='coerce').clip(40, 80)
            df[f'eff_{col}'] = num_col.groupby(df['馬名_clean']).apply(
                lambda x: x.shift(1).rolling(3, min_periods=1).median()
            ).reset_index(level=0, drop=True)

    df['time_idx_diff'] = df['eff_my_time_idx'] - df.get('horse_avg_time_idx', 0)
    df['last3f_idx_diff'] = df['eff_my_last3f_idx'] - df.get('horse_avg_last3f_idx', 0)
    df['pace_idx_diff'] = df['eff_my_pace_idx'] - df.get('horse_avg_pace_idx', 0)
    df['start_idx_diff'] = df['eff_my_start_idx'] - df.get('horse_avg_start_idx', 0)

    z_cols = [
        'eff_rank_avg', 'eff_top3_rate', 'prev_prize', 'eff_my_time_idx', 
        'eff_my_last3f_idx', 'eff_my_pace_idx', 'jockey_win_rate', 'jockey_track_win_rate',
        'horse_win_rate', 'horse_avg_time_idx', 'horse_avg_last3f_idx', 'horse_avg_pace_idx', 'horse_avg_start_idx',
        'age', 'kinryo_num', 'body_weight', 'prev_rank', 'first_pos'
    ]
    for c in z_cols:
        if c in df.columns:
            s_val = pd.to_numeric(df[c], errors='coerce')
            df[f'{c}_z'] = df.groupby('race_id')[c].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-5) if len(x) > 1 else 0.0
            ).fillna(0.0)

    rank_cols = ['eff_my_time_idx', 'horse_avg_time_idx', 'jockey_win_rate', 'horse_win_rate']
    for c in rank_cols:
        if c in df.columns:
            s_val = pd.to_numeric(df[c], errors='coerce')
            df[f'{c}_rank_in_race'] = s_val.rank(ascending=False, method='min')

    df_test = df[df['race_id'].isin(target_races)].copy()

    X_test = pd.DataFrame(index=df_test.index)
    for f in features:
        X_test[f] = pd.to_numeric(df_test[f], errors='coerce') if f in df_test.columns else np.nan

    df_test['raw_score'] = model.predict(X_test)

    total_invest = 0
    total_return = 0.0
    total_hits = 0
    
    pattern_stats = {
        "① 1強 (3連単 6点)": {"races": 0, "hits": 0, "invest": 0, "return": 0.0},
        "② 2強 (3連単ダブル軸12点)": {"races": 0, "hits": 0, "invest": 0, "return": 0.0},
        "③ 混戦 (3連複 ◎軸1頭10点)": {"races": 0, "hits": 0, "invest": 0, "return": 0.0},
        "④ 波乱 (3連複 5頭BOX10点)": {"races": 0, "hits": 0, "invest": 0, "return": 0.0},
    }

    for race_id, group in df_test.groupby('race_id'):
        group = group.copy()
        raw_scores = group['raw_score'].values
        s_std = np.std(raw_scores)
        if pd.notna(s_std) and s_std > 0:
            z_scores = (raw_scores - np.mean(raw_scores)) / s_std
            base_probs = 1.0 / (1.0 + np.exp(-1.2 * z_scores))
            group['win_prob'] = base_probs * 0.35 + 0.01
        else:
            group['win_prob'] = 0.10

        group = group.sort_values(by=['win_prob'], ascending=[False]).reset_index(drop=True)
        probs = group['win_prob'].values
        p1, p2, p3, p4 = probs[0], probs[1], probs[2], probs[3] if len(probs)>3 else 0.05
        
        gap_1_2 = p1 - p2
        gap_1_3 = p1 - p3

        # 🔥 修正箇所: 先に印を付与してから着順（r1, r2, r3）を取得する
        marks = ["◎", "◯", "▲", "△", "☆1", "☆2"]
        group['印'] = "消"
        for i in range(min(len(group), len(marks))):
            group.loc[i, '印'] = marks[i]

        r1 = group[group['rank_num_target'] == 1]
        r2 = group[group['rank_num_target'] == 2]
        r3 = group[group['rank_num_target'] == 3]

        if r1.empty or r2.empty or r3.empty: continue

        v1 = r1['単勝_num'].iloc[0]
        v2 = r2['単勝_num'].iloc[0]
        v3 = r3['単勝_num'].iloc[0]

        o1 = v1 if pd.notna(v1) and v1 > 0 else 3.5
        o2 = v2 if pd.notna(v2) and v2 > 0 else 5.0
        o3 = v3 if pd.notna(v3) and v3 > 0 else 8.0

        est_sanrenpuku = min(max(o1 * o2 * o3 * 25, 400), 80000)
        est_sanrentan = min(max(o1 * o2 * o3 * 120, 1000), 250000)

        m1, m2, m3 = r1['印'].iloc[0], r2['印'].iloc[0], r3['印'].iloc[0]
        top5 = {"◎", "◯", "▲", "△", "☆1"}
        top6 = {"◎", "◯", "▲", "△", "☆1", "☆2"}

        # スコア差による買い目の自動分岐
        if gap_1_2 >= 0.07:
            pat = "① 1強 (3連単 6点)"
            invest = 600
            hit = (m1 == "◎" and m2 in {"◯", "▲", "△"} and m3 in {"◯", "▲", "△"})
            pay = est_sanrentan if hit else 0.0

        elif gap_1_2 < 0.035 and gap_1_3 >= 0.06:
            pat = "② 2強 (3連単ダブル軸12点)"
            invest = 1200
            hit = (m1 in {"◎", "◯"} and m2 in {"◎", "◯", "▲"} and m3 in top5)
            pay = est_sanrentan if hit else 0.0

        elif (p1 - p4) < 0.08:
            pat = "④ 波乱 (3連複 5頭BOX10点)"
            invest = 1000
            hit = all(m in top5 for m in [m1, m2, m3])
            pay = est_sanrenpuku if hit else 0.0

        else:
            pat = "③ 混戦 (3連複 ◎軸1頭10点)"
            invest = 1000
            hit = ("◎" in [m1, m2, m3] and all(o in top6 for o in [m for m in [m1, m2, m3] if m != "◎"]))
            pay = est_sanrenpuku if hit else 0.0

        pattern_stats[pat]["races"] += 1
        pattern_stats[pat]["invest"] += invest
        pattern_stats[pat]["return"] += pay
        if hit: pattern_stats[pat]["hits"] += 1

        total_invest += invest
        total_return += pay
        if hit: total_hits += 1

    print("\n" + "="*85)
    print(f"🧠 【{TARGET_YEAR}年{TARGET_MONTH}月{TARGET_DAY}日 AI気配察知・適応購入 検証結果】")
    print("="*85)
    for pat, st in pattern_stats.items():
        r = st["races"]
        h = st["hits"]
        inv = st["invest"]
        ret = st["return"]
        rec = (ret / inv * 100) if inv > 0 else 0.0
        print(f"・ {pat:<28} : 該当{r:2d}R | 的中{h:2d}R ({(h/r*100 if r>0 else 0):5.1f}%) | 投資{inv:6,d}円 | 払戻{int(ret):6,d}円 | 回収率: {rec:6.1f}%")
    
    print("-" * 85)
    total_races = sum([st["races"] for st in pattern_stats.values()])
    total_rec = (total_return / total_invest * 100) if total_invest > 0 else 0.0
    print(f"🏆 【総合トータル成績】  : 全{total_races}R | 的中{total_hits}R ({(total_hits/total_races*100):.1f}%) | 投資{total_invest:,}円 | 払戻{int(total_return):,}円")
    print(f"🔥 【最終合計収支】      : {int(total_return - total_invest):+,}円 (総合回収率: {total_rec:.1f}%)")
    print("="*85)

if __name__ == "__main__":
    main()