import os
import re
import joblib
import unicodedata
import numpy as np
import pandas as pd

# ==========================================
# 🎯 「勝負気配」自動察知＆適応購入 検証スクリプト (Optuna最強版対応)
# ==========================================

MODEL_PATHS = ["keiba_ai_model.pkl", "勝ちパカくん.pkl"]
DATA_FILE = "ml_target_data.csv"

EVAL_MODE = "RECENT"
TARGET_YEAR = 2026
TARGET_MONTH = 8
TARGET_DAY = 2
TARGET_RACE_COUNT = 3000
EXCLUDE_NEWCOMER = True

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

def preprocess_features(df):
    df_feat = df.copy()

    df_feat['馬名_clean'] = df_feat['馬名'].astype(str).apply(clean_horse_name)
    df_feat['date_parsed'] = pd.to_datetime(df_feat['date'], errors='coerce')
    df_feat = df_feat.dropna(subset=['date_parsed']).sort_values(['馬名_clean', 'date_parsed'])

    sex_age = df_feat['性齢'].apply(parse_sex_age)
    df_feat['sex_code'] = [s[0] for s in sex_age]
    df_feat['age'] = [s[1] for s in sex_age]

    df_feat['distance_num'] = pd.to_numeric(df_feat.get('distance'), errors='coerce')
    df_feat['dist_cat'] = df_feat['distance_num'].apply(get_dist_cat)
    df_feat['rank_num'] = pd.to_numeric(df_feat.get('着順'), errors='coerce')
    df_feat['is_top3_past'] = (df_feat['rank_num'] <= 3).astype(int)
    df_feat['is_win_past'] = (df_feat['rank_num'] == 1).astype(int)

    passing = df_feat['通過'].apply(parse_passing)
    df_feat['first_corner_pos'] = [p[0] for p in passing]
    df_feat['last_corner_pos'] = [p[1] for p in passing]
    df_feat['pos_gain'] = [p[2] for p in passing]

    df_feat['eff_rank_avg'] = df_feat.groupby('馬名_clean')['rank_num'].apply(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    
    df_feat['eff_top5_rate'] = df_feat.groupby('馬名_clean')['rank_num'].apply(
        lambda x: (x.shift(1) <= 5).astype(float).rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    df_feat['eff_top3_rate'] = df_feat.groupby('馬名_clean')['is_top3_past'].apply(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    df_feat['kinryo_num'] = pd.to_numeric(df_feat.get('斤量'), errors='coerce')
    df_feat['wakuban_num'] = pd.to_numeric(df_feat.get('枠番'), errors='coerce')
    df_feat['umaban_num'] = pd.to_numeric(df_feat.get('馬番'), errors='coerce')

    # 🌟 NEW 2: トラックバイアス精密化をテストデータ側でも計算
    df_feat['meet_day_num'] = pd.to_numeric(df_feat.get('meet_day_num'), errors='coerce').fillna(1.0)
    df_feat['race_num'] = df_feat['race_id'].astype(str).str[-2:]
    df_feat['race_num'] = pd.to_numeric(df_feat['race_num'], errors='coerce').fillna(1.0)
    df_feat['track_degradation'] = df_feat['meet_day_num'] * df_feat['race_num']

    weights_parsed = df_feat.get('馬体重', pd.Series()).apply(parse_weight)
    df_feat['body_weight'] = [p[0] for p in weights_parsed]
    df_feat['body_weight_diff'] = [p[1] for p in weights_parsed]
    df_feat['kinryo_body_ratio'] = df_feat['kinryo_num'] / df_feat['body_weight'].fillna(470)

    num_cols = [
        'horse_runs', 'horse_wins', 'horse_win_rate', 'prev_rank',
        'horse_avg_time_idx', 'horse_avg_last3f_idx', 'horse_avg_pace_idx', 'horse_avg_start_idx',
        'jockey_runs', 'jockey_wins', 'jockey_win_rate', 'jockey_track_win_rate',
        'jp_runs', 'jp_wins', 'first_pos'
    ]
    for c in num_cols:
        if c in df_feat.columns:
            df_feat[c] = pd.to_numeric(df_feat[c], errors='coerce')

    df_feat['jp_win_rate'] = df_feat['jp_wins'] / (df_feat['jp_runs'] + 1e-5)

    place_code = df_feat.get('place_code', pd.Series(['00']*len(df_feat)))
    surface = df_feat.get('surface', pd.Series(['芝']*len(df_feat)))
    df_feat['course_id'] = place_code.astype(str) + "_" + surface.astype(str) + "_" + df_feat['distance_num'].fillna(0).astype(int).astype(str)
    df_feat['course_frame_id'] = df_feat['course_id'] + "_frame_" + df_feat['wakuban_num'].fillna(0).astype(int).astype(str)

    course_frame_win_rates = df_feat.groupby('course_frame_id')['is_win_past'].mean().to_dict()
    df_feat['course_frame_win_rate'] = df_feat['course_frame_id'].map(course_frame_win_rates).fillna(0.08)

    kinryo_diffs, is_same_jockeys, dist_diffs = [], [], []
    cat_win_rates, cat_runs_list, prev_prizes = [], [], []

    for horse, group in df_feat.groupby('馬名_clean', sort=False):
        for i in range(len(group)):
            curr_row = group.iloc[i]
            past_rows = group.iloc[:i]

            curr_dist = curr_row['distance_num']
            curr_cat = curr_row['dist_cat']
            curr_jockey = str(curr_row.get('騎手', '')).strip()
            curr_kinryo = curr_row['kinryo_num']

            if past_rows.empty:
                kinryo_diffs.append(0.0)
                is_same_jockeys.append(0)
                dist_diffs.append(np.nan)
                cat_win_rates.append(np.nan)
                cat_runs_list.append(0)
                prev_prizes.append(np.nan)
            else:
                prev_row = past_rows.iloc[-1]
                prev_kinryo = prev_row['kinryo_num']
                kinryo_diffs.append(curr_kinryo - prev_kinryo if pd.notna(curr_kinryo) and pd.notna(prev_kinryo) else 0.0)

                prev_jockey = str(prev_row.get('騎手', '')).strip()
                is_same_jockeys.append(1 if (curr_jockey and curr_jockey == prev_jockey) else 0)

                top3_past = past_rows[past_rows['is_top3_past'] == 1]
                if not top3_past.empty and pd.notna(curr_dist):
                    best_dist_avg = top3_past['distance_num'].mean()
                    dist_diffs.append(abs(curr_dist - best_dist_avg))
                else:
                    dist_diffs.append(np.nan)

                cat_past = past_rows[past_rows['dist_cat'] == curr_cat]
                c_runs = len(cat_past)
                cat_runs_list.append(c_runs)
                cat_win_rates.append(cat_past['is_win_past'].sum() / c_runs if c_runs > 0 else np.nan)
                prev_prizes.append(pd.to_numeric(prev_row.get('賞金(万円)'), errors='coerce'))

    df_feat['kinryo_diff'] = kinryo_diffs
    df_feat['is_same_jockey'] = is_same_jockeys
    df_feat['dist_diff'] = dist_diffs
    df_feat['cat_win_rate'] = cat_win_rates
    df_feat['cat_runs'] = cat_runs_list
    df_feat['prev_prize'] = prev_prizes

    df_feat['interval_days'] = df_feat.groupby('馬名_clean')['date_parsed'].diff().dt.days.fillna(30)
    df_feat['is_long_rest'] = (df_feat['interval_days'] >= 180).astype(int)

    target_cols = ['my_time_idx', 'my_last3f_idx', 'my_pace_idx', 'my_start_idx', 'hybrid_power_idx']
    for col in target_cols:
        if col in df_feat.columns:
            num_col = pd.to_numeric(df_feat[col], errors='coerce').clip(40, 80)
            df_feat[f'eff_{col}'] = num_col.groupby(df_feat['馬名_clean']).apply(
                lambda x: x.shift(1).rolling(3, min_periods=1).median()
            ).reset_index(level=0, drop=True)
        else:
            df_feat[f'eff_{col}'] = np.nan

    # 🌟 NEW 1: ペース展開の相対評価をテストデータ側でも計算
    df_feat['race_avg_start_idx'] = df_feat.groupby('race_id')['eff_my_start_idx'].transform('mean').fillna(50.0)
    df_feat['pace_scenario_idx'] = df_feat['eff_my_last3f_idx'].fillna(50.0) * (df_feat['race_avg_start_idx'] / 50.0)

    course_front_rates = df_feat.groupby('course_id')['is_top3_past'].mean().to_dict()
    df_feat['course_front_rate'] = df_feat['course_id'].map(course_front_rates).fillna(0.3)
    df_feat['style_course_fit'] = df_feat['eff_my_start_idx'].fillna(50) * df_feat['course_front_rate']

    df_feat['prev_rank_num'] = df_feat['rank_num'].groupby(df_feat['馬名_clean']).shift(1)

    df_feat['time_idx_diff'] = df_feat['eff_my_time_idx'] - df_feat['horse_avg_time_idx']
    df_feat['last3f_idx_diff'] = df_feat['eff_my_last3f_idx'] - df_feat['horse_avg_last3f_idx']
    df_feat['pace_idx_diff'] = df_feat['eff_my_pace_idx'] - df_feat['horse_avg_pace_idx']
    df_feat['start_idx_diff'] = df_feat['eff_my_start_idx'] - df_feat['horse_avg_start_idx']

    z_cols = [
        'eff_rank_avg', 'eff_top3_rate', 'prev_prize', 'eff_my_time_idx', 
        'eff_my_last3f_idx', 'eff_my_pace_idx', 'jockey_win_rate', 'jockey_track_win_rate',
        'horse_win_rate', 'horse_avg_time_idx', 'horse_avg_last3f_idx', 'horse_avg_pace_idx', 'horse_avg_start_idx',
        'age', 'kinryo_num', 'body_weight', 'prev_rank', 'first_pos', 'eff_hybrid_power_idx',
        'pace_scenario_idx' # 🌟 NEW 1
    ]
    for c in z_cols:
        if c in df_feat.columns:
            df_feat[f'{c}_z'] = df_feat.groupby('race_id')[c].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-5) if len(x) > 1 else 0.0
            ).fillna(0.0)

    rank_cols = ['eff_my_time_idx', 'horse_avg_time_idx', 'jockey_win_rate', 'horse_win_rate', 'eff_hybrid_power_idx']
    for c in rank_cols:
        if c in df_feat.columns:
            df_feat[f'{c}_rank_in_race'] = df_feat.groupby('race_id')[c].rank(ascending=False, method='min')

    return df_feat

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
    df['単勝_num'] = df['単勝'].apply(extract_valid_odds)

    print(f"⚙️ 特徴量 ({len(features)}個) を生成中... (過去データから偏差値や新・展開指数を計算)")
    df_prep = preprocess_features(df)
    
    df_prep['date_parsed'] = pd.to_datetime(df_prep['date'], format='mixed', errors='coerce')
    
    if EXCLUDE_NEWCOMER and 'race_name' in df_prep.columns:
        df_prep = df_prep[~df_prep['race_name'].astype(str).str.contains("新馬", na=False)]

    if EVAL_MODE == "DAILY":
        df_test = df_prep[
            (df_prep['date_parsed'].dt.year == TARGET_YEAR) & 
            (df_prep['date_parsed'].dt.month == TARGET_MONTH) & 
            (df_prep['date_parsed'].dt.day == TARGET_DAY)
        ].copy()
        print(f"🎯 モード: {TARGET_YEAR}年{TARGET_MONTH}月{TARGET_DAY}日の特定日検証")
    else:
        recent_races = df_prep[['race_id', 'date_parsed']].drop_duplicates().sort_values('date_parsed', ascending=False)
        target_races = recent_races.head(TARGET_RACE_COUNT)['race_id'].tolist()
        df_test = df_prep[df_prep['race_id'].isin(target_races)].copy()
        print(f"🎯 モード: 直近 {TARGET_RACE_COUNT} レースの一括検証")

    if df_test.empty:
        print("⚠️ 該当するテストデータが見つかりませんでした。")
        return

    X_test = pd.DataFrame(index=df_test.index)
    for f in features:
        X_test[f] = pd.to_numeric(df_test[f], errors='coerce') if f in df_test.columns else np.nan

    print("🤖 AI推論を実行中...")
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

    processed_races = 0
    total_target_races = df_test['race_id'].nunique()

    for race_id, group in df_test.groupby('race_id'):
        processed_races += 1
        if processed_races % 500 == 0:
            print(f"  ... {processed_races} / {total_target_races} レース処理完了")

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
    print(f"🧠 【AI気配察知・適応購入 3000レース一括検証結果】")
    print("="*85)
    for pat, st in pattern_stats.items():
        r = st["races"]
        h = st["hits"]
        inv = st["invest"]
        ret = st["return"]
        rec = (ret / inv * 100) if inv > 0 else 0.0
        print(f"・ {pat:<28} : 該当{r:4d}R | 的中{h:4d}R ({(h/r*100 if r>0 else 0):5.1f}%) | 投資{inv:8,d}円 | 払戻{int(ret):9,d}円 | 回収率: {rec:6.1f}%")
    
    print("-" * 85)
    total_races = sum([st["races"] for st in pattern_stats.values()])
    total_rec = (total_return / total_invest * 100) if total_invest > 0 else 0.0
    print(f"🏆 【総合トータル成績】  : 全{total_races}R | 的中{total_hits}R ({(total_hits/total_races*100):.1f}%) | 投資{total_invest:,}円 | 払戻{int(total_return):,}円")
    print(f"🔥 【最終合計収支】      : {int(total_return - total_invest):+,}円 (総合回収率: {total_rec:.1f}%)")
    print("="*85)

if __name__ == "__main__":
    main()