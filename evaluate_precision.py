import os
import re
import joblib
import unicodedata
import numpy as np
import pandas as pd

# ==========================================
# 🐴 上位4頭（◎・◯・▲・△1）限定 検証スクリプト
# ==========================================

MODEL_FILE = "keiba_ai_model.pkl"
DATA_FILE = "ml_target_data.csv"
TEST_RACE_COUNT = 2220  # 直近2,220レースで検証

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

def main():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
        print("❌ 必要なファイルが見つかりません。")
        return

    print("🔄 モデルとデータを読み込んでいます...")
    model_data = joblib.load(MODEL_FILE)
    model = model_data['model']
    features = model_data['features']

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['rank_num_target'] = pd.to_numeric(df['着順'], errors='coerce')
    df = df.dropna(subset=['rank_num_target', 'race_id']).copy()

    df['馬名_clean'] = df['馬名'].astype(str).apply(clean_horse_name)
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df['distance_num'] = pd.to_numeric(df.get('distance'), errors='coerce')
    df['dist_cat'] = df['distance_num'].apply(get_dist_cat)
    df['is_top3_past'] = (df['rank_num_target'] <= 3).astype(int)
    df['is_win_past'] = (df['rank_num_target'] == 1).astype(int)

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

    passing = df['通過'].apply(parse_passing)
    df['first_corner_pos'] = [p[0] for p in passing]
    df['last_corner_pos'] = [p[1] for p in passing]
    df['pos_gain'] = [p[2] for p in passing]

    df['単勝_num'] = pd.to_numeric(df.get('単勝'), errors='coerce').fillna(10.0)

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

    races = df['race_id'].unique()
    test_races = races[-TEST_RACE_COUNT:]
    df_test = df[df['race_id'].isin(test_races)].copy()

    X_test = pd.DataFrame(index=df_test.index)
    for f in features:
        X_test[f] = pd.to_numeric(df_test[f], errors='coerce') if f in df_test.columns else np.nan

    df_test['raw_score'] = model.predict(X_test)

    # 上位4頭（◎, ◯, ▲, △）用の各種カウンタ
    sanrenpuku_4box_hits = 0  # 4頭BOX (4点)
    sanrenpuku_jiku_hits = 0  # ◎ 軸 -> ◯▲△ (3点)
    sanrentan_4box_hits = 0   # 4頭BOX (24点)
    sanrentan_1착固定_hits = 0# ◎ 1着固定 -> ◯▲△ (6点)
    umaren_4box_hits = 0      # 4頭BOX (6点)
    umaren_jiku_hits = 0      # ◎ 軸 -> ◯▲△ (3点)

    total_valid_races = 0

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
        group['印'] = "消"

        # 上位4頭だけに個別の印を付与
        if len(group) > 0: group.loc[0, '印'] = "◎"
        if len(group) > 1: group.loc[1, '印'] = "◯"
        if len(group) > 2: group.loc[2, '印'] = "▲"
        if len(group) > 3: group.loc[3, '印'] = "△"

        r1 = group[group['rank_num_target'] == 1]
        r2 = group[group['rank_num_target'] == 2]
        r3 = group[group['rank_num_target'] == 3]

        if not r1.empty and not r2.empty and not r3.empty:
            total_valid_races += 1
            m1 = r1['印'].iloc[0]
            m2 = r2['印'].iloc[0]
            m3 = r3['印'].iloc[0]

            top4_marks = {"◎", "◯", "▲", "△"}
            race_top3_marks = [m1, m2, m3]

            # 1. 3連複 4頭BOX (1,2,3着が上位4頭で独占)
            if all(m in top4_marks for m in race_top3_marks):
                sanrenpuku_4box_hits += 1
                sanrentan_4box_hits += 1

                # 2. 3連複 ◎軸 (◎が入っていて残り2頭も上位4頭)
                if "◎" in race_top3_marks:
                    sanrenpuku_jiku_hits += 1

            # 3. 3連単 ◎1着固定 (1着が◎、2・3着が◯▲△)
            if m1 == "◎" and m2 in top4_marks and m3 in top4_marks:
                sanrentan_1착固定_hits += 1

            # 4. 馬連 4頭BOX (1,2着が上位4頭)
            if m1 in top4_marks and m2 in top4_marks:
                umaren_4box_hits += 1

                # 5. 馬連 ◎軸 (1着または2着に◎があり、相方も上位4頭)
                if "◎" in [m1, m2]:
                    umaren_jiku_hits += 1

    print("\n" + "="*70)
    print(f"🎯 【Python上位4頭（◎・◯・▲・△）限定】 馬券別 的中率 ({total_valid_races:,} R)")
    print("="*70)
    print(f"① 3連複 【4頭BOX】       ( 4点)  : 的中率 {(sanrenpuku_4box_hits/total_valid_races)*100:5.2f}% ({sanrenpuku_4box_hits}R)")
    print(f"② 3連複 【◎軸 -> ◯▲△】  ( 3点)  : 的中率 {(sanrenpuku_jiku_hits/total_valid_races)*100:5.2f}% ({sanrenpuku_jiku_hits}R)")
    print(f"③ 3連単 【4頭BOX】       (24点)  : 的中率 {(sanrentan_4box_hits/total_valid_races)*100:5.2f}% ({sanrentan_4box_hits}R)")
    print(f"④ 3連単 【◎1着固定】     ( 6点)  : 的中率 {(sanrentan_1착固定_hits/total_valid_races)*100:5.2f}% ({sanrentan_1착固定_hits}R)")
    print(f"⑤ 馬連   【4頭BOX】       ( 6点)  : 的中率 {(umaren_4box_hits/total_valid_races)*100:5.2f}% ({umaren_4box_hits}R)")
    print(f"⑥ 馬連   【◎軸 -> ◯▲△】  ( 3点)  : 的中率 {(umaren_jiku_hits/total_valid_races)*100:5.2f}% ({umaren_jiku_hits}R)")
    print("="*70)

if __name__ == "__main__":
    main()