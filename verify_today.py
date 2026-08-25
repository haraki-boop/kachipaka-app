import os
import re
import random
import joblib
import pandas as pd
import numpy as np
import unicodedata

# ==========================================
# 🎯 勝ちぱかくん: 6〜8月完全検証（v2新特徴量・最強AI対応版）
# ==========================================

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

MODEL_FILE = "keiba_ai_model.pkl"
# 🌟 変更点: 新特徴量が入った v2 データを指定
DATA_FILE = "ml_target_data_v2.csv"

def clean_horse_name(name):
    if pd.isna(name): return ""
    return re.sub(r'[\s・･.\-ー_]+', '', unicodedata.normalize('NFKC', str(name))).strip().upper()

def normalize_date(d_str):
    nums = re.findall(r'\d+', str(d_str))
    if len(nums) >= 3:
        return f"{nums[0]}{int(nums[1]):02d}{int(nums[2]):02d}"
    return ""

def get_payout_estimate_fixed(group, pat_type):
    top3 = group[group['rank_num'] <= 3].sort_values('rank_num')
    if len(top3) < 3: return 0
    
    odds_list = []
    for idx, r in top3.iterrows():
        val = pd.to_numeric(str(r.get('単勝', '')).replace('倍', '').strip(), errors='coerce')
        pop = pd.to_numeric(r.get('人気', np.nan), errors='coerce')
        if pd.isna(val) or val <= 1.0:
            val = (pop * 2.5) if pd.notna(pop) else 6.0
        odds_list.append(val)
    
    o1, o2, o3 = odds_list[0], odds_list[1], odds_list[2]
    mult = 5.5 if "3連単" in pat_type else 2.1
    min_val = 600 if "3連単" in pat_type else 300
    
    payout = max((o1 * o2 * o3) * mult * 10, min_val)
    return int(payout / 10) * 10

def main():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
        print("❌ ファイルが見つかりません。")
        return

    print("🔄 モデルおよび全データ(v2)を読み込み中...")
    model_data = joblib.load(MODEL_FILE)
    model = model_data['model']
    features = model_data['features']

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['馬名_clean'] = df['馬名'].astype(str).apply(clean_horse_name)
    df['rank_num'] = pd.to_numeric(df['着順'], errors='coerce')
    df['umaban_num'] = pd.to_numeric(df.get('馬番', 99), errors='coerce').fillna(99)
    df['date_norm'] = df['date'].apply(normalize_date)

    target_df = df[df['date_norm'].str.contains(r'20260[678]', na=False)].copy()
    unique_dates = sorted(target_df['date_norm'].unique())

    if not unique_dates:
        print("❌ 6〜8月のデータが見つかりませんでした。")
        return

    months = ["202606", "202607", "202608"]
    
    def create_month_dict():
        return {m: {'races': 0, 'skipped': 0, 'bet': 0, 'cost': 0, 'pay': 0, 'hit': 0} for m in months}

    monthly_stats_A = create_month_dict()
    monthly_stats_B = create_month_dict()

    ticket_stats_A = {
        '3連単(軸1頭24点)': {'bet': 0, 'hit': 0, 'cost': 0, 'pay': 0},
        '3連複(従来型10点)': {'bet': 0, 'hit': 0, 'cost': 0, 'pay': 0}
    }
    ticket_stats_B = {
        '3連単(軸2頭18点)': {'bet': 0, 'hit': 0, 'cost': 0, 'pay': 0},
        '3連複(従来型10点)': {'bet': 0, 'hit': 0, 'cost': 0, 'pay': 0}
    }

    marks = ["◎", "◯", "▲", "△", "☆1", "☆2", "☆3"]
    top5 = {"◎", "◯", "▲", "△", "☆1"}
    top6 = {"◎", "◯", "▲", "△", "☆1", "☆2"}
    top7 = {"◎", "◯", "▲", "△", "☆1", "☆2", "☆3"}

    for target_date in unique_dates:
        m_key = target_date[:6]
        df_today = df[df['date_norm'] == target_date].copy()
        if df_today.empty: continue

        df_today['race_num'] = df_today['race_id'].astype(str).str[-2:]
        df_today['race_num'] = pd.to_numeric(df_today['race_num'], errors='coerce').fillna(1.0)
        df_today['meet_day_num'] = pd.to_numeric(df_today.get('meet_day_num', 1.0), errors='coerce').fillna(1.0)
        df_today['track_degradation'] = df_today['meet_day_num'] * df_today['race_num']

        X_test = pd.DataFrame(index=df_today.index)
        for f in features:
            X_test[f] = pd.to_numeric(df_today[f], errors='coerce') if f in df_today.columns else np.nan
            X_test[f] = X_test[f].fillna(0)

        df_today['raw_score'] = model.predict(X_test)

        for race_id, group in df_today.groupby('race_id'):
            monthly_stats_A[m_key]['races'] += 1
            monthly_stats_B[m_key]['races'] += 1
            group = group.copy()

            # 見送り判定：新馬戦のみ除外
            if '新馬' in str(group.get('race_name', pd.Series()).iloc[0]):
                monthly_stats_A[m_key]['skipped'] += 1
                monthly_stats_B[m_key]['skipped'] += 1
                continue
            
            raw_scores = group['raw_score'].values
            s_std = np.std(raw_scores)
            if pd.notna(s_std) and s_std > 0:
                z_scores = (raw_scores - np.mean(raw_scores)) / s_std
                base_probs = 1.0 / (1.0 + np.exp(-1.2 * z_scores))
                group['win_prob'] = base_probs * 0.35 + 0.01
            else:
                group['win_prob'] = 0.10

            group = group.sort_values(by=['win_prob', 'umaban_num'], ascending=[False, True]).reset_index(drop=True)
            probs = group['win_prob'].values
            p1, p2, p3, p4 = probs[0], probs[1], probs[2], probs[3] if len(probs)>3 else 0.05
            gap_1_2, gap_1_3 = p1 - p2, p1 - p3

            group['印'] = "消"
            for i in range(min(len(group), len(marks))):
                group.loc[i, '印'] = marks[i]

            actual_top3 = group[group['rank_num'] <= 3].sort_values('rank_num')
            actual_marks = actual_top3['印'].tolist()
            if len(actual_marks) < 3: continue
            
            actual_top3_set = set(actual_marks)

            monthly_stats_A[m_key]['bet'] += 1
            monthly_stats_B[m_key]['bet'] += 1

            # 3連単を買う明確な本命・対抗決着条件
            is_3ren_tan_race = (gap_1_2 >= 0.07) or (gap_1_2 < 0.035 and gap_1_3 >= 0.06)

            # ====================================================
            # 【パターンA】 3連単(軸1頭24点) vs 3連複(従来型10点)
            # ====================================================
            pA_hit, pA_cost = False, 0
            if is_3ren_tan_race:
                ticket_A = "3連単(軸1頭24点)"
                pA_cost = 2400
                if "◎" in actual_top3_set and actual_top3_set.issubset(top5): pA_hit = True
            elif (p1 - p4) < 0.08: # 3連複波乱 (5頭BOX 10点)
                ticket_A = "3連複(従来型10点)"
                pA_cost = 1000
                if actual_top3_set.issubset(top5): pA_hit = True
            else: # 3連複混戦 (1頭軸5頭流し 10点)
                ticket_A = "3連複(従来型10点)"
                pA_cost = 1000
                if "◎" in actual_top3_set and actual_top3_set.issubset(top6): pA_hit = True

            monthly_stats_A[m_key]['cost'] += pA_cost
            ticket_stats_A[ticket_A]['bet'] += 1
            ticket_stats_A[ticket_A]['cost'] += pA_cost

            if pA_hit:
                monthly_stats_A[m_key]['hit'] += 1
                ticket_stats_A[ticket_A]['hit'] += 1
                pay_type = "3連単" if "3連単" in ticket_A else "3連複"
                pA_pay = get_payout_estimate_fixed(group, pay_type)
                monthly_stats_A[m_key]['pay'] += pA_pay
                ticket_stats_A[ticket_A]['pay'] += pA_pay

            # ====================================================
            # 【パターンB】 3連単(軸2頭18点) vs 3連複(従来型10点)
            # ====================================================
            pB_hit, pB_cost = False, 0
            if is_3ren_tan_race:
                ticket_B = "3連単(軸2頭18点)"
                pB_cost = 1800
                if "◎" in actual_top3_set and "◯" in actual_top3_set and actual_top3_set.issubset(top7): pB_hit = True
            elif (p1 - p4) < 0.08: # 3連複波乱 (5頭BOX 10点)
                ticket_B = "3連複(従来型10点)"
                pB_cost = 1000
                if actual_top3_set.issubset(top5): pB_hit = True
            else: # 3連複混戦 (1頭軸5頭流し 10点)
                ticket_B = "3連複(従来型10点)"
                pB_cost = 1000
                if "◎" in actual_top3_set and actual_top3_set.issubset(top6): pB_hit = True

            monthly_stats_B[m_key]['cost'] += pB_cost
            ticket_stats_B[ticket_B]['bet'] += 1
            ticket_stats_B[ticket_B]['cost'] += pB_cost

            if pB_hit:
                monthly_stats_B[m_key]['hit'] += 1
                ticket_stats_B[ticket_B]['hit'] += 1
                pay_type = "3連単" if "3連単" in ticket_B else "3連複"
                pB_pay = get_payout_estimate_fixed(group, pay_type)
                monthly_stats_B[m_key]['pay'] += pB_pay
                ticket_stats_B[ticket_B]['pay'] += pB_pay

    # Output
    print("\n" + "="*85)
    print("🏆 【6月〜8月 3連単マルチ2パターン比較 フルスペック修正版(v2)】")
    print("="*85)

    def print_full_report(title, monthly_dict, ticket_dict):
        print(f"\n📢 【{title}】")
        print("─" * 85)
        print("📅 月別収支推移:")
        tot_c, tot_p, tot_b, tot_h = 0, 0, 0, 0
        for m in months:
            d = monthly_dict[m]
            m_name = f"{m[4:6]}月"
            rec = (d['pay'] / d['cost'] * 100) if d['cost'] > 0 else 0
            prof = d['pay'] - d['cost']
            prof_s = f"+{int(prof):,}円" if prof >= 0 else f"{int(prof):,}円"
            rate = (d['hit'] / d['bet'] * 100) if d['bet'] > 0 else 0
            print(f"   ▶ {m_name} | 勝負:{d['bet']}R (見送り:{d['skipped']}R) | 的中:{d['hit']}R ({rate:.1f}%) | 投資:{d['cost']:,}円 | 払戻:{int(d['pay']):,}円 | 損益:{prof_s} | 回収率:{rec:.1f}%")
            tot_c += d['cost']
            tot_p += d['pay']
            tot_b += d['bet']
            tot_h += d['hit']

        print("🎫 券種別成績:")
        for t_type, t_d in ticket_dict.items():
            t_rec = (t_d['pay'] / t_d['cost'] * 100) if t_d['cost'] > 0 else 0
            t_prof = t_d['pay'] - t_d['cost']
            t_prof_s = f"+{int(t_prof):,}円" if t_prof >= 0 else f"{int(t_prof):,}円"
            t_rate = (t_d['hit'] / t_d['bet'] * 100) if t_d['bet'] > 0 else 0
            print(f"   ▶ {t_type} | 勝負:{t_d['bet']}R | 的中:{t_d['hit']}R ({t_rate:.1f}%) | 投資:{t_d['cost']:,}円 | 払戻:{int(t_d['pay']):,}円 | 損益:{t_prof_s} | 回収率:{t_rec:.1f}%")

        tot_rec = (tot_p / tot_c * 100) if tot_c > 0 else 0
        tot_prof = tot_p - tot_c
        tot_prof_s = f"+{int(tot_prof):,}円" if tot_prof >= 0 else f"{int(tot_prof):,}円"
        print(f"🔥 通算成績: 勝負 {tot_b} R | 的中 {tot_h} R ({tot_h/tot_b*100:.1f}%) | 投資:{tot_c:,}円 | 払戻:{int(tot_p):,}円 | 損益:{tot_prof_s} | 通算回収率:{tot_rec:.1f}%")

    print_full_report("パターンA: 3連単【◎ 軸1頭 相手4頭マルチ (24点/2,400円)】", monthly_stats_A, ticket_stats_A)
    print_full_report("パターンB: 3連単【◎◯ 軸2頭 相手5頭マルチ (18点/1,800円)】", monthly_stats_B, ticket_stats_B)
    print("="*85)

if __name__ == "__main__":
    main()