import os
import re
import random
import joblib
import pandas as pd
import numpy as np
import unicodedata

# ==========================================
# 🎯 勝ちぱかくん: 3連複3パターン比較検証スクリプト
# ==========================================

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

MODEL_FILE = "keiba_ai_model.pkl"
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
    
    def create_stats_dict():
        return {
            'months': {m: {'bet': 0, 'cost': 0, 'pay': 0, 'hit': 0, 'skipped': 0} for m in months},
            'tickets': {
                '3連単(軸1頭24点)': {'bet': 0, 'hit': 0, 'cost': 0, 'pay': 0},
                '3連複': {'bet': 0, 'hit': 0, 'cost': 0, 'pay': 0}
            }
        }

    # ① 従来型（波乱=BOX / 混戦=流し）
    stats_1 = create_stats_dict()
    # ② 全て5頭BOX
    stats_2 = create_stats_dict()
    # ③ 全て1軸5頭流し
    stats_3 = create_stats_dict()

    marks = ["◎", "◯", "▲", "△", "☆1", "☆2", "☆3"]
    top5 = {"◎", "◯", "▲", "△", "☆1"}
    top6 = {"◎", "◯", "▲", "△", "☆1", "☆2"}

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
            group = group.copy()

            # 見送り判定：新馬戦のみ除外
            if '新馬' in str(group.get('race_name', pd.Series()).iloc[0]):
                for st in [stats_1, stats_2, stats_3]:
                    st['months'][m_key]['skipped'] += 1
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

            is_3ren_tan_race = (gap_1_2 >= 0.07) or (gap_1_2 < 0.035 and gap_1_3 >= 0.06)

            if is_3ren_tan_race:
                cost_tan = 2400
                hit_tan = ("◎" in actual_top3_set and actual_top3_set.issubset(top5))
                pay_tan = get_payout_estimate_fixed(group, "3連単") if hit_tan else 0
                
                for st in [stats_1, stats_2, stats_3]:
                    st['months'][m_key]['bet'] += 1
                    st['months'][m_key]['cost'] += cost_tan
                    st['tickets']['3連単(軸1頭24点)']['bet'] += 1
                    st['tickets']['3連単(軸1頭24点)']['cost'] += cost_tan
                    if hit_tan:
                        st['months'][m_key]['hit'] += 1
                        st['months'][m_key]['pay'] += pay_tan
                        st['tickets']['3連単(軸1頭24点)']['hit'] += 1
                        st['tickets']['3連単(軸1頭24点)']['pay'] += pay_tan
            else:
                cost_fuku = 1000
                pay_fuku_raw = get_payout_estimate_fixed(group, "3連複")
                
                hit_box = actual_top3_set.issubset(top5)
                hit_nagashi = ("◎" in actual_top3_set and actual_top3_set.issubset(top6))
                
                is_harAN = (p1 - p4) < 0.08
                
                hit_1 = hit_box if is_harAN else hit_nagashi
                hit_2 = hit_box
                hit_3 = hit_nagashi
                
                hits_and_stats = [(hit_1, stats_1), (hit_2, stats_2), (hit_3, stats_3)]
                for hit_f, st in hits_and_stats:
                    st['months'][m_key]['bet'] += 1
                    st['months'][m_key]['cost'] += cost_fuku
                    st['tickets']['3連複']['bet'] += 1
                    st['tickets']['3連複']['cost'] += cost_fuku
                    if hit_f:
                        st['months'][m_key]['hit'] += 1
                        st['months'][m_key]['pay'] += pay_fuku_raw
                        st['tickets']['3連複']['hit'] += 1
                        st['tickets']['3連複']['pay'] += pay_fuku_raw

    # Output
    print("\n" + "="*85)
    print("🏆 【6月〜8月 3連複買い方3パターン検証結果】")
    print("="*85)

    def print_summary(title, st):
        tot_c = sum(st['months'][m]['cost'] for m in months)
        tot_p = sum(st['months'][m]['pay'] for m in months)
        tot_b = sum(st['months'][m]['bet'] for m in months)
        tot_h = sum(st['months'][m]['hit'] for m in months)
        
        t_fuku = st['tickets']['3連複']
        f_rate = (t_fuku['hit']/t_fuku['bet']*100) if t_fuku['bet']>0 else 0
        f_rec = (t_fuku['pay']/t_fuku['cost']*100) if t_fuku['cost']>0 else 0
        f_prof = t_fuku['pay'] - t_fuku['cost']
        
        rec = (tot_p / tot_c * 100) if tot_c > 0 else 0
        prof = tot_p - tot_c
        
        print(f"\n📢 【{title}】")
        print("─" * 85)
        print(f"   ▶ 3連複のみ成績 : 勝負:{t_fuku['bet']}R | 的中:{t_fuku['hit']}R ({f_rate:.1f}%) | 投資:{t_fuku['cost']:,}円 | 払戻:{t_fuku['pay']:,}円 | 損益:{'+' if f_prof>=0 else ''}{f_prof:,}円 | 回収率:{f_rec:.1f}%")
        print(f"   🔥 通算回収率  : 勝負:{tot_b}R | 的中:{tot_h}R ({tot_h/tot_b*100:.1f}%) | 投資:{tot_c:,}円 | 払戻:{tot_p:,}円 | 損益:{'+' if prof>=0 else ''}{prof:,}円 | 通算回収率:{rec:.1f}%")

    print_summary("① 従来型（波乱気配:5頭BOX / 混戦気配:1軸5頭流し）", stats_1)
    print_summary("② 全て5頭BOX（波乱・混戦ともにBOX）", stats_2)
    print_summary("③ 全て1軸5頭流し（波乱・混戦ともに◎軸流し）", stats_3)
    print("="*85)

if __name__ == "__main__":
    main()