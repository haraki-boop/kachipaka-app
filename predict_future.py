import os
import re
import joblib
import unicodedata
import numpy as np
import pandas as pd

# ==========================================
# 🧠 勝ちパカくん: AI気配察知・自動買い目予想エンジン
# ==========================================

MODEL_PATHS = ["keiba_ai_model.pkl", "勝ちパカくん.pkl"]
FUTURE_CSV = "future_races.csv"

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip().upper()

def main():
    model_path = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            model_path = p
            break

    if not model_path or not os.path.exists(FUTURE_CSV):
        print("❌ 必要なモデルファイルまたは出馬表 (future_races.csv) が見つかりません。")
        return

    print("🔄 モデルと最新出馬表を読み込んでいます...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    features = model_data['features']

    df = pd.read_csv(FUTURE_CSV, low_memory=False)
    if 'race_id' not in df.columns or df.empty:
        print("❌ 対象となる未来のレースデータが存在しません。")
        return

    # 特徴量の計算
    X_future = pd.DataFrame(index=df.index)
    for f in features:
        X_future[f] = pd.to_numeric(df[f], errors='coerce') if f in df.columns else np.nan

    df['raw_score'] = model.predict(X_future)

    print("\n" + "="*85)
    print("🏇 【勝ちパカくん】本気配察知・適応買い目 予想レポート")
    print("="*85)

    total_races = 0

    for race_id, group in df.groupby('race_id'):
        total_races += 1
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
        
        # 気配判別のスコアギャップ
        gap_1_2 = p1 - p2
        gap_1_3 = p1 - p3

        marks = ["◎", "◯", "▲", "△", "☆1", "☆2"]
        group['印'] = "消"
        for i in range(min(len(group), len(marks))):
            group.loc[i, '印'] = marks[i]

        r_name = group['race_name'].iloc[0] if 'race_name' in group.columns and pd.notna(group['race_name'].iloc[0]) else f"Race {race_id}"
        date_str = group['date'].iloc[0] if 'date' in group.columns and pd.notna(group['date'].iloc[0]) else ""

        # 気配判定と最適な買い目の割り振りを自動実行
        if gap_1_2 >= 0.07:
            pat = "① 1強気配 (軸圧倒)"
            rec_ticket = "3連単 1着固定 (6点 / 600円)"
            buy_detail = f"1着: {group.loc[0, '馬番']}(◎) -> 2・3着: {group.loc[1, '馬番']}(◯), {group.loc[2, '馬番']}(▲), {group.loc[3, '馬番']}(△)"
        elif gap_1_2 < 0.035 and gap_1_3 >= 0.06:
            pat = "② 2強気配 (頭分け対抗)"
            rec_ticket = "3連単 ダブル軸 (12点 / 1,200円)"
            buy_detail = f"1着: {group.loc[0, '馬番']}(◎), {group.loc[1, '馬番']}(◯) -> 2着: ◎, ◯, {group.loc[2, '馬番']}(▲) -> 3着: ◎, ◯, ▲, {group.loc[3, '馬番']}(△), {group.loc[4, '馬番'] if len(group)>4 else ''}(☆1)"
        elif (p1 - p4) < 0.08:
            pat = "④ 波乱気配 (大混戦)"
            rec_ticket = "3連複 5頭BOX (10点 / 1,000円)"
            u_list = [f"{group.loc[k, '馬番']}({group.loc[k, '印']})" for k in range(min(5, len(group)))]
            buy_detail = f"BOX: {', '.join(u_list)}"
        else:
            pat = "③ 混戦気配 (標準展開)"
            rec_ticket = "3連複 ◎1頭軸流し (10点 / 1,000円)"
            u_others = [f"{group.loc[k, '馬番']}({group.loc[k, '印']})" for k in range(1, min(6, len(group)))]
            buy_detail = f"軸: {group.loc[0, '馬番']}(◎) -> 相手: {', '.join(u_others)}"

        print(f"\n📍 レースID: {race_id} | {date_str} {r_name}")
        print(f"  ├ 🧠 勝負気配  : {pat}")
        print(f"  ├ 🎟️ 推奨買い目: {rec_ticket}")
        print(f"  ├ 📝 買目詳細  : {buy_detail}")
        top_str = " | ".join([f"{group.loc[k, '印']}:{group.loc[k, '馬名']}({group.loc[k, '馬番']})" for k in range(min(4, len(group)))])
        print(f"  └ 🐴 上位評価  : {top_str}")

    print("\n" + "="*85)
    print(f"✅ 全 {total_races} レースの予想および気配察知アドバイスの出力完了！")
    print("="*85)

if __name__ == "__main__":
    main()