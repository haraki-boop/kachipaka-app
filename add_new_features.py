import pandas as pd
import numpy as np

def main():
    input_file = "ml_target_data.csv"
    output_file = "ml_target_data_v2.csv"

    print("🔄 データを読み込んでいます...")
    df = pd.read_csv(input_file, low_memory=False)

    # タイムトラベル（未来のデータカンニング）を防ぐため日付順にソート
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date_parsed').reset_index(drop=True)

    print("📊 1. 調教師（厩舎）スコアを計算中...")
    # 調教師の過去の勝率
    df['trainer_win_rate'] = df.groupby('調教師')['is_win'].transform(lambda x: x.shift().expanding().mean()).fillna(0.08)
    # 調教師 × 騎手の黄金タッグ勝率（ヤリ気配の検知）
    df['trainer_jockey_combo'] = df.groupby(['調教師', '騎手'])['is_win'].transform(lambda x: x.shift().expanding().mean()).fillna(0.08)

    print("🏟️ 2. コース・馬場適性を計算中...")
    # その馬の、当該競馬場（place_code）での過去勝率
    df['horse_track_win_rate'] = df.groupby(['馬名', 'place_code'])['is_win'].transform(lambda x: x.shift().expanding().mean()).fillna(0.0)
    
    print("🎲 3. 枠順バイアス（有利不利）を計算中...")
    # 競馬場 × 距離 における各枠番（1〜8枠）の勝率
    df['frame_win_rate'] = df.groupby(['place_code', 'distance', '枠番'])['is_win'].transform(lambda x: x.shift().expanding().mean()).fillna(0.08)

    # NaN（初回出走など）を平均的な数値で埋める
    cols_to_fill = ['trainer_win_rate', 'trainer_jockey_combo', 'horse_track_win_rate', 'frame_win_rate']
    df[cols_to_fill] = df[cols_to_fill].fillna(0)

    print("💾 新しいデータを保存しています...")
    df.to_csv(output_file, index=False)
    print(f"✅ 完了！ 新特徴量を追加した '{output_file}' を作成しました。")

if __name__ == "__main__":
    main()