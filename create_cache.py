import os
import re
import gc
import joblib
import numpy as np
import pandas as pd
import unicodedata
import warnings
warnings.filterwarnings('ignore')

DB_CSV = "keiba_database.csv"
CACHE_FILE = "app_cache.pkl"

def clean_name(text):
    if pd.isna(text): return ""
    s = unicodedata.normalize('NFKC', str(text))
    s = re.sub(r'[\s・･.\-ー_]+', '', s).strip()
    return s.upper()

def main():
    if not os.path.exists(DB_CSV):
        print(f"❌ エラー: {DB_CSV} が見つかりません。")
        return

    print("🔄 過去マスターデータベースを読み込み中...")
    df = pd.read_csv(DB_CSV, low_memory=False, encoding='utf-8-sig')

    # 1. カラム名の正規化と基礎データのクレンジング
    rename_dict = {}
    for col in df.columns:
        if 'ﾀｲﾑ指数M' in col: rename_dict[col] = 'time_idx_m'
        elif 'ﾀｲﾑ指数' in col: rename_dict[col] = 'time_idx'
        elif 'ｽﾀｰﾄ' in col: rename_dict[col] = 'start_idx'
        elif '追走' in col: rename_dict[col] = 'pace_idx'
        elif '上がり' in col: rename_dict[col] = 'last3f_idx'
        elif '賞金' in col: rename_dict[col] = 'prize'
    df = df.rename(columns=rename_dict)

    df['rank_num'] = pd.to_numeric(df['着順'], errors='coerce')
    df['target_win'] = (df['rank_num'] == 1).astype(int)
    df['target_place'] = (df['rank_num'] <= 3).astype(int)

    date_clean = df['date'].astype(str).str.replace('年', '-').str.replace('月', '-').str.replace('日', '').str.replace('/', '-')
    df['date_parsed'] = pd.to_datetime(date_clean, errors='coerce')

    for c in ['time_idx', 'time_idx_m', 'start_idx', 'pace_idx', 'last3f_idx']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).replace('**', np.nan), errors='coerce')

    if '馬体重' in df.columns:
        weight_extract = df['馬体重'].astype(str).str.extract(r'(\d+)\s*\(([-+]?\d+)\)')
        df['weight'] = pd.to_numeric(weight_extract[0], errors='coerce')
        df['weight_change'] = pd.to_numeric(weight_extract[1], errors='coerce')

    df['prize'] = pd.to_numeric(df.get('prize', '0').astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)

    # 2. キー情報の名寄せ
    df['馬名_clean'] = df['馬名'].astype(str).apply(clean_name)
    df['騎手_clean'] = df['騎手'].astype(str).apply(clean_name)
    df['調教師_clean'] = df['調教師'].astype(str).str.replace(r'\[.*?\]\n', '', regex=True).apply(clean_name)
    
    df['race_id_str'] = df['race_id'].astype(str)
    places = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京',
              '06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
    df['place_name'] = df['race_id_str'].str[4:6].map(places).fillna('不明')

    df = df.sort_values('date_parsed').reset_index(drop=True)
    valid_db = df.dropna(subset=['rank_num']).copy()

    # 3. 騎手・調教師の「未来に向けた」勝率辞書
    # ※未来のレースから見れば全データが「過去」になるため、単純なmean()が正解となります
    print("🧠 騎手・調教師の成績マップを作成中...")
    jockey_win_map = valid_db.groupby('騎手_clean')['target_win'].mean().to_dict()
    jockey_place_map = valid_db.groupby('騎手_clean')['target_place'].mean().to_dict()
    trainer_win_map = valid_db.groupby('調教師_clean')['target_win'].mean().to_dict()
    trainer_place_map = valid_db.groupby('調教師_clean')['target_place'].mean().to_dict()

    jp_win_map = valid_db.groupby(['騎手_clean', 'place_name'])['target_win'].mean().to_dict()
    jp_place_map = valid_db.groupby(['騎手_clean', 'place_name'])['target_place'].mean().to_dict()
    tp_win_map = valid_db.groupby(['調教師_clean', 'place_name'])['target_win'].mean().to_dict()

    # 4. 各馬の直近ラグ・EMA辞書を計算
    print("🐴 競走馬の最新ラグ・EMAキャッシュを作成中...")
    horse_dict = {}
    
    lag_cols = ['rank_num', 'time_idx', 'start_idx', 'pace_idx', 'last3f_idx', 'prize', 'weight', 'weight_change']
    ema_cols = ['time_idx', 'start_idx', 'pace_idx', 'last3f_idx', 'prize']

    grouped = valid_db.groupby('馬名_clean')
    
    for horse, group in grouped:
        h_data = {}
        h_data['last_date'] = group['date_parsed'].iloc[-1]
        h_data['horse_win_rate'] = group['target_win'].mean()
        h_data['horse_place_rate'] = group['target_place'].mean()

        # 過去1走前〜5走前のラグ取得
        for c in lag_cols:
            if c in group.columns:
                vals = group[c].values
                for lag in range(1, 6):
                    h_data[f'prev{lag}_{c}'] = vals[-lag] if len(vals) >= lag else np.nan

        # 最新レース時点でのEMA（＝次走にとっての事前EMA）
        for c in ema_cols:
            if c in group.columns:
                ts = group[c].dropna()
                if len(ts) > 0:
                    ema3 = ts.ewm(span=3, min_periods=1).mean().iloc[-1]
                    ema5 = ts.ewm(span=5, min_periods=1).mean().iloc[-1]
                    ema10 = ts.ewm(span=10, min_periods=1).mean().iloc[-1]
                    ema20 = ts.ewm(span=20, min_periods=1).mean().iloc[-1]
                    cmax = ts.max()
                    prev1 = ts.iloc[-1]
                else:
                    ema3 = ema5 = ema10 = ema20 = cmax = prev1 = np.nan

                h_data[f'ema3_{c}'] = ema3
                h_data[f'ema5_{c}'] = ema5
                h_data[f'ema10_{c}'] = ema10
                h_data[f'ema20_{c}'] = ema20
                h_data[f'trend_3_10_{c}'] = (ema3 - ema10) if pd.notna(ema3) and pd.notna(ema10) else np.nan
                h_data[f'trend_5_20_{c}'] = (ema5 - ema20) if pd.notna(ema5) and pd.notna(ema20) else np.nan
                h_data[f'max_{c}'] = cmax
                h_data[f'ratio_to_max_{c}'] = (prev1 / cmax) if cmax and cmax > 0 else np.nan

        horse_dict[horse] = h_data

    print(f"💾 キャッシュファイル（{CACHE_FILE}）を保存中...")
    joblib.dump({
        'horse_dict': horse_dict,
        'jockey_win_map': jockey_win_map,
        'jockey_place_map': jockey_place_map,
        'trainer_win_map': trainer_win_map,
        'trainer_place_map': trainer_place_map,
        'jp_win_map': jp_win_map,
        'jp_place_map': jp_place_map,
        'tp_win_map': tp_win_map
    }, CACHE_FILE)

    print(f"🎉 【完了】推論用キャッシュ（{CACHE_FILE}）の作成に成功しました！")

if __name__ == "__main__":
    main()