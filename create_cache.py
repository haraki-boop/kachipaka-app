import os
import re
import gc
import joblib
import numpy as np
import pandas as pd
import unicodedata
import warnings
warnings.filterwarnings('ignore')

DB_CSV = "keiba_database2.csv"
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

    print(f"🔄 過去マスターデータベース ({DB_CSV}) を読み込み中...")
    try:
        df = pd.read_csv(DB_CSV, low_memory=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(DB_CSV, low_memory=False, encoding='cp932')

    print("  -> [1/4] クレンジング & プレミアムデータの基礎変数化...")
    
    df['rank_num'] = pd.to_numeric(df['着順'], errors='coerce')
    df['target_win'] = (df['rank_num'] == 1).astype(int)
    df['target_place'] = (df['rank_num'] <= 3).astype(int)
    
    date_clean = df['date'].astype(str).str.replace('年', '-').str.replace('月', '-').str.replace('日', '').str.replace('/', '-')
    df['date_parsed'] = pd.to_datetime(date_clean, errors='coerce')
    
    num_cols = ['タイム指数', '調子偏差値', '同条件_馬場指数', '同距離_馬場指数', '他距離_馬場指数', '通過', '上り']
    for c in num_cols:
        if c in df.columns:
            if c == '通過':
                df[c] = pd.to_numeric(df[c].astype(str).str.split('-').str[0], errors='coerce')
            else:
                df[c] = pd.to_numeric(df[c].astype(str).replace('**', np.nan).replace('', np.nan), errors='coerce')

    if '馬体重' in df.columns:
        weight_extract = df['馬体重'].astype(str).str.extract(r'(\d+)\s*\(([-+]?\d+)\)')
        df['weight'] = pd.to_numeric(weight_extract[0], errors='coerce')
        df['weight_change'] = pd.to_numeric(weight_extract[1], errors='coerce')
    else:
        df['weight'], df['weight_change'] = np.nan, np.nan

    df['prize'] = pd.to_numeric(df.get('賞金(万円)', '0').astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
    
    df['race_id_str'] = df['race_id'].astype(str)
    places = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
    df['place_name'] = df['race_id_str'].str[4:6].map(places).fillna('不明')
    
    df['surface'] = df.get('surface', pd.Series(index=df.index)).fillna('不明')
    df['distance'] = pd.to_numeric(df.get('distance', pd.Series(index=df.index)), errors='coerce').fillna(0)
    df['dist_category'] = pd.cut(df['distance'], bins=[0, 1300, 1700, 2100, 4000], labels=['短距離', 'マイル', '中距離', '長距離']).astype(str)

    # 2. キー情報の名寄せ
    print("  -> [2/4] キー情報（馬名・騎手・調教師）の正規化...")
    df['馬名_clean'] = df['馬名'].astype(str).apply(clean_name)
    df['騎手_clean'] = df['騎手'].astype(str).apply(clean_name)
    
    if '調教師' in df.columns:
        df['調教師_clean'] = df['調教師'].astype(str).str.replace(r'\[.*?\]\n', '', regex=True).apply(clean_name)
    elif 'trainer' in df.columns:
        df['調教師_clean'] = df['trainer'].astype(str).str.replace(r'\[.*?\]\n', '', regex=True).apply(clean_name)
    else:
        df['調教師_clean'] = '不明'

    df = df.sort_values('date_parsed').reset_index(drop=True)
    valid_db = df.dropna(subset=['rank_num']).copy()

    # 3. 騎手・調教師の成績マップ作成
    print("  -> [3/4] 騎手・調教師のフルコンボ成績辞書を作成中...")
    
    def safe_to_dict(series):
        return series.dropna().to_dict()

    jockey_win_map = safe_to_dict(valid_db.groupby('騎手_clean')['target_win'].mean())
    jockey_place_map = safe_to_dict(valid_db.groupby('騎手_clean')['target_place'].mean())
    trainer_win_map = safe_to_dict(valid_db.groupby('調教師_clean')['target_win'].mean())
    trainer_place_map = safe_to_dict(valid_db.groupby('調教師_clean')['target_place'].mean())

    jp_win_map = safe_to_dict(valid_db.groupby(['騎手_clean', 'place_name'])['target_win'].mean())
    jp_place_map = safe_to_dict(valid_db.groupby(['騎手_clean', 'place_name'])['target_place'].mean())
    tp_win_map = safe_to_dict(valid_db.groupby(['調教師_clean', 'place_name'])['target_win'].mean())

    js_win_map = safe_to_dict(valid_db.groupby(['騎手_clean', 'surface'])['target_win'].mean())
    js_place_map = safe_to_dict(valid_db.groupby(['騎手_clean', 'surface'])['target_place'].mean())
    jd_win_map = safe_to_dict(valid_db.groupby(['騎手_clean', 'dist_category'])['target_win'].mean())
    jd_place_map = safe_to_dict(valid_db.groupby(['騎手_clean', 'dist_category'])['target_place'].mean())

    ts_win_map = safe_to_dict(valid_db.groupby(['調教師_clean', 'surface'])['target_win'].mean())
    ts_place_map = safe_to_dict(valid_db.groupby(['調教師_clean', 'surface'])['target_place'].mean())

    jt_win_map = safe_to_dict(valid_db.groupby(['騎手_clean', '調教師_clean'])['target_win'].mean())
    jt_place_map = safe_to_dict(valid_db.groupby(['騎手_clean', '調教師_clean'])['target_place'].mean())

    # 4. 競走馬の直近データ・EMA辞書作成
    print("  -> [4/4] 競走馬の最新ラグ（過去5走）・EMA・安定度（Std）フルキャスト辞書を作成中...")
    horse_dict = {}
    
    lag_cols = ['rank_num', 'タイム指数', '調子偏差値', '同条件_馬場指数', '同距離_馬場指数', '他距離_馬場指数', 'prize', 'weight', 'weight_change', '通過', '上り']
    ema_cols = ['タイム指数', '調子偏差値', 'prize', '通過', '上り']

    grouped = valid_db.groupby('馬名_clean')
    
    for horse, group in grouped:
        h_data = {}
        h_data['last_date'] = group['date_parsed'].iloc[-1]
        h_data['horse_win_rate'] = group['target_win'].mean()
        h_data['horse_place_rate'] = group['target_place'].mean()

        for c in lag_cols:
            if c in group.columns:
                vals = group[c].values
                for lag in range(1, 6):
                    h_data[f'prev{lag}_{c}'] = vals[-lag] if len(vals) >= lag else np.nan

        for c in ema_cols:
            if c in group.columns:
                ts = group[c].dropna()
                if len(ts) > 0:
                    ema3 = ts.ewm(span=3, min_periods=1).mean().iloc[-1]
                    ema5 = ts.ewm(span=5, min_periods=1).mean().iloc[-1]
                    ema10 = ts.ewm(span=10, min_periods=1).mean().iloc[-1]
                    ema20 = ts.ewm(span=20, min_periods=1).mean().iloc[-1]
                    std3 = ts.rolling(window=3, min_periods=1).std().iloc[-1] if len(ts) >= 3 else 0
                    std5 = ts.rolling(window=5, min_periods=1).std().iloc[-1] if len(ts) >= 5 else 0
                    std10 = ts.rolling(window=10, min_periods=1).std().iloc[-1] if len(ts) >= 10 else 0
                    std20 = ts.rolling(window=20, min_periods=1).std().iloc[-1] if len(ts) >= 20 else 0
                    cmax = ts.max()
                    prev1 = ts.iloc[-1]
                else:
                    ema3 = ema5 = ema10 = ema20 = cmax = prev1 = np.nan
                    std3 = std5 = std10 = std20 = 0

                h_data[f'ema3_{c}'] = ema3
                h_data[f'ema5_{c}'] = ema5
                h_data[f'ema10_{c}'] = ema10
                h_data[f'ema20_{c}'] = ema20
                h_data[f'std3_{c}'] = std3
                h_data[f'std5_{c}'] = std5
                h_data[f'std10_{c}'] = std10
                h_data[f'std20_{c}'] = std20
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
        'tp_win_map': tp_win_map,
        'js_win_map': js_win_map,
        'js_place_map': js_place_map,
        'jd_win_map': jd_win_map,
        'jd_place_map': jd_place_map,
        'ts_win_map': ts_win_map,
        'ts_place_map': ts_place_map,
        'jt_win_map': jt_win_map,
        'jt_place_map': jt_place_map
    }, CACHE_FILE, compress=3)

    print(f"🎉 【完了】推論用キャッシュ（{CACHE_FILE}）の作成に成功しました！")

if __name__ == "__main__":
    main()