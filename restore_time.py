import pandas as pd

def restore_time_column():
    print("データの読み込みを開始します...")
    
    # 壊れたファイルの読み込み
    try:
        df_ml = pd.read_csv("ml_target_data.csv", low_memory=False, encoding="utf-8-sig")
    except:
        df_ml = pd.read_csv("ml_target_data.csv", low_memory=False, encoding="cp932")

    # 原本ファイル（keiba_database.csv）の読み込み
    try:
        df_db = pd.read_csv("keiba_database.csv", low_memory=False, encoding="utf-8-sig")
    except:
        df_db = pd.read_csv("keiba_database.csv", low_memory=False, encoding="cp932")

    # ml_target_data の空欄になった「タイム」列を一旦削除
    if 'タイム' in df_ml.columns:
        df_ml = df_ml.drop(columns=['タイム'])

    # keiba_database から結合キー（race_id, 馬番）と「タイム」列だけを抽出
    # ※同名馬の重複を避けるため race_id と 馬番 をキーにします
    df_time_source = df_db[['race_id', '馬番', 'タイム']].drop_duplicates()

    # 正しいタイム列を結合して復旧
    df_ml = pd.merge(df_ml, df_time_source, on=['race_id', '馬番'], how='left')

    # 上書き保存
    df_ml.to_csv("ml_target_data.csv", index=False, encoding="utf-8-sig")
    print("タイム列の復旧が完了しました。")

if __name__ == "__main__":
    restore_time_column()