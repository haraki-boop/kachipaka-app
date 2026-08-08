import os
import re
import sys
import time
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup

INPUT_CSV = "cleaned_keiba_data.csv"
OUTPUT_CSV = "enhanced_keiba_data.csv"
MAX_CONSECUTIVE_ERRORS = 5  # 連続通信エラーの上限

def safe_read_csv(filepath):
    """全ての列を文字列型(str)として安全に読み込む"""
    for enc in ['utf-8-sig', 'cp932', 'euc-jp', 'utf-8']:
        try:
            return pd.read_csv(filepath, encoding=enc, low_memory=False, dtype=str)
        except Exception:
            continue
    return None

def get_race_info(race_id, session):
    """【Phase 1用】コース条件（芝ダ/距離/馬場）の取得"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        res = session.get(url, timeout=10)
        if res.status_code != 200:
            return None, res.status_code
            
        soup = BeautifulSoup(res.content, "html.parser", from_encoding="euc-jp")
        diary = soup.find(class_="data_intro")
        if not diary:
            return None, 200
            
        diary_text = diary.text.replace('\n', '').replace(' ', '')
        surf_match = re.search(r'(芝|ダート|ダ|障)[^\d]*(\d+)m', diary_text)
        surface = surf_match.group(1) if surf_match else "不明"
        if surface == "ダート": surface = "ダ"
        distance = surf_match.group(2) if surf_match else "0"
        
        cond_match = re.search(r'(良|稍重|重|不良)', diary_text)
        condition = cond_match.group(1) if cond_match else "不明"
        
        return {"surface": str(surface), "distance": str(distance), "condition": str(condition)}, 200
    except Exception:
        return None, 500

def get_clean_horse_data(race_id, session):
    """【Phase 2用】EUC-JPで正確な文字（馬名・騎手・陣営）を再取得"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        res = session.get(url, timeout=10)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.content, "html.parser", from_encoding="euc-jp")
        table = soup.find("table", class_="race_table_01")
        if not table:
            return None
        
        horse_data = {}
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) >= 15:
                raw_ub = cols[2].text.strip()
                umaban = str(int(raw_ub)) if raw_ub.isdigit() else raw_ub
                horse_name = cols[3].text.strip()
                jockey = cols[6].text.strip()
                trainer = cols[18].text.strip() if len(cols) > 18 else ""
                owner = cols[19].text.strip() if len(cols) > 19 else ""
                horse_data[umaban] = {
                    "馬名": horse_name,
                    "騎手": jockey,
                    "調教師": trainer,
                    "馬主": owner
                }
        return horse_data
    except Exception:
        return None

def phase1_collect_data(df, session):
    """Phase 1: 未処理データの収集（中断・再開対応）"""
    print("\n==========================================")
    print(" PHASE 1: 未処理データの収集（コース・距離・馬場）")
    print("==========================================")
    
    for col in ['surface', 'distance', 'condition']:
        if col not in df.columns:
            df[col] = "不明"
        df[col] = df[col].fillna("不明").astype(str)

    all_ids = list(df['race_id'].dropna().unique())
    done_ids = set(df[df['surface'].str.strip().ne("不明") & df['surface'].str.strip().ne("") & df['surface'].notna()]['race_id'].unique())
    target_ids = [rid for rid in all_ids if rid not in done_ids]

    total_all = len(all_ids)
    total_done = len(done_ids)
    total_target = len(target_ids)

    print(f"📊 総レース数: {total_all} 件 | 取得済み: {total_done} 件 | 未処理: {total_target} 件")

    if total_target == 0:
        print("✅ フェーズ1完了: 全レースの舞台設定が取得済みです。")
        return df, True

    if total_done > 0:
        print(f"🔄 【再開モード】前回からの続き（{total_done + 1}件目）より再開します。")

    consecutive_errors = 0
    save_counter = 0

    try:
        for i, rid in enumerate(target_ids):
            current_idx = total_done + i + 1
            print(f"\r⏳ [Phase1 {current_idx}/{total_all}] (今回残り: {i+1}/{total_target}) ID: {rid} | 連続エラー: {consecutive_errors}", end="")

            info, status_code = get_race_info(rid, session)

            if info is not None:
                mask = df['race_id'] == rid
                df.loc[mask, 'surface'] = info['surface']
                df.loc[mask, 'distance'] = info['distance']
                df.loc[mask, 'condition'] = info['condition']
                consecutive_errors = 0
                save_counter += 1
            else:
                consecutive_errors += 1
                if status_code == 403:
                    print(f"\n⚠️ 403 Access Denied（アクセス制限）を検知しました。")

            # 20件ごとに自動保存
            if save_counter % 20 == 0 and save_counter > 0:
                df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

            # 連続エラー停止
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n\n🛑 連続エラーが{MAX_CONSECUTIVE_ERRORS}回に達したため一時停止します。")
                print("💾 ここまでの進捗を保存しました。再実行で続きから再開できます。")
                df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
                return df, False

            time.sleep(random.uniform(1.2, 1.8))

    except KeyboardInterrupt:
        print("\n\n⛔ ユーザー操作により中断されました。データを保存しています...")
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print("💾 保存完了。次回実行時にこの続きから再開できます。")
        sys.exit(0)

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ フェーズ1完了: {OUTPUT_CSV} に保存しました。")
    return df, True

def phase2_repair_data(df, session):
    """Phase 2: 文字化けデータの修復（中断・再開対応）"""
    print("\n==========================================")
    print(" PHASE 2: 文字化けデータの検証・修復")
    print("==========================================")

    df_str = df.astype(str)
    corrupted_mask = df_str.apply(lambda x: x.str.contains(r'|\?{2,}', na=False)).any(axis=1)
    corrupted_race_ids = df[corrupted_mask]['race_id'].unique()

    total_corrupted = len(corrupted_race_ids)
    print(f"📊 残り文字化けレース数: {total_corrupted} 件")

    if total_corrupted == 0:
        print("🎉 文字化けデータはゼロです！すべての処理が正常に完了しました。")
        return df

    consecutive_errors = 0
    fixed_count = 0

    def norm_ub(val):
        s = str(val).strip().replace('.0', '')
        return str(int(s)) if s.isdigit() else s

    df['馬番_norm'] = df['馬番'].apply(norm_ub)

    try:
        for i, rid in enumerate(corrupted_race_ids):
            print(f"\r⏳ [Phase2 {i+1}/{total_corrupted}] 修復中 ID: {rid} | 連続エラー: {consecutive_errors}", end="")

            h_data = get_clean_horse_data(rid, session)
            if h_data:
                for umaban, info in h_data.items():
                    mask = (df['race_id'] == rid) & (df['馬番_norm'] == umaban)
                    if mask.any():
                        for col, val in info.items():
                            if col in df.columns:
                                df.loc[mask, col] = str(val)
                fixed_count += 1
                consecutive_errors = 0
            else:
                consecutive_errors += 1

            if fixed_count % 20 == 0 and fixed_count > 0:
                if '馬番_norm' in df.columns:
                    df = df.drop(columns=['馬番_norm'])
                df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
                df['馬番_norm'] = df['馬番'].apply(norm_ub)

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n\n🛑 連続エラーが{MAX_CONSECUTIVE_ERRORS}回に達したため一時停止します。")
                print("💾 ここまでの修復進捗を保存しました。再実行で続きから再開できます。")
                break

            time.sleep(random.uniform(1.2, 1.8))

    except KeyboardInterrupt:
        print("\n\n⛔ ユーザー操作により中断されました。修復データを保存しています...")
        if '馬番_norm' in df.columns:
            df = df.drop(columns=['馬番_norm'])
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print("💾 保存完了。次回実行時に残りの文字化け修復から再開できます。")
        sys.exit(0)

    if '馬番_norm' in df.columns:
        df = df.drop(columns=['馬番_norm'])
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉 修復完了！最新データを {OUTPUT_CSV} に保存しました。")
    return df

def main():
    filepath = OUTPUT_CSV if os.path.exists(OUTPUT_CSV) else INPUT_CSV
    if not os.path.exists(filepath):
        print(f"❌ 入力ファイルが見つかりません: {filepath}")
        return

    print(f"📂 データベースを読み込んでいます: {filepath}")
    df = safe_read_csv(filepath)
    if df is None:
        print("❌ CSVファイルの読み込みに失敗しました。")
        return

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://db.netkeiba.com/",
    })

    # Phase 1: 未処理データの収集
    df, phase1_success = phase1_collect_data(df, session)

    # Phase 2: 文字化け修復（Phase 1 が完了した後に実行）
    if phase1_success:
        df = phase2_repair_data(df, session)

if __name__ == "__main__":
    main()