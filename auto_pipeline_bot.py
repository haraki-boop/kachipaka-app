import os
import re
import sys
import time
import joblib
import subprocess
import unicodedata
import numpy as np
import pandas as pd
from datetime import datetime
from colorama import init, Fore, Style

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Windows環境の文字色初期化
init(autoreset=True)

# モジュール読み込み用にパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

FUTURE_CSV = "future_races.csv"
ML_TARGET_CSV = "ml_target_data.csv"
MODEL_PATHS = ["keiba_ai_model.pkl", "勝ちパカくん.pkl"]

# ==========================================
# 🌟 EnsembleModel クラスの定義（joblib読み込み用）
# ==========================================
class EnsembleModel:
    def __init__(self, lgb_model, xgb_model, cat_model, weights=(0.4, 0.3, 0.3)):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.weights = weights

    def predict(self, X):
        X_num = X.copy()
        for col in X_num.columns:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce').fillna(0)

        # LightGBM
        lgb_pred = self.lgb_model.predict(X_num)
        lgb_std = np.std(lgb_pred)
        lgb_norm = (lgb_pred - np.mean(lgb_pred)) / (lgb_std + 1e-5) if lgb_std > 0 else lgb_pred

        # XGBoost
        xgb_pred = self.xgb_model.predict(xgb.DMatrix(X_num))
        xgb_std = np.std(xgb_pred)
        xgb_norm = (xgb_pred - np.mean(xgb_pred)) / (xgb_std + 1e-5) if xgb_std > 0 else xgb_pred

        # CatBoost
        cat_pred = self.cat_model.predict(X_num)
        cat_std = np.std(cat_pred)
        cat_norm = (cat_pred - np.mean(cat_pred)) / (cat_std + 1e-5) if cat_std > 0 else cat_pred

        w1, w2, w3 = self.weights
        return w1 * lgb_norm + w2 * xgb_norm + w3 * cat_norm

import __main__
__main__.EnsembleModel = EnsembleModel


def print_header(title):
    print(f"\n{Fore.YELLOW}{'='*60}")
    print(f"{Fore.YELLOW} {title}")
    print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")

def step1_fetch_future_races():
    print_header("🚀 STEP 1: 最新出馬表データ(未来のレース)の自動取得")
    
    if os.path.exists("scrape_shutsuba.py"):
        try:
            print(f"{Fore.CYAN}  [🤖 取得中] scrape_shutsuba.py を実行します...")
            res = subprocess.run([sys.executable, "scrape_shutsuba.py"], check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"{Fore.RED}❌ scrape_shutsuba.py の実行中にエラーが発生しました。")
            return False
    else:
        try:
            from predict_with_gemini import run_scraper
            class DummyProgress:
                def progress(self, val): pass
            class DummyText:
                def text(self, msg): 
                    print(f"{Fore.CYAN}  [🤖 取得中] {msg}")
            
            p_text = DummyText()
            p_bar = DummyProgress()
            run_scraper(p_text, p_bar)
        except Exception as e:
            print(f"{Fore.RED}❌ 出馬表データの取得中にエラーが発生しました: {e}")
            return False

    if os.path.exists(FUTURE_CSV) and os.path.getsize(FUTURE_CSV) > 0:
        print(f"{Fore.GREEN}✅ 最新出馬表(＋オッズ)の準備完了: {FUTURE_CSV}")
        return True
    else:
        print(f"{Fore.RED}❌ {FUTURE_CSV} が見つからないか、ファイルが空です。")
        return False

def step1_5_fetch_past_results():
    print_header("📥 STEP 1.5: 確定レース結果(過去データ)の取得・上書き")
    
    result_scripts = ["scrape_results.py", "update_results.py", "scrape_past_races.py"]
    target_script = None
    for script in result_scripts:
        if os.path.exists(script):
            target_script = script
            break

    if target_script:
        try:
            print(f"{Fore.CYAN}  [🤖 更新中] {target_script} を実行して確定着順データを追記・上書きします...")
            subprocess.run([sys.executable, target_script], check=True, text=True)
            print(f"{Fore.GREEN}✅ 確定レース結果の取得・更新が完了しました。")
            return True
        except subprocess.CalledProcessError as e:
            print(f"{Fore.RED}❌ レース結果の更新スクリプト実行時にエラーが発生しました。")
            return False
    else:
        print(f"{Fore.CYAN}ℹ️  個別の着順取得スクリプトが見つからないため、次の特徴量生成ステップで更新を行います。")
        return True

def step2_update_features():
    print_header("📊 STEP 2: 蓄積データの更新・68特徴量生成 (create_features.py)")
    if not os.path.exists("create_features.py"):
        print(f"{Fore.YELLOW}⚠️ create_features.py が見つかりません。既存データで続行します。")
        return True

    try:
        print(f"{Fore.CYAN}  [🤖 計算中] 過去データの集計と全特徴量を作成しています...")
        subprocess.run([sys.executable, "create_features.py"], check=True, text=True)
        print(f"{Fore.GREEN}✅ 蓄積データの集計・特徴量作成が完了しました。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ 特徴量生成中にエラーが発生しました。")
        return False

def step3_retrain_ai():
    print_header("🧠 STEP 3: 蓄積データのAI再学習 (train_lightgbm.py)")
    if not os.path.exists("train_lightgbm.py"):
        print(f"{Fore.YELLOW}⚠️ train_lightgbm.py が見つかりません。再学習をスキップします。")
        return True

    try:
        print(f"{Fore.CYAN}  [🤖 学習中] AIモデルを最新データで再トレーニングしています...")
        subprocess.run([sys.executable, "train_lightgbm.py"], check=True, text=True)
        print(f"{Fore.GREEN}✅ AIモデルの再学習が完了しました。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ AI再学習中にエラーが発生しました。")
        return False

def step3_5_predict_with_ai_sense():
    print_header("🎯 STEP 3.5: AI勝負気配察知＆適応買い目生成エンジンの実行")
    
    model_path = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            model_path = p
            break

    if not model_path or not os.path.exists(FUTURE_CSV):
        print(f"{Fore.RED}❌ モデルファイルまたは {FUTURE_CSV} が存在しません。予想をスキップします。")
        return False

    try:
        model_data = joblib.load(model_path)
        
        # 辞書形式か、そのままモデルクラスかを判定して取得
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            features = model_data.get('features', [])
        else:
            model = model_data
            features = [] # 特徴量リストがない場合は空にする

        df = pd.read_csv(FUTURE_CSV, low_memory=False)
        if 'race_id' not in df.columns or df.empty:
            print(f"{Fore.YELLOW}⚠️ 出馬表に有効なデータがありません。")
            return True

        X_future = pd.DataFrame(index=df.index)
        
        # 特徴量リストがある場合は指定の列だけを抽出し、ない場合は除外列以外を使用
        if features:
            for f in features:
                X_future[f] = pd.to_numeric(df[f], errors='coerce') if f in df.columns else np.nan
        else:
            exclude_cols = ['date', 'race_id', '馬名', '騎手', '調教師', 'place_name', 'race_name', 'day_label']
            use_cols = [c for c in df.columns if c not in exclude_cols]
            for c in use_cols:
                X_future[c] = pd.to_numeric(df[c], errors='coerce')

        df['raw_score'] = model.predict(X_future)

        print(f"\n{Fore.GREEN}🏇 【勝ちパカくん】勝負気配＆最適買い目 レポート")
        print(f"{Fore.GREEN}" + "-"*60)

        results = []
        for race_id, group in df.groupby('race_id'):
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

            r_name = group['race_name'].iloc[0] if 'race_name' in group.columns and pd.notna(group['race_name'].iloc[0]) else f"Race {race_id}"
            date_str = group['date'].iloc[0] if 'date' in group.columns and pd.notna(group['date'].iloc[0]) else ""

            # 気配判別ロジック
            if gap_1_2 >= 0.07:
                pat = "① 1強気配 (軸圧倒)"
                rec_ticket = "3連単 1着固定 (6点/600円)"
                buy_detail = f"1着: {group.loc[0, '馬番']}(◎) -> 2・3着: {group.loc[1, '馬番']}(◯), {group.loc[2, '馬番']}(▲), {group.loc[3, '馬番']}(△)"
            elif gap_1_2 < 0.035 and gap_1_3 >= 0.06:
                pat = "② 2強気配 (頭分け対抗)"
                rec_ticket = "3連単 ダブル軸 (12点/1,200円)"
                buy_detail = f"1着: {group.loc[0, '馬番']}(◎), {group.loc[1, '馬番']}(◯) -> 2着: ◎, ◯, {group.loc[2, '馬番']}(▲) -> 3着: ◎, ◯, ▲, {group.loc[3, '馬番']}(△), {group.loc[4, '馬番'] if len(group)>4 else ''}(☆1)"
            elif (p1 - p4) < 0.08:
                pat = "④ 波乱気配 (大混戦)"
                rec_ticket = "3連複 5頭BOX (10点/1,000円)"
                u_list = [f"{group.loc[k, '馬番']}({group.loc[k, '印']})" for k in range(min(5, len(group)))]
                buy_detail = f"BOX: {', '.join(u_list)}"
            else:
                pat = "③ 混戦気配 (標準展開)"
                rec_ticket = "3連複 ◎1頭軸流し (10点/1,000円)"
                u_others = [f"{group.loc[k, '馬番']}({group.loc[k, '印']})" for k in range(1, min(6, len(group)))]
                buy_detail = f"軸: {group.loc[0, '馬番']}(◎) -> 相手: {', '.join(u_others)}"

            print(f"\n📍 {Fore.CYAN}レースID: {race_id} | {date_str} {r_name}{Style.RESET_ALL}")
            print(f"  ├ 🧠 勝負気配  : {Fore.MAGENTA}{pat}{Style.RESET_ALL}")
            print(f"  ├ 🎟️ 推奨買い目: {Fore.YELLOW}{rec_ticket}{Style.RESET_ALL}")
            print(f"  ├ 📝 買目詳細  : {buy_detail}")
            top_str = " | ".join([f"{group.loc[k, '印']}:{group.loc[k, '馬名']}({group.loc[k, '馬番']})" for k in range(min(4, len(group)))])
            print(f"  └ 🐴 上位評価  : {top_str}")

            results.append({
                'race_id': race_id, 'date': date_str, 'race_name': r_name,
                'pattern': pat, 'ticket': rec_ticket, 'buy_detail': buy_detail
            })

        # 予想結果を prediction_history.csv へ保存
        if results:
            res_df = pd.DataFrame(results)
            res_df.to_csv("prediction_history.csv", index=False, encoding='utf-8-sig')
            print(f"\n{Fore.GREEN}✅ 予想結果を prediction_history.csv に自動出力しました。")

        return True
    except Exception as e:
        print(f"{Fore.RED}❌ 予想生成中にエラーが発生しました: {e}")
        return False

def step4_push_to_github():
    print_header("☁️ STEP 4: GitHubへの自動コミット＆Push (Web版同期)")
    files_to_add = ["future_races.csv", "ml_target_data.csv", "keiba_ai_model.pkl", "prediction_history.csv"]
    existing_files = [f for f in files_to_add if os.path.exists(f)]

    if not existing_files:
        print(f"{Fore.YELLOW}⚠️ 追加する対象ファイルが見つかりません。")
        return

    try:
        print(f"{Fore.CYAN}  [🤖 送信中] 変更ファイルをステージング中...")
        subprocess.run(["git", "add"] + existing_files, check=True)
        commit_msg = f"Auto-pipeline update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        subprocess.run(["git", "commit", "-m", commit_msg])
        
        print(f"{Fore.CYAN}  [🤖 送信中] GitHubへデータをPush中...")
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"{Fore.GREEN}✅ GitHub同期完了！Streamlit Cloud側も最新化されます。")
    except subprocess.CalledProcessError as e:
        print(f"{Fore.YELLOW}⚠️ Git Push時にエラー・警告が発生しました。")

def main():
    start_time = time.time()
    
    print(f"\n{Fore.MAGENTA}==========================================")
    print(f"{Fore.MAGENTA} 🤖 勝ちパカくん 全自動パイプラインBOT 起動")
    print(f"{Fore.MAGENTA}=========================================={Style.RESET_ALL}\n")

    # ① 出馬表データ取得
    if not step1_fetch_future_races():
        print(f"{Fore.RED}⛔ STEP 1 で失敗したため処理を中断します。")
        return

    # ①.5 確定レース結果（着順データ）の取得
    step1_5_fetch_past_results()

    # ② 蓄積データ・特徴量更新
    if not step2_update_features():
        print(f"{Fore.RED}⛔ STEP 2 で失敗したため処理を中断します。")
        return

    # ③ AIモデル再学習
    if not step3_retrain_ai():
        print(f"{Fore.RED}⛔ STEP 3 で失敗したため処理を中断します。")
        return

    # ③.5 AI勝負気配察知・適応予想の実行
    step3_5_predict_with_ai_sense()

    # ④ GitHubへの自動同期
    step4_push_to_github()

    elapsed = round(time.time() - start_time, 1)
    print(f"\n{Fore.MAGENTA}==========================================")
    print(f"{Fore.GREEN} 🎉 全パイプライン処理完了 (所要時間: {elapsed} 秒)")
    print(f"{Fore.MAGENTA}=========================================={Style.RESET_ALL}\n")
    print(f"{Fore.WHITE}このウィンドウは数秒後に自動で閉じます...")
    time.sleep(5)

if __name__ == "__main__":
    main()