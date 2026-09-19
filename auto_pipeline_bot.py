import os
import sys
import time
import subprocess
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

def print_header(title):
    print(f"\n{Fore.YELLOW}{'='*60}")
    print(f"{Fore.YELLOW} {title}")
    print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")

def run_script(script_name, step_name):
    print_header(step_name)
    if not os.path.exists(script_name):
        print(f"{Fore.YELLOW}⚠️ {script_name} が見つかりません。スキップします。")
        return True
    try:
        print(f"{Fore.CYAN}  [🤖 実行中] {script_name} を実行しています...")
        subprocess.run([sys.executable, script_name], check=True, text=True)
        print(f"{Fore.GREEN}✅ {script_name} の実行が完了しました。")
        return True
    except subprocess.CalledProcessError:
        print(f"{Fore.RED}❌ {script_name} の実行中にエラーが発生しました。")
        return False

def step_git_push():
    print_header("☁️ 最終ステップ: GitHubへの自動コミット＆Push")
    
    # 修正: UI表示に必要な future_races.csv をPush対象に戻しています
    files_to_add = ["future_races.csv", "keiba_ai_model.pkl", "app_cache.pkl", "prediction_history.csv"]
    existing_files = [f for f in files_to_add if os.path.exists(f)]

    if not existing_files:
        print(f"{Fore.YELLOW}⚠️ 追加する対象ファイルが見つかりません。")
        return

    try:
        print(f"{Fore.CYAN}  [🤖 送信中] 完成版モデルおよび予測キャッシュ(app_cache.pkl)をステージング中...")
        subprocess.run(["git", "add"] + existing_files, check=True)
        commit_msg = f"Auto-pipeline update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        subprocess.run(["git", "commit", "-m", commit_msg])
        
        print(f"{Fore.CYAN}  [🤖 送信中] GitHubへデータをPush中...")
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"{Fore.GREEN}✅ GitHub同期完了！Streamlit Cloud側も最新予測データで同期されました。")
    except subprocess.CalledProcessError:
        print(f"{Fore.YELLOW}⚠️ Git Push時にエラー（または変更なし）が発生しました。")

def main():
    start_time = time.time()
    
    print(f"\n{Fore.MAGENTA}==========================================")
    print(f"{Fore.MAGENTA} 🤖 勝ちパカくん 全自動パイプラインBOT 起動")
    print(f"{Fore.MAGENTA}=========================================={Style.RESET_ALL}\n")

    steps = [
        ("scrape_results.py", "STEP 1: 確定レース結果の取得・最新日付に更新"),
        ("create_rich_features.py", "STEP 2: 蓄積データの更新 (ml_target_data_v2.csv 生成)"),
        ("train_lightgbm.py", "STEP 3: AIモデルの再学習 (脳みそ更新)"),
        ("create_cache.py", "STEP 4: 推論用辞書の作成 (app_cache.pkl 生成)"),
        ("scrape_shutsuba.py", "STEP 5: 最新出馬表データの取得 (future_races.csv 生成)"),
        ("predict_future.py", "STEP 6: AI特徴量検証・推論・脚質・買い目の一括算出 (app_cache.pkl保存)")
    ]

    for script, desc in steps:
        if not run_script(script, desc):
            print(f"{Fore.RED}⛔ {script} で致命的なエラーが発生したため、後続の処理を中断します。")
            return

    step_git_push()

    elapsed = round(time.time() - start_time, 1)
    print(f"\n{Fore.MAGENTA}==========================================")
    print(f"{Fore.GREEN} 🎉 全パイプライン処理完了 (所要時間: {elapsed} 秒)")
    print(f"{Fore.MAGENTA}=========================================={Style.RESET_ALL}\n")
    time.sleep(3)

if __name__ == "__main__":
    main()