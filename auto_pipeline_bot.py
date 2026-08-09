import os
import sys
import time
import subprocess
from datetime import datetime
from colorama import init, Fore, Style

# Windows環境の文字色初期化
init(autoreset=True)

# モジュール読み込み用にパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predict_with_gemini import run_scraper, FUTURE_CSV
    class DummyProgress:
        def progress(self, val): pass
    class DummyText:
        def text(self, msg): 
            print(f"{Fore.CYAN}  [🤖 取得中] {msg}")
except Exception as e:
    print(f"{Fore.RED}❌ predict_with_gemini.py の読み込みに失敗しました: {e}")
    sys.exit(1)


def print_header(title):
    print(f"\n{Fore.YELLOW}{'='*50}")
    print(f"{Fore.YELLOW} {title}")
    print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}")

def step1_fetch_future_races():
    print_header("🚀 STEP 1: 最新レースデータの自動取得")
    p_text = DummyText()
    p_bar = DummyProgress()
    
    success = run_scraper(p_text, p_bar)
    if success and os.path.exists(FUTURE_CSV) and os.path.getsize(FUTURE_CSV) > 0:
        print(f"{Fore.GREEN}✅ 最新出馬表(＋オッズ)の取得完了: {FUTURE_CSV}")
        return True
    else:
        print(f"{Fore.RED}❌ 出馬表データの取得に失敗しました。")
        return False


def step2_update_features():
    print_header("📊 STEP 2: 蓄積データの更新・特徴量生成 (create_features.py)")
    if not os.path.exists("create_features.py"):
        print(f"{Fore.YELLOW}⚠️ create_features.py が見つかりません。スキップします。")
        return False

    try:
        print(f"{Fore.CYAN}  [🤖 計算中] 過去データの集計と特徴量を作成しています...")
        res = subprocess.run([sys.executable, "create_features.py"], check=True, text=True, capture_output=True)
        print(f"{Fore.GREEN}✅ 蓄積データの集計・特徴量作成が完了しました。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ 特徴量生成中にエラーが発生しました:\n{e.stderr}")
        return False


def step3_retrain_ai():
    print_header("🧠 STEP 3: 蓄積データのAI再学習 (train_lightgbm.py)")
    if not os.path.exists("train_lightgbm.py"):
        print(f"{Fore.YELLOW}⚠️ train_lightgbm.py が見つかりません。スキップします。")
        return False

    try:
        print(f"{Fore.CYAN}  [🤖 学習中] AIモデルを最新データで再トレーニングしています...")
        res = subprocess.run([sys.executable, "train_lightgbm.py"], check=True, text=True, capture_output=True)
        print(f"{Fore.GREEN}✅ AIモデルの再学習が完了しました。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ AI再学習中にエラーが発生しました:\n{e.stderr}")
        return False


def step4_push_to_github():
    print_header("☁️ STEP 4: GitHubへの自動コミット＆Push (Web版同期)")
    files_to_add = ["future_races.csv", "ml_target_data.csv", "keiba_ai_model.pkl", "prediction_history.csv"]
    existing_files = [f for f in files_to_add if os.path.exists(f)]

    if not existing_files:
        print(f"{Fore.YELLOW}⚠️ 追加する対象ファイルが見つかりません。")
        return

    try:
        print(f"{Fore.CYAN}  [🤖 送信中] ファイルをパッケージングしています...")
        subprocess.run(["git", "add"] + existing_files, check=True, capture_output=True)
        commit_msg = f"Auto-pipeline update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True)
        
        print(f"{Fore.CYAN}  [🤖 送信中] GitHubへデータをPush中...")
        res = subprocess.run(["git", "push"], check=True, text=True, capture_output=True)
        print(f"{Fore.GREEN}✅ GitHub同期完了！Web上のStreamlit Cloud側も最新化されます。")
    except subprocess.CalledProcessError as e:
        print(f"{Fore.YELLOW}⚠️ Git Push時に警告・エラーが発生しました (※変更がない場合は無視してOKです):\n{e.stderr}")


def main():
    start_time = time.time()
    
    print(f"\n{Fore.MAGENTA}==========================================")
    print(f"{Fore.MAGENTA} 🤖 勝ちぱかくん 全自動パイプラインBOT 起動")
    print(f"{Fore.MAGENTA}=========================================={Style.RESET_ALL}\n")

    # ① 最新レースデータ取得
    step1_fetch_future_races()

    # ② 蓄積データ更新
    step2_update_features()

    # ③ AIモデル再学習
    step3_retrain_ai()

    # ④ 成果物をGitHubへ同期
    step4_push_to_github()

    elapsed = round(time.time() - start_time, 1)
    print(f"\n{Fore.MAGENTA}==========================================")
    print(f"{Fore.GREEN} 🎉 全パイプライン処理完了 (所要時間: {elapsed} 秒)")
    print(f"{Fore.MAGENTA}=========================================={Style.RESET_ALL}\n")
    print(f"{Fore.WHITE}このウィンドウは数秒後に自動で閉じます...")
    time.sleep(5)


if __name__ == "__main__":
    main()