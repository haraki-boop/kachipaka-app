import os
import sys
import time
import subprocess
from datetime import datetime

# モジュール読み込み用にパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predict_with_gemini import run_scraper, FUTURE_CSV
    class DummyProgress:
        def progress(self, val): pass
    class DummyText:
        def text(self, msg): print(f"  [スクレイパー] {msg}")
except Exception as e:
    print(f"❌ predict_with_gemini.py の読み込みに失敗しました: {e}")
    sys.exit(1)


def step1_fetch_future_races():
    print("\n==========================================")
    print("🚀 STEP 1: 最新レースデータの自動取得")
    print("==========================================")
    p_text = DummyText()
    p_bar = DummyProgress()
    
    success = run_scraper(p_text, p_bar)
    if success and os.path.exists(FUTURE_CSV) and os.path.getsize(FUTURE_CSV) > 0:
        print(f"✅ 最新出馬表の取得完了: {FUTURE_CSV}")
        return True
    else:
        print("❌ 出馬表データの取得に失敗しました。")
        return False


def step2_update_features():
    print("\n==========================================")
    print("📊 STEP 2: 蓄積データの更新・特徴量生成 (create_features.py)")
    print("==========================================")
    if not os.path.exists("create_features.py"):
        print("⚠️ create_features.py が見つかりません。スキップします。")
        return False

    try:
        res = subprocess.run([sys.executable, "create_features.py"], check=True, text=True, capture_output=True)
        print(res.stdout)
        print("✅ 蓄積データの集計・特徴量作成が完了しました。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 特徴量生成中にエラーが発生しました:\n{e.stderr}")
        return False


def step3_retrain_ai():
    print("\n==========================================")
    print("🧠 STEP 3: 蓄積データのAI再学習 (train_lightgbm.py)")
    print("==========================================")
    if not os.path.exists("train_lightgbm.py"):
        print("⚠️ train_lightgbm.py が見つかりません。スキップします。")
        return False

    try:
        res = subprocess.run([sys.executable, "train_lightgbm.py"], check=True, text=True, capture_output=True)
        print(res.stdout)
        print("✅ AIモデルの再学習が完了しました。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ AI再学習中にエラーが発生しました:\n{e.stderr}")
        return False


def step4_push_to_github():
    print("\n==========================================")
    print("☁️ STEP 4: GitHubへの自動コミット＆Push (Web版同期)")
    print("==========================================")
    files_to_add = ["future_races.csv", "ml_target_data.csv", "keiba_ai_model.pkl", "prediction_history.csv"]
    existing_files = [f for f in files_to_add if os.path.exists(f)]

    if not existing_files:
        print("⚠️ 追加する対象ファイルが見つかりません。")
        return

    try:
        subprocess.run(["git", "add"] + existing_files, check=True, capture_output=True)
        commit_msg = f"Auto-pipeline update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True)
        
        print("📤 GitHubへデータを送信中...")
        res = subprocess.run(["git", "push"], check=True, text=True, capture_output=True)
        print("✅ GitHub同期完了！Web上のStreamlit Cloud側も最新化されます。")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Git Push時に警告・エラーが発生しました:\n{e.stderr}")


def main():
    start_time = time.time()
    print("🤖 全自動パイプラインBOTを開始します...")

    # ① 最新レースデータ取得
    s1 = step1_fetch_future_races()

    # ② 蓄積データ更新
    s2 = step2_update_features()

    # ③ AIモデル再学習
    s3 = step3_retrain_ai()

    # ④ 成果物をGitHubへ同期
    step4_push_to_github()

    elapsed = round(time.time() - start_time, 1)
    print("\n==========================================")
    print(f"🎉 全パイプライン処理完了 (所要時間: {elapsed} 秒)")
    print("==========================================")


if __name__ == "__main__":
    main()