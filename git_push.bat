@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==========================================
echo ☁️ GitHub (Streamlit Cloud) への同期を開始します
echo ==========================================

echo [1/3] 変更されたファイルを追加しています...
git add .

echo [2/3] コミットを作成しています...
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do (set mydate=%%a-%%b-%%c)
for /f "tokens=1-2 delims=: " %%a in ('time /t') do (set mytime=%%a:%%b)
git commit -m "Auto-update: %mydate% %mytime% (AIロジックと蓄積データの最新化)"

echo [3/3] GitHubへデータを送信(Push)しています...
git push origin main

echo.
echo ==========================================
echo ✅ GitHubへの同期が完了しました！
echo 数分以内にStreamlitのWebアプリ側にも最新のAIロジックとデータが反映されます。
echo ==========================================
pause