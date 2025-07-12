@echo off
echo WSL2アクセス設定用バッチファイル
echo 管理者権限で実行してください
echo.

REM WSL2のIPアドレスを取得
for /f %%i in ('wsl hostname -I') do set WSL_IP=%%i

echo WSL2 IP Address: %WSL_IP%
echo.

echo ポートプロキシ設定を追加中...
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=%WSL_IP%

echo ファイアウォール設定を追加中...
netsh advfirewall firewall add rule name="WSL2-Port8000" dir=in action=allow protocol=TCP localport=8000

echo.
echo 設定完了！
echo ブラウザで http://localhost:8000 にアクセスしてください
echo.
echo 現在のポートプロキシ設定:
netsh interface portproxy show all

pause