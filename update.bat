@echo off
echo [EVO PUSH] Přidávám změny...
git add .
git commit -m "Automatický update"
git push
pause