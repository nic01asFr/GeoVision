@echo off
echo 🔧 Installation automatique du plugin QGIS YOLO Detector
echo.

REM Détection du répertoire QGIS utilisateur
set QGIS_PROFILE=%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins
if not exist "%QGIS_PROFILE%" (
    echo ❌ Répertoire QGIS non trouvé: %QGIS_PROFILE%
    echo Veuillez vérifier votre installation QGIS
    pause
    exit /b 1
)

echo 📁 Répertoire QGIS trouvé: %QGIS_PROFILE%

REM Suppression de l'ancienne version si elle existe
if exist "%QGIS_PROFILE%\qgis_yolo_detector" (
    echo 🗑️  Suppression de l'ancienne version...
    rmdir /s /q "%QGIS_PROFILE%\qgis_yolo_detector"
)

REM Extraction du plugin
echo 📦 Extraction du plugin...
powershell -command "Expand-Archive -Path '%~dp0*.zip' -DestinationPath '%QGIS_PROFILE%' -Force"

echo ✅ Installation terminée!
echo 📋 Prochaines étapes:
echo    1. Redémarrer QGIS
echo    2. Activer l'extension dans Extensions ^> Gestionnaire d'extensions
echo    3. L'outil apparaîtra dans le menu Extensions

pause
