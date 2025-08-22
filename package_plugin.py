#!/usr/bin/env python3
"""
Script de packaging automatique pour le plugin QGIS YOLO Interactive Detector
Usage: python package_plugin.py [version]

Crée un fichier ZIP prêt à installer dans QGIS avec versioning automatique
"""

import os
import sys
import zipfile
import shutil
from datetime import datetime
from pathlib import Path

# Configuration
PLUGIN_NAME = "qgis_yolo_detector"
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc", 
    "*.pyo",
    ".git",
    ".vscode",
    "tests",
    "*.zip",
    "package_plugin.py",
    ".pytest_cache",
    "*.log"
]

def should_exclude(file_path, exclude_patterns):
    """Vérifie si un fichier doit être exclu du package"""
    path_str = str(file_path)
    for pattern in exclude_patterns:
        if pattern.startswith("*."):
            # Pattern d'extension
            if path_str.endswith(pattern[1:]):
                return True
        elif pattern in path_str:
            # Pattern de nom/dossier
            return True
    return False

def get_version_info():
    """Lit la version depuis metadata.txt"""
    metadata_path = Path(PLUGIN_NAME) / "metadata.txt"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('version='):
                    return line.split('=')[1].strip()
    return "1.0.0"

def update_version_in_metadata(new_version):
    """Met à jour la version dans metadata.txt"""
    metadata_path = Path(PLUGIN_NAME) / "metadata.txt"
    if metadata_path.exists():
        # Lire le fichier
        with open(metadata_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Modifier la ligne version
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for line in lines:
                if line.startswith('version='):
                    f.write(f'version={new_version}\n')
                else:
                    f.write(line)

def create_package(version=None):
    """Crée le package ZIP du plugin"""
    
    # Détermine la version
    if version is None:
        current_version = get_version_info()
        print(f"Version actuelle: {current_version}")
        
        # Auto-incrémente la version build
        version_parts = current_version.split('.')
        if len(version_parts) == 3:
            build_num = int(version_parts[2]) + 1
            version = f"{version_parts[0]}.{version_parts[1]}.{build_num}"
        else:
            version = f"{current_version}.1"
    
    # Met à jour la version dans metadata.txt
    update_version_in_metadata(version)
    
    # Nom du fichier ZIP
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_filename = f"{PLUGIN_NAME}_v{version}_{timestamp}.zip"
    
    print(f"🔧 Création du package: {zip_filename}")
    print(f"📁 Dossier source: {PLUGIN_NAME}/")
    
    # Vérifications préalables
    plugin_dir = Path(PLUGIN_NAME)
    if not plugin_dir.exists():
        print(f"❌ Erreur: Le dossier {PLUGIN_NAME} n'existe pas!")
        return False
    
    required_files = ["__init__.py", "plugin_main.py", "metadata.txt"]
    for req_file in required_files:
        if not (plugin_dir / req_file).exists():
            print(f"⚠️  Attention: Fichier manquant: {req_file}")
    
    # Création du ZIP
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Parcours récursif du dossier plugin
            for root, dirs, files in os.walk(plugin_dir):
                # Filtrage des dossiers à exclure
                dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d, EXCLUDE_PATTERNS)]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    # Vérification exclusion
                    if should_exclude(file_path, EXCLUDE_PATTERNS):
                        continue
                    
                    # Chemin dans l'archive (relatif au dossier plugin)
                    arcname = file_path.relative_to(plugin_dir.parent)
                    
                    # Ajout au ZIP
                    zipf.write(file_path, arcname)
                    print(f"  ✅ {arcname}")
        
        # Informations finales
        zip_size = Path(zip_filename).stat().st_size / 1024  # KB
        print(f"\n🎉 Package créé avec succès!")
        print(f"📦 Fichier: {zip_filename}")
        print(f"📏 Taille: {zip_size:.1f} KB")
        print(f"🔖 Version: {version}")
        
        # Instructions d'installation
        print(f"\n📋 Instructions d'installation:")
        print(f"1. Ouvrir QGIS")
        print(f"2. Extensions → Installer depuis un ZIP")
        print(f"3. Sélectionner: {zip_filename}")
        print(f"4. Redémarrer QGIS")
        print(f"5. Activer l'extension dans le gestionnaire d'extensions")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la création du package: {str(e)}")
        return False

def create_install_script():
    """Crée un script d'installation automatique"""
    install_script = """@echo off
echo 🔧 Installation automatique du plugin QGIS YOLO Detector
echo.

REM Détection du répertoire QGIS utilisateur
set QGIS_PROFILE=%APPDATA%\\QGIS\\QGIS3\\profiles\\default\\python\\plugins
if not exist "%QGIS_PROFILE%" (
    echo ❌ Répertoire QGIS non trouvé: %QGIS_PROFILE%
    echo Veuillez vérifier votre installation QGIS
    pause
    exit /b 1
)

echo 📁 Répertoire QGIS trouvé: %QGIS_PROFILE%

REM Suppression de l'ancienne version si elle existe
if exist "%QGIS_PROFILE%\\qgis_yolo_detector" (
    echo 🗑️  Suppression de l'ancienne version...
    rmdir /s /q "%QGIS_PROFILE%\\qgis_yolo_detector"
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
"""
    
    with open("install_plugin.bat", 'w', encoding='utf-8') as f:
        f.write(install_script)
    
    print("📜 Script d'installation créé: install_plugin.bat")

def main():
    """Fonction principale"""
    print("🎯 QGIS YOLO Interactive Detector - Package Builder V1.6.0 OPTIMIZED")
    print("=" * 70)
    
    # Version depuis argument ou auto-increment
    version = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Création du package
    success = create_package(version)
    
    if success:
        # Création du script d'installation
        create_install_script()
        
        print(f"\n🚀 Package V1.6.0 OPTIMIZED prêt pour test dans QGIS!")
        print(f"💡 Nouveautés:")
        print(f"   • Sélection candidats intelligente avec métriques géospatiales")
        print(f"   • Gestionnaire qualité annotation avec suivi temps réel")
        print(f"   • Interface annotation rapide avec mode batch")
        print(f"   • Pipeline entraînement optimisé géospatial")
        print(f"   • Décision SAM adaptative selon contexte objets")
    else:
        print(f"\n❌ Échec de la création du package")
        sys.exit(1)

if __name__ == "__main__":
    main()