"""
Installateur automatique de dépendances pour le plugin YOLO Interactive Detector

Ce module gère l'installation automatique des dépendances Python nécessaires
au fonctionnement du plugin dans l'environnement QGIS.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QProgressBar, QTextEdit, QCheckBox, QMessageBox,
                           QGroupBox, QScrollArea, QWidget)
from qgis.PyQt.QtCore import QThread, pyqtSignal, QTimer, Qt
from qgis.PyQt.QtGui import QFont, QPixmap, QIcon


class DependencyChecker:
    """Vérificateur de dépendances avec diagnostics détaillés"""
    
    REQUIRED_PACKAGES = {
        'torch': {
            'name': 'PyTorch',
            'import_name': 'torch',
            'pip_name': 'torch',
            'description': 'Framework de deep learning - Requis pour YOLO',
            'critical': True,
            'install_url': 'https://pytorch.org/get-started/locally/',
            'check_function': '_check_torch'
        },
        'ultralytics': {
            'name': 'Ultralytics YOLO',
            'import_name': 'ultralytics',
            'pip_name': 'ultralytics',
            'description': 'Framework YOLO v8 + FastSAM - Cœur du système de détection',
            'critical': True,
            'install_url': 'https://docs.ultralytics.com/quickstart/',
            'check_function': '_check_ultralytics'
        },
        'cv2': {
            'name': 'OpenCV',
            'import_name': 'cv2',
            'pip_name': 'opencv-python',
            'description': 'Traitement d\'images - Requis pour la manipulation d\'images',
            'critical': True,
            'install_url': 'https://opencv.org/',
            'check_function': '_check_opencv'
        },
        'PIL': {
            'name': 'Pillow',
            'import_name': 'PIL',
            'pip_name': 'Pillow',
            'description': 'Manipulation d\'images Python',
            'critical': True,
            'install_url': 'https://pillow.readthedocs.io/',
            'check_function': '_check_pillow'
        },
        'numpy': {
            'name': 'NumPy',
            'import_name': 'numpy',
            'pip_name': 'numpy',
            'description': 'Calcul numérique - Généralement déjà installé avec QGIS',
            'critical': True,
            'install_url': 'https://numpy.org/',
            'check_function': '_check_numpy'
        },
        
        # NEW v1.5.0: Smart Assistant Dependencies (Optional but recommended)
        'segment_anything': {
            'name': 'Segment Anything (Meta SAM)',
            'import_name': 'segment_anything',
            'pip_name': 'segment-anything',
            'description': 'Model SAM de Meta - Pour Smart Mode précision maximale',
            'critical': False,  # Optionnel
            'install_url': 'https://github.com/facebookresearch/segment-anything',
            'check_function': '_check_sam'
        },
        'mobile_sam': {
            'name': 'MobileSAM',
            'import_name': 'mobile_sam',
            'pip_name': 'mobile-sam',
            'description': 'SAM léger optimisé CPU - Alternative pour Smart Mode',
            'critical': False,  # Optionnel
            'install_url': 'https://github.com/ChaoningZhang/MobileSAM',
            'check_function': '_check_mobile_sam'
        }
    }
    
    def __init__(self):
        self.python_executable = sys.executable
        self.pip_available = self._check_pip_available()
        
        # SÉCURITÉ: Configurer Ultralytics pour éviter les téléchargements dès l'initialisation
        self._configure_ultralytics_offline_mode()
        
    def _check_pip_available(self) -> bool:
        """Vérifie si pip est disponible"""
        try:
            result = subprocess.run([self.python_executable, '-m', 'pip', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _configure_ultralytics_offline_mode(self):
        """Configure Ultralytics en mode offline pour éviter les téléchargements"""
        try:
            import os
            from pathlib import Path
            
            # SÉCURITÉ: Utiliser répertoire du plugin au lieu de cwd()
            plugin_dir = Path(__file__).parent.parent
            secure_dir = plugin_dir.parent / 'qgis_yolo_detector_data'
            config_dir = secure_dir / 'config'
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Variables d'environnement Ultralytics
            os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
            os.environ['ULTRALYTICS_CONFIG_DIR'] = str(config_dir)
            
            # Forcer le mode offline 
            os.environ['YOLO_OFFLINE'] = '1'
            os.environ['ULTRALYTICS_OFFLINE'] = '1'
            
            # Désactiver les téléchargements automatiques
            os.environ['YOLO_NO_DOWNLOADS'] = '1'
            
            print("🔒 Mode offline Ultralytics configuré")
            
        except Exception as e:
            print(f"⚠️ Erreur configuration offline: {e}")
    
    def check_all_dependencies(self) -> Dict[str, Dict]:
        """
        Vérifie toutes les dépendances
        
        Returns:
            Dict: État de chaque dépendance
        """
        results = {}
        
        for package_key, package_info in self.REQUIRED_PACKAGES.items():
            check_func = getattr(self, package_info['check_function'], self._check_generic)
            results[package_key] = check_func(package_info)
        
        return results
    
    def _check_generic(self, package_info: Dict) -> Dict:
        """Vérification générique d'un package"""
        try:
            spec = importlib.util.find_spec(package_info['import_name'])
            if spec is None:
                return {
                    'available': False,
                    'version': None,
                    'error': f"Module {package_info['import_name']} non trouvé",
                    'info': package_info
                }
            
            # Tentative d'import pour vérifier la validité
            module = importlib.import_module(package_info['import_name'])
            version = getattr(module, '__version__', 'Version inconnue')
            
            return {
                'available': True,
                'version': version,
                'error': None,
                'info': package_info
            }
            
        except Exception as e:
            return {
                'available': False,
                'version': None,
                'error': f"Erreur d'import: {str(e)}",
                'info': package_info
            }
    
    def _check_torch(self, package_info: Dict) -> Dict:
        """Vérification spécialisée pour PyTorch"""
        result = self._check_generic(package_info)
        
        if result['available']:
            try:
                import torch
                result['cuda_available'] = torch.cuda.is_available()
                result['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else None
                result['device_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            except:
                result['cuda_available'] = False
                result['cuda_version'] = None
                result['device_count'] = 0
        
        return result
    
    def _check_ultralytics(self, package_info: Dict) -> Dict:
        """Vérification spécialisée pour Ultralytics"""
        result = self._check_generic(package_info)
        
        if result['available']:
            try:
                from ultralytics import YOLO
                # SÉCURITÉ: Test basique sans téléchargement de modèle
                # Vérifier que les classes principales existent
                yolo_classes = ['YOLO']
                for cls_name in yolo_classes:
                    if not hasattr(YOLO, '__init__'):
                        raise AttributeError(f"Classe {cls_name} manquante")
                
                result['yolo_functional'] = True
                result['model_test'] = 'Import OK - Évite téléchargement'
            except Exception as e:
                result['yolo_functional'] = False
                result['model_test'] = f'Erreur: {str(e)}'
        
        return result
    
    def _check_opencv(self, package_info: Dict) -> Dict:
        """Vérification spécialisée pour OpenCV"""
        result = self._check_generic(package_info)
        
        if result['available']:
            try:
                import cv2
                # Test des fonctionnalités de base
                cv2.imread  # Vérification que les fonctions sont accessibles
                result['build_info'] = cv2.getBuildInformation().split('\n')[0]
            except Exception as e:
                result['available'] = False
                result['error'] = f'OpenCV non fonctionnel: {str(e)}'
        
        return result
    
    def _check_pillow(self, package_info: Dict) -> Dict:
        """Vérification spécialisée pour Pillow"""
        result = self._check_generic(package_info)
        
        if result['available']:
            try:
                from PIL import Image
                # Test de base
                result['formats'] = len(Image.registered_extensions())
            except Exception as e:
                result['available'] = False
                result['error'] = f'Pillow non fonctionnel: {str(e)}'
        
        return result
    
    def _check_numpy(self, package_info: Dict) -> Dict:
        """Vérification spécialisée pour NumPy"""
        result = self._check_generic(package_info)
        
        if result['available']:
            try:
                import numpy as np
                result['blas_info'] = 'BLAS disponible' if np.show_config is not None else 'BLAS non détecté'
            except:
                result['blas_info'] = 'Inconnu'
        
        return result
    
    def _check_sam(self, package_info: Dict) -> Dict:
        """Vérification spécialisée pour Segment Anything Model"""
        result = self._check_generic(package_info)
        
        if result['available']:
            try:
                from segment_anything import SamPredictor, sam_model_registry
                result['sam_models'] = list(sam_model_registry.keys())
                result['sam_functional'] = True
            except ImportError:
                result['available'] = False
                result['error'] = 'segment-anything non fonctionnel'
            except Exception as e:
                result['sam_functional'] = False
                result['error'] = f'SAM test échoué: {str(e)}'
        
        return result
    
    def _check_mobile_sam(self, package_info: Dict) -> Dict:
        """Vérification spécialisée pour MobileSAM"""
        result = self._check_generic(package_info)
        
        if result['available']:
            try:
                from mobile_sam import sam_model_registry, SamPredictor
                result['mobile_sam_models'] = list(sam_model_registry.keys())
                result['mobile_sam_functional'] = True
            except ImportError:
                result['available'] = False
                result['error'] = 'mobile-sam non fonctionnel'
            except Exception as e:
                result['mobile_sam_functional'] = False
                result['error'] = f'MobileSAM test échoué: {str(e)}'
        
        return result


class DependencyInstaller(QThread):
    """Thread d'installation des dépendances"""
    
    progress_updated = pyqtSignal(int, str)  # (progress, message)
    installation_finished = pyqtSignal(bool, str)  # (success, message)
    
    def __init__(self, packages_to_install: List[str]):
        super().__init__()
        self.packages_to_install = packages_to_install
        self.python_executable = sys.executable
        
    def run(self):
        """Exécute l'installation des packages"""
        total_packages = len(self.packages_to_install)
        
        if total_packages == 0:
            self.installation_finished.emit(True, "Aucune installation nécessaire")
            return
        
        success_count = 0
        failed_packages = []
        
        for i, package in enumerate(self.packages_to_install):
            progress = int((i / total_packages) * 100)
            self.progress_updated.emit(progress, f"Installation de {package}...")
            
            if self._install_package(package):
                success_count += 1
                self.progress_updated.emit(
                    int(((i + 1) / total_packages) * 100), 
                    f"✅ {package} installé avec succès"
                )
            else:
                failed_packages.append(package)
                self.progress_updated.emit(
                    int(((i + 1) / total_packages) * 100), 
                    f"❌ Échec installation {package}"
                )
        
        # Résultat final
        if success_count == total_packages:
            self.installation_finished.emit(True, f"✅ Toutes les dépendances installées ({success_count}/{total_packages})")
        else:
            message = f"⚠️ Installation partielle: {success_count}/{total_packages} réussies"
            if failed_packages:
                message += f"\nÉchecs: {', '.join(failed_packages)}"
            self.installation_finished.emit(False, message)
    
    def _install_package(self, package: str) -> bool:
        """Installation réelle des packages requis"""
        print(f"🔧 Tentative d'installation: {package}")
        
        # Méthodes d'installation en ordre de préférence
        install_methods = [
            self._install_with_pip_subprocess,
            self._install_with_pip_api,
            self._install_with_temp_script
        ]
        
        for method in install_methods:
            try:
                print(f"🔄 Essai méthode: {method.__name__}")
                if method(package):
                    print(f"✅ {package} installé avec succès via {method.__name__}")
                    return True
            except Exception as e:
                print(f"⚠️ Échec {method.__name__}: {e}")
                continue
        
        # Si toutes les méthodes échouent
        print(f"❌ Impossible d'installer {package} automatiquement")
        print("🛡️ Si votre centre de sécurité bloque l'installation:")
        print("   1. Ajoutez pip.exe aux exceptions de votre antivirus")
        print("   2. Ou installez manuellement: pip install --user " + package)
        print("   3. Ou utilisez conda install au lieu de pip")
        return False
    
    def _install_with_pip_subprocess(self, package: str) -> bool:
        """Installation via subprocess sécurisé"""
        try:
            import subprocess
            import sys
            
            # Commande d'installation sécurisée
            cmd = [sys.executable, '-m', 'pip', 'install', '--user', '--upgrade', package]
            print(f"🚀 Exécution: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes max
                cwd=None  # Répertoire courant
            )
            
            if result.returncode == 0:
                print(f"✅ Installation réussie: {package}")
                if result.stdout:
                    print(f"Output: {result.stdout[-200:]}...")  # Derniers 200 caractères
                return True
            else:
                print(f"❌ Échec installation {package}: code {result.returncode}")
                if result.stderr:
                    print(f"Erreur: {result.stderr[-200:]}...")  # Derniers 200 caractères
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout installation {package}")
            return False
        except Exception as e:
            print(f"❌ Exception subprocess: {e}")
            return False
    
    def _install_with_pip_api(self, package: str) -> bool:
        """Tentative d'installation avec l'API pip directe"""
        try:
            import pip
            # Utiliser l'API interne de pip (compatible QGIS)
            if hasattr(pip, 'main'):
                # pip < 10.0
                result = pip.main(['install', '--user', '--upgrade', package])
                return result == 0
            else:
                # pip >= 10.0 - pas d'API directe, utiliser subprocess
                print("🔄 pip moderne détecté, utilisation subprocess...")
                return self._install_with_pip_subprocess(package)
        except Exception as e:
            print(f"❌ API pip échouée: {e}")
            return False
    
    def _install_with_temp_script(self, package: str) -> bool:
        """Installation via script temporaire (fallback)"""
        try:
            import tempfile
            import os
            
            # Créer un script batch Windows pour éviter les hooks QGIS
            if os.name == 'nt':  # Windows
                script_content = f'''@echo off
"{self.python_executable}" -m pip install --user --upgrade "{package}"
exit %ERRORLEVEL%'''
                script_ext = '.bat'
            else:  # Linux/Mac
                script_content = f'''#!/bin/bash
"{self.python_executable}" -m pip install --user --upgrade "{package}"
exit $?'''
                script_ext = '.sh'
            
            # Écrire le script temporaire
            with tempfile.NamedTemporaryFile(mode='w', suffix=script_ext, delete=False) as f:
                f.write(script_content)
                temp_script = f.name
            
            try:
                # Rendre le script exécutable sur Unix
                if os.name != 'nt':
                    os.chmod(temp_script, 0o755)
                
                # Exécuter le script
                result = subprocess.run([temp_script], capture_output=True, text=True, timeout=300)
                return result.returncode == 0
            finally:
                # Nettoyer le fichier temporaire
                try:
                    os.unlink(temp_script)
                except:
                    pass
                    
        except Exception:
            return False


class DependencyDialog(QDialog):
    """Dialog d'installation des dépendances"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Detector - Gestionnaire de Dépendances")
        self.setMinimumSize(800, 600)
        
        self.checker = DependencyChecker()
        self.installer = None
        
        self.init_ui()
        self.check_dependencies()
        
    def init_ui(self):
        """Initialise l'interface utilisateur"""
        layout = QVBoxLayout(self)
        
        # En-tête
        header_label = QLabel("🔧 Gestionnaire de Dépendances YOLO Interactive Detector")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)
        
        # Zone de statut
        self.status_group = QGroupBox("État des Dépendances")
        self.status_layout = QVBoxLayout(self.status_group)
        
        # Scroll area pour les dépendances
        scroll = QScrollArea()
        scroll_widget = QWidget()
        self.dependencies_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        self.status_layout.addWidget(scroll)
        
        layout.addWidget(self.status_group)
        
        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Zone de log
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        # Boutons
        buttons_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("🔄 Actualiser")
        self.refresh_button.clicked.connect(self.check_dependencies)
        buttons_layout.addWidget(self.refresh_button)
        
        self.install_button = QPushButton("📦 Installer les Dépendances Manquantes")
        self.install_button.clicked.connect(self.install_missing_dependencies)
        self.install_button.setEnabled(False)
        buttons_layout.addWidget(self.install_button)
        
        buttons_layout.addStretch()
        
        self.close_button = QPushButton("Fermer")
        self.close_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.close_button)
        
        layout.addLayout(buttons_layout)
        
    def check_dependencies(self):
        """Vérifie l'état des dépendances"""
        self.log_text.append("🔍 Vérification des dépendances...")
        
        # Clear previous results
        for i in reversed(range(self.dependencies_layout.count())): 
            self.dependencies_layout.itemAt(i).widget().setParent(None)
        
        self.dependency_results = self.checker.check_all_dependencies()
        missing_packages = []
        
        for package_key, result in self.dependency_results.items():
            widget = self._create_dependency_widget(package_key, result)
            self.dependencies_layout.addWidget(widget)
            
            if not result['available'] and result['info']['critical']:
                missing_packages.append(result['info']['pip_name'])
        
        # État global
        if missing_packages:
            self.install_button.setEnabled(True)
            self.install_button.setText(f"📦 Installer {len(missing_packages)} dépendance(s) manquante(s)")
            self.log_text.append(f"⚠️ {len(missing_packages)} dépendances critiques manquantes")
        else:
            self.install_button.setEnabled(False)
            self.install_button.setText("✅ Toutes les dépendances sont installées")
            self.log_text.append("✅ Toutes les dépendances critiques sont disponibles")
            
    def _create_dependency_widget(self, package_key: str, result: Dict) -> QWidget:
        """Crée un widget pour afficher l'état d'une dépendance"""
        from qgis.PyQt.QtWidgets import QWidget, QHBoxLayout, QLabel
        from qgis.PyQt.QtCore import Qt
        
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Icône de statut
        status_icon = "✅" if result['available'] else "❌"
        status_label = QLabel(status_icon)
        status_label.setMinimumWidth(30)
        layout.addWidget(status_label)
        
        # Nom du package
        name_label = QLabel(result['info']['name'])
        name_label.setMinimumWidth(150)
        font = QFont()
        font.setBold(True)
        name_label.setFont(font)
        layout.addWidget(name_label)
        
        # Version
        version_text = result.get('version', 'Non installé')
        version_label = QLabel(version_text)
        version_label.setMinimumWidth(100)
        layout.addWidget(version_label)
        
        # Description
        desc_label = QLabel(result['info']['description'])
        layout.addWidget(desc_label)
        
        # Informations supplémentaires pour certains packages
        if package_key == 'torch' and result['available']:
            cuda_info = "🚀 CUDA" if result.get('cuda_available') else "💻 CPU"
            cuda_label = QLabel(cuda_info)
            layout.addWidget(cuda_label)
        
        layout.addStretch()
        
        # Style selon l'état
        if not result['available'] and result['info']['critical']:
            widget.setStyleSheet("QWidget { background-color: #ffebee; border-left: 3px solid #f44336; }")
        elif result['available']:
            widget.setStyleSheet("QWidget { background-color: #e8f5e8; border-left: 3px solid #4caf50; }")
        
        return widget
        
    def install_missing_dependencies(self):
        """Lance l'installation des dépendances manquantes"""
        missing_packages = []
        
        for package_key, result in self.dependency_results.items():
            if not result['available'] and result['info']['critical']:
                missing_packages.append(result['info']['pip_name'])
        
        if not missing_packages:
            QMessageBox.information(self, "Information", "Aucune dépendance à installer")
            return
            
        if not self.checker.pip_available:
            QMessageBox.critical(self, "Erreur", 
                               "pip n'est pas disponible dans cet environnement Python.\n"
                               "Veuillez installer les dépendances manuellement.")
            return
        
        # Confirmation
        reply = QMessageBox.question(self, "Confirmation", 
                                   f"Installer {len(missing_packages)} dépendance(s) ?\n\n"
                                   f"Packages: {', '.join(missing_packages)}\n\n"
                                   f"L'installation peut prendre plusieurs minutes.",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
            
        # Lancement de l'installation
        self.install_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.installer = DependencyInstaller(missing_packages)
        self.installer.progress_updated.connect(self._on_installation_progress)
        self.installer.installation_finished.connect(self._on_installation_finished)
        self.installer.start()
        
    def _on_installation_progress(self, progress: int, message: str):
        """Met à jour la progression de l'installation"""
        self.progress_bar.setValue(progress)
        self.log_text.append(message)
        
    def _on_installation_finished(self, success: bool, message: str):
        """Gère la fin de l'installation"""
        self.progress_bar.setVisible(False)
        self.install_button.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.log_text.append(message)
        
        if success:
            QMessageBox.information(self, "Succès", 
                                  "Installation terminée avec succès !\n\n"
                                  "Redémarrez QGIS pour que les changements prennent effet.")
        else:
            # Proposer l'installation manuelle en cas d'échec
            reply = QMessageBox.question(self, "Installation incomplète", 
                              "L'installation automatique a échoué.\n\n"
                              "Voulez-vous voir les commandes d'installation manuelle ?",
                              QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self._show_manual_installation_guide()
        
        # Actualisation automatique
        QTimer.singleShot(1000, self.check_dependencies)
    
    def _show_manual_installation_guide(self):
        """Affiche un guide d'installation manuelle"""
        import sys
        
        # Obtenir les packages manquants
        missing_packages = []
        for package_key, result in self.dependency_results.items():
            if not result['available'] and result['info']['critical']:
                missing_packages.append(result['info']['pip_name'])
        
        if not missing_packages:
            return
        
        # Créer les commandes d'installation
        python_exe = sys.executable
        commands = []
        
        for package in missing_packages:
            if package == 'torch':
                commands.append(f'"{python_exe}" -m pip install --user torch torchvision')
            else:
                commands.append(f'"{python_exe}" -m pip install --user {package}')
        
        # Créer le dialog d'installation manuelle
        dialog = QDialog(self)
        dialog.setWindowTitle("Guide d'Installation Manuelle")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Instructions
        instructions = QLabel("""
<h3>Installation Manuelle des Dépendances</h3>
<p>Ouvrez une invite de commande (CMD sur Windows) ou un terminal et exécutez les commandes suivantes :</p>
        """)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Zone de texte avec les commandes
        commands_text = QTextEdit()
        commands_text.setPlainText('\n'.join(commands))
        commands_text.selectAll()  # Sélectionner tout pour faciliter la copie
        layout.addWidget(commands_text)
        
        # Instructions additionnelles
        additional_info = QLabel("""
<p><b>Instructions :</b></p>
<ul>
<li>Copiez-collez chaque commande dans votre invite de commande</li>
<li>Appuyez sur Entrée pour exécuter chaque commande</li>
<li>Attendez que chaque installation se termine avant la suivante</li>
<li>Redémarrez QGIS après l'installation</li>
</ul>
        """)
        additional_info.setWordWrap(True)
        layout.addWidget(additional_info)
        
        # Boutons
        button_layout = QHBoxLayout()
        
        copy_button = QPushButton("📋 Copier les Commandes")
        copy_button.clicked.connect(lambda: self._copy_to_clipboard('\n'.join(commands)))
        button_layout.addWidget(copy_button)
        
        close_button = QPushButton("Fermer")
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()
    
    def _copy_to_clipboard(self, text: str):
        """Copie le texte dans le presse-papiers"""
        try:
            from qgis.PyQt.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            QMessageBox.information(self, "Copié", "Commandes copiées dans le presse-papiers !")
        except Exception:
            pass


def show_dependency_manager(parent=None):
    """Affiche le gestionnaire de dépendances"""
    dialog = DependencyDialog(parent)
    return dialog.exec_()


def check_dependencies_silent() -> Tuple[bool, List[str]]:
    """
    Vérification silencieuse des dépendances
    
    Returns:
        Tuple[bool, List[str]]: (all_available, missing_critical_packages)
    """
    checker = DependencyChecker()
    results = checker.check_all_dependencies()
    
    missing_critical = []
    for package_key, result in results.items():
        if not result['available'] and result['info']['critical']:
            missing_critical.append(result['info']['name'])
    
    return len(missing_critical) == 0, missing_critical


if __name__ == "__main__":
    # Test autonome
    from qgis.PyQt.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    show_dependency_manager()
    sys.exit(app.exec_())