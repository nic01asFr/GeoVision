"""
Plugin principal YOLO Interactive Object Detector

Ce fichier contient la classe principale du plugin qui gère:
- L'initialisation de l'interface QGIS
- La création des menus et barres d'outils
- L'instanciation des composants principaux
- La gestion du cycle de vie du plugin
"""

import os
import sys
from pathlib import Path

from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox

from qgis.core import QgsProject, QgsApplication
from qgis.gui import QgisInterface

# Import des composants principaux du plugin
# NOTE: Imports déplacés dans les fonctions pour éviter les erreurs au chargement du module
# Les vérifications de disponibilité se font dynamiquement


class YOLODetectorPlugin:
    """
    Classe principale du plugin YOLO Interactive Object Detector
    """

    def __init__(self, iface: QgisInterface):
        """
        Initialise le plugin.

        Args:
            iface: Interface QGIS
        """
        self.iface = iface
        
        # Initialisation des chemins
        self.plugin_dir = os.path.dirname(__file__)
        
        # SÉCURITÉ: Configuration globale anti-téléchargement dès l'initialisation
        self._configure_global_offline_mode()
        
        # COMPATIBILITÉ: Protection NumPy 1.x/2.x
        self._configure_numpy_compatibility()
        
        # Initialisation des composants
        self.actions = []
        self.menu = "&YOLO Object Detector"
        self.toolbar = None
        self.main_dialog = None
        self.processing_provider = None
        
        # État du plugin
        self.first_start = None

    def tr(self, message):
        """
        Traduction des chaînes de caractères.
        
        Args:
            message: Message à traduire
            
        Returns:
            Message traduit
        """
        return QCoreApplication.translate('YOLODetectorPlugin', message)

    def _configure_global_offline_mode(self):
        """Configure le mode offline global pour éviter tous les téléchargements"""
        try:
            import os
            from pathlib import Path
            
            # SÉCURITÉ: Utiliser répertoire du plugin au lieu de cwd()
            plugin_dir = Path(__file__).parent
            secure_dir = plugin_dir.parent / 'qgis_yolo_detector_data'
            secure_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"🔧 Plugin dir: {plugin_dir}")
            print(f"🔧 Secure dir: {secure_dir}")
            
            # Configuration Ultralytics/YOLO
            os.environ['YOLO_CONFIG_DIR'] = str(secure_dir / 'config')
            os.environ['ULTRALYTICS_CONFIG_DIR'] = str(secure_dir / 'config')
            os.environ['YOLO_OFFLINE'] = '1'
            os.environ['ULTRALYTICS_OFFLINE'] = '1'
            os.environ['YOLO_NO_DOWNLOADS'] = '1'
            
            # Configuration PyTorch Hub
            os.environ['TORCH_HOME'] = str(secure_dir / 'torch')
            os.environ['TORCH_HUB_OFFLINE'] = '1'
            
            # Configuration générale
            os.environ['HF_OFFLINE'] = '1'  # Hugging Face offline
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            print("🔒 Mode offline global configuré")
            print(f"📁 Répertoire sécurisé: {secure_dir}")
            
        except Exception as e:
            print(f"⚠️ Erreur configuration offline globale: {e}")

    def _configure_numpy_compatibility(self):
        """Configure la compatibilité NumPy 1.x/2.x pour éviter les erreurs de modules"""
        try:
            # Variables d'environnement pour éviter les conflits
            os.environ['NPY_DISABLE_OPTIMIZATION'] = '1'
            os.environ['NUMPY_EXPERIMENTAL_DTYPE_API'] = '0'
            
            # Configuration warnings pour masquer les avertissements NumPy 1.x/2.x
            import warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
            warnings.filterwarnings('ignore', message='.*compiled using NumPy 1.x.*')
            
            # Test de chargement NumPy
            try:
                import numpy as np
                print(f"🔧 NumPy {np.__version__} - Compatibilité 1.x/2.x configurée")
            except Exception as numpy_error:
                print(f"⚠️ NumPy indisponible: {numpy_error}")
            
            print("✅ Compatibilité NumPy 1.x/2.x configurée")
            
        except Exception as e:
            print(f"⚠️ Erreur configuration NumPy: {e}")

    def add_action(self, icon_path, text, callback, enabled_flag=True, 
                   add_to_menu=True, add_to_toolbar=True, status_tip=None,
                   whats_this=None, parent=None):
        """
        Ajoute une action à la barre d'outils et au menu.
        
        Args:
            icon_path: Chemin vers l'icône
            text: Texte de l'action
            callback: Fonction de callback
            enabled_flag: Si l'action est activée
            add_to_menu: Ajouter au menu
            add_to_toolbar: Ajouter à la barre d'outils
            status_tip: Texte de statut
            whats_this: Texte d'aide
            parent: Widget parent
            
        Returns:
            QAction créée
        """
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar and self.toolbar is not None:
            # Ajout à la barre d'outils
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        """
        Initialise l'interface graphique du plugin.
        
        Crée les menus, barres d'outils et connexions nécessaires.
        """
        # Barre d'outils personnalisée (DOIT être créée EN PREMIER)
        self.toolbar = self.iface.addToolBar('YOLO Object Detector')
        self.toolbar.setObjectName('YOLOObjectDetectorToolbar')
        
        # Création des icônes (chemin relatif au plugin)
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        
        # Action principale - Ouvrir l'interface YOLO
        self.add_action(
            icon_path,
            text=self.tr('YOLO Object Detector'),
            callback=self.run,
            parent=self.iface.mainWindow(),
            status_tip=self.tr('Open YOLO Interactive Object Detector'),
            whats_this=self.tr('Create custom object detection models through interactive annotation')
        )

        # Intégration avec Processing Framework
        self.initProcessing()

        # Premier démarrage
        self.first_start = True

    def initProcessing(self):
        """
        Initialise l'intégration avec QGIS Processing Framework.
        """
        try:
            from .processing.provider import YOLOProcessingProvider
            self.processing_provider = YOLOProcessingProvider()
            QgsApplication.processingRegistry().addProvider(self.processing_provider)
            print("✅ YOLO Processing Provider initialisé")
        except ImportError:
            # Processing provider pas encore implémenté
            print("⏳ YOLO Processing Provider pas encore disponible - sera ajouté dans une future version")
        except Exception as e:
            QMessageBox.warning(
                self.iface.mainWindow(),
                self.tr('YOLO Processing Error'),
                self.tr(f'Could not initialize Processing algorithms: {str(e)}')
            )

    def unload(self):
        """
        Nettoie les ressources lors de la désactivation du plugin.
        """
        # Suppression des actions du menu
        for action in self.actions:
            self.iface.removePluginMenu(self.menu, action)
            
        # Suppression de la barre d'outils
        if self.toolbar:
            del self.toolbar

        # Suppression du fournisseur Processing
        if self.processing_provider:
            QgsApplication.processingRegistry().removeProvider(self.processing_provider)

        # Fermeture des dialogues/dock widgets ouverts
        if self.main_dialog:
            # Si c'est un dock widget, le retirer de l'interface
            if hasattr(self.main_dialog, 'setVisible'):
                try:
                    self.iface.removeDockWidget(self.main_dialog)
                except:
                    pass  # Peut échouer si déjà retiré
            
            self.main_dialog.close()
            self.main_dialog = None

    def run(self):
        """
        Lance l'interface principale du plugin.
        """
        # VERSION MINIMALE TESTABLE - Pas de vérification dépendances pour l'instant
        # self.check_dependencies() sera activé plus tard
        
        # CORRECTION: Vérifier si le dialog existe déjà pour éviter les doublons
        if self.main_dialog is not None and not self.main_dialog.isHidden():
            # Si le dialog existe et est visible, le mettre au premier plan
            self.main_dialog.raise_()
            self.main_dialog.activateWindow()
            print("📌 Dialog existant remis au premier plan")
            return
        
        # Interface minimale temporaire pour tests
        if self.main_dialog is None:
            # Essaie d'importer l'interface complète
            try:
                from .ui.main_dialog import YOLOMainDialog
                self.main_dialog = YOLOMainDialog(self.iface)
                
                # NOUVEAUTÉ: Intégrer le dock widget dans QGIS
                self.iface.addDockWidget(Qt.RightDockWidgetArea, self.main_dialog)
                print("✅ Interface complète chargée et intégrée comme dock widget")
            except ImportError:
                # Dialog temporaire si l'interface complète n'existe pas encore
                self.main_dialog = self.create_minimal_dialog()
                print("⏳ Interface minimale chargée - développement en cours")
            except Exception as e:
                print(f"❌ Erreur chargement interface: {e}")
                # Fallback vers interface minimale
                self.main_dialog = self.create_minimal_dialog()
                print("⚠️ Interface minimale chargée (fallback après erreur)")

        # NOUVEAUTÉ: Affichage du dock widget ou dialog selon le type
        if hasattr(self.main_dialog, 'setVisible'):
            # C'est un dock widget
            self.main_dialog.setVisible(True)
            self.main_dialog.raise_()
        else:
            # C'est un dialog classique (fallback)
            self.main_dialog.show()
            self.main_dialog.raise_()
            self.main_dialog.activateWindow()

        # CORRECTION: Message de bienvenue APRES affichage de l'interface
        # pour éviter le blocage derrière la fenêtre principale
        if self.first_start:
            # Délai court pour laisser l'interface se stabiliser
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(200, self.show_welcome_message)
            self.first_start = False
    
    def create_minimal_dialog(self):
        """Crée un dialog minimal pour les tests initiaux"""
        from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit
        
        dialog = QDialog(self.iface.mainWindow())
        dialog.setWindowTitle("YOLO Object Detector - Version Test")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Message de statut
        status_label = QLabel("🎯 Plugin YOLO Object Detector - Version de Test")
        status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2E8B57;")
        layout.addWidget(status_label)
        
        # Zone d'informations
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h3>🚀 Status du Plugin</h3>
        <p><b>✅ Plugin chargé avec succès</b></p>
        <p><b>✅ Interface de base fonctionnelle</b></p>
        <p><b>⏳ Fonctionnalités en développement :</b></p>
        <ul>
            <li>🎨 Interface principale avec onglets</li>
            <li>🎯 Outil d'annotation canvas</li>
            <li>🧠 Moteur YOLO d'entraînement</li>
            <li>🔍 Système de détection massive</li>
        </ul>
        
        <h3>📋 Pour Tester :</h3>
        <p>1. Vérifiez que ce dialog s'ouvre correctement</p>
        <p>2. Vérifiez que le plugin apparaît dans le menu</p>
        <p>3. Testez la fermeture/réouverture</p>
        
        <h3>🔧 Version Actuelle :</h3>
        <p>Plugin de base - Prêt pour développement incrémental</p>
        """)
        layout.addWidget(info_text)
        
        # Bouton de test
        test_button = QPushButton("🧪 Tester les Fonctionnalités Disponibles")
        test_button.clicked.connect(self.run_basic_tests)
        layout.addWidget(test_button)
        
        # Bouton fermer
        close_button = QPushButton("✅ Fermer")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)
        
        return dialog
    
    def run_basic_tests(self):
        """Exécute des tests de base du plugin"""
        from qgis.PyQt.QtWidgets import QMessageBox
        
        test_results = []
        
        # Test 1: Vérification du répertoire de projet
        try:
            project_dir = self.get_project_directory()
            test_results.append(f"✅ Répertoire projet créé: {project_dir}")
        except Exception as e:
            test_results.append(f"❌ Erreur répertoire projet: {str(e)}")
        
        # Test 2: Statut des modules du plugin
        test_results.append("--- Modules Plugin ---")
        
        # Test UI
        try:
            from .ui.main_dialog import YOLOMainDialog
            test_results.append("✅ Interface principale (ui.main_dialog)")
        except ImportError:
            test_results.append("⏳ Interface principale (ui.main_dialog)")
        
        # Test YOLO Engine
        try:
            from .core.yolo_engine import YOLOEngine
            test_results.append("✅ Moteur YOLO (core.yolo_engine)")
        except ImportError:
            test_results.append("⏳ Moteur YOLO (core.yolo_engine)")
        
        # Test Annotation Manager
        try:
            from .core.annotation_manager import AnnotationManager
            test_results.append("✅ Gestionnaire annotations (core.annotation_manager)")
        except ImportError:
            test_results.append("⏳ Gestionnaire annotations (core.annotation_manager)")
            
        # Test Processing Provider
        try:
            from .processing.provider import YOLOProcessingProvider
            test_results.append("✅ Processing Framework (processing.provider)")
        except ImportError:
            test_results.append("⏳ Processing Framework (processing.provider)")
        
        # Test 3: Vérification des dépendances critiques
        try:
            import torch
            test_results.append(f"✅ PyTorch disponible: {torch.__version__}")
        except ImportError:
            test_results.append("⚠️ PyTorch non installé")
            
        try:
            import ultralytics
            test_results.append(f"✅ Ultralytics disponible: {ultralytics.__version__}")
        except ImportError:
            test_results.append("⚠️ Ultralytics non installé")
        
        # Affichage des résultats
        results_text = "\n".join(test_results)
        QMessageBox.information(
            self.iface.mainWindow(),
            "🧪 Résultats des Tests",
            f"Tests du plugin YOLO Detector:\n\n{results_text}"
        )

    def check_dependencies(self):
        """
        Vérifie que les dépendances critiques sont installées.
        
        Returns:
            bool: True si toutes les dépendances sont disponibles
        """
        missing_deps = []
        
        try:
            import torch
        except ImportError:
            missing_deps.append('PyTorch')
            
        try:
            import ultralytics
        except ImportError:
            missing_deps.append('Ultralytics YOLO')
            
        try:
            import cv2
        except ImportError:
            missing_deps.append('OpenCV')

        if missing_deps:
            QMessageBox.critical(
                self.iface.mainWindow(),
                self.tr('Missing Dependencies'),
                self.tr(
                    f'The following required packages are missing:\n\n'
                    f'{", ".join(missing_deps)}\n\n'
                    f'Please install them using:\n'
                    f'pip install torch ultralytics opencv-python\n\n'
                    f'Or use the automatic installer from the plugin menu.'
                )
            )
            return False
            
        return True

    def show_welcome_message(self):
        """
        Affiche un message de bienvenue au premier démarrage.
        """
        from qgis.PyQt.QtWidgets import QMessageBox
        
        # CORRECTION: S'assurer que le parent est bien la fenêtre principale du plugin
        # et non pas QGIS pour éviter les problèmes de superposition
        parent_window = self.main_dialog if self.main_dialog else self.iface.mainWindow()
        
        # Création du message box avec le bon parent
        msgBox = QMessageBox(parent_window)
        msgBox.setWindowTitle(self.tr('Welcome to YOLO Object Detector'))
        msgBox.setText(self.tr(
            'Welcome to YOLO Interactive Object Detector!\n\n'
            'This plugin allows you to create custom object detection models '
            'by simply clicking on objects in your raster data.\n\n'
            'Quick Start:\n'
            '1. Load a raster layer in QGIS\n'
            '2. Create a new object class\n'
            '3. Click on 10-20 examples of your target objects\n'
            '4. Train your custom model\n'
            '5. Apply it to detect similar objects automatically\n\n'
                'For detailed instructions, check the documentation tab.'
        ))
        msgBox.setIcon(QMessageBox.Information)
        
        # CORRECTION: Configuration pour rester au-dessus de la fenêtre du plugin
        if self.main_dialog:
            msgBox.setWindowModality(2)  # Qt.WindowModal
        
        msgBox.exec_()

    def get_project_directory(self):
        """
        Retourne le répertoire de projet pour stocker les données du plugin.
        
        Returns:
            str: Chemin vers le répertoire de projet
        """
        # Utilise le répertoire du projet QGIS ou un répertoire par défaut
        project = QgsProject.instance()
        if project.fileName():
            project_dir = Path(project.fileName()).parent / 'yolo_detector_data'
        else:
            # SÉCURITÉ: Utiliser répertoire de travail courant au lieu de Documents
            project_dir = Path.cwd() / 'qgis_yolo_detector' / 'default_project'
            
        # Création du répertoire s'il n'existe pas
        project_dir.mkdir(parents=True, exist_ok=True)
        
        return str(project_dir)

    def show_about(self):
        """
        Affiche la boîte de dialogue À propos.
        """
        QMessageBox.about(
            self.iface.mainWindow(),
            self.tr('About YOLO Object Detector'),
            self.tr(
                '<h3>YOLO Interactive Object Detector v1.0</h3>'
                '<p>This plugin revolutionizes object detection in QGIS by allowing '
                'users to create custom YOLO models through interactive annotation.</p>'
                
                '<p><b>Key Features:</b></p>'
                '<ul>'
                '<li>Interactive canvas annotation</li>'
                '<li>Automatic model training with transfer learning</li>'
                '<li>Batch processing capabilities</li>'
                '<li>GPU acceleration with CPU fallback</li>'
                '<li>Integration with QGIS Processing Framework</li>'
                '</ul>'
                
                '<p><b>Developed by:</b> Claude Code Agent</p>'
                '<p><b>License:</b> GPL v3</p>'
                '<p><b>Repository:</b> <a href="https://github.com/example/qgis-yolo-detector">'
                'GitHub</a></p>'
            )
        )


# TODO pour Claude Code:
# 1. Implémenter YOLOMainDialog dans ui/main_dialog.py
# 2. Implémenter YOLOEngine dans core/yolo_engine.py  
# 3. Implémenter AnnotationManager dans core/annotation_manager.py
# 4. Implémenter YOLOProcessingProvider dans processing/provider.py
# 5. Créer l'icône icon.png (24x24 pixels)
# 6. Tester le chargement du plugin dans QGIS
