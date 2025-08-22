"""
Interface principale du plugin YOLO Interactive Object Detector

Cette interface contient :
- Onglet 1: Gestion des Classes d'Objets
- Onglet 2: Annotation Interactive
- Onglet 3: Entraînement de Modèles
- Onglet 4: Détection et Application

Workflow utilisateur :
1. Créer des classes d'objets (ex: "Poteaux électriques")
2. Annoter des exemples sur le canvas QGIS
3. Entraîner automatiquement un modèle YOLO
4. Appliquer le modèle sur de vastes zones
"""

import os
from pathlib import Path
from datetime import datetime

from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QIcon, QPixmap, QFont
from qgis.PyQt.QtWidgets import (
    QDialog, QDockWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QWidget, QLabel, QPushButton, QLineEdit,
    QComboBox, QListWidget, QListWidgetItem, QTextEdit,
    QGroupBox, QFrame, QProgressBar, QCheckBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QSlider,
    QSplitter, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QFileDialog
)

from qgis.gui import QgisInterface
from qgis.core import QgsProject

# Import de l'outil d'annotation
try:
    from .annotation_tool import InteractiveAnnotationTool
    ANNOTATION_TOOL_AVAILABLE = True
except ImportError:
    ANNOTATION_TOOL_AVAILABLE = False

# Import du gestionnaire de dépendances
try:
    from ..utils.dependency_installer import check_dependencies_silent, show_dependency_manager
    DEPENDENCY_MANAGER_AVAILABLE = True
except ImportError:
    DEPENDENCY_MANAGER_AVAILABLE = False

# DatasetManager intégré dans l'onglet Classes au lieu d'un onglet séparé


class YOLOMainDialog(QDockWidget):
    """Interface principale du plugin YOLO Interactive Object Detector"""
    
    # Signaux
    class_created = pyqtSignal(str)  # Nouvelle classe créée
    annotation_requested = pyqtSignal(str)  # Annotation demandée pour une classe
    training_requested = pyqtSignal(str)  # Entraînement demandé
    detection_requested = pyqtSignal(str, dict)  # Détection demandée
    
    def __init__(self, iface: QgisInterface, parent=None):
        """
        Initialise l'interface principale
        
        Args:
            iface: Interface QGIS
            parent: Widget parent
        """
        super().__init__("🎯 YOLO Interactive Object Detector", parent)
        self.iface = iface
        
        # Configuration du dock widget
        self.setObjectName("YOLOMainDock")
        self.setFeatures(
            QDockWidget.DockWidgetMovable | 
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        
        # Taille par défaut
        self.setMinimumWidth(400)
        self.resize(800, 600)
        
        # Données
        self.object_classes = {}  # Dict des classes d'objets
        self.current_class = None  # Classe active
        
        # État de l'entraînement
        self.training_in_progress = False
        self.current_dataset_info = None
        # SUPPRIMÉ: self.trained_models et self.class_models → maintenant en DB
        
        # Outil d'annotation
        self.annotation_tool = None
        if ANNOTATION_TOOL_AVAILABLE:
            self.annotation_tool = InteractiveAnnotationTool(self.iface.mapCanvas())
            # NOUVEAU: Référence bidirectionnelle pour Smart Mode
            self.annotation_tool.set_main_dialog(self)
        
        # Gestionnaire d'annotations (AVANT le chargement des données)
        self.annotation_manager = None
        try:
            from ..core.annotation_manager import get_annotation_manager
            self.annotation_manager = get_annotation_manager()
            print("✅ AnnotationManager initialisé")
        except ImportError:
            print("⚠️ AnnotationManager non disponible")
        
        # Chargement depuis la base au démarrage (APRÈS l'init de annotation_manager)
        self._load_persistent_data()
        
        # Vérification non-bloquante des dépendances (en arrière-plan)
        self.dependencies_ok = False
        if DEPENDENCY_MANAGER_AVAILABLE:
            try:
                all_available, missing = check_dependencies_silent()
                if all_available:
                    self.dependencies_ok = True
                    print("✅ Toutes les dépendances sont disponibles")
                else:
                    print(f"ℹ️ Dépendances manquantes (mode annotation disponible): {', '.join(missing)}")
            except Exception as e:
                print(f"⚠️ Erreur vérification des dépendances: {e}")
        
        # Moteur YOLO (initialisation avec gestion d'erreur détaillée)
        self.yolo_engine = None
        try:
            from ..core.yolo_engine import YOLOEngine
            self.yolo_engine = YOLOEngine()
            print("✅ YOLOEngine initialisé avec succès")
        except ImportError as e:
            print(f"❌ YOLOEngine non disponible - Dépendances manquantes: {e}")
            print("💡 Installez: pip install ultralytics torch opencv-python")
        except Exception as e:
            print(f"❌ Erreur critique initialisation YOLOEngine : {e}")
            import traceback
            traceback.print_exc()
        
        # Générateur de datasets YOLO
        self.dataset_generator = None
        try:
            from ..core.yolo_dataset_generator import YOLODatasetGenerator
            self.dataset_generator = YOLODatasetGenerator(self.annotation_manager)
            print("✅ YOLODatasetGenerator (complet) initialisé")
        except Exception as e:
            print(f"⚠️ YOLODatasetGenerator complet non disponible : {e}")
            # Fallback vers générateur simplifié
            try:
                from ..core.simple_dataset_generator import SimpleYOLODatasetGenerator
                self.dataset_generator = SimpleYOLODatasetGenerator(self.annotation_manager)
                print("✅ SimpleYOLODatasetGenerator (fallback) initialisé")
            except Exception as e2:
                print(f"❌ Aucun générateur disponible : {e2}")
        
        # NOUVEAU: Smart Annotation Engine (lazy loading)
        self.smart_engine = None
        self.smart_mode_enabled = False
        
        # Interface
        self.setup_ui()
        
        # Connexions
        self.setup_connections()
        
        # Initialisation - chargement des classes existantes et mise à jour interface
        print("🔄 Initialisation de l'interface - chargement des classes existantes...")
        self.load_existing_classes()
        self.update_interface_state()
        
        # Mise à jour immédiate de l'affichage après initialisation complète
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(100, self.refresh_interface_display)
        print("✅ Initialisation de l'interface terminée")
    
    def load_existing_classes(self):
        """Charge les classes existantes depuis l'AnnotationManager"""
        if not self.annotation_manager:
            print("⚠️ AnnotationManager non disponible pour le chargement des classes")
            return
        
        try:
            existing_classes = self.annotation_manager.get_all_classes()
            print(f"📋 Chargement de {len(existing_classes)} classes existantes depuis la base")
            
            classes_loaded = 0
            for class_name in existing_classes:
                if class_name not in self.object_classes:
                    # Récupérer les statistiques réelles de la classe
                    try:
                        stats = self.annotation_manager.get_class_statistics(class_name)
                        class_data = {
                            'name': class_name,
                            'description': f"Classe existante: {class_name}",
                            'examples': [],  # Sera mis à jour par les statistiques
                            'model_path': None,
                            'created_at': datetime.now(),
                            'color': '#FF0000',
                            'stats': stats  # Ajouter les statistiques
                        }
                        self.object_classes[class_name] = class_data
                        classes_loaded += 1
                        print(f"✅ Classe '{class_name}' chargée ({stats.example_count} exemples)")
                    except Exception as e:
                        print(f"⚠️ Erreur chargement statistiques pour '{class_name}': {e}")
                        # Charger quand même la classe avec des données minimales
                        class_data = {
                            'name': class_name,
                            'description': f"Classe existante: {class_name}",
                            'examples': [],
                            'model_path': None,
                            'created_at': datetime.now(),
                            'color': '#FF0000'
                        }
                        self.object_classes[class_name] = class_data
                        classes_loaded += 1
                        print(f"✅ Classe '{class_name}' chargée (statistiques non disponibles)")
            
            print(f"📊 Total: {classes_loaded} classes chargées dans l'interface")
            
            # CORRECTION CRITIQUE: Mise à jour immédiate de l'interface après chargement
            self.update_classes_tree_with_real_stats()
            self.update_class_combos()
            print("🔄 Interface mise à jour avec les classes chargées")
                    
        except Exception as e:
            print(f"❌ Erreur chargement classes existantes: {e}")
            import traceback
            traceback.print_exc()
        
    def setup_ui(self):
        """Construit l'interface utilisateur"""
        
        # Widget central pour le dock widget
        central_widget = QWidget()
        self.setWidget(central_widget)
        
        # Layout principal sur le widget central
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        
        # En-tête avec titre et informations
        header_frame = self.create_header()
        layout.addWidget(header_frame)
        
        # Widget à onglets principal
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        
        # Onglet 1: Gestion des Classes (AMÉLIORÉE)
        self.classes_tab = self.create_classes_tab()
        self.tab_widget.addTab(self.classes_tab, "📁 Classes d'Objets")
        
        # Onglet 2: Annotation Interactive
        self.annotation_tab = self.create_annotation_tab()
        self.tab_widget.addTab(self.annotation_tab, "🎯 Annotation")
        
        # Onglet 3: Entraînement
        self.training_tab = self.create_training_tab()
        self.tab_widget.addTab(self.training_tab, "🧠 Entraînement")
        
        # Onglet 4: Détection
        self.detection_tab = self.create_detection_tab()
        self.tab_widget.addTab(self.detection_tab, "🔍 Détection")
        
        layout.addWidget(self.tab_widget)
        
        # Barre de statut
        status_frame = self.create_status_bar()
        layout.addWidget(status_frame)
        
        # Boutons de contrôle
        controls_frame = self.create_controls()
        layout.addWidget(controls_frame)
        
    def create_header(self):
        """Crée l'en-tête de l'interface"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a90e2, stop:1 #50c8a3);
                border-radius: 8px;
                color: white;
            }
        """)
        
        layout = QHBoxLayout(frame)
        
        # Titre principal
        title_label = QLabel("🎯 YOLO Interactive Object Detector")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: white;")
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        # Informations de version
        version_label = QLabel("v1.0.0 - Développement")
        version_label.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 10px;")
        layout.addWidget(version_label)
        
        return frame
    
    def create_classes_tab(self):
        """Crée l'onglet de gestion des classes d'objets"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Section création de classe
        create_group = QGroupBox("➕ Créer une Nouvelle Classe")
        create_layout = QFormLayout(create_group)
        
        self.class_name_input = QLineEdit()
        self.class_name_input.setPlaceholderText("Ex: Poteaux électriques")
        create_layout.addRow("Nom de la classe:", self.class_name_input)
        
        self.class_description_input = QTextEdit()
        self.class_description_input.setPlaceholderText(
            "Description détaillée pour améliorer la détection:\n"
            "Ex: 'Structures verticales en béton ou métal,\n"
            "     rectangulaires, hauteur 3-8m, supportant des lignes électriques'"
        )
        self.class_description_input.setMaximumHeight(80)
        create_layout.addRow("Description sémantique:", self.class_description_input)
        
        self.create_class_btn = QPushButton("🎯 Créer Classe")
        self.create_class_btn.setMinimumHeight(35)
        self.create_class_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        create_layout.addRow(self.create_class_btn)
        
        layout.addWidget(create_group)
        
        # Section liste des classes
        list_group = QGroupBox("📋 Classes Existantes")
        list_layout = QVBoxLayout(list_group)
        
        self.classes_tree = QTreeWidget()
        self.classes_tree.setHeaderLabels(["Classe", "Exemples", "Statut", "Actions"])
        self.classes_tree.setRootIsDecorated(False)
        self.classes_tree.setAlternatingRowColors(True)
        list_layout.addWidget(self.classes_tree)
        
        layout.addWidget(list_group)
        
        # Boutons d'action
        actions_frame = QFrame()
        actions_layout = QHBoxLayout(actions_frame)
        
        self.view_detail_btn = QPushButton("📊 Vue Détaillée")
        self.edit_class_btn = QPushButton("✏️ Modifier")
        self.duplicate_class_btn = QPushButton("📋 Dupliquer")
        self.delete_class_btn = QPushButton("🗑️ Supprimer")
        self.export_class_btn = QPushButton("📤 Exporter")
        
        actions_layout.addWidget(self.view_detail_btn)
        actions_layout.addWidget(self.edit_class_btn)
        actions_layout.addWidget(self.duplicate_class_btn)
        actions_layout.addWidget(self.delete_class_btn)
        actions_layout.addStretch()
        actions_layout.addWidget(self.export_class_btn)
        
        # Connecter le bouton vue détaillée
        self.view_detail_btn.clicked.connect(self.show_class_detail)
        
        layout.addWidget(actions_frame)
        
        return widget
    
    def create_annotation_tab(self):
        """Crée l'onglet d'annotation interactive"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sélection de classe active
        class_group = QGroupBox("🎯 Classe Active")
        class_layout = QFormLayout(class_group)
        
        self.active_class_combo = QComboBox()
        self.active_class_combo.addItem("--- Aucune classe sélectionnée ---")
        # CORRECTION: Connexion du signal de changement de sélection
        self.active_class_combo.currentTextChanged.connect(self.on_active_class_changed)
        class_layout.addRow("Classe à annoter:", self.active_class_combo)
        
        layout.addWidget(class_group)
        
        # Outils d'annotation
        tools_group = QGroupBox("🛠️ Outils d'Annotation")
        tools_layout = QGridLayout(tools_group)
        
        # Mode d'annotation géométrique
        tools_layout.addWidget(QLabel("Forme:"), 0, 0)
        
        self.bbox_mode_btn = QPushButton("🔲 Rectangle")
        self.bbox_mode_btn.setCheckable(True)
        self.bbox_mode_btn.setChecked(True)
        tools_layout.addWidget(self.bbox_mode_btn, 0, 1)
        
        self.polygon_mode_btn = QPushButton("📐 Polygone")
        self.polygon_mode_btn.setCheckable(True)
        tools_layout.addWidget(self.polygon_mode_btn, 0, 2)
        
        # NOUVEAU: Mode d'assistance IA
        tools_layout.addWidget(QLabel("Intelligence:"), 1, 0)
        
        self.manual_mode_btn = QPushButton("✋ Manuel")
        self.manual_mode_btn.setCheckable(True)
        self.manual_mode_btn.setChecked(True)
        self.manual_mode_btn.setToolTip("Annotation manuelle classique")
        tools_layout.addWidget(self.manual_mode_btn, 1, 1)
        
        self.smart_mode_btn = QPushButton("🤖 Smart Assistant")
        self.smart_mode_btn.setCheckable(True)
        self.smart_mode_btn.setToolTip("IA assistée : YOLO + SAM pour précision optimale")
        self.smart_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #2E7D32;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
        """)
        tools_layout.addWidget(self.smart_mode_btn, 1, 2)
        
        # Contrôles principaux
        self.start_annotation_btn = QPushButton("🎯 Commencer l'Annotation")
        self.start_annotation_btn.setMinimumHeight(40)
        self.start_annotation_btn.setCheckable(True)
        self.start_annotation_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #FF5722;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        tools_layout.addWidget(self.start_annotation_btn, 2, 0, 1, 2)
        
        # NOUVEAU: Bouton détection automatique Smart Mode
        self.auto_detect_btn = QPushButton("🤖 Détection Auto")
        self.auto_detect_btn.setMinimumHeight(40)
        self.auto_detect_btn.setToolTip("Smart Mode : YOLO détecte automatiquement les objets dans la zone visible")
        self.auto_detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #BA68C8;
            }
            QPushButton:disabled {
                background-color: #E0E0E0;
                color: #9E9E9E;
            }
        """)
        self.auto_detect_btn.setEnabled(False)  # Activé uniquement en Smart Mode
        tools_layout.addWidget(self.auto_detect_btn, 2, 2)
        
        layout.addWidget(tools_group)
        
        # NOUVEAU: Panneau configuration Smart Mode (masqué par défaut)
        self.smart_config_group = QGroupBox("🤖 Configuration Smart Assistant")
        self.smart_config_group.setVisible(False)  # Masqué par défaut
        smart_config_layout = QFormLayout(self.smart_config_group)
        
        # Seuil de confiance YOLO
        self.yolo_confidence_slider = QSlider(Qt.Horizontal)
        self.yolo_confidence_slider.setMinimum(10)
        self.yolo_confidence_slider.setMaximum(50)
        self.yolo_confidence_slider.setValue(20)  # 20% par défaut
        self.yolo_confidence_slider.setToolTip("Seuil de confiance pour la détection YOLO préalable")
        
        self.yolo_confidence_label = QLabel("20%")
        yolo_conf_layout = QHBoxLayout()
        yolo_conf_layout.addWidget(self.yolo_confidence_slider)
        yolo_conf_layout.addWidget(self.yolo_confidence_label)
        smart_config_layout.addRow("Confiance YOLO:", yolo_conf_layout)
        
        # Options SAM
        self.enable_sam_checkbox = QCheckBox("Activer raffinement SAM")
        self.enable_sam_checkbox.setChecked(True)
        self.enable_sam_checkbox.setToolTip("Utilise SAM pour optimiser la précision des contours")
        smart_config_layout.addRow("", self.enable_sam_checkbox)
        
        # NOUVEAU: Option contours précis
        self.precise_contours_checkbox = QCheckBox("🔺 Générer contours précis (polygones)")
        self.precise_contours_checkbox.setChecked(True)
        self.precise_contours_checkbox.setToolTip("Extrait les contours polygonaux précis depuis les masques SAM pour l'entraînement")
        smart_config_layout.addRow("", self.precise_contours_checkbox)
        
        # Validation automatique
        self.auto_validation_checkbox = QCheckBox("Validation automatique (confiance > 80%)")
        self.auto_validation_checkbox.setChecked(True)
        self.auto_validation_checkbox.setToolTip("Accepte automatiquement les détections très confiantes")
        smart_config_layout.addRow("", self.auto_validation_checkbox)
        
        # Mode debug
        self.debug_mode_checkbox = QCheckBox("Mode debug (logs détaillés)")
        self.debug_mode_checkbox.setChecked(False)
        self.debug_mode_checkbox.setToolTip("Affiche des informations détaillées sur le processus IA")
        smart_config_layout.addRow("", self.debug_mode_checkbox)
        
        # Statut Smart Engine
        self.smart_status_label = QLabel("⏳ Smart Engine non initialisé")
        self.smart_status_label.setStyleSheet("color: #FF9800; font-style: italic; font-size: 10px;")
        smart_config_layout.addRow("Statut:", self.smart_status_label)
        
        layout.addWidget(self.smart_config_group)
        
        # Progression et exemples
        progress_group = QGroupBox("📊 Progression")
        progress_layout = QVBoxLayout(progress_group)
        
        # Barre de progression
        self.annotation_progress = QProgressBar()
        self.annotation_progress.setMinimum(0)
        self.annotation_progress.setMaximum(20)  # 20 exemples recommandés
        self.annotation_progress.setValue(0)
        self.annotation_progress.setFormat("Exemples: %v / %m (Minimum: 10)")
        progress_layout.addWidget(self.annotation_progress)
        
        # Liste des exemples
        self.examples_list = QListWidget()
        self.examples_list.setMaximumHeight(120)
        progress_layout.addWidget(self.examples_list)
        
        # Statistiques
        self.stats_label = QLabel("📈 Aucun exemple collecté")
        self.stats_label.setStyleSheet("color: #666; font-style: italic;")
        progress_layout.addWidget(self.stats_label)
        
        layout.addWidget(progress_group)
        
        return widget
        
    def create_training_tab(self):
        """Crée l'onglet d'entraînement de modèles"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sélection du modèle à entraîner
        model_group = QGroupBox("🧠 Configuration d'Entraînement")
        model_layout = QFormLayout(model_group)
        
        self.training_class_combo = QComboBox()
        model_layout.addRow("Classe à entraîner:", self.training_class_combo)
        
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItems([
            "yolo11n.pt (Ultra-rapide, 5MB - Recommandé pour tests)",
            "yolo11s.pt (Équilibré, 19MB - Optimal qualité/vitesse)", 
            "yolo11m.pt (Haute précision, 39MB - Maximum performances)"
        ])
        # Sélection par défaut : modèle équilibré
        self.base_model_combo.setCurrentIndex(1)
        model_layout.addRow("Modèle de base:", self.base_model_combo)
        
        layout.addWidget(model_group)
        
        # Paramètres d'entraînement
        params_group = QGroupBox("⚙️ Paramètres")
        params_layout = QFormLayout(params_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 200)
        self.epochs_spin.setValue(50)
        params_layout.addRow("Nombre d'époques:", self.epochs_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(8)
        params_layout.addRow("Taille de batch:", self.batch_size_spin)
        
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 0.1)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setDecimals(4)
        params_layout.addRow("Taux d'apprentissage:", self.learning_rate_spin)
        
        layout.addWidget(params_group)
        
        # Contrôles d'entraînement
        training_group = QGroupBox("🚀 Entraînement")
        training_layout = QVBoxLayout(training_group)
        
        self.train_btn = QPushButton("🚀 Générer Dataset")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        training_layout.addWidget(self.train_btn)
        
        # Barre de progression d'entraînement
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        training_layout.addWidget(self.training_progress)
        
        # Zone de logs
        self.training_logs = QTextEdit()
        self.training_logs.setMaximumHeight(100)
        self.training_logs.setPlaceholderText("Les logs d'entraînement apparaîtront ici...")
        training_layout.addWidget(self.training_logs)
        
        layout.addWidget(training_group)
        
        return widget
        
    def create_detection_tab(self):
        """Crée l'onglet de détection et application"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sélection du modèle
        model_group = QGroupBox("🔍 Configuration de Détection")
        model_layout = QFormLayout(model_group)
        
        # NOUVEAUTÉ: Sélection par classe d'objet
        self.detection_class_combo = QComboBox()
        self.detection_class_combo.addItem("--- Sélectionner une classe ---")
        self.detection_class_combo.currentIndexChanged.connect(self.on_detection_class_index_changed)
        model_layout.addRow("🎯 Détecter la classe:", self.detection_class_combo)
        
        # OU sélection par modèle (mode expert)
        self.detection_model_combo = QComboBox()
        self.detection_model_combo.addItem("--- Aucun modèle disponible ---")
        model_layout.addRow("🔧 Modèle expert:", self.detection_model_combo)
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(10)  # ✅ Valeur par défaut plus permissive
        
        self.confidence_label = QLabel("0.10")
        confidence_frame = QFrame()
        confidence_layout = QHBoxLayout(confidence_frame)
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_label)
        
        model_layout.addRow("Seuil de confiance:", confidence_frame)
        
        layout.addWidget(model_group)
        
        # NOUVEAUTÉ: Informations d'échelle
        scale_group = QGroupBox("📐 Informations d'Échelle")
        scale_layout = QFormLayout(scale_group)
        
        self.scale_info_label = QLabel("Sélectionner une classe pour voir les informations d'échelle")
        self.scale_info_label.setWordWrap(True)
        self.scale_info_label.setStyleSheet("""
            QLabel {
                background-color: #f0f8ff;
                border: 1px solid #b0c4de;
                border-radius: 4px;
                padding: 8px;
                color: #333;
            }
        """)
        scale_layout.addRow(self.scale_info_label)
        
        layout.addWidget(scale_group)
        
        # Zone de traitement
        processing_group = QGroupBox("🎯 Zone de Traitement")
        processing_layout = QVBoxLayout(processing_group)
        
        # Options de zone
        zone_frame = QFrame()
        zone_layout = QHBoxLayout(zone_frame)
        
        self.current_view_radio = QCheckBox("Vue actuelle du canvas")
        self.current_view_radio.setChecked(True)
        zone_layout.addWidget(self.current_view_radio)
        
        self.custom_extent_radio = QCheckBox("Emprise personnalisée")
        zone_layout.addWidget(self.custom_extent_radio)
        
        self.full_layer_radio = QCheckBox("Couche complète")
        zone_layout.addWidget(self.full_layer_radio)
        
        processing_layout.addWidget(zone_frame)
        
        layout.addWidget(processing_group)
        
        # Bouton de détection
        detect_group = QGroupBox("🚀 Lancer la Détection")
        detect_layout = QVBoxLayout(detect_group)
        
        self.detect_btn = QPushButton("🔍 Détecter les Objets")
        self.detect_btn.setMinimumHeight(40)
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        detect_layout.addWidget(self.detect_btn)
        
        # Barre de progression de détection
        self.detection_progress = QProgressBar()
        self.detection_progress.setVisible(False)
        detect_layout.addWidget(self.detection_progress)
        
        layout.addWidget(detect_group)
        
        # Résultats
        results_group = QGroupBox("📊 Résultats")
        results_layout = QVBoxLayout(results_group)
        
        self.results_label = QLabel("Aucune détection effectuée")
        self.results_label.setStyleSheet("color: #666; font-style: italic;")
        results_layout.addWidget(self.results_label)
        
        layout.addWidget(results_group)
        
        return widget
    
    def create_status_bar(self):
        """Crée la barre de statut"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout(frame)
        
        self.status_label = QLabel("🟢 Plugin chargé - Prêt à utiliser")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Indicateur GPU/CPU
        self.device_label = QLabel("💻 CPU")
        layout.addWidget(self.device_label)
        
        return frame
    
    def create_controls(self):
        """Crée les boutons de contrôle"""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        # Bouton aide
        help_btn = QPushButton("❓ Aide")
        help_btn.clicked.connect(self.show_help)
        layout.addWidget(help_btn)
        
        # Bouton paramètres
        settings_btn = QPushButton("⚙️ Paramètres")
        settings_btn.clicked.connect(self.show_settings)
        layout.addWidget(settings_btn)
        
        # Bouton gestionnaire de dépendances
        if DEPENDENCY_MANAGER_AVAILABLE:
            deps_btn = QPushButton("📦 Dépendances")
            deps_btn.clicked.connect(self.show_dependency_manager)
            layout.addWidget(deps_btn)
        
        layout.addStretch()
        
        # Bouton fermer
        close_btn = QPushButton("✅ Fermer")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        return frame
        
    def setup_connections(self):
        """Configure les connexions de signaux"""
        
        # Onglet Classes
        self.create_class_btn.clicked.connect(self.create_new_class)
        self.classes_tree.itemSelectionChanged.connect(self.on_class_selection_changed)
        
        # CORRECTION: Connexions manquantes pour les boutons de gestion des classes
        self.edit_class_btn.clicked.connect(self.edit_selected_class)
        self.duplicate_class_btn.clicked.connect(self.duplicate_selected_class)
        self.delete_class_btn.clicked.connect(self.delete_selected_class)
        self.export_class_btn.clicked.connect(self.export_selected_class)
        
        # Onglet Annotation
        self.start_annotation_btn.toggled.connect(self.toggle_annotation_mode)
        self.bbox_mode_btn.toggled.connect(self.on_annotation_mode_changed)
        self.polygon_mode_btn.toggled.connect(self.on_annotation_mode_changed)
        
        # NOUVEAU: Connexions Smart Mode
        self.manual_mode_btn.toggled.connect(self.on_intelligence_mode_changed)
        self.smart_mode_btn.toggled.connect(self.on_intelligence_mode_changed)
        self.auto_detect_btn.clicked.connect(self.start_smart_auto_detection)
        
        # NOUVEAU: Connexions configuration Smart Mode
        self.yolo_confidence_slider.valueChanged.connect(self.update_yolo_confidence_label)
        self.enable_sam_checkbox.toggled.connect(self.on_smart_config_changed)
        self.precise_contours_checkbox.toggled.connect(self.on_smart_config_changed)
        self.auto_validation_checkbox.toggled.connect(self.on_smart_config_changed)
        self.debug_mode_checkbox.toggled.connect(self.on_smart_config_changed)
        
        # Connexions outil d'annotation
        if self.annotation_tool:
            self.annotation_tool.annotation_created.connect(self.on_annotation_created)
            self.annotation_tool.tool_activated.connect(self.on_annotation_tool_activated)
            self.annotation_tool.tool_deactivated.connect(self.on_annotation_tool_deactivated)
        
        # Connexions gestionnaire d'annotations
        if self.annotation_manager:
            self.annotation_manager.annotation_added.connect(self.on_annotation_added)
            self.annotation_manager.statistics_updated.connect(self.on_statistics_updated)
        
        # Connexions générateur de datasets
        if self.dataset_generator:
            self.dataset_generator.dataset_generation_started.connect(self.on_dataset_generation_started)
            self.dataset_generator.dataset_generation_progress.connect(self.on_dataset_generation_progress)
            self.dataset_generator.dataset_generation_completed.connect(self.on_dataset_generation_completed)
            self.dataset_generator.dataset_generation_error.connect(self.on_dataset_generation_error)
        
        # Onglet Entraînement
        self.train_btn.clicked.connect(self.start_training)
        
        # Onglet Détection
        self.detect_btn.clicked.connect(self.start_detection)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        
        # Mises à jour d'interface
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
    def create_new_class(self):
        """Crée une nouvelle classe d'objet"""
        name = self.class_name_input.text().strip()
        description = self.class_description_input.toPlainText().strip()
        
        if not name:
            QMessageBox.warning(self, "Erreur", "Veuillez saisir un nom de classe.")
            return
            
        if name in self.object_classes:
            QMessageBox.warning(self, "Erreur", f"La classe '{name}' existe déjà.")
            return
        
        # Création de la classe
        class_data = {
            'name': name,
            'description': description,
            'examples': [],
            'model_path': None,
            'created_at': datetime.now(),
            'color': '#FF0000'  # Rouge par défaut
        }
        
        self.object_classes[name] = class_data
        
        # Enregistrement dans la base de données
        if self.annotation_manager:
            try:
                self.annotation_manager.create_class(name, description, class_data['color'])
                print(f"✅ Classe '{name}' enregistrée dans la base de données")
            except Exception as e:
                print(f"⚠️ Erreur enregistrement classe dans la base: {e}")
        
        # Mise à jour de l'interface
        if self.annotation_manager:
            self.update_classes_tree_with_real_stats()
        else:
            self.update_classes_tree()
        self.update_class_combos()
        
        # Sélection automatique de la nouvelle classe
        items = self.classes_tree.findItems(name, Qt.MatchExactly, 0)
        if items:
            self.classes_tree.setCurrentItem(items[0])
            self.current_class = name
            
            # Mise à jour des combos avec la nouvelle sélection
            self.active_class_combo.setCurrentText(name)
        
        # Force la mise à jour de l'état de l'interface
        self.update_interface_state()
        
        # CORRECTION: Ajouter un modèle fictif pour test si pas de YOLOEngine
        trained_models = self.annotation_manager.get_trained_models()
        if not self.yolo_engine and not trained_models:
            self._add_test_model(name)
        
        # Nettoyage du formulaire
        self.class_name_input.clear()
        self.class_description_input.setPlainText("")
        
        # Message de succès
        self.status_label.setText(f"✅ Classe '{name}' créée avec succès - Prête pour l'annotation")
        
        # Émet le signal
        self.class_created.emit(name)
    
    def update_classes_tree(self):
        """Met à jour l'arbre des classes"""
        self.classes_tree.clear()
        
        for name, data in self.object_classes.items():
            item = QTreeWidgetItem([
                name,
                str(len(data['examples'])),
                "🟢 Prêt" if len(data['examples']) >= 10 else "⚠️ Incomplet",
                ""
            ])
            self.classes_tree.addTopLevelItem(item)
    
    def update_class_combos(self):
        """Met à jour les combos de sélection de classe"""
        # Sauvegarde des sélections actuelles
        current_active = self.active_class_combo.currentText()
        current_training = self.training_class_combo.currentText()
        
        # Vidage et remplissage
        self.active_class_combo.clear()
        self.training_class_combo.clear()
        
        if not self.object_classes:
            self.active_class_combo.addItem("--- Aucune classe disponible ---")
            self.training_class_combo.addItem("--- Aucune classe disponible ---")
        else:
            self.active_class_combo.addItem("--- Sélectionner une classe ---")
            self.training_class_combo.addItem("--- Sélectionner une classe ---")
            
            for name in self.object_classes.keys():
                self.active_class_combo.addItem(name)
                self.training_class_combo.addItem(name)
        
        # Restauration des sélections si possibles
        active_index = self.active_class_combo.findText(current_active)
        if active_index >= 0:
            self.active_class_combo.setCurrentIndex(active_index)
        else:
            # CORRECTION: Si pas de sélection restaurée, prendre la première classe disponible
            if len(self.object_classes) > 0:
                first_class = list(self.object_classes.keys())[0]
                first_index = self.active_class_combo.findText(first_class)
                if first_index >= 0:
                    # Bloquer les signaux temporairement pour éviter la double activation
                    self.active_class_combo.blockSignals(True)
                    self.active_class_combo.setCurrentIndex(first_index)
                    self.active_class_combo.blockSignals(False)
                    
                    # Définir manuellement la classe active
                    self.current_class = first_class
                    print(f"🎯 Classe active auto-sélectionnée: {first_class}")
                    
                    # Mise à jour de l'interface
                    self.update_annotation_progress()
                    self.update_examples_list()
            
        training_index = self.training_class_combo.findText(current_training)
        if training_index >= 0:
            self.training_class_combo.setCurrentIndex(training_index)
    
    def _load_persistent_data(self):
        """Charge les données persistantes depuis la base"""
        try:
            # Chargement des datasets et modèles depuis la DB
            datasets = self.annotation_manager.get_datasets()
            models = self.annotation_manager.get_trained_models()
            
            print(f"🔄 Chargement données persistantes:")
            print(f"📊 Datasets: {len(datasets)}")
            print(f"🤖 Modèles: {len(models)}")
            
            # Logs détaillés des modèles disponibles
            for model in models:
                print(f"  • {model['name']} → classes: {model['class_names']}")
                
        except Exception as e:
            print(f"❌ Erreur chargement données persistantes: {e}")

    def update_interface_state(self):
        """Met à jour l'état général de l'interface"""
        has_classes = bool(self.object_classes)
        has_active_class = self.current_class is not None
        
        # NOUVEAUTÉ: Utilisation de la persistance DB
        trained_models = self.annotation_manager.get_trained_models()
        has_trained_models = bool(trained_models)
        
        # DEBUG: Affichage des états pour diagnostiquer
        print(f"🔍 Mise à jour interface - Classes: {has_classes}, Active: {has_active_class}, Modèles: {has_trained_models}")
        print(f"🔍 Modèles disponibles: {[m['name'] for m in trained_models]}")
        
        # Activation/désactivation des contrôles
        self.start_annotation_btn.setEnabled(has_active_class)
        self.train_btn.setEnabled(has_classes)
        # CORRECTION: Activer le bouton détection si des modèles sont disponibles
        self.detect_btn.setEnabled(has_trained_models)
        
        # DEBUG: Vérification de l'état du bouton
        print(f"🔍 Bouton détection activé: {self.detect_btn.isEnabled()}")
        
        # Mise à jour de la liste des modèles si nécessaire
        if has_trained_models:
            self._update_detection_models_list()
        
        # Mise à jour du statut
        if not has_classes:
            self.status_label.setText("📝 Créez votre première classe d'objet pour commencer")
        elif not has_active_class:
            self.status_label.setText("🎯 Sélectionnez une classe pour commencer l'annotation")
        else:
            examples_count = len(self.object_classes[self.current_class]['examples'])
            if examples_count < 10:
                self.status_label.setText(f"📊 {examples_count}/10 exemples minimum - Continuez l'annotation")
            else:
                self.status_label.setText(f"✅ {examples_count} exemples - Prêt pour l'entraînement")
    
    def on_class_selection_changed(self):
        """Gestion du changement de sélection de classe"""
        current_item = self.classes_tree.currentItem()
        
        if current_item:
            class_name = current_item.text(0)
            self.current_class = class_name
            
            # Mise à jour des combos
            self.active_class_combo.setCurrentText(class_name)
            
            # Mise à jour des progressions et listes
            self.update_annotation_progress()
            self.update_examples_list()
        else:
            self.current_class = None
            
        self.update_interface_state()
    
    def toggle_annotation_mode(self, checked):
        """Active/désactive le mode annotation"""
        if checked:
            if not self.current_class:
                QMessageBox.warning(self, "Attention", "Veuillez sélectionner une classe d'objet avant de commencer l'annotation.")
                self.start_annotation_btn.setChecked(False)
                return
            
            if not ANNOTATION_TOOL_AVAILABLE or not self.annotation_tool:
                QMessageBox.critical(self, "Erreur", "Outil d'annotation non disponible.")
                self.start_annotation_btn.setChecked(False)
                return
                
            # Activation de l'outil d'annotation
            try:
                self.annotation_tool.set_active_class(self.current_class)
                mode = 'bbox' if self.bbox_mode_btn.isChecked() else 'polygon'
                self.annotation_tool.set_annotation_mode(mode)
                
                # Activation dans QGIS
                self.iface.mapCanvas().setMapTool(self.annotation_tool)
                
                self.status_label.setText(f"🎯 Mode annotation actif - Classe: {self.current_class} | Cliquez et glissez pour dessiner des rectangles autour des objets")
                self.start_annotation_btn.setText("⏸️ Arrêter l'Annotation")
                
            except Exception as e:
                error_msg = f"Erreur lors de l'activation de l'outil d'annotation:\n{str(e)}"
                QMessageBox.critical(self, "Erreur d'Activation", error_msg)
                self.start_annotation_btn.setChecked(False)
                
        else:
            # Désactivation de l'outil
            if self.annotation_tool:
                self.iface.mapCanvas().unsetMapTool(self.annotation_tool)
            
            self.status_label.setText("⏹️ Mode annotation arrêté")
            self.start_annotation_btn.setText("🎯 Commencer l'Annotation")
    
    def on_annotation_mode_changed(self):
        """Gestion du changement de mode d'annotation"""
        if self.bbox_mode_btn.isChecked():
            self.polygon_mode_btn.setChecked(False)
        elif self.polygon_mode_btn.isChecked():
            self.bbox_mode_btn.setChecked(False)
        else:
            # Au moins un mode doit être sélectionné
            self.bbox_mode_btn.setChecked(True)
    
    def start_training(self):
        """Lance la génération de dataset ou l'entraînement selon le contexte"""
        
        # Si nous avons déjà un dataset et que le bouton dit "Lancer Entraînement"
        if (self.current_dataset_info and 
            self.train_btn.text() == "🧠 Lancer Entraînement"):
            self._start_actual_training(self.current_dataset_info)
            return
        
        # Sinon, procédure normale de génération de dataset
        selected_class = self.training_class_combo.currentText()
        
        if selected_class.startswith("---"):
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner une classe à entraîner.")
            return
        
        if not self.dataset_generator:
            QMessageBox.critical(self, "Erreur", "Générateur de datasets non disponible.")
            return
        
        # Vérification des données
        if self.annotation_manager:
            stats = self.annotation_manager.get_class_statistics(selected_class)
            if not stats.ready_for_training:
                reply = QMessageBox.question(
                    self, 
                    "Données Insuffisantes", 
                    f"La classe '{selected_class}' n'a que {stats.example_count} exemples.\n"
                    f"Minimum recommandé : 10 exemples de qualité.\n"
                    f"Score qualité actuel : {stats.quality_score:.1%}\n\n"
                    f"Continuer quand même ?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
        
        # Dialog de configuration du dataset
        try:
            dataset_name = f"dataset_{selected_class}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Vérification du type de générateur pour adapter le message
            has_augmentation = hasattr(self.dataset_generator, 'augmentation_config')
            
            if has_augmentation:
                # Configuration d'augmentation complète
                from ..core.yolo_dataset_generator import AugmentationConfig
                aug_config = AugmentationConfig(
                    enabled=True,
                    augmentation_factor=3
                )
                
                message = (f"Génération du dataset d'entraînement pour '{selected_class}':\n\n"
                          f"• Générateur: Complet avec augmentation\n"
                          f"• Augmentation de données: Activée (3x multiplicateur)\n"
                          f"• Division: 70% train / 20% val / 10% test\n"
                          f"• Format: YOLO standard\n\n"
                          f"Continuer ?")
            else:
                # Générateur simplifié
                aug_config = None
                message = (f"Génération du dataset d'entraînement pour '{selected_class}':\n\n"
                          f"• Générateur: Simplifié (sans augmentation)\n"
                          f"• Augmentation de données: Non disponible\n"
                          f"• Division: 70% train / 20% val / 10% test\n"
                          f"• Format: YOLO standard\n\n"
                          f"Continuer ?")
            
            # Message de confirmation
            reply = QMessageBox.question(
                self,
                "Génération du Dataset",
                message,
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Lancement de la génération
                self.train_btn.setEnabled(False)
                self.train_btn.setText("🔄 Génération en cours...")
                self.training_progress.setVisible(True)
                self.training_progress.setRange(0, 0)  # Mode indéterminé
                
                # Génération asynchrone
                from qgis.PyQt.QtCore import QTimer
                QTimer.singleShot(100, lambda: self._generate_dataset_for_training(
                    dataset_name, [selected_class], aug_config
                ))
                
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la préparation : {str(e)}")
    
    def _generate_dataset_for_training(self, dataset_name: str, selected_classes: list, aug_config):
        """Génère le dataset en arrière-plan"""
        try:
            # Adaptation selon le type de générateur
            if hasattr(self.dataset_generator, 'augmentation_config') and aug_config:
                # Générateur complet avec augmentation
                dataset_info = self.dataset_generator.generate_dataset(
                    dataset_name=dataset_name,
                    selected_classes=selected_classes,
                    augmentation_config=aug_config
                )
            else:
                # Générateur simplifié
                dataset_info = self.dataset_generator.generate_dataset(
                    dataset_name=dataset_name,
                    selected_classes=selected_classes
                )
            # Le succès sera géré par le signal dataset_generation_completed
            
        except Exception as e:
            self.dataset_generator.dataset_generation_error.emit(dataset_name, str(e))
    
    def start_detection(self):
        """Démarre la détection d'objets sur le raster actif"""
        
        # CORRECTION: Vérifications préalables plus flexibles
        print(f"🔍 Début détection - YOLOEngine disponible: {self.yolo_engine is not None}")
        trained_models = self.annotation_manager.get_trained_models()
        print(f"🔍 Modèles entraînés disponibles: {len(trained_models)}")
        
        # Vérification absolue du YOLOEngine
        if not self.yolo_engine:
            QMessageBox.critical(self, "Erreur", 
                "YOLOEngine non disponible. Vérifiez l'installation des dépendances:\n"
                "- PyTorch\n"
                "- Ultralytics\n"
                "- OpenCV\n\n"
                "Utilisez le gestionnaire de dépendances pour les installer.")
            return
        
        # NOUVEAUTÉ: Détection du mode de sélection (classe ou modèle)
        selected_class = self.detection_class_combo.currentData()
        selected_model_text = self.detection_model_combo.currentText()
        
        model_data = None
        detection_mode = None
        
        # Priorité 1: Sélection par classe
        if selected_class and not selected_class.startswith("---"):
            model = self.get_model_for_class(selected_class)
            if model:
                model_data = model['id']  # Utilise l'ID du modèle
                model_path = model['path']
                detection_mode = f"classe '{selected_class}'"
                print(f"🎯 Détection par classe: {selected_class} → modèle: {model['name']}")
            else:
                QMessageBox.warning(self, "Erreur", f"Aucun modèle disponible pour la classe '{selected_class}'")
                return
        
        # Priorité 2: Sélection par modèle (mode expert)
        elif not selected_model_text.startswith("---"):
            model_id = self.detection_model_combo.currentData()
            if model_id:
                # Trouver le modèle par ID
                trained_models = self.annotation_manager.get_trained_models()
                model = next((m for m in trained_models if m['id'] == model_id), None)
                if model:
                    model_data = model['id']
                    model_path = model['path']
                    detection_mode = f"modèle '{model['name']}'"
                    print(f"🔧 Détection par modèle expert: {model['name']}")
                else:
                    QMessageBox.warning(self, "Erreur", f"Modèle non trouvé: {model_id}")
                    return
            else:
                QMessageBox.warning(self, "Erreur", "Sélection de modèle invalide")
                return
        
        # Aucune sélection valide
        else:
            # Auto-sélection si une seule classe/modèle disponible
            available_classes = self.get_available_classes_with_models()
            trained_models = self.annotation_manager.get_trained_models()
            
            if len(available_classes) == 1:
                selected_class = available_classes[0]
                model = self.get_model_for_class(selected_class)
                if model:
                    model_data = model['id']
                    model_path = model['path']
                    detection_mode = f"classe '{selected_class}' (auto-sélection)"
                    print(f"🔄 Auto-sélection: {detection_mode}")
                else:
                    QMessageBox.warning(self, "Erreur", "Modèle auto-sélectionné introuvable")
                    return
            elif len(trained_models) == 1:
                model = trained_models[0]
                model_data = model['id']
                model_path = model['path']
                detection_mode = f"modèle '{model['name']}' (auto-sélection)"
                print(f"🔄 Auto-sélection: {detection_mode}")
            else:
                QMessageBox.warning(self, "Attention", 
                    "Veuillez sélectionner:\n"
                    "• Une classe d'objet à détecter, OU\n"
                    "• Un modèle spécifique (mode expert)")
                return
        
        # Vérification finale du chemin du modèle
        if not model_data or not model_path:
            QMessageBox.warning(self, "Erreur", "Informations de modèle incomplètes")
            return
        
        print(f"🔍 Mode de détection: {detection_mode}")
        print(f"🔍 Chemin modèle: {model_path}")
        
        # CORRECTION: Vérification intelligente de l'existence du modèle
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Erreur", f"Fichier modèle non trouvé :\n{model_path}")
            return
        
        # Diagnostic complet de l'état du système
        print("🔍 === DIAGNOSTIC DÉTECTION ===")
        
        # Vérification du canvas
        canvas = self.iface.mapCanvas()
        print(f"🔍 Canvas: {canvas}")
        print(f"🔍 Canvas layers: {canvas.layerCount()}")
        
        # Vérification de la couche raster active
        active_layer = canvas.currentLayer()
        print(f"🔍 Couche active: {active_layer}")
        print(f"🔍 Type couche active: {type(active_layer)}")
        
        from qgis.core import QgsRasterLayer, QgsProject
        
        # Liste toutes les couches du projet
        project_layers = QgsProject.instance().mapLayers()
        raster_layers = [layer for layer in project_layers.values() if isinstance(layer, QgsRasterLayer)]
        print(f"🔍 Couches raster dans le projet: {len(raster_layers)}")
        for i, layer in enumerate(raster_layers):
            print(f"  {i+1}. {layer.name()} - Valide: {layer.isValid()}")
        
        if not isinstance(active_layer, QgsRasterLayer):
            if len(raster_layers) > 0:
                # Utiliser la première couche raster disponible
                active_layer = raster_layers[0]
                print(f"🔍 Utilisation automatique de: {active_layer.name()}")
                QMessageBox.information(self, "Couche Sélectionnée",
                    f"Aucune couche raster active détectée.\n"
                    f"Utilisation automatique de: {active_layer.name()}")
            else:
                QMessageBox.warning(self, "Couche Manquante",
                    "Aucune couche raster trouvée dans le projet.\n"
                    "Veuillez charger une couche raster (GeoTIFF, etc.) pour lancer la détection.")
                return
        
        # Vérifications de validité de la couche
        if not active_layer.isValid():
            QMessageBox.critical(self, "Couche Invalide",
                f"La couche raster '{active_layer.name()}' n'est pas valide.\n"
                f"Vérifiez que le fichier source existe et est accessible.")
            return
        
        print(f"🔍 Couche finale sélectionnée: {active_layer.name()}")
        print(f"🔍 Étendue couche: {active_layer.extent()}")
        print(f"🔍 CRS couche: {active_layer.crs().authid()}")
        
        # Configuration de la détection
        confidence = self.confidence_slider.value() / 100.0  # Conversion 0-100 -> 0.0-1.0
        
        # Détermination de la zone à traiter
        if self.current_view_radio.isChecked():
            extent = self.iface.mapCanvas().extent()
            zone_description = "vue actuelle du canvas"
        elif self.full_layer_radio.isChecked():
            extent = active_layer.extent()
            zone_description = "couche complète"
        else:  # custom_extent_radio
            extent = self.iface.mapCanvas().extent()  # Pour l'instant, comme current_view
            zone_description = "emprise personnalisée"
        
        # Confirmation avant lancement
        reply = QMessageBox.question(
            self,
            "Lancer la Détection",
            f"🔍 Paramètres de détection :\n\n"
            f"• Modèle : {model_data}\n"
            f"• Couche : {active_layer.name()}\n"
            f"• Zone : {zone_description}\n"
            f"• Seuil confiance : {confidence:.1%}\n\n"
            f"Lancer la détection ?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._run_detection(active_layer, model_path, extent, confidence, zone_description)
    
    def update_confidence_label(self, value):
        """Met à jour le label du seuil de confiance"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
    
    def update_scale_info(self, class_name: str):
        """Met à jour les informations d'échelle pour une classe"""
        try:
            stats = self.annotation_manager.get_class_statistics(class_name)
            
            if stats.example_count == 0:
                self.scale_info_label.setText(f"❌ Aucun exemple d'entraînement pour '{class_name}'")
                return
            
            # Formatage des informations d'échelle
            info_text = f"📊 <b>{class_name}</b> ({stats.example_count} exemples)<br/>"
            
            if stats.optimal_pixel_size > 0:
                info_text += f"🎯 <b>Résolution optimale:</b> {stats.optimal_pixel_size:.3f} m/pixel<br/>"
                info_text += f"📐 <b>Plage d'échelles:</b> 1:{stats.scale_range[0]:,} - 1:{stats.scale_range[1]:,}<br/>"
                info_text += f"🔍 <b>Zoom QGIS:</b> niveau {stats.zoom_level_range[0]} - {stats.zoom_level_range[1]}<br/>"
                
                # Indication de cohérence
                scale_variance = stats.pixel_size_range[1] / stats.pixel_size_range[0] if stats.pixel_size_range[0] > 0 else float('inf')
                if scale_variance < 1.5:
                    info_text += f"✅ <b>Cohérence:</b> Excellente (variance {scale_variance:.1f}x)"
                elif scale_variance < 2.5:
                    info_text += f"⚠️ <b>Cohérence:</b> Bonne (variance {scale_variance:.1f}x)"
                else:
                    info_text += f"❌ <b>Cohérence:</b> Faible (variance {scale_variance:.1f}x)"
            else:
                info_text += "⚠️ Données d'échelle incomplètes"
            
            self.scale_info_label.setText(info_text)
            
        except Exception as e:
            print(f"❌ Erreur mise à jour infos échelle: {e}")
            self.scale_info_label.setText(f"❌ Erreur récupération données pour '{class_name}'")
    
    def on_detection_class_index_changed(self, index):
        """Gestion du changement de classe de détection par index"""
        if index <= 0:  # Index 0 = "--- Sélectionner une classe ---"
            self.detect_btn.setEnabled(False)
            self.status_label.setText("❌ Aucune classe sélectionnée")
            self.scale_info_label.setText("Sélectionner une classe pour voir les informations d'échelle")
            return
        
        # Récupération du vrai nom de classe depuis itemData()
        class_name = self.detection_class_combo.itemData(index)
        display_name = self.detection_class_combo.itemText(index)
        
        if class_name:
            print(f"🎯 Classe sélectionnée: '{class_name}' (affiché: '{display_name}')")
            model = self.get_model_for_class(class_name)
            if model:
                print(f"✅ Modèle trouvé: '{model['name']}'")
                
            # NOUVEAUTÉ: Mise à jour des informations d'échelle
            self.update_scale_info(class_name)
            
            # Synchroniser la sélection du modèle expert
            if model:
                model_id = model['id']
                for i in range(self.detection_model_combo.count()):
                    if self.detection_model_combo.itemData(i) == model_id:
                        self.detection_model_combo.setCurrentIndex(i)
                        break
                
                # Activer le bouton de détection
                self.detect_btn.setEnabled(True)
                self.status_label.setText(f"🎯 Prêt à détecter: {class_name}")
            else:
                print(f"⚠️ Aucun modèle disponible pour la classe '{class_name}'")
                self.detect_btn.setEnabled(False)
                self.status_label.setText(f"❌ Aucun modèle entraîné pour: {class_name}")
    
    def on_tab_changed(self, index):
        """Gestion du changement d'onglet"""
        tab_names = ["Classes", "Annotation", "Entraînement", "Détection"]
        if index < len(tab_names):
            self.status_label.setText(f"📂 Onglet {tab_names[index]} actif")
            
            # CORRECTION: Si on accède à l'onglet détection, mettre à jour les listes
            if index == 3:  # Index 3 = onglet détection
                print("🔍 Accès onglet détection - Mise à jour des listes")
                self._update_detection_models_list()  # IMPORTANT: Mettre à jour les dropdowns
                trained_models = self.annotation_manager.get_trained_models()
                if trained_models:
                    print("🔍 Modèles trouvés - Activation du bouton")
                    self.detect_btn.setEnabled(True)
                else:
                    print("⚠️ Aucun modèle trouvé")
                self.update_interface_state()
    
    def show_help(self):
        """Affiche l'aide"""
        help_text = """
        <h3>🎯 YOLO Interactive Object Detector - Guide Rapide</h3>
        
        <h4>1. 📁 Créer des Classes d'Objets</h4>
        <p>• Définissez le type d'objet à détecter (ex: "Poteaux électriques")<br>
        • Une description optionnelle aide à clarifier l'objectif</p>
        
        <h4>2. 🎯 Annoter des Exemples</h4>
        <p>• Sélectionnez une classe active<br>
        • Activez le mode annotation<br>
        • Cliquez sur 10-20 exemples d'objets sur le canvas QGIS</p>
        
        <h4>3. 🧠 Entraîner le Modèle</h4>
        <p>• Choisissez la classe à entraîner<br>
        • Ajustez les paramètres si nécessaire<br>
        • Lancez l'entraînement (quelques minutes)</p>
        
        <h4>4. 🔍 Détecter Massivement</h4>
        <p>• Sélectionnez le modèle entraîné<br>
        • Définissez la zone de traitement<br>
        • Lancez la détection automatique</p>
        
        <p><b>💡 Astuce :</b> Plus vous fournissez d'exemples variés, meilleur sera le modèle !</p>
        """
        
        QMessageBox.information(self, "Aide", help_text)
    
    def show_settings(self):
        """Affiche les paramètres"""
        QMessageBox.information(self, "Paramètres", "Interface de paramètres en cours de développement...")
    
    def show_dependency_manager(self):
        """Affiche le gestionnaire de dépendances"""
        if DEPENDENCY_MANAGER_AVAILABLE:
            show_dependency_manager(self)
        else:
            QMessageBox.warning(self, "Gestionnaire non disponible", 
                              "Le gestionnaire de dépendances n'est pas disponible.")
    
    def on_annotation_created(self, annotation_data):
        """
        Gestion d'une nouvelle annotation créée
        
        Args:
            annotation_data: Données de l'annotation
        """
        try:
            class_name = annotation_data.get('class_name')
            if class_name in self.object_classes:
                # Ajout de l'exemple à la classe
                self.object_classes[class_name]['examples'].append(annotation_data)
                
                # Mise à jour de l'interface
                self.update_annotation_progress()
                self.update_classes_tree()
                
                # Mise à jour de la liste des exemples
                self.update_examples_list()
                
                # Message de succès
                examples_count = len(self.object_classes[class_name]['examples'])
                self.status_label.setText(
                    f"✅ Exemple #{examples_count} ajouté à '{class_name}'"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'ajout de l'exemple:\n{str(e)}")
    
    def on_annotation_tool_activated(self):
        """Outil d'annotation activé"""
        if hasattr(self, 'status_label'):
            self.status_label.setText("🎯 Outil d'annotation activé sur le canvas")
    
    def on_annotation_tool_deactivated(self):
        """Outil d'annotation désactivé"""
        # Synchronisation avec l'interface
        if self.start_annotation_btn.isChecked():
            self.start_annotation_btn.setChecked(False)
    
    def on_annotation_added(self, class_name: str, annotation_data: dict):
        """Gestion de l'ajout d'une nouvelle annotation via le gestionnaire"""
        print(f"📊 Nouvelle annotation ajoutée pour '{class_name}'")
        
        # Mise à jour de l'affichage si c'est la classe active
        if class_name == self.current_class:
            self.update_annotation_progress()
            self.update_examples_list()
    
    def on_statistics_updated(self, class_name: str, stats_data: dict):
        """Gestion de la mise à jour des statistiques"""
        print(f"📈 Statistiques mises à jour pour '{class_name}': {stats_data['example_count']} exemples")
        
        # Mise à jour de l'arbre des classes
        self.update_classes_tree_with_real_stats()
        
        # Mise à jour de l'interface si c'est la classe active
        if class_name == self.current_class:
            self.update_annotation_progress()
    
    def update_classes_tree_with_real_stats(self):
        """Met à jour l'arbre des classes avec les vraies statistiques"""
        if not self.annotation_manager:
            print("⚠️ AnnotationManager non disponible pour les statistiques")
            # Fallback vers méthode basique
            self.update_classes_tree()
            return
            
        try:
            print("🔄 Mise à jour arbre des classes avec statistiques réelles...")
            
            # Récupération de toutes les classes avec annotations
            classes_with_annotations = self.annotation_manager.get_all_classes()
            print(f"📊 Classes dans la base: {classes_with_annotations}")
            
            # Ajout des classes sans annotations (seulement dans self.object_classes)
            all_classes = set(self.object_classes.keys()) | set(classes_with_annotations)
            print(f"📋 Toutes les classes à afficher: {sorted(all_classes)}")
            
            self.classes_tree.clear()
            
            items_added = 0
            for class_name in sorted(all_classes):
                try:
                    stats = self.annotation_manager.get_class_statistics(class_name)
                    
                    # Statut basé sur les vraies statistiques
                    if stats.ready_for_training:
                        status = "✅ Prêt"
                    elif stats.example_count > 0:
                        status = f"⚠️ {stats.example_count}/10"
                    else:
                        status = "⭕ Vide"
                    
                    # Score qualité
                    quality = f"{stats.quality_score:.1%}" if stats.example_count > 0 else "0%"
                    
                    item = QTreeWidgetItem([
                        class_name,
                        str(stats.example_count),
                        status,
                        quality
                    ])
                    
                    # Couleur selon le statut
                    if stats.ready_for_training:
                        item.setForeground(2, item.foreground(2))  # Vert par défaut
                    elif stats.example_count > 0:
                        from qgis.PyQt.QtGui import QColor
                        item.setForeground(2, QColor(255, 165, 0))  # Orange
                    
                    self.classes_tree.addTopLevelItem(item)
                    items_added += 1
                    
                except Exception as e:
                    print(f"⚠️ Erreur statistiques pour classe '{class_name}': {e}")
                    # Ajouter quand même la classe avec des infos basiques
                    item = QTreeWidgetItem([
                        class_name,
                        "?",
                        "❓ Erreur",
                        "0%"
                    ])
                    self.classes_tree.addTopLevelItem(item)
                    items_added += 1
            
            print(f"✅ {items_added} classes ajoutées à l'arbre")
            
            # Force le rafraîchissement de l'affichage
            self.classes_tree.update()
                
        except Exception as e:
            print(f"❌ Erreur mise à jour arbre : {e}")
            import traceback
            traceback.print_exc()
            # Fallback vers l'ancienne méthode
            self.update_classes_tree()
    
    def on_dataset_generation_started(self, dataset_name: str):
        """Gestion du début de génération de dataset"""
        self.training_logs.append(f"📊 Début de génération du dataset '{dataset_name}'...")
        print(f"📊 Génération dataset démarrée : {dataset_name}")
    
    def on_dataset_generation_progress(self, class_name: str, current: int, total: int):
        """Gestion du progrès de génération"""
        progress_msg = f"🔄 Traitement classe '{class_name}' ({current}/{total})"
        self.training_logs.append(progress_msg)
        print(progress_msg)
    
    def on_dataset_generation_completed(self, dataset_name: str, dataset_info: dict):
        """Gestion de la fin de génération de dataset"""
        success_msg = (f"✅ Dataset '{dataset_name}' généré avec succès !\n"
                      f"• {dataset_info['total_images']} images totales\n"
                      f"• Train: {dataset_info['train_images']} | "
                      f"Val: {dataset_info['val_images']} | "
                      f"Test: {dataset_info['test_images']}")
        
        self.training_logs.append(success_msg)
        
        # Stockage des informations dataset pour l'entraînement
        self.current_dataset_info = dataset_info
        
        # Réinitialisation de l'interface de génération
        self.training_progress.setVisible(False)
        
        # Proposition d'entraînement si YOLOEngine disponible
        if self.yolo_engine:
            reply = QMessageBox.question(
                self,
                "Dataset Prêt - Lancer l'Entraînement ?",
                f"Dataset '{dataset_name}' généré avec succès !\n\n"
                f"• {dataset_info['total_images']} images générées\n"
                f"• Chemin : {dataset_info['dataset_path']}\n\n"
                f"🧠 Lancer l'entraînement du modèle YOLO maintenant ?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Lancer l'entraînement directement
                self._start_actual_training(dataset_info)
            else:
                # L'utilisateur peut lancer l'entraînement plus tard
                self.train_btn.setText("🧠 Lancer Entraînement")
                self.train_btn.setEnabled(True)
                self.status_label.setText(f"✅ Dataset prêt - Cliquez pour entraîner le modèle")
        else:
            # Pas de YOLOEngine - retour à l'état initial
            self.train_btn.setText("🚀 Générer Dataset")
            self.train_btn.setEnabled(True)
            
            QMessageBox.information(
                self,
                "Dataset Généré",
                f"Dataset '{dataset_name}' créé avec succès !\n\n"
                f"Images générées : {dataset_info['total_images']}\n"
                f"Chemin : {dataset_info['dataset_path']}\n\n"
                f"⚠️ YOLOEngine non disponible - Impossible de lancer l'entraînement."
            )
            
            self.status_label.setText(f"✅ Dataset généré - YOLOEngine requis pour entraînement")
        
        print(f"✅ Dataset généré : {dataset_name}")
    
    def on_dataset_generation_error(self, dataset_name: str, error_message: str):
        """Gestion des erreurs de génération"""
        error_msg = f"❌ Erreur génération dataset '{dataset_name}': {error_message}"
        self.training_logs.append(error_msg)
        
        # Réinitialisation de l'interface
        self.training_progress.setVisible(False)
        self.train_btn.setEnabled(True)
        self.train_btn.setText("🚀 Générer Dataset")
        
        # Message d'erreur
        QMessageBox.critical(
            self,
            "Erreur de Génération",
            f"Échec de la génération du dataset '{dataset_name}':\n\n{error_message}"
        )
        
        self.status_label.setText("❌ Erreur génération dataset")
        print(f"❌ Erreur génération dataset : {error_message}")
    
    def update_annotation_progress(self):
        """Met à jour la barre de progression d'annotation"""
        if not self.current_class:
            self.annotation_progress.setValue(0)
            self.stats_label.setText("📈 Aucune classe sélectionnée")
            return
        
        # Utilisation des vraies statistiques si disponibles
        if self.annotation_manager:
            try:
                stats = self.annotation_manager.get_class_statistics(self.current_class)
                examples_count = stats.example_count
                quality_score = stats.quality_score
                ready_for_training = stats.ready_for_training
            except Exception as e:
                print(f"❌ Erreur récupération stats : {e}")
                examples_count = 0
                quality_score = 0.0
                ready_for_training = False
        else:
            # Fallback vers anciennes données
            if self.current_class in self.object_classes:
                examples_count = len(self.object_classes[self.current_class]['examples'])
            else:
                examples_count = 0
            quality_score = 0.0
            ready_for_training = examples_count >= 10
        
        # Mise à jour de la barre de progression
        self.annotation_progress.setValue(min(examples_count, 20))  # Max 20 pour l'affichage
        
        # Mise à jour du texte de statut avec qualité
        if ready_for_training:
            self.stats_label.setText(f"✅ {examples_count} exemples - Qualité: {quality_score:.1%} - Prêt pour l'entraînement !")
        elif examples_count > 0:
            self.stats_label.setText(f"📊 {examples_count}/10 exemples - Qualité: {quality_score:.1%} - Continuez l'annotation")
        else:
            self.stats_label.setText("📈 Aucun exemple collecté - Commencez l'annotation")
    
    def update_examples_list(self):
        """Met à jour la liste des exemples"""
        self.examples_list.clear()
        
        if not self.current_class:
            return
        
        # Utiliser les vraies données de l'AnnotationManager si disponible
        examples = []
        if self.annotation_manager:
            try:
                examples = self.annotation_manager.get_class_annotations(self.current_class)
            except Exception as e:
                print(f"❌ Erreur récupération exemples : {e}")
                examples = []
        
        # Fallback vers les données locales si pas d'AnnotationManager
        if not examples and self.current_class in self.object_classes:
            examples = self.object_classes[self.current_class]['examples']
        
        for i, example in enumerate(examples):
            # Informations sur l'exemple
            if hasattr(example, 'timestamp'):
                # AnnotationExample object
                timestamp = example.timestamp
                dimensions = getattr(example, 'metadata', {}).get('dimensions_m', {})
                item_text = f"Exemple {example.id[:8]}... - {timestamp[:16]}"
            else:
                # Dict format (legacy)
                timestamp = example.get('timestamp', 'Inconnu')
                dimensions = example.get('dimensions_m', {})
                width = dimensions.get('width', 0)
                height = dimensions.get('height', 0)
                item_text = f"Exemple #{i+1} - {width:.1f}x{height:.1f}m - {timestamp[:16]}"
            
            # Ajout à la liste
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, example)  # Stockage des données
            self.examples_list.addItem(item)
    
    def _start_actual_training(self, dataset_info: dict):
        """Lance l'entraînement YOLO effectif"""
        if not self.yolo_engine:
            QMessageBox.critical(self, "Erreur", "YOLOEngine non disponible pour l'entraînement.")
            return
        
        try:
            # Récupération des paramètres depuis l'interface
            base_model_text = self.base_model_combo.currentText()
            base_model = base_model_text.split()[0]  # "yolo11n.pt (Rapide, léger)" -> "yolo11n.pt"
            epochs = self.epochs_spin.value()
            batch_size = self.batch_size_spin.value()
            learning_rate = self.learning_rate_spin.value()
            
            # Configuration pour l'entraînement
            config = {
                'dataset_path': dataset_info['config_path'],
                'base_model': base_model,
                'epochs': epochs, 
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'dataset_name': dataset_info.get('name', 'unknown')
            }
            
            # Mise à jour de l'interface pour l'entraînement
            self.training_in_progress = True
            self.train_btn.setText("🧠 Entraînement en cours...")
            self.train_btn.setEnabled(False)
            self.training_progress.setVisible(True)
            self.training_progress.setRange(0, epochs)
            
            # Message de début
            self.training_logs.append(f"🚀 Début entraînement YOLO avec {base_model}")
            self.training_logs.append(f"📊 Paramètres: {epochs} époques, batch={batch_size}, lr={learning_rate}")
            
            # Lancement de l'entraînement asynchrone
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._run_yolo_training_async(config))
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du lancement de l'entraînement:\n{str(e)}")
            self._reset_training_interface()
    
    def _run_yolo_training_async(self, config: dict):
        """Exécute l'entraînement YOLO de manière asynchrone"""
        try:
            # Callback de progression
            def on_training_progress(progress: float, info: dict):
                epoch = info.get('epoch', 0)
                total_epochs = info.get('total_epochs', config['epochs'])
                loss = info.get('loss', 0.0)
                
                # Mise à jour interface depuis le thread principal
                from qgis.PyQt.QtCore import QTimer
                QTimer.singleShot(0, lambda: self._update_training_progress(epoch, total_epochs, loss))
            
            # Lancement de l'entraînement
            results = self.yolo_engine.train_custom_model(
                dataset_config_path=config['dataset_path'],
                base_model=config['base_model'],
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                progress_callback=on_training_progress
            )
            
            # Gestion du succès
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._on_training_completed(results, config))
            
        except Exception as e:
            # Gestion de l'erreur
            error_message = str(e)  # Capturer le message d'erreur dans une variable locale
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._on_training_error(error_message))
    
    def _update_training_progress(self, epoch: int, total_epochs: int, loss: float):
        """Met à jour la progression de l'entraînement"""
        if self.training_progress.isVisible():
            self.training_progress.setValue(epoch)
            
            # Mise à jour des logs de manière limitée pour éviter le spam
            if epoch % 5 == 0 or epoch == total_epochs:  # Log toutes les 5 époques
                self.training_logs.append(f"📈 Époque {epoch}/{total_epochs} - Loss: {loss:.4f}")
                
                # Auto-scroll vers le bas
                cursor = self.training_logs.textCursor()
                cursor.movePosition(cursor.End)
                self.training_logs.setTextCursor(cursor)
        
        # Mise à jour du statut
        progress_percent = (epoch / total_epochs) * 100 if total_epochs > 0 else 0
        self.status_label.setText(f"🧠 Entraînement en cours... {progress_percent:.1f}% - Époque {epoch}/{total_epochs}")
    
    def _on_training_completed(self, results: dict, config: dict):
        """Gestion du succès de l'entraînement"""
        try:
            best_model_path = results.get('best_model_path', '')
            final_metrics = results.get('final_metrics', {})
            training_time = results.get('training_time', 0)
            
            # Messages de succès
            self.training_logs.append("🎉 Entraînement terminé avec succès !")
            self.training_logs.append(f"💾 Modèle sauvegardé : {best_model_path}")
            
            if training_time:
                self.training_logs.append(f"⏱️ Temps d'entraînement : {training_time:.1f}s")
            
            # Stockage du modèle entraîné avec lien direct aux classes
            classes_detected = list(config.get('class_names', {}).values())
            if not classes_detected:
                # CORRECTION: Si pas de classes détectées, utiliser le nom du dataset
                dataset_name = config.get('dataset_name', 'Unknown')
                if 'dataset_' in dataset_name:
                    class_info = dataset_name.replace('dataset_', '').split('_')[0]
                else:
                    class_info = dataset_name
            elif len(classes_detected) == 1:
                class_info = classes_detected[0]
            elif len(classes_detected) <= 3:
                class_info = "_".join(classes_detected)
            else:
                class_info = f"{len(classes_detected)}Classes"
            
            model_name = f"{class_info}_Model_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # CORRECTION: Copier le modèle vers un nom unique pour éviter l'écrasement
            import shutil
            unique_model_filename = f"{model_name}.pt"
            models_dir = self.annotation_manager.project_dir / "trained_models"
            models_dir.mkdir(parents=True, exist_ok=True)
            unique_model_path = models_dir / unique_model_filename
            
            try:
                # Copier le modèle temporaire vers l'emplacement permanent
                shutil.copy2(best_model_path, str(unique_model_path))
                print(f"💾 Modèle copié vers: {unique_model_path}")
                final_model_path = str(unique_model_path)
            except Exception as copy_error:
                print(f"⚠️ Erreur copie modèle: {copy_error}, utilisation chemin original")
                final_model_path = best_model_path
            
            model_info = {
                'path': final_model_path,
                'dataset_name': config['dataset_name'],
                'classes': classes_detected,
                'created_at': datetime.now(),
                'metrics': final_metrics,
                'config': config
            }
            
            # NOUVEAUTÉ: Sauvegarde en base de données
            dataset_id = config.get('dataset_name', f'dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            
            # Si pas de classes détectées, utiliser classe inférée du dataset
            if not classes_detected:
                classes_detected = [class_info]
            
            # Sauvegarde du dataset si pas déjà fait
            dataset_path = str(config.get('dataset_path', ''))
            if dataset_path:
                self.annotation_manager.save_dataset(
                    dataset_id=dataset_id,
                    name=config['dataset_name'],
                    path=dataset_path, 
                    class_names=classes_detected,
                    image_count=39,  # TODO: récupérer le vrai nombre
                    config=config
                )
            
            # Sauvegarde du modèle entraîné en DB
            model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            success = self.annotation_manager.save_trained_model(
                model_id=model_id,
                name=model_name,
                path=final_model_path,
                dataset_id=dataset_id,
                class_names=classes_detected,
                metrics=final_metrics,
                config=config
            )
            
            if success:
                print(f"💾 Modèle et dataset sauvegardés en DB")
            else:
                print(f"⚠️ Erreur sauvegarde en DB - continuité en mémoire assurée")
            
            # Mise à jour de l'interface
            self._reset_training_interface()
            self.train_btn.setText("✅ Entraînement Terminé")
            
            # FORCER la mise à jour de la liste des modèles pour la détection
            print("🔄 Force mise à jour des listes de détection après entraînement")
            self._update_detection_models_list()
            
            # CORRECTION: Mise à jour de l'état de l'interface pour activer le bouton détection
            self.update_interface_state()
            
            # CORRECTION SUPPLÉMENTAIRE: Force l'activation du bouton détection
            trained_models = self.annotation_manager.get_trained_models()
            if trained_models:
                print("🔍 Force activation du bouton détection")
                self.detect_btn.setEnabled(True)
                self.detect_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2E8B57;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        padding: 8px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #228B22;
                    }
                """)
            
            # Message de succès avec options
            reply = QMessageBox.question(
                self,
                "Entraînement Terminé !",
                f"🎉 Modèle YOLO entraîné avec succès !\n\n"
                f"📁 Modèle : {model_name}\n"
                f"💾 Sauvegardé : {final_model_path}\n\n"
                f"🔍 Passer à l'onglet Détection pour tester le modèle ?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Basculer vers l'onglet détection
                self.tab_widget.setCurrentIndex(3)  # Index 3 = onglet détection
                # Sélectionner le modèle qui vient d'être entraîné
                self.detection_model_combo.setCurrentText(model_name)
            
            self.status_label.setText(f"✅ Modèle {model_name} entraîné - Prêt pour détection")
            
        except Exception as e:
            print(f"Erreur traitement succès entraînement: {e}")
            self._on_training_error(f"Erreur post-entraînement: {str(e)}")
    
    def _on_training_error(self, error_message: str):
        """Gestion des erreurs d'entraînement"""
        self.training_logs.append(f"❌ Erreur entraînement : {error_message}")
        
        # Réinitialisation interface
        self._reset_training_interface()
        
        # Message d'erreur
        QMessageBox.critical(
            self,
            "Erreur d'Entraînement",
            f"L'entraînement du modèle YOLO a échoué :\n\n{error_message}\n\n"
            f"Vérifiez les logs pour plus de détails."
        )
        
        self.status_label.setText("❌ Erreur entraînement - Vérifiez les paramètres")
    
    def _reset_training_interface(self):
        """Remet l'interface d'entraînement à l'état initial"""
        self.training_in_progress = False
        self.train_btn.setEnabled(True)
        self.training_progress.setVisible(False)
        
        # Le texte du bouton sera mis à jour par le contexte (génération ou nouvelle classe)
    
    def get_model_for_class(self, class_name):
        """Récupère le modèle entraîné pour une classe spécifique"""
        models = self.annotation_manager.get_models_for_class(class_name)
        return models[0] if models else None
    
    def get_available_classes_with_models(self):
        """Retourne la liste des classes qui ont un modèle entraîné"""
        models = self.annotation_manager.get_trained_models()
        classes = set()
        for model in models:
            classes.update(model.get('class_names', []))
        return list(classes)
    
    def can_detect_for_class(self, class_name):
        """Vérifie si on peut faire de la détection pour une classe"""
        models = self.annotation_manager.get_models_for_class(class_name)
        return len(models) > 0
    
    def _update_detection_models_list(self):
        """Met à jour la liste des modèles et classes dans l'onglet détection"""
        print("🔄 === MISE À JOUR LISTES DÉTECTION ===")
        
        # Sauvegarder les sélections actuelles
        current_class_selection = self.detection_class_combo.currentText()
        current_model_selection = self.detection_model_combo.currentText()
        
        # NOUVEAUTÉ: Données depuis la base
        trained_models = self.annotation_manager.get_trained_models()
        available_classes = self.get_available_classes_with_models()
        
        print(f"📊 Modèles en DB: {len(trained_models)}")
        print(f"🎯 Classes avec modèles: {len(available_classes)}")
        
        for model in trained_models:
            print(f"  • {model['name']} → {model['class_names']} (path: {model['path']})")
        
        # MISE À JOUR DES CLASSES DISPONIBLES
        self.detection_class_combo.clear()
        if not available_classes:
            self.detection_class_combo.addItem("--- Aucune classe entraînée ---")
        else:
            self.detection_class_combo.addItem("--- Sélectionner une classe ---")
            
            # Ajouter les classes avec modèles disponibles
            for class_name in sorted(available_classes):
                model = self.get_model_for_class(class_name)
                if model:
                    created_date = model['created_at']
                    if isinstance(created_date, str):
                        # Parse ISO format datetime
                        from datetime import datetime
                        try:
                            dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                            created_date = dt.strftime('%d/%m %H:%M')
                        except:
                            created_date = created_date[:16]  # Fallback
                    display_name = f"{class_name} (modèle du {created_date})"
                    self.detection_class_combo.addItem(display_name, class_name)
        
        # MISE À JOUR DES MODÈLES (mode expert)
        self.detection_model_combo.clear()
        if not trained_models:
            self.detection_model_combo.addItem("--- Aucun modèle entraîné ---")
        else:
            self.detection_model_combo.addItem("--- Sélectionner un modèle ---")
            
            # Ajouter les modèles entraînés (plus récents en premier)
            for model in trained_models:
                model_name = model['name']
                created_date = model['created_at']
                if isinstance(created_date, str):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                        created_date = dt.strftime('%d/%m %H:%M')
                    except:
                        created_date = created_date[:16]
                
                classes_info = model.get('class_names', [])
                
                if classes_info:
                    if len(classes_info) == 1:
                        class_description = f"détecte: {classes_info[0]}"
                    elif len(classes_info) <= 3:
                        class_description = f"détecte: {', '.join(classes_info)}"
                    else:
                        class_description = f"détecte {len(classes_info)} classes"
                else:
                    class_description = "modèle générique"
                
                display_name = f"{model_name} ({class_description}, {created_date})"
                self.detection_model_combo.addItem(display_name, model['id'])
        
        # Restaurer les sélections si possible
        if current_class_selection:
            for i in range(self.detection_class_combo.count()):
                if current_class_selection in self.detection_class_combo.itemText(i):
                    self.detection_class_combo.setCurrentIndex(i)
                    break
        
        if current_model_selection:
            index = self.detection_model_combo.findText(current_model_selection)
            if index >= 0:
                self.detection_model_combo.setCurrentIndex(index)
    
    def _run_detection(self, raster_layer, model_path: str, extent, confidence: float, zone_description: str):
        """Lance la détection d'objets sur la zone spécifiée"""
        try:
            # Chargement du modèle YOLO
            if not self.yolo_engine.load_model(model_path):
                QMessageBox.critical(self, "Erreur", f"Impossible de charger le modèle :\n{model_path}")
                return
            
            # Configuration de l'interface pour la détection
            self.detect_btn.setText("🔍 Détection en cours...")
            self.detect_btn.setEnabled(False)
            self.detection_progress.setVisible(True)
            self.detection_progress.setRange(0, 0)  # Mode indéterminé au début
            
            # Lancement de la détection asynchrone
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._process_detection_async(
                raster_layer, extent, confidence, zone_description
            ))
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du lancement de la détection:\n{str(e)}")
            self._reset_detection_interface()
    
    def _process_detection_async(self, raster_layer, extent, confidence: float, zone_description: str):
        """Traite la détection de manière asynchrone par tiles avec optimisation d'échelle"""
        try:
            from ..core.raster_extractor import RasterPatchExtractor
            
            # NOUVEAUTÉ: Récupération métadonnées d'échelle du modèle actuel
            optimal_pixel_size = None
            current_class = None
            
            # Déterminer la classe/modèle utilisé pour récupérer ses métadonnées
            if hasattr(self, 'detection_class_combo') and self.detection_class_combo.currentData():
                current_class = self.detection_class_combo.currentData()
                if current_class and not current_class.startswith("---"):
                    try:
                        stats = self.annotation_manager.get_class_statistics(current_class)
                        optimal_pixel_size = stats.optimal_pixel_size
                        print(f"🎯 CLASSE: {current_class}")
                        print(f"🎯 RÉSOLUTION OPTIMALE: {optimal_pixel_size:.3f}m/px")
                        print(f"🎯 PLAGE ÉCHELLES: 1:{stats.scale_range[0]} - 1:{stats.scale_range[1]}")
                    except Exception as e:
                        print(f"⚠️ Erreur métadonnées classe: {e}")
            
            # Calcul des tiles pour traitement avec adaptation d'échelle
            tiles = self._calculate_detection_tiles(extent, raster_layer, 
                                                  max_tile_size=1024, 
                                                  optimal_pixel_size=optimal_pixel_size)
            total_tiles = len(tiles)
            
            if total_tiles == 0:
                self._on_detection_error("Aucune zone à traiter")
                return
            
            # Mise à jour de la barre de progression
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._update_detection_progress_setup(total_tiles))
            
            # Extracteur de patches
            extractor = RasterPatchExtractor(target_size=(640, 640))
            
            all_detections = []
            processed_tiles = 0
            
            for tile_extent in tiles:
                # Extraction du patch
                patch_data = extractor.extract_patch(tile_extent, raster_layer)
                
                if patch_data:
                    # Détection YOLO sur le patch
                    detections = self.yolo_engine.detect_objects(
                        patch_data['image_array'],
                        confidence_threshold=confidence
                    )
                    
                    # Conversion des coordonnées vers le système de la carte
                    map_detections = self._convert_detections_to_map_coords(
                        detections, patch_data, tile_extent
                    )
                    
                    all_detections.extend(map_detections)
                
                processed_tiles += 1
                
                # Mise à jour de la progression
                QTimer.singleShot(0, lambda pt=processed_tiles, tt=total_tiles: 
                    self._update_detection_progress(pt, tt))
            
            # Finalisation
            QTimer.singleShot(0, lambda: self._on_detection_completed(
                all_detections, zone_description, total_tiles
            ))
            
        except Exception as e:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._on_detection_error(str(e)))
    
    def _calculate_detection_tiles(self, extent, raster_layer, max_tile_size: int = 1024, 
                                  optimal_pixel_size: float = None):
        """
        Calcule les tiles pour la détection avec adaptation d'échelle
        
        Args:
            extent: Zone à traiter
            raster_layer: Couche raster
            max_tile_size: Taille max tile en pixels
            optimal_pixel_size: Résolution optimale du modèle (None = auto)
        """
        tiles = []
        
        # Taille des pixels actuels
        layer_extent = raster_layer.extent()
        layer_width = raster_layer.width()
        layer_height = raster_layer.height()
        
        current_pixel_size_x = layer_extent.width() / layer_width
        current_pixel_size_y = layer_extent.height() / layer_height
        current_pixel_size = (current_pixel_size_x + current_pixel_size_y) / 2
        
        # NOUVEAUTÉ: Adaptation taille tile selon résolution optimale
        adaptive_tile_size = max_tile_size
        if optimal_pixel_size and optimal_pixel_size > 0:
            scale_ratio = current_pixel_size / optimal_pixel_size
            
            # Si résolution actuelle plus fine → tiles plus grandes
            # Si résolution actuelle plus grossière → tiles plus petites
            adaptive_tile_size = int(max_tile_size * scale_ratio)
            adaptive_tile_size = max(320, min(2048, adaptive_tile_size))  # Limites
            
            print(f"🔍 ÉCHELLE: Résolution actuelle: {current_pixel_size:.3f}m/px")
            print(f"🔍 ÉCHELLE: Résolution optimale: {optimal_pixel_size:.3f}m/px") 
            print(f"🔍 ÉCHELLE: Ratio: {scale_ratio:.2f}x")
            print(f"🔍 ÉCHELLE: Taille tile adaptée: {adaptive_tile_size}px (base: {max_tile_size}px)")
        
        pixel_size_x = current_pixel_size_x
        pixel_size_y = current_pixel_size_y
        
        # Taille de la zone en pixels
        zone_width_px = extent.width() / pixel_size_x
        zone_height_px = extent.height() / pixel_size_y
        
        # Si la zone est petite, traiter en une seule tile
        if zone_width_px <= adaptive_tile_size and zone_height_px <= adaptive_tile_size:
            tiles.append(extent)
            return tiles
        
        # Sinon, découper en tiles adaptatives
        tile_width_map = pixel_size_x * adaptive_tile_size
        tile_height_map = pixel_size_y * adaptive_tile_size
        
        x_start = extent.xMinimum()
        y_start = extent.yMinimum()
        x_end = extent.xMaximum()
        y_end = extent.yMaximum()
        
        y = y_start
        while y < y_end:
            x = x_start
            while x < x_end:
                tile_x_max = min(x + tile_width_map, x_end)
                tile_y_max = min(y + tile_height_map, y_end)
                
                from qgis.core import QgsRectangle
                tile_extent = QgsRectangle(x, y, tile_x_max, tile_y_max)
                tiles.append(tile_extent)
                
                x = tile_x_max
            y = tile_y_max
        
        return tiles
    
    def _convert_detections_to_map_coords(self, detections: list, patch_data: dict, tile_extent) -> list:
        """Convertit les détections YOLO en coordonnées carte"""
        map_detections = []
        
        extracted_bbox = patch_data['extracted_bbox']
        
        print(f"🔍 DEBUG COORD: Conversion {len(detections)} détections")
        print(f"🔍 DEBUG COORD: Extracted bbox: {extracted_bbox}")
        print(f"🔍 DEBUG COORD: Tile extent: {tile_extent}")
        
        for i, detection in enumerate(detections):
            # Coordonnées YOLO normalisées [center_x, center_y, width, height]
            yolo_bbox = detection['bbox_normalized']
            print(f"🔍 DEBUG COORD: Detection {i+1} YOLO bbox: {yolo_bbox}")
            
            # Conversion vers coordonnées carte
            img_width = extracted_bbox['xmax'] - extracted_bbox['xmin']
            img_height = extracted_bbox['ymax'] - extracted_bbox['ymin']
            
            print(f"🔍 DEBUG COORD: Image dimensions: {img_width} x {img_height}")
            
            # Centre de l'objet en coordonnées carte
            center_x_map = extracted_bbox['xmin'] + (yolo_bbox[0] * img_width)
            center_y_map = extracted_bbox['ymin'] + (yolo_bbox[1] * img_height)
            
            # Dimensions de l'objet en coordonnées carte  
            width_map = yolo_bbox[2] * img_width
            height_map = yolo_bbox[3] * img_height
            
            print(f"🔍 DEBUG COORD: Centre carte: ({center_x_map}, {center_y_map})")
            print(f"🔍 DEBUG COORD: Dimensions carte: {width_map} x {height_map}")
            
            # Bounding box finale
            bbox_map = {
                'xmin': center_x_map - width_map/2,
                'ymin': center_y_map - height_map/2,
                'xmax': center_x_map + width_map/2,
                'ymax': center_y_map + height_map/2
            }
            
            print(f"🔍 DEBUG COORD: BBox finale: {bbox_map}")
            
            map_detection = {
                'bbox_map': bbox_map,
                'confidence': detection['confidence'],
                'class_id': detection['class_id'],
                'class_name': detection['class_name'],
                'center_x': center_x_map,
                'center_y': center_y_map
            }
            
            map_detections.append(map_detection)
        
        print(f"🔍 DEBUG COORD: {len(map_detections)} détections converties")
        return map_detections
    
    def _update_detection_progress_setup(self, total_tiles: int):
        """Configure la barre de progression pour la détection"""
        self.detection_progress.setRange(0, total_tiles)
        self.detection_progress.setValue(0)
        self.status_label.setText(f"🔍 Détection sur {total_tiles} zones...")
    
    def _update_detection_progress(self, processed: int, total: int):
        """Met à jour la progression de la détection"""
        self.detection_progress.setValue(processed)
        progress_percent = (processed / total) * 100 if total > 0 else 0
        self.status_label.setText(f"🔍 Détection... {progress_percent:.1f}% ({processed}/{total} zones)")
    
    def _on_detection_completed(self, detections: list, zone_description: str, total_tiles: int):
        """Gestion de la fin de détection"""
        try:
            num_detections = len(detections)
            
            print("=" * 80)
            print("🚨🚨🚨 _ON_DETECTION_COMPLETED v1.3.4 - CLAUDE CODE 🚨🚨🚨")
            print("=" * 80)
            print(f"🔍 DEBUG FINAL: Détection terminée avec {num_detections} détections")
            print(f"🔍 DEBUG FINAL: Zone: {zone_description}, Tiles: {total_tiles}")
            
            # Réinitialisation de l'interface
            self._reset_detection_interface()
            
            # Création de la couche de résultats
            if num_detections > 0:
                print(f"🔍 DEBUG FINAL: Création de la couche de résultats...")
                result_layer = self._create_detection_results_layer(detections)
                
                if result_layer:
                    from qgis.core import QgsProject
                    project = QgsProject.instance()
                    
                    print(f"🔍 DEBUG FINAL: Ajout de la couche au projet...")
                    print(f"🔍 DEBUG FINAL: Couche valide: {result_layer.isValid()}")
                    print(f"🔍 DEBUG FINAL: Nombre de features: {result_layer.featureCount()}")
                    print(f"🔍 DEBUG FINAL: Extent de la couche: {result_layer.extent()}")
                    
                    project.addMapLayer(result_layer)
                    
                    print(f"🔍 DEBUG FINAL: Couche ajoutée au projet")
                    print(f"🔍 DEBUG FINAL: Couches dans le projet: {len(project.mapLayers())}")
                    
                    # Forcer le rafraîchissement du canvas
                    from qgis.utils import iface
                    if iface:
                        iface.mapCanvas().refresh()
                        print(f"🔍 DEBUG FINAL: Canvas rafraîchi")
                    
                    # Zoom sur les résultats
                    self.iface.mapCanvas().setExtent(result_layer.extent())
                    self.iface.mapCanvas().refresh()
            
            # Message de résultats
            if num_detections > 0:
                message = (f"🎉 Détection terminée avec succès !\n\n"
                          f"• {num_detections} objets détectés\n"
                          f"• Zone : {zone_description}\n"  
                          f"• {total_tiles} zones traitées\n\n"
                          f"Les résultats ont été ajoutés comme nouvelle couche vectorielle.")
                
                self.results_label.setText(f"✅ {num_detections} objets détectés")
                self.status_label.setText(f"✅ Détection terminée - {num_detections} objets trouvés")
            else:
                message = (f"ℹ️ Détection terminée.\n\n"
                          f"• Aucun objet détecté\n"
                          f"• Zone : {zone_description}\n"
                          f"• {total_tiles} zones traitées\n\n"
                          f"Essayez de réduire le seuil de confiance ou vérifiez que le modèle "
                          f"correspond au type d'objets présents dans l'image.")
                
                self.results_label.setText("ℹ️ Aucun objet détecté")
                self.status_label.setText("ℹ️ Détection terminée - Aucun objet trouvé")
            
            QMessageBox.information(self, "Détection Terminée", message)
            
        except Exception as e:
            self._on_detection_error(f"Erreur post-détection: {str(e)}")
    
    def _on_detection_error(self, error_message: str):
        """Gestion des erreurs de détection"""
        self._reset_detection_interface()
        
        QMessageBox.critical(
            self,
            "Erreur de Détection", 
            f"La détection d'objets a échoué :\n\n{error_message}"
        )
        
        self.results_label.setText("❌ Erreur lors de la détection")
        self.status_label.setText("❌ Erreur détection - Vérifiez les paramètres")
    
    def _reset_detection_interface(self):
        """Remet l'interface de détection à l'état initial"""
        self.detect_btn.setText("🔍 Détecter les Objets")
        self.detect_btn.setEnabled(True)
        self.detection_progress.setVisible(False)
    
    def _create_detection_results_layer(self, detections: list):
        """Crée une couche vectorielle avec les résultats de détection (bbox polygonales)"""
        try:
            from qgis.core import (QgsVectorLayer, QgsField, QgsFeature, QgsGeometry, 
                                   QgsPointXY, QgsRectangle, QgsSymbol, QgsSingleSymbolRenderer,
                                   QgsFillSymbol)
            from qgis.PyQt.QtCore import QVariant
            from qgis.PyQt.QtGui import QColor
            from datetime import datetime
            
            print("=" * 80)
            print("🚨🚨🚨 VERSION DEBUG v1.3.4 ACTIVÉE - CLAUDE CODE 🚨🚨🚨")
            print("=" * 80)
            print(f"🔍 DEBUG: Création couche pour {len(detections)} détections")
            
            # Création de la couche vectorielle (POLYGONES) avec le CRS du projet
            from qgis.core import QgsProject
            project_crs = QgsProject.instance().crs()
            print(f"🔍 DEBUG: CRS du projet: {project_crs.authid()}")
            
            layer = QgsVectorLayer(f"Polygon?crs={project_crs.authid()}", "YOLO_Detections", "memory")
            
            if not layer.isValid():
                print("❌ DEBUG: Erreur création couche de résultats")
                return None
            
            print(f"✅ DEBUG: Couche créée avec succès")
            
            # Ajout des champs avec métadonnées complètes
            provider = layer.dataProvider()
            provider.addAttributes([
                QgsField("class_name", QVariant.String),
                QgsField("confidence", QVariant.Double),
                QgsField("confidence_pct", QVariant.String),  # Pourcentage formaté
                QgsField("center_x", QVariant.Double),
                QgsField("center_y", QVariant.Double),
                QgsField("bbox_xmin", QVariant.Double),
                QgsField("bbox_ymin", QVariant.Double), 
                QgsField("bbox_xmax", QVariant.Double),
                QgsField("bbox_ymax", QVariant.Double),
                QgsField("bbox_width", QVariant.Double),
                QgsField("bbox_height", QVariant.Double),
                QgsField("detection_time", QVariant.String),
                QgsField("model_used", QVariant.String)
            ])
            layer.updateFields()
            
            print(f"🔍 DEBUG: Champs ajoutés: {[field.name() for field in layer.fields()]}")
            
            # Récupération du modèle utilisé pour les métadonnées
            current_model_name = "Unknown"
            if hasattr(self, 'detection_class_combo') and self.detection_class_combo.currentData():
                model = self.get_model_for_class(self.detection_class_combo.currentData())
                if model:
                    current_model_name = model.get('name', 'Unknown')
            
            # Validation des détections
            if not detections:
                print("❌ DEBUG: Aucune détection à traiter")
                return None
                
            print(f"🔍 DEBUG: Exemple de détection: {detections[0]}")
            
            # Ajout des features avec géométries bbox
            features = []
            detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"🔍 DEBUG: Début création de {len(detections)} features")
            
            for i, detection in enumerate(detections):
                # Validation de la détection
                if 'bbox_map' not in detection:
                    print(f"❌ DEBUG: Feature {i+1} sans bbox_map")
                    continue
                    
                feature = QgsFeature(layer.fields())
                
                # Géométrie : rectangle (bbox) au lieu de point
                bbox = detection['bbox_map']
                print(f"🔍 DEBUG: Bbox: {bbox}")
                
                # Vérification de la validité des coordonnées
                if (bbox['xmin'] >= bbox['xmax'] or bbox['ymin'] >= bbox['ymax']):
                    print(f"❌ DEBUG: Bbox invalide pour feature {i+1}: xmin={bbox['xmin']}, xmax={bbox['xmax']}, ymin={bbox['ymin']}, ymax={bbox['ymax']}")
                    continue
                
                # Conversion des types NumPy vers types Python natifs
                bbox_native = {}
                for key, value in bbox.items():
                    # Conversion explicite des types NumPy vers float Python
                    if hasattr(value, 'item'):  # NumPy types have .item() method
                        bbox_native[key] = float(value.item())
                    else:
                        bbox_native[key] = float(value)
                    
                    # Vérification des valeurs NaN ou infinies
                    if str(bbox_native[key]).lower() in ['nan', 'inf', '-inf']:
                        print(f"❌ DEBUG: Valeur invalide dans bbox: {key}={bbox_native[key]}")
                        continue
                
                bbox = bbox_native  # Utiliser les valeurs converties
                print(f"🔍 DEBUG: Bbox convertie: {bbox}")
                
                # NOUVEAU: Utilisation des polygones SAM si disponibles
                if detection.get('polygon_points') and detection.get('polygon_available'):
                    print(f"🔺 DEBUG: Utilisation polygone SAM avec {len(detection['polygon_points'])} vertices")
                    
                    # Créer un polygone précis à partir des points SAM
                    polygon_points = detection['polygon_points']
                    qgs_points = []
                    
                    for point in polygon_points:
                        # Les points sont en coordonnées map (déjà transformés)
                        qgs_points.append(QgsPointXY(float(point[0]), float(point[1])))
                    
                    # Fermer le polygone si nécessaire
                    if qgs_points and (qgs_points[0].x() != qgs_points[-1].x() or qgs_points[0].y() != qgs_points[-1].y()):
                        qgs_points.append(qgs_points[0])
                    
                    geometry = QgsGeometry.fromPolygonXY([qgs_points])
                    print(f"✅ DEBUG: Géométrie polygone SAM créée avec {len(qgs_points)} points")
                    
                else:
                    print(f"📦 DEBUG: Utilisation bbox rectangulaire (pas de polygone SAM)")
                    
                    # Fallback bbox rectangulaire
                    rectangle = QgsRectangle(
                        bbox['xmin'], bbox['ymin'],
                        bbox['xmax'], bbox['ymax']
                    )
                    
                    print(f"🔍 DEBUG: QgsRectangle créé: {rectangle.toString()}")
                    print(f"🔍 DEBUG: Rectangle vide: {rectangle.isEmpty()}")
                    print(f"🔍 DEBUG: Rectangle valide: {not rectangle.isNull()}")
                    
                    geometry = QgsGeometry.fromRect(rectangle)
                
                print(f"🔍 DEBUG: Géométrie valide: {not geometry.isNull()}")
                print(f"🔍 DEBUG: Type géométrie: {geometry.type()}")
                print(f"🔍 DEBUG: Géométrie vide: {geometry.isEmpty()}")
                
                if geometry.isNull() or geometry.isEmpty():
                    print(f"❌ DEBUG: Géométrie nulle ou vide pour feature {i+1}")
                    continue
                    
                feature.setGeometry(geometry)
                print(f"✅ DEBUG: Géométrie définie pour feature {i+1}")
                
                # Calcul des dimensions
                width = bbox['xmax'] - bbox['xmin']
                height = bbox['ymax'] - bbox['ymin']
                
                # Conversion de toutes les valeurs NumPy vers types Python natifs
                confidence = float(detection['confidence'].item()) if hasattr(detection['confidence'], 'item') else float(detection['confidence'])
                center_x = float(detection['center_x'].item()) if hasattr(detection['center_x'], 'item') else float(detection['center_x'])
                center_y = float(detection['center_y'].item()) if hasattr(detection['center_y'], 'item') else float(detection['center_y'])
                
                # Attributs complets avec métadonnées - TOUS TYPES PYTHON NATIFS
                attributes = [
                    str(detection['class_name']),                   # Classe détectée (string)
                    round(confidence, 4),                           # Confiance (float)
                    f"{confidence*100:.1f}%",                      # Pourcentage formaté (string)
                    round(center_x, 2),                            # Centre X (float)
                    round(center_y, 2),                            # Centre Y (float)
                    round(bbox['xmin'], 2),                        # Bbox limites (float)
                    round(bbox['ymin'], 2),                        # (float)
                    round(bbox['xmax'], 2),                        # (float)
                    round(bbox['ymax'], 2),                        # (float)
                    round(width, 2),                               # Dimensions (float)
                    round(height, 2),                              # (float)
                    str(detection_time),                           # Timestamp (string)
                    str(current_model_name)                        # Modèle utilisé (string)
                ]
                
                print(f"🔍 DEBUG: Attributs types: {[type(attr).__name__ for attr in attributes]}")
                
                feature.setAttributes(attributes)
                print(f"✅ DEBUG: Attributs définis pour feature {i+1}: {attributes[:3]}...")
                
                features.append(feature)
            
            print(f"🔍 DEBUG: {len(features)} features créées")
            
            # Ajout des features au provider
            if features:
                print(f"🔍 DEBUG: Tentative d'ajout de {len(features)} features")
                
                # Vérification détaillée avant ajout
                valid_features = []
                for i, feature in enumerate(features):
                    print(f"🔍 DEBUG: Feature {i+1} - Valide: {feature.isValid()}")
                    print(f"🔍 DEBUG: Feature {i+1} - A géométrie: {feature.hasGeometry()}")
                    if feature.hasGeometry():
                        geom = feature.geometry()
                        print(f"🔍 DEBUG: Feature {i+1} - Géométrie valide: {not geom.isNull()}")
                        print(f"🔍 DEBUG: Feature {i+1} - Géométrie vide: {geom.isEmpty()}")
                        print(f"🔍 DEBUG: Feature {i+1} - Aire: {geom.area()}")
                        if not geom.isNull() and not geom.isEmpty():
                            valid_features.append(feature)
                        else:
                            print(f"❌ DEBUG: Feature {i+1} rejetée - géométrie invalide")
                    else:
                        print(f"❌ DEBUG: Feature {i+1} rejetée - pas de géométrie")
                
                print(f"🔍 DEBUG: {len(valid_features)} features valides sur {len(features)}")
                
                if valid_features:
                    result = provider.addFeatures(valid_features)
                    print(f"🔍 DEBUG: Ajout features au provider: {result}")
                    print(f"🔍 DEBUG: Nombre features dans la couche après ajout: {layer.featureCount()}")
                    
                    # Vérification finale des features dans la couche
                    feature_iter = layer.getFeatures()
                    count = 0
                    for feature in feature_iter:
                        count += 1
                        print(f"🔍 DEBUG: Feature {count} dans couche - ID: {feature.id()}, Attributs: {len(feature.attributes())}")
                    print(f"🔍 DEBUG: Total itéré: {count} features")
                else:
                    print("❌ DEBUG: Aucune feature valide à ajouter")
            else:
                print("❌ DEBUG: Aucune feature à ajouter")
                
            layer.updateExtents()
            print(f"🔍 DEBUG: Extent de la couche: {layer.extent()}")
            
            # Style visuel optimisé pour les bbox
            fill_symbol = QgsFillSymbol.createSimple({
                'color': '255,0,0,60',           # Rouge semi-transparent
                'outline_color': '255,0,0,255',  # Contour rouge opaque
                'outline_width': '2',            # Contour épais
                'outline_style': 'solid'         # Ligne continue
            })
            
            renderer = QgsSingleSymbolRenderer(fill_symbol)
            layer.setRenderer(renderer)
            
            # Nom descriptif avec timestamp et nombre de détections
            timestamp = datetime.now().strftime("%H:%M:%S")
            layer.setName(f"YOLO_Detections_{len(detections)}obj_{timestamp}")
            
            print(f"✅ Couche créée: {len(detections)} détections avec bbox polygonales")
            
            return layer
            
        except Exception as e:
            print(f"Erreur création couche résultats: {e}")
            return None
    
    def refresh_interface_display(self):
        """Force la mise à jour complète de l'affichage de l'interface"""
        try:
            print("🔄 Rafraîchissement complet de l'interface...")
            
            # Mise à jour des arbres et combos
            if self.annotation_manager:
                self.update_classes_tree_with_real_stats() 
            else:
                self.update_classes_tree()
            
            self.update_class_combos()
            self.update_interface_state()
            
            # Informations de debug
            print(f"📊 Classes en mémoire: {list(self.object_classes.keys())}")
            print(f"📋 Items dans l'arbre: {self.classes_tree.topLevelItemCount()}")
            print(f"🎯 Classes dans combo annotation: {[self.active_class_combo.itemText(i) for i in range(self.active_class_combo.count())]}")
            
            print("✅ Rafraîchissement terminé")
            
        except Exception as e:
            print(f"❌ Erreur rafraîchissement interface: {e}")
            import traceback
            traceback.print_exc()
    
    def on_active_class_changed(self, class_name):
        """Gestion du changement de classe active dans le combo"""
        print(f"🔄 Signal changement classe: '{class_name}'")
        
        if class_name and not class_name.startswith("---"):
            # Vérifier que la classe existe bien
            if class_name in self.object_classes:
                self.current_class = class_name
                print(f"🎯 Classe active changée: {class_name}")
                
                # Mise à jour de l'interface
                self.update_annotation_progress()
                self.update_examples_list()
                self.update_interface_state()
                
                # Synchronisation avec l'arbre
                items = self.classes_tree.findItems(class_name, Qt.MatchExactly, 0)
                if items:
                    # Bloquer temporairement les signaux pour éviter la récursion
                    self.classes_tree.blockSignals(True)
                    self.classes_tree.setCurrentItem(items[0])
                    self.classes_tree.blockSignals(False)
                
                # Message de confirmation
                self.status_label.setText(f"🎯 Classe '{class_name}' sélectionnée pour annotation")
            else:
                print(f"⚠️ Classe '{class_name}' non trouvée dans object_classes")
                self.current_class = None
        else:
            self.current_class = None
            print("⚪ Aucune classe active sélectionnée")
            self.status_label.setText("⚪ Sélectionnez une classe pour commencer l'annotation")
        
        # CORRECTION: Mise à jour immédiate de l'état des boutons
        self.update_interface_state()
        print(f"⚙️ Bouton annotation activé: {self.start_annotation_btn.isEnabled()}")
    
    def edit_selected_class(self):
        """Modifie la classe sélectionnée"""
        current_item = self.classes_tree.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner une classe à modifier.")
            return
        
        class_name = current_item.text(0)
        if class_name not in self.object_classes:
            QMessageBox.warning(self, "Erreur", f"Classe '{class_name}' non trouvée.")
            return
        
        # Dialog de modification
        from qgis.PyQt.QtWidgets import QInputDialog
        new_description, ok = QInputDialog.getText(
            self, "Modifier Classe", 
            f"Description pour '{class_name}':",
            text=self.object_classes[class_name].get('description', '')
        )
        
        if ok:
            self.object_classes[class_name]['description'] = new_description
            if self.annotation_manager:
                self.annotation_manager.create_class(class_name, new_description)
            
            self.update_classes_tree_with_real_stats()
            self.status_label.setText(f"✅ Classe '{class_name}' modifiée")
    
    def duplicate_selected_class(self):
        """Duplique la classe sélectionnée"""
        current_item = self.classes_tree.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner une classe à dupliquer.")
            return
        
        class_name = current_item.text(0)
        if class_name not in self.object_classes:
            QMessageBox.warning(self, "Erreur", f"Classe '{class_name}' non trouvée.")
            return
        
        # Dialog pour nouveau nom
        from qgis.PyQt.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, "Dupliquer Classe", 
            f"Nouveau nom pour la copie de '{class_name}':",
            text=f"{class_name}_copie"
        )
        
        if ok and new_name:
            if new_name in self.object_classes:
                QMessageBox.warning(self, "Erreur", f"La classe '{new_name}' existe déjà.")
                return
            
            # Duplication des données
            original_data = self.object_classes[class_name].copy()
            original_data['name'] = new_name
            original_data['description'] = f"Copie de {class_name}: " + original_data.get('description', '')
            original_data['examples'] = []  # Nouvelle classe sans exemples
            original_data['created_at'] = datetime.now()
            
            self.object_classes[new_name] = original_data
            
            if self.annotation_manager:
                self.annotation_manager.create_class(new_name, original_data['description'], original_data['color'])
            
            self.update_classes_tree_with_real_stats()
            self.update_class_combos()
            self.status_label.setText(f"✅ Classe '{new_name}' créée par duplication")
    
    def delete_selected_class(self):
        """Supprime la classe sélectionnée"""
        current_item = self.classes_tree.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner une classe à supprimer.")
            return
        
        class_name = current_item.text(0)
        if class_name not in self.object_classes:
            QMessageBox.warning(self, "Erreur", f"Classe '{class_name}' non trouvée.")
            return
        
        # Confirmation avec nombre d'exemples
        example_count = 0
        if self.annotation_manager:
            try:
                stats = self.annotation_manager.get_class_statistics(class_name)
                example_count = stats.example_count
            except:
                pass
        
        reply = QMessageBox.question(
            self, "Confirmer Suppression",
            f"Supprimer la classe '{class_name}' ?\n\n"
            f"Cette action supprimera aussi:\n"
            f"- {example_count} exemples d'annotation\n"
            f"- Toutes les données associées\n\n"
            f"Cette action est irréversible.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Suppression de l'interface
            del self.object_classes[class_name]
            
            # TODO: Suppression de la base de données (nécessite implémentation dans AnnotationManager)
            # if self.annotation_manager:
            #     self.annotation_manager.delete_class(class_name)
            
            # Mise à jour interface
            self.update_classes_tree_with_real_stats()
            self.update_class_combos()
            
            # Réinitialiser la classe active si nécessaire
            if self.current_class == class_name:
                self.current_class = None
                self.update_interface_state()
            
            self.status_label.setText(f"✅ Classe '{class_name}' supprimée")
    
    def export_selected_class(self):
        """Exporte la classe sélectionnée"""
        current_item = self.classes_tree.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner une classe à exporter.")
            return
        
        class_name = current_item.text(0)
        if class_name not in self.object_classes:
            QMessageBox.warning(self, "Erreur", f"Classe '{class_name}' non trouvée.")
            return
        
        # Dialog de sélection du fichier
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Exporter Classe '{class_name}'",
            f"{class_name}_export.json",
            "Fichiers JSON (*.json);;Tous les fichiers (*)"
        )
        
        if file_path:
            try:
                # Préparation des données d'export
                export_data = {
                    'class_info': self.object_classes[class_name],
                    'export_date': datetime.now().isoformat(),
                    'plugin_version': '1.1.11'
                }
                
                # Ajout des statistiques si disponibles
                if self.annotation_manager:
                    try:
                        stats = self.annotation_manager.get_class_statistics(class_name)
                        annotations = self.annotation_manager.get_class_annotations(class_name)
                        export_data['statistics'] = {
                            'example_count': stats.example_count,
                            'quality_score': stats.quality_score,
                            'ready_for_training': stats.ready_for_training
                        }
                        export_data['annotations_count'] = len(annotations)
                    except Exception as e:
                        export_data['export_note'] = f"Erreur récupération statistiques: {str(e)}"
                
                # Sauvegarde
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                
                QMessageBox.information(self, "Export Réussi", 
                    f"Classe '{class_name}' exportée vers:\n{file_path}")
                self.status_label.setText(f"✅ Classe '{class_name}' exportée")
                
            except Exception as e:
                QMessageBox.critical(self, "Erreur Export", 
                    f"Impossible d'exporter la classe '{class_name}':\n{str(e)}")
                self.status_label.setText(f"❌ Erreur export '{class_name}'")

    # ============================================================================
    # NOUVEAU: Méthodes de gestion Smart Mode
    # ============================================================================
    
    def on_intelligence_mode_changed(self, checked):
        """Gestion du changement de mode d'intelligence (Manuel/Smart)"""
        sender = self.sender()
        
        if sender == self.manual_mode_btn and checked:
            # Mode manuel activé
            self.smart_mode_btn.setChecked(False)
            self.smart_mode_enabled = False
            self.smart_config_group.setVisible(False)
            self.smart_status_label.setText("✋ Mode manuel actif")
            self.auto_detect_btn.setEnabled(False)  # Désactiver détection auto
            print("🖱️ Mode annotation manuel activé")
            
        elif sender == self.smart_mode_btn and checked:
            # Mode smart activé
            self.manual_mode_btn.setChecked(False)
            self.smart_mode_enabled = True
            self.smart_config_group.setVisible(True)
            self._initialize_smart_engine()
            self.auto_detect_btn.setEnabled(True)  # Activer détection auto
            print("🤖 Mode smart assistant activé")
            print(f"🔍 DEBUG SMART MODE: smart_mode_enabled={self.smart_mode_enabled}")
            print(f"🔍 DEBUG SMART MODE: precise_contours_checkbox={self.precise_contours_checkbox.isChecked()}")
        
        # Mise à jour de l'outil d'annotation si actif
        if self.annotation_tool and hasattr(self.annotation_tool, 'set_smart_mode'):
            print(f"🔍 DEBUG TRANSMISSION: Envoi smart_mode_enabled={self.smart_mode_enabled} vers annotation_tool")
            self.annotation_tool.set_smart_mode(self.smart_mode_enabled)
            print(f"🔍 DEBUG TRANSMISSION: annotation_tool.smart_mode_enabled={getattr(self.annotation_tool, 'smart_mode_enabled', 'MISSING')}")
    
    def _initialize_smart_engine(self):
        """Initialise le SmartAnnotationEngine de manière asynchrone"""
        if self.smart_engine is not None:
            # Déjà initialisé
            self.smart_status_label.setText("✅ Smart Engine prêt")
            return
        
        self.smart_status_label.setText("⏳ Initialisation Smart Engine...")
        
        # Utilisation d'un QTimer pour initialisation non-bloquante
        from qgis.PyQt.QtCore import QTimer
        
        def init_async():
            try:
                # Vérification des prérequis avant initialisation
                if not self.yolo_engine:
                    self.smart_status_label.setText("❌ YOLOEngine non disponible")
                    return
                
                if not self.annotation_manager:
                    self.smart_status_label.setText("❌ AnnotationManager non disponible")
                    return
                
                # Import et initialisation sécurisés
                from ..core.smart_annotation_engine import SmartAnnotationEngine
                self.smart_engine = SmartAnnotationEngine(
                    yolo_engine=self.yolo_engine,
                    annotation_manager=self.annotation_manager
                )
                
                # Configuration selon les paramètres UI
                self.smart_engine.enable_debug_mode(self.debug_mode_checkbox.isChecked())
                
                # Mise à jour du statut
                cpu_profile = self.smart_engine.cpu_profile.level
                self.smart_status_label.setText(f"✅ Smart Engine prêt (CPU: {cpu_profile})")
                
                print(f"🤖 SmartAnnotationEngine initialisé avec profil CPU: {cpu_profile}")
                
            except Exception as e:
                print(f"❌ Erreur initialisation SmartAnnotationEngine: {e}")
                self.smart_status_label.setText("❌ Erreur initialisation")
                
                # Fallback vers mode manuel
                self.manual_mode_btn.setChecked(True)
                self.smart_mode_btn.setChecked(False)
                self.smart_mode_enabled = False
                self.smart_config_group.setVisible(False)
        
        QTimer.singleShot(100, init_async)
    
    def update_yolo_confidence_label(self, value):
        """Met à jour le label de confiance YOLO"""
        self.yolo_confidence_label.setText(f"{value}%")
    
    def on_smart_config_changed(self):
        """Gestion des changements de configuration Smart Mode"""
        if self.smart_engine is None:
            return
        
        try:
            # Mise à jour de la configuration du Smart Engine
            if hasattr(self.smart_engine, 'cpu_profile'):
                # Mise à jour du seuil de confiance YOLO
                confidence_pct = self.yolo_confidence_slider.value()
                self.smart_engine.cpu_profile.confidence_threshold_yolo = confidence_pct / 100.0
                
                # Activation/désactivation SAM
                self.smart_engine.cpu_profile.enable_sam = self.enable_sam_checkbox.isChecked()
                
                # Contours précis (polygones SAM)
                if hasattr(self.smart_engine, 'enable_precise_contours'):
                    checkbox_value = self.precise_contours_checkbox.isChecked()
                    self.smart_engine.enable_precise_contours = checkbox_value
                    print(f"🔍 DEBUG CONTOURS: precise_contours_checkbox={checkbox_value}")
                    print(f"🔍 DEBUG CONTOURS: smart_engine.enable_precise_contours={self.smart_engine.enable_precise_contours}")
                else:
                    print("⚠️ DEBUG CONTOURS: smart_engine n'a pas d'attribut 'enable_precise_contours'")
                
                # Mode debug
                self.smart_engine.enable_debug_mode(self.debug_mode_checkbox.isChecked())
                
                contours_status = "Polygones" if self.precise_contours_checkbox.isChecked() else "BBox"
                print(f"🔧 Smart Engine configuré: YOLO={confidence_pct}%, SAM={self.enable_sam_checkbox.isChecked()}, Contours={contours_status}")
                
        except Exception as e:
            print(f"⚠️ Erreur configuration Smart Engine: {e}")
    
    def get_smart_detection_result(self, user_rect, raster_patch, target_class):
        """
        Interface pour obtenir un résultat de détection intelligente
        
        Args:
            user_rect: Rectangle utilisateur (x1, y1, x2, y2)
            raster_patch: Patch raster extrait
            target_class: Classe cible
            
        Returns:
            SmartDetectionResult ou None si mode manuel
        """
        if not self.smart_mode_enabled or self.smart_engine is None:
            return None
        
        try:
            result = self.smart_engine.smart_detect_from_user_rectangle(
                user_rect, raster_patch, target_class
            )
            
            # Mise à jour des statistiques dans l'interface
            if hasattr(result, 'processing_time'):
                stats = self.smart_engine.get_performance_stats()
                stats_text = (f"📊 Smart Stats: {stats['total_detections']} détections, "
                             f"SAM: {stats['sam_usage_rate']:.1f}%, "
                             f"Auto: {stats['auto_acceptance_rate']:.1f}%, "
                             f"Temps moy: {stats['avg_processing_time_ms']:.1f}ms")
                
                # Affichage temporaire dans le status
                self.status_label.setText(stats_text)
                
                # Retour au status normal après 3s
                from qgis.PyQt.QtCore import QTimer
                QTimer.singleShot(3000, lambda: self.status_label.setText("✅ Annotation intelligente terminée"))
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur détection intelligente: {e}")
            return None
    
    def cleanup_smart_engine(self):
        """Nettoie le Smart Engine"""
        if self.smart_engine is not None:
            try:
                self.smart_engine.cleanup()
                print("✅ Smart Engine nettoyé")
            except Exception as e:
                print(f"⚠️ Erreur nettoyage Smart Engine: {e}")
            finally:
                self.smart_engine = None
    
    def start_smart_auto_detection(self):
        """
        NOUVEAU: Lance la détection automatique Smart Mode
        
        YOLO scanne la zone visible et propose automatiquement des bbox
        """
        if not self.smart_mode_enabled or not self.smart_engine:
            QMessageBox.warning(
                self, 
                "Smart Mode Requis", 
                "La détection automatique nécessite le Smart Mode activé.\n\n"
                "Activez d'abord '🤖 Smart Assistant' ci-dessus."
            )
            return
        
        # Vérification classe active
        if not self.current_class:
            QMessageBox.warning(
                self,
                "Classe Manquante",
                "Sélectionnez d'abord une classe à détecter dans le dropdown 'Classe à annoter'"
            )
            return
        
        # Déléguer à l'annotation tool
        if self.annotation_tool and hasattr(self.annotation_tool, 'start_smart_auto_detection'):
            try:
                # S'assurer que l'outil a la bonne classe active
                self.annotation_tool.set_active_class(self.current_class)
                
                # Lancer la détection automatique
                self.annotation_tool.start_smart_auto_detection()
                
            except Exception as e:
                print(f"❌ Erreur détection automatique: {e}")
                QMessageBox.critical(
                    self,
                    "Erreur Détection",
                    f"Erreur lors de la détection automatique:\n\n{str(e)}"
                )
        else:
            QMessageBox.warning(
                self,
                "Fonctionnalité Indisponible",
                "L'outil d'annotation n'est pas correctement initialisé.\n\n"
                "Essayez de redémarrer le plugin."
            )

    def accept(self):
        """Méthode de compatibilité avec QDialog - masque le dock widget"""
        self.setVisible(False)
        
    def reject(self):
        """Méthode de compatibilité avec QDialog - masque le dock widget"""
        self.setVisible(False)
        
    def closeEvent(self, event):
        """Gestion de la fermeture de la fenêtre"""
        # Nettoyage si nécessaire
        if self.start_annotation_btn.isChecked():
            self.start_annotation_btn.setChecked(False)
        
        # Désactivation de l'outil d'annotation
        if self.annotation_tool and self.iface.mapCanvas().mapTool() == self.annotation_tool:
            self.iface.mapCanvas().unsetMapTool(self.annotation_tool)
        
        # NOUVEAU: Nettoyage Smart Engine
        self.cleanup_smart_engine()
        
        # Nettoyage YOLOEngine si en cours d'entraînement
        if self.training_in_progress and self.yolo_engine:
            self.yolo_engine.cleanup()
        
        event.accept()
    
    def show_class_detail(self):
        """Affiche les détails d'une classe dans un dialog simple"""
        # Récupérer la classe sélectionnée
        current_item = self.classes_tree.currentItem()
        if not current_item:
            QMessageBox.information(self, "Information", "Veuillez sélectionner une classe")
            return
        
        class_name = current_item.text(0)
        
        try:
            # Récupérer les exemples de cette classe
            examples = self.annotation_manager.get_class_examples(class_name)
            
            # Créer le dialog de détails
            from .class_detail_dialog import ClassDetailDialog
            dialog = ClassDetailDialog(class_name, examples, self)
            dialog.exec_()
            
        except Exception as e:
            # Fallback simple si le dialog détaillé n'est pas disponible
            example_count = len(examples) if 'examples' in locals() else 0
            
            QMessageBox.information(
                self, 
                f"Détails - {class_name}",
                f"Classe: {class_name}\n"
                f"Exemples: {example_count}\n"
                f"Status: Prêt pour entraînement" if example_count >= 10 else f"Status: {10-example_count} exemples manquants"
            )
    
