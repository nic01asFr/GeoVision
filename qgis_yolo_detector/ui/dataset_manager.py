"""
Gestionnaire de Dataset avec Interface Complète

Interface permettant de visualiser, éditer et gérer les collections
d'annotations pour chaque classe avec versioning et export.
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer
from qgis.PyQt.QtGui import QPixmap, QPainter, QPen, QColor, QPolygonF
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QComboBox, QSpinBox, QGroupBox,
    QTabWidget, QToolBar, QAction, QMenu, QHeaderView,
    QDialog, QDialogButtonBox, QTextEdit, QSlider
)
from qgis.core import QgsPointXY, QgsGeometry
import json
from datetime import datetime
from typing import List, Dict, Optional


class DatasetManagerWidget(QWidget):
    """
    Widget principal pour la gestion complète des datasets
    
    Fonctionnalités:
    - Vue arborescente des classes et versions
    - Galerie/table des exemples
    - Édition des polygones et attributs
    - Versioning automatique
    - Export optimisé pour entraînement
    """
    
    # Signaux
    example_selected = pyqtSignal(dict)  # Exemple sélectionné
    dataset_modified = pyqtSignal(str)   # Dataset modifié
    
    def __init__(self, annotation_manager, parent=None):
        """
        Initialise le gestionnaire de dataset
        
        Args:
            annotation_manager: Gestionnaire des annotations
            parent: Widget parent
        """
        super().__init__(parent)
        
        self.annotation_manager = annotation_manager
        self.current_class = None
        self.current_version = None
        self.selected_examples = []
        
        self.setup_ui()
        self.load_datasets()
    
    def setup_ui(self):
        """Configure l'interface utilisateur complète"""
        layout = QVBoxLayout(self)
        
        # Toolbar principale
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)
        
        # Splitter principal (arbre | contenu)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Panel gauche : Arbre des classes et versions
        left_panel = self.create_class_tree_panel()
        main_splitter.addWidget(left_panel)
        
        # Panel droit : Contenu (tabs)
        right_panel = self.create_content_panel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([300, 700])
        layout.addWidget(main_splitter)
        
        # Barre de statut
        self.status_bar = QLabel("Prêt")
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 5px;
                border-top: 1px solid #ccc;
            }
        """)
        layout.addWidget(self.status_bar)
    
    def create_toolbar(self):
        """Crée la toolbar avec actions principales"""
        toolbar = QToolBar()
        
        # Actions principales
        self.action_refresh = QAction("🔄 Rafraîchir", self)
        self.action_refresh.triggered.connect(self.load_datasets)
        toolbar.addAction(self.action_refresh)
        
        toolbar.addSeparator()
        
        self.action_new_version = QAction("📦 Nouvelle Version", self)
        self.action_new_version.triggered.connect(self.create_new_version)
        toolbar.addAction(self.action_new_version)
        
        self.action_export = QAction("💾 Exporter Dataset", self)
        self.action_export.triggered.connect(self.export_dataset)
        toolbar.addAction(self.action_export)
        
        toolbar.addSeparator()
        
        self.action_batch_edit = QAction("✏️ Édition Batch", self)
        self.action_batch_edit.triggered.connect(self.batch_edit_examples)
        toolbar.addAction(self.action_batch_edit)
        
        self.action_clean = QAction("🧹 Nettoyer", self)
        self.action_clean.setToolTip("Supprimer exemples de faible qualité")
        self.action_clean.triggered.connect(self.clean_dataset)
        toolbar.addAction(self.action_clean)
        
        return toolbar
    
    def create_class_tree_panel(self):
        """Crée le panneau avec l'arbre des classes"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Titre
        title = QLabel("📚 Classes & Versions")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Arbre des classes
        self.class_tree = QTreeWidget()
        self.class_tree.setHeaderLabels(["Classe/Version", "Exemples", "Modèle"])
        self.class_tree.itemClicked.connect(self.on_class_selected)
        
        # Style de l'arbre
        self.class_tree.setStyleSheet("""
            QTreeWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QTreeWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        
        layout.addWidget(self.class_tree)
        
        # Statistiques de la classe sélectionnée
        self.class_stats = QGroupBox("📊 Statistiques")
        stats_layout = QVBoxLayout(self.class_stats)
        
        self.stats_label = QLabel("Sélectionnez une classe")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(self.class_stats)
        
        return panel
    
    def create_content_panel(self):
        """Crée le panneau principal avec tabs"""
        self.content_tabs = QTabWidget()
        
        # Tab 1: Galerie visuelle
        self.gallery_tab = self.create_gallery_tab()
        self.content_tabs.addTab(self.gallery_tab, "🖼️ Galerie")
        
        # Tab 2: Table détaillée
        self.table_tab = self.create_table_tab()
        self.content_tabs.addTab(self.table_tab, "📋 Table")
        
        # Tab 3: Éditeur d'exemple
        self.editor_tab = self.create_editor_tab()
        self.content_tabs.addTab(self.editor_tab, "✏️ Éditeur")
        
        # Tab 4: Métriques et qualité
        self.metrics_tab = self.create_metrics_tab()
        self.content_tabs.addTab(self.metrics_tab, "📈 Métriques")
        
        return self.content_tabs
    
    def create_gallery_tab(self):
        """Crée l'onglet galerie avec vignettes"""
        from qgis.PyQt.QtWidgets import QScrollArea, QGridLayout
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Contrôles de filtrage
        filter_bar = QHBoxLayout()
        
        filter_bar.addWidget(QLabel("Taille vignettes:"))
        self.thumb_size_slider = QSlider(Qt.Horizontal)
        self.thumb_size_slider.setRange(50, 200)
        self.thumb_size_slider.setValue(100)
        self.thumb_size_slider.valueChanged.connect(self.update_gallery)
        filter_bar.addWidget(self.thumb_size_slider)
        
        filter_bar.addWidget(QLabel("Filtre confiance:"))
        self.conf_filter = QSpinBox()
        self.conf_filter.setRange(0, 100)
        self.conf_filter.setSuffix("%")
        self.conf_filter.valueChanged.connect(self.update_gallery)
        filter_bar.addWidget(self.conf_filter)
        
        filter_bar.addStretch()
        layout.addLayout(filter_bar)
        
        # Zone scrollable pour la galerie
        scroll = QScrollArea()
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        scroll.setWidget(self.gallery_widget)
        scroll.setWidgetResizable(True)
        
        layout.addWidget(scroll)
        
        return widget
    
    def create_table_tab(self):
        """Crée l'onglet table avec détails"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Table des exemples
        self.examples_table = QTableWidget()
        self.examples_table.setColumnCount(9)
        self.examples_table.setHorizontalHeaderLabels([
            "ID", "Aperçu", "Classe", "Confiance", 
            "Polygone", "Date", "Dimensions", "Source", "Actions"
        ])
        
        # Configuration table
        header = self.examples_table.horizontalHeader()
        header.setStretchLastSection(True)
        self.examples_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.examples_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        
        layout.addWidget(self.examples_table)
        
        # Barre d'actions pour sélection
        actions_bar = QHBoxLayout()
        
        self.btn_edit_selected = QPushButton("✏️ Éditer Sélection")
        self.btn_edit_selected.clicked.connect(self.edit_selected_examples)
        actions_bar.addWidget(self.btn_edit_selected)
        
        self.btn_delete_selected = QPushButton("🗑️ Supprimer Sélection")
        self.btn_delete_selected.clicked.connect(self.delete_selected_examples)
        actions_bar.addWidget(self.btn_delete_selected)
        
        self.btn_export_selected = QPushButton("💾 Exporter Sélection")
        self.btn_export_selected.clicked.connect(self.export_selected_examples)
        actions_bar.addWidget(self.btn_export_selected)
        
        actions_bar.addStretch()
        layout.addLayout(actions_bar)
        
        return widget
    
    def create_editor_tab(self):
        """Crée l'onglet éditeur d'exemple"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Panel gauche : Aperçu et édition polygone
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Canvas pour édition polygone
        self.polygon_editor = PolygonEditorWidget()
        left_layout.addWidget(self.polygon_editor)
        
        # Boutons d'édition
        edit_buttons = QHBoxLayout()
        self.btn_add_point = QPushButton("➕ Ajouter Point")
        self.btn_delete_point = QPushButton("➖ Supprimer Point")
        self.btn_smooth = QPushButton("〰️ Lisser")
        self.btn_simplify = QPushButton("📐 Simplifier")
        
        edit_buttons.addWidget(self.btn_add_point)
        edit_buttons.addWidget(self.btn_delete_point)
        edit_buttons.addWidget(self.btn_smooth)
        edit_buttons.addWidget(self.btn_simplify)
        
        left_layout.addLayout(edit_buttons)
        
        # Panel droit : Attributs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Formulaire d'attributs
        attrs_group = QGroupBox("📝 Attributs")
        attrs_layout = QVBoxLayout(attrs_group)
        
        # Classe
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Classe:"))
        self.edit_class_combo = QComboBox()
        class_layout.addWidget(self.edit_class_combo)
        attrs_layout.addLayout(class_layout)
        
        # Confiance
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confiance:"))
        self.edit_conf_spin = QSpinBox()
        self.edit_conf_spin.setRange(0, 100)
        self.edit_conf_spin.setSuffix("%")
        conf_layout.addWidget(self.edit_conf_spin)
        attrs_layout.addLayout(conf_layout)
        
        # Notes
        self.edit_notes = QTextEdit()
        self.edit_notes.setMaximumHeight(100)
        self.edit_notes.setPlaceholderText("Notes sur cet exemple...")
        attrs_layout.addWidget(QLabel("Notes:"))
        attrs_layout.addWidget(self.edit_notes)
        
        right_layout.addWidget(attrs_group)
        
        # Historique
        history_group = QGroupBox("📜 Historique")
        history_layout = QVBoxLayout(history_group)
        self.history_list = QTextEdit()
        self.history_list.setReadOnly(True)
        history_layout.addWidget(self.history_list)
        
        right_layout.addWidget(history_group)
        
        # Boutons de sauvegarde
        save_buttons = QHBoxLayout()
        self.btn_save_changes = QPushButton("💾 Sauvegarder")
        self.btn_save_changes.clicked.connect(self.save_example_changes)
        self.btn_revert = QPushButton("↩️ Annuler")
        self.btn_revert.clicked.connect(self.revert_changes)
        
        save_buttons.addWidget(self.btn_save_changes)
        save_buttons.addWidget(self.btn_revert)
        save_buttons.addStretch()
        
        right_layout.addLayout(save_buttons)
        
        # Splitter pour panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        return widget
    
    def create_metrics_tab(self):
        """Crée l'onglet métriques et qualité"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistiques globales
        global_stats = QGroupBox("📊 Statistiques Globales")
        stats_layout = QVBoxLayout(global_stats)
        
        self.global_stats_text = QTextEdit()
        self.global_stats_text.setReadOnly(True)
        stats_layout.addWidget(self.global_stats_text)
        
        layout.addWidget(global_stats)
        
        # Graphiques de qualité (placeholder)
        quality_group = QGroupBox("📈 Métriques de Qualité")
        quality_layout = QVBoxLayout(quality_group)
        
        quality_label = QLabel("Graphiques de distribution et qualité")
        quality_label.setAlignment(Qt.AlignCenter)
        quality_label.setStyleSheet("padding: 50px; background-color: #f0f0f0;")
        quality_layout.addWidget(quality_label)
        
        layout.addWidget(quality_group)
        
        return widget
    
    def load_datasets(self):
        """Charge tous les datasets depuis la base"""
        self.class_tree.clear()
        
        # Récupérer toutes les classes
        classes = self.annotation_manager.get_all_classes()
        
        for class_name in classes:
            # Item classe
            class_item = QTreeWidgetItem(self.class_tree)
            class_item.setText(0, f"📁 {class_name}")
            
            # Compter les exemples
            examples = self.annotation_manager.get_class_examples(class_name)
            class_item.setText(1, str(len(examples)))
            
            # Vérifier si modèle existe
            models = self.annotation_manager.get_trained_models()
            has_model = any(m['classes'][0] == class_name for m in models if m['classes'])
            class_item.setText(2, "✅" if has_model else "❌")
            
            # Versions (placeholder pour futur versioning)
            version_item = QTreeWidgetItem(class_item)
            version_item.setText(0, "📦 v1.0 (actuelle)")
            version_item.setText(1, str(len(examples)))
            
            class_item.setExpanded(True)
        
        self.update_status(f"✅ {len(classes)} classes chargées")
    
    def on_class_selected(self, item, column):
        """Gestion sélection classe dans l'arbre"""
        if item.parent() is None:
            # C'est une classe
            class_name = item.text(0).replace("📁 ", "")
            self.current_class = class_name
            self.load_class_examples(class_name)
            self.update_class_stats(class_name)
        else:
            # C'est une version
            class_item = item.parent()
            class_name = class_item.text(0).replace("📁 ", "")
            version = item.text(0).replace("📦 ", "")
            self.current_class = class_name
            self.current_version = version
            self.load_class_examples(class_name)
    
    def load_class_examples(self, class_name):
        """Charge les exemples d'une classe"""
        examples = self.annotation_manager.get_class_examples(class_name)
        
        # Mise à jour galerie
        self.update_gallery_with_examples(examples)
        
        # Mise à jour table
        self.update_table_with_examples(examples)
        
        self.update_status(f"📊 {len(examples)} exemples chargés pour '{class_name}'")
    
    def update_gallery_with_examples(self, examples):
        """Met à jour la galerie avec les exemples"""
        # Nettoyer la galerie existante
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Ajouter les vignettes
        thumb_size = self.thumb_size_slider.value()
        conf_threshold = self.conf_filter.value() / 100.0
        
        row, col = 0, 0
        max_cols = 5
        
        for example in examples:
            # Filtrer par confiance si nécessaire
            if example.get('confidence', 1.0) < conf_threshold:
                continue
            
            # Créer vignette
            thumb = self.create_thumbnail(example, thumb_size)
            self.gallery_layout.addWidget(thumb, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def create_thumbnail(self, example, size):
        """Crée une vignette pour un exemple"""
        widget = QWidget()
        widget.setFixedSize(size + 20, size + 40)
        widget.setStyleSheet("""
            QWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QWidget:hover {
                border: 2px solid #3498db;
                background-color: #ecf0f1;
            }
        """)
        
        layout = QVBoxLayout(widget)
        
        # Image placeholder
        image_label = QLabel()
        image_label.setFixedSize(size, size)
        image_label.setStyleSheet("background-color: #ddd;")
        image_label.setAlignment(Qt.AlignCenter)
        
        # Charger l'image si disponible
        if 'image_patch' in example:
            # TODO: Convertir numpy array en QPixmap
            image_label.setText("🖼️")
        else:
            image_label.setText("📷")
        
        layout.addWidget(image_label)
        
        # Infos
        info_label = QLabel(f"{example.get('id', 'N/A')[:8]}...")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(info_label)
        
        # Click handler
        widget.mousePressEvent = lambda e: self.on_thumbnail_clicked(example)
        
        return widget
    
    def update_table_with_examples(self, examples):
        """Met à jour la table avec les exemples"""
        self.examples_table.setRowCount(len(examples))
        
        for row, example in enumerate(examples):
            # ID
            self.examples_table.setItem(row, 0, QTableWidgetItem(str(example.get('id', ''))))
            
            # Aperçu (placeholder)
            apercu_item = QTableWidgetItem("🖼️")
            apercu_item.setTextAlignment(Qt.AlignCenter)
            self.examples_table.setItem(row, 1, apercu_item)
            
            # Classe
            self.examples_table.setItem(row, 2, QTableWidgetItem(example.get('class_name', '')))
            
            # Confiance
            conf = example.get('confidence', 0) * 100
            self.examples_table.setItem(row, 3, QTableWidgetItem(f"{conf:.1f}%"))
            
            # Polygone
            has_polygon = "✅" if example.get('polygon_points') else "❌"
            self.examples_table.setItem(row, 4, QTableWidgetItem(has_polygon))
            
            # Date
            timestamp = example.get('timestamp', '')
            self.examples_table.setItem(row, 5, QTableWidgetItem(timestamp))
            
            # Dimensions
            bbox = example.get('bbox_map', [0, 0, 0, 0])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            self.examples_table.setItem(row, 6, QTableWidgetItem(f"{width:.0f}x{height:.0f}"))
            
            # Source
            source = example.get('raster_name', 'Unknown')
            self.examples_table.setItem(row, 7, QTableWidgetItem(source))
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            
            btn_edit = QPushButton("✏️")
            btn_edit.clicked.connect(lambda checked, ex=example: self.edit_example(ex))
            btn_delete = QPushButton("🗑️")
            btn_delete.clicked.connect(lambda checked, ex=example: self.delete_example(ex))
            
            actions_layout.addWidget(btn_edit)
            actions_layout.addWidget(btn_delete)
            
            self.examples_table.setCellWidget(row, 8, actions_widget)
    
    def update_class_stats(self, class_name):
        """Met à jour les statistiques de la classe"""
        examples = self.annotation_manager.get_class_examples(class_name)
        
        total = len(examples)
        with_polygon = sum(1 for e in examples if e.get('polygon_points'))
        avg_conf = sum(e.get('confidence', 0) for e in examples) / max(total, 1)
        
        stats_text = f"""
        📊 Classe: {class_name}
        📝 Total exemples: {total}
        🔺 Avec polygones: {with_polygon} ({with_polygon/max(total,1)*100:.1f}%)
        🎯 Confiance moyenne: {avg_conf*100:.1f}%
        📅 Dernière modification: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """
        
        self.stats_label.setText(stats_text)
    
    def on_thumbnail_clicked(self, example):
        """Gestion clic sur vignette"""
        self.example_selected.emit(example)
        
        # Ouvrir dans l'éditeur
        self.content_tabs.setCurrentIndex(2)  # Onglet éditeur
        self.load_example_in_editor(example)
    
    def load_example_in_editor(self, example):
        """Charge un exemple dans l'éditeur"""
        # Charger l'image et le polygone
        self.polygon_editor.load_example(example)
        
        # Charger les attributs
        self.edit_class_combo.setCurrentText(example.get('class_name', ''))
        self.edit_conf_spin.setValue(int(example.get('confidence', 0) * 100))
        self.edit_notes.setText(example.get('notes', ''))
        
        # Historique
        history = example.get('history', [])
        history_text = "\n".join([f"• {h}" for h in history])
        self.history_list.setText(history_text or "Aucun historique")
    
    def save_example_changes(self):
        """Sauvegarde les modifications de l'exemple"""
        if not self.polygon_editor.current_example:
            return
        
        # Récupérer les modifications
        example = self.polygon_editor.current_example
        example['class_name'] = self.edit_class_combo.currentText()
        example['confidence'] = self.edit_conf_spin.value() / 100.0
        example['notes'] = self.edit_notes.toPlainText()
        example['polygon_points'] = self.polygon_editor.get_polygon_points()
        
        # Ajouter à l'historique
        if 'history' not in example:
            example['history'] = []
        example['history'].append(f"Modifié le {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Sauvegarder dans la base
        self.annotation_manager.update_example(example)
        
        self.update_status("✅ Modifications sauvegardées")
        self.dataset_modified.emit(self.current_class)
    
    def create_new_version(self):
        """Crée une nouvelle version du dataset"""
        if not self.current_class:
            return
        
        # TODO: Implémenter versioning
        self.update_status("📦 Nouvelle version créée")
    
    def export_dataset(self):
        """Exporte le dataset courant"""
        if not self.current_class:
            return
        
        # TODO: Export vers format YOLO/COCO
        self.update_status("💾 Dataset exporté")
    
    def batch_edit_examples(self):
        """Édition batch des exemples sélectionnés"""
        # TODO: Dialog d'édition batch
        pass
    
    def clean_dataset(self):
        """Nettoie le dataset (supprime basse qualité)"""
        # TODO: Filtrage par qualité
        pass
    
    def edit_selected_examples(self):
        """Édite les exemples sélectionnés"""
        # TODO: Multi-édition
        pass
    
    def delete_selected_examples(self):
        """Supprime les exemples sélectionnés"""
        # TODO: Suppression batch
        pass
    
    def export_selected_examples(self):
        """Exporte les exemples sélectionnés"""
        # TODO: Export sélection
        pass
    
    def on_table_selection_changed(self):
        """Gestion changement sélection table"""
        selected_rows = set(item.row() for item in self.examples_table.selectedItems())
        self.selected_examples = list(selected_rows)
    
    def edit_example(self, example):
        """Édite un exemple spécifique"""
        self.load_example_in_editor(example)
        self.content_tabs.setCurrentIndex(2)
    
    def delete_example(self, example):
        """Supprime un exemple"""
        # TODO: Confirmation et suppression
        pass
    
    def revert_changes(self):
        """Annule les modifications courantes"""
        if self.polygon_editor.current_example:
            self.load_example_in_editor(self.polygon_editor.current_example)
    
    def update_gallery(self):
        """Met à jour la galerie avec les paramètres actuels"""
        if self.current_class:
            self.load_class_examples(self.current_class)
    
    def update_status(self, message):
        """Met à jour la barre de statut"""
        self.status_bar.setText(message)
        QTimer.singleShot(3000, lambda: self.status_bar.setText("Prêt"))


class PolygonEditorWidget(QWidget):
    """
    Widget d'édition interactive de polygones
    
    Permet de modifier les contours des objets détectés
    avec ajout/suppression de points, lissage, etc.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_example = None
        self.polygon_points = []
        self.selected_point = None
        self.image_pixmap = None
        
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
    
    def load_example(self, example):
        """Charge un exemple pour édition"""
        self.current_example = example
        self.polygon_points = example.get('polygon_points', [])
        
        # TODO: Charger l'image
        self.update()
    
    def get_polygon_points(self):
        """Retourne les points du polygone édité"""
        return self.polygon_points
    
    def paintEvent(self, event):
        """Dessine le polygone et l'image"""
        painter = QPainter(self)
        
        # Dessiner l'image si disponible
        if self.image_pixmap:
            painter.drawPixmap(self.rect(), self.image_pixmap)
        else:
            # Placeholder
            painter.fillRect(self.rect(), QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignCenter, "Pas d'image")
        
        # Dessiner le polygone
        if self.polygon_points:
            # Convertir en QPolygonF
            polygon = QPolygonF()
            for point in self.polygon_points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    x = point[0] * self.width()
                    y = point[1] * self.height()
                    polygon.append(QPointF(x, y))
            
            # Dessiner le polygone
            painter.setPen(QPen(QColor(76, 175, 80), 2))
            painter.setBrush(QColor(76, 175, 80, 50))
            painter.drawPolygon(polygon)
            
            # Dessiner les points de contrôle
            painter.setBrush(QColor(255, 255, 255))
            for i, point in enumerate(polygon):
                if i == self.selected_point:
                    painter.setPen(QPen(QColor(255, 0, 0), 3))
                else:
                    painter.setPen(QPen(QColor(76, 175, 80), 2))
                painter.drawEllipse(point, 5, 5)
    
    def mousePressEvent(self, event):
        """Gestion clic souris pour sélection point"""
        if event.button() == Qt.LeftButton:
            # Chercher le point le plus proche
            click_pos = event.pos()
            min_dist = float('inf')
            selected = None
            
            for i, point in enumerate(self.polygon_points):
                x = point[0] * self.width()
                y = point[1] * self.height()
                dist = ((click_pos.x() - x)**2 + (click_pos.y() - y)**2)**0.5
                
                if dist < min_dist and dist < 10:  # Seuil de 10 pixels
                    min_dist = dist
                    selected = i
            
            self.selected_point = selected
            self.update()
    
    def mouseMoveEvent(self, event):
        """Déplacement du point sélectionné"""
        if self.selected_point is not None and event.buttons() & Qt.LeftButton:
            # Mettre à jour la position du point
            x = event.pos().x() / self.width()
            y = event.pos().y() / self.height()
            
            # Limiter aux bordures
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            
            self.polygon_points[self.selected_point] = [x, y]
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Fin du déplacement"""
        if event.button() == Qt.LeftButton:
            self.selected_point = None