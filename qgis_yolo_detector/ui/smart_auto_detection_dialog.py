"""
Dialog de Sélection des Détections Automatiques Smart Mode

Interface pour valider les détections automatiques YOLO sur la zone visible.
Permet à l'utilisateur de sélectionner quelles détections conserver pour l'entraînement.
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer
from qgis.PyQt.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QBrush, QImage
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QCheckBox, QGroupBox, QGridLayout,
    QMessageBox, QProgressBar, QTextEdit, QSplitter
)

import numpy as np
import cv2
from PIL import Image
import numpy as np


class DetectionItem(QWidget):
    """Widget représentant une détection individuelle"""
    
    selection_changed = pyqtSignal(dict, bool)  # (detection, selected)
    
    def __init__(self, detection, image_patch, index, parent=None):
        super().__init__(parent)
        
        self.detection = detection
        self.image_patch = image_patch
        self.index = index
        self.selected = True  # Sélectionné par défaut
        
        self.setup_ui()
        self.update_preview()
    
    def setup_ui(self):
        """Configure l'interface de l'item"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Checkbox de sélection
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        self.checkbox.toggled.connect(self.on_selection_changed)
        layout.addWidget(self.checkbox)
        
        # Aperçu miniature
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(80, 80)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)
        
        # Informations détection
        info_layout = QVBoxLayout()
        
        self.class_label = QLabel(f"Classe: {self.detection['class_name']}")
        self.class_label.setFont(QFont("Arial", 9, QFont.Bold))
        info_layout.addWidget(self.class_label)
        
        self.confidence_label = QLabel(f"Confiance: {self.detection['confidence']:.1%}")
        info_layout.addWidget(self.confidence_label)
        
        bbox = self.detection['bbox']
        self.size_label = QLabel(f"Taille: {bbox[2]-bbox[0]:.0f}×{bbox[3]-bbox[1]:.0f}px")
        info_layout.addWidget(self.size_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Style selon la confiance
        confidence = self.detection['confidence']
        if confidence > 0.8:
            color = "#E8F5E8"  # Vert clair
        elif confidence > 0.5:
            color = "#FFF8E1"  # Jaune clair
        else:
            color = "#FFEBEE"  # Rouge clair
        
        self.setStyleSheet(f"QWidget {{ background-color: {color}; border-radius: 4px; }}")
    
    def update_preview(self):
        """Met à jour l'aperçu miniature avec bbox"""
        try:
            bbox = self.detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Extraction de la zone d'intérêt avec padding
            padding = 10
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(self.image_patch.shape[1], x2 + padding)
            crop_y2 = min(self.image_patch.shape[0], y2 + padding)
            
            cropped = self.image_patch[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if cropped.size == 0:
                self.preview_label.setText("Erreur\nAperçu")
                return
            
            # Conversion pour Qt (conversion memoryview vers bytes)
            if len(cropped.shape) == 3:
                height, width, channel = cropped.shape
                bytes_per_line = 3 * width
                # Conversion memoryview vers bytes pour compatibilité QImage
                image_data = bytes(cropped.data)
                q_image = QImage(image_data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                height, width = cropped.shape
                bytes_per_line = width
                # Conversion memoryview vers bytes pour compatibilité QImage
                image_data = bytes(cropped.data)
                q_image = QImage(image_data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            # Redimensionnement
            pixmap = QPixmap.fromImage(q_image)
            pixmap = pixmap.scaled(76, 76, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Dessin de la bbox relative
            painter = QPainter(pixmap)
            
            # Calcul des coordonnées relatives dans le crop
            rel_x1 = max(0, (x1 - crop_x1) * pixmap.width() / (crop_x2 - crop_x1))
            rel_y1 = max(0, (y1 - crop_y1) * pixmap.height() / (crop_y2 - crop_y1))
            rel_x2 = min(pixmap.width(), (x2 - crop_x1) * pixmap.width() / (crop_x2 - crop_x1))
            rel_y2 = min(pixmap.height(), (y2 - crop_y1) * pixmap.height() / (crop_y2 - crop_y1))
            
            # Style bbox selon confiance
            confidence = self.detection['confidence']
            if confidence > 0.8:
                pen_color = QColor(76, 175, 80)  # Vert
            elif confidence > 0.5:
                pen_color = QColor(255, 193, 7)   # Orange
            else:
                pen_color = QColor(244, 67, 54)   # Rouge
            
            pen = QPen(pen_color, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(int(rel_x1), int(rel_y1), int(rel_x2 - rel_x1), int(rel_y2 - rel_y1))
            
            painter.end()
            
            self.preview_label.setPixmap(pixmap)
            
        except Exception as e:
            print(f"⚠️ Erreur aperçu détection {self.index}: {e}")
            self.preview_label.setText("Erreur\nAperçu")
    
    def on_selection_changed(self, checked):
        """Gestion du changement de sélection"""
        self.selected = checked
        self.selection_changed.emit(self.detection, checked)
        
        # Style visuel selon sélection
        if checked:
            self.setStyleSheet(self.styleSheet() + " border: 2px solid #2196F3;")
        else:
            self.setStyleSheet(self.styleSheet().replace(" border: 2px solid #2196F3;", ""))


class SmartAutoDetectionDialog(QDialog):
    """
    Dialog principal pour la validation des détections automatiques
    """
    
    detections_validated = pyqtSignal(list)  # Liste des détections validées
    
    def __init__(self, detections, image_patch, target_class, parent=None):
        super().__init__(parent)
        
        self.detections = detections
        self.image_patch = image_patch
        self.target_class = target_class
        self.detection_items = []
        
        # Configuration fenêtre
        self.setWindowTitle(f"🤖 Détections Automatiques - {len(detections)} objets trouvés")
        self.setModal(True)
        self.resize(800, 600)
        
        self.setup_ui()
        self.populate_detections()
        
        # Auto-sélection des détections de haute confiance
        self.auto_select_high_confidence()
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        layout = QVBoxLayout(self)
        
        # Header avec informations
        header_group = QGroupBox(f"Détections YOLO - Classe cible: {self.target_class}")
        header_layout = QHBoxLayout(header_group)
        
        self.info_label = QLabel()
        header_layout.addWidget(self.info_label)
        
        # Boutons de sélection rapide
        select_all_btn = QPushButton("✅ Tout sélectionner")
        select_all_btn.clicked.connect(self.select_all)
        header_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("❌ Tout désélectionner")
        select_none_btn.clicked.connect(self.select_none)
        header_layout.addWidget(select_none_btn)
        
        select_high_btn = QPushButton("⭐ Seulement haute confiance")
        select_high_btn.clicked.connect(self.select_high_confidence)
        header_layout.addWidget(select_high_btn)
        
        layout.addWidget(header_group)
        
        # Zone de défilement pour les détections
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.detections_widget = QWidget()
        self.detections_layout = QVBoxLayout(self.detections_widget)
        self.detections_layout.setAlignment(Qt.AlignTop)
        
        scroll_area.setWidget(self.detections_widget)
        layout.addWidget(scroll_area)
        
        # Barre de statut
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Boutons d'action
        buttons_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("❌ Annuler")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        
        buttons_layout.addStretch()
        
        self.validate_btn = QPushButton("✅ Valider les détections sélectionnées")
        self.validate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
        """)
        self.validate_btn.clicked.connect(self.validate_selections)
        buttons_layout.addWidget(self.validate_btn)
        
        layout.addLayout(buttons_layout)
    
    def populate_detections(self):
        """Remplit la liste des détections"""
        for i, detection in enumerate(self.detections):
            item = DetectionItem(detection, self.image_patch, i, self)
            item.selection_changed.connect(self.on_item_selection_changed)
            
            self.detections_layout.addWidget(item)
            self.detection_items.append(item)
        
        self.update_status()
    
    def update_status(self):
        """Met à jour les informations de statut"""
        total = len(self.detections)
        selected = sum(1 for item in self.detection_items if item.selected)
        
        self.info_label.setText(f"Total: {total} détections | Sélectionnées: {selected}")
        
        if selected == 0:
            self.status_label.setText("Aucune détection sélectionnée")
            self.validate_btn.setEnabled(False)
        else:
            self.status_label.setText(f"{selected} détection(s) seront ajoutées aux exemples d'entraînement")
            self.validate_btn.setEnabled(True)
    
    def on_item_selection_changed(self, detection, selected):
        """Gestion du changement de sélection d'un item"""
        self.update_status()
    
    def select_all(self):
        """Sélectionne toutes les détections"""
        for item in self.detection_items:
            item.checkbox.setChecked(True)
    
    def select_none(self):
        """Désélectionne toutes les détections"""
        for item in self.detection_items:
            item.checkbox.setChecked(False)
    
    def select_high_confidence(self):
        """Sélectionne uniquement les détections de haute confiance (>70%)"""
        for item in self.detection_items:
            high_conf = item.detection['confidence'] > 0.7
            item.checkbox.setChecked(high_conf)
    
    def auto_select_high_confidence(self):
        """Auto-sélection des détections de très haute confiance au démarrage"""
        high_count = 0
        for item in self.detection_items:
            if item.detection['confidence'] > 0.85:
                item.checkbox.setChecked(True)
                high_count += 1
            else:
                item.checkbox.setChecked(False)
        
        if high_count > 0:
            self.status_label.setText(f"Auto-sélection: {high_count} détections de très haute confiance (>85%)")
    
    def validate_selections(self):
        """Valide les sélections et émet le signal"""
        selected_detections = []
        
        for item in self.detection_items:
            if item.selected:
                selected_detections.append(item.detection)
        
        if not selected_detections:
            QMessageBox.warning(
                self,
                "Aucune Sélection",
                "Veuillez sélectionner au moins une détection à valider."
            )
            return
        
        # Confirmation
        reply = QMessageBox.question(
            self,
            "Confirmation",
            f"Valider {len(selected_detections)} détection(s) ?\n\n"
            f"Ces détections seront ajoutées aux exemples d'entraînement "
            f"pour la classe '{self.target_class}'.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.detections_validated.emit(selected_detections)
            self.accept()
    
    def keyPressEvent(self, event):
        """Gestion des raccourcis clavier"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.validate_selections()
        elif event.key() == Qt.Key_Escape:
            self.reject()
        elif event.key() == Qt.Key_A and event.modifiers() == Qt.ControlModifier:
            self.select_all()
        elif event.key() == Qt.Key_D and event.modifiers() == Qt.ControlModifier:
            self.select_none()
        else:
            super().keyPressEvent(event)