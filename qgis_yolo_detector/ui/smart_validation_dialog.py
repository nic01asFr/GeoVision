"""
Mini Dialog de Validation Smart Mode

Interface légère et non-bloquante pour valider les détections Smart Assistant.
Affiche rapidement le résultat de la détection YOLO+SAM avec options de validation.
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer, QPointF
from qgis.PyQt.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage, QBrush, QPolygonF
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QProgressBar, QWidget
)

import numpy as np
import cv2
from PIL import Image
import numpy as np


class SmartValidationDialog(QDialog):
    """
    Mini dialog de validation pour les détections Smart Mode
    
    Interface ultra-légère qui :
    - Affiche l'aperçu de la détection (bbox sur image)
    - Montre les métriques de confiance YOLO/SAM
    - Propose validation rapide : ✅ ❌ ✏️
    """
    
    # Signaux
    detection_accepted = pyqtSignal(dict)  # Détection acceptée
    detection_rejected = pyqtSignal()      # Détection rejetée
    detection_manual_edit = pyqtSignal()   # Demande édition manuelle
    
    def __init__(self, smart_result, image_patch, parent=None):
        """
        Initialise le dialog de validation
        
        Args:
            smart_result: SmartDetectionResult à valider
            image_patch: Patch image avec détection
            parent: Widget parent
        """
        super().__init__(parent)
        
        self.smart_result = smart_result
        self.image_patch = image_patch
        
        # Configuration fenêtre
        self.setWindowTitle("🤖 Smart Detection")
        self.setWindowFlags(
            Qt.Dialog | 
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint  # Sans bordure pour aspect popup
        )
        self.setModal(False)  # Non-bloquant
        self.resize(300, 200)
        
        # Auto-fermeture après délai si haute confiance
        self.auto_close_timer = QTimer()
        self.auto_close_timer.timeout.connect(self._auto_accept)
        
        self.setup_ui()
        self.update_content()
        
        # Auto-fermeture si confiance très élevée (>90%)
        if smart_result.confidence_yolo > 0.9:
            self.auto_close_timer.start(2000)  # 2 secondes
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Header avec titre
        header_label = QLabel("🤖 Détection Intelligente")
        header_label.setStyleSheet("""
            QLabel {
                background-color: #2E7D32;
                color: white;
                padding: 6px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        layout.addWidget(header_label)
        
        # Zone d'aperçu (sera remplie par update_content)
        self.preview_label = QLabel()
        self.preview_label.setMinimumHeight(80)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)
        
        # Informations détection
        info_group = QGroupBox("📊 Détails")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(2)
        
        self.class_label = QLabel()
        self.confidence_label = QLabel()
        self.processing_time_label = QLabel()
        
        info_layout.addWidget(self.class_label)
        info_layout.addWidget(self.confidence_label)
        info_layout.addWidget(self.processing_time_label)
        
        layout.addWidget(info_group)
        
        # Boutons de validation
        buttons_layout = QHBoxLayout()
        
        self.accept_btn = QPushButton("✅")
        self.accept_btn.setToolTip("Accepter la détection")
        self.accept_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                min-width: 40px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
        """)
        self.accept_btn.clicked.connect(self._on_accept)
        
        self.reject_btn = QPushButton("❌")
        self.reject_btn.setToolTip("Rejeter la détection")
        self.reject_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                min-width: 40px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #EF5350;
            }
        """)
        self.reject_btn.clicked.connect(self._on_reject)
        
        self.edit_btn = QPushButton("✏️")
        self.edit_btn.setToolTip("Éditer manuellement")
        self.edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                min-width: 40px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #FFB74D;
            }
        """)
        self.edit_btn.clicked.connect(self._on_edit)
        
        buttons_layout.addWidget(self.accept_btn)
        buttons_layout.addWidget(self.edit_btn)
        buttons_layout.addWidget(self.reject_btn)
        
        layout.addLayout(buttons_layout)
        
        # Auto-fermeture progress bar (si applicable)
        self.auto_close_progress = QProgressBar()
        self.auto_close_progress.setVisible(False)
        self.auto_close_progress.setMaximum(100)
        self.auto_close_progress.setFormat("Auto-acceptation dans %v%")
        layout.addWidget(self.auto_close_progress)
    
    def update_content(self):
        """Met à jour le contenu avec les données de détection"""
        # Informations textuelles avec mapping
        class_text = f"Classe: {self.smart_result.class_name}"
        if hasattr(self.smart_result, 'original_coco_class') and self.smart_result.original_coco_class:
            class_text += f" (via {self.smart_result.original_coco_class})"
        self.class_label.setText(class_text)
        
        confidence_text = f"YOLO: {self.smart_result.confidence_yolo:.1%}"
        if self.smart_result.confidence_sam is not None:
            confidence_text += f" | SAM: {self.smart_result.confidence_sam:.1%}"
        self.confidence_label.setText(confidence_text)
        
        time_text = f"Temps: {self.smart_result.processing_time:.1f}ms"
        if hasattr(self.smart_result, 'original_coco_class') and self.smart_result.original_coco_class:
            time_text += f" • Mapping intelligent"
        self.processing_time_label.setText(time_text)
        
        # Aperçu visuel (bbox sur image)
        try:
            preview_pixmap = self._create_preview_image()
            if preview_pixmap:
                self.preview_label.setPixmap(preview_pixmap)
            else:
                self.preview_label.setText("Aperçu non disponible")
        except Exception as e:
            self.preview_label.setText(f"Erreur aperçu: {str(e)}")
        
        # Configuration auto-fermeture si confiance élevée
        if self.smart_result.confidence_yolo > 0.9:
            self.auto_close_progress.setVisible(True)
            self._start_auto_close_animation()
    
    def _create_preview_image(self):
        """
        Crée une image d'aperçu avec la bbox dessinée
        
        Returns:
            QPixmap: Image d'aperçu ou None
        """
        try:
            # Conversion image patch vers format Qt (méthode directe numpy→QImage)
            if len(self.image_patch.shape) == 3:
                # RGB
                height, width, channel = self.image_patch.shape
                bytes_per_line = 3 * width
                # Conversion memoryview vers bytes pour compatibilité QImage
                image_data = bytes(self.image_patch.data)
                q_image = QImage(image_data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                # Grayscale
                height, width = self.image_patch.shape
                bytes_per_line = width
                # Conversion memoryview vers bytes pour compatibilité QImage
                image_data = bytes(self.image_patch.data)
                q_image = QImage(image_data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            # Création QPixmap
            pixmap = QPixmap.fromImage(q_image)
            
            # Redimensionnement pour aperçu
            preview_size = 120
            pixmap = pixmap.scaled(preview_size, preview_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Dessin de la bbox
            painter = QPainter(pixmap)
            
            # Calcul des coordonnées redimensionnées
            scale_x = pixmap.width() / self.image_patch.shape[1]
            scale_y = pixmap.height() / self.image_patch.shape[0]
            
            x1, y1, x2, y2 = self.smart_result.bbox
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            
            # Affichage avec polygone précis si disponible
            print(f"🔍 DEBUG APERÇU: hasattr polygon_points: {hasattr(self.smart_result, 'polygon_points')}")
            if hasattr(self.smart_result, 'polygon_points'):
                print(f"🔍 DEBUG APERÇU: polygon_points is not None: {self.smart_result.polygon_points is not None}")
                if self.smart_result.polygon_points:
                    print(f"🔍 DEBUG APERÇU: nombre de points: {len(self.smart_result.polygon_points)}")
            
            if hasattr(self.smart_result, 'polygon_points') and self.smart_result.polygon_points:
                print(f"🔺 DEBUG APERÇU: Affichage polygone SAM avec {len(self.smart_result.polygon_points)} vertices")
                # POLYGONE PRÉCIS SAM (vert, rempli semi-transparent)
                polygon_points = self.smart_result.polygon_points
                
                # Création du polygone Qt
                polygon = QPolygonF()
                
                # Conversion des points en coordonnées écran
                print(f"🔍 DEBUG COORDS: image_patch shape: {self.image_patch.shape}")
                print(f"🔍 DEBUG COORDS: scale_x={scale_x}, scale_y={scale_y}")
                
                for i, point in enumerate(polygon_points):
                    if len(point) >= 2:
                        # Les points sont normalisés [0,1], donc on multiplie par dimensions image
                        px = point[0] * self.image_patch.shape[1] * scale_x
                        py = point[1] * self.image_patch.shape[0] * scale_y
                        
                        if i < 3:  # Afficher seulement les 3 premiers points
                            print(f"🔍 DEBUG COORDS: Point {i}: ({point[0]:.3f}, {point[1]:.3f}) -> ({px:.1f}, {py:.1f})")
                        
                        polygon.append(QPointF(px, py))
                
                # Dessin du polygone avec remplissage
                painter.setBrush(QBrush(QColor(76, 175, 80, 60)))  # Vert semi-transparent
                painter.setPen(QPen(QColor(76, 175, 80), 2, Qt.SolidLine))  # Contour vert
                painter.drawPolygon(polygon)
                
                print(f"✅ DEBUG APERÇU: Polygone SAM dessiné avec {polygon.size()} points")
                print(f"🔺 DEBUG APERÇU: Utilisateur devrait voir un polygone VERT dans l'aperçu!")
                
                # Bbox originale en pointillés rouges pour comparaison
                if hasattr(self.smart_result, 'bbox_original'):
                    orig_x1, orig_y1, orig_x2, orig_y2 = self.smart_result.bbox_original
                    orig_x1_scaled = int(orig_x1 * scale_x)
                    orig_y1_scaled = int(orig_y1 * scale_y)
                    orig_x2_scaled = int(orig_x2 * scale_x)
                    orig_y2_scaled = int(orig_y2 * scale_y)
                    
                    pen_original = QPen(QColor(244, 67, 54), 1, Qt.DashLine)
                    painter.setPen(pen_original)
                    painter.drawRect(orig_x1_scaled, orig_y1_scaled, 
                                   orig_x2_scaled - orig_x1_scaled, orig_y2_scaled - orig_y1_scaled)
                
                # Légende
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawText(x1_scaled + 2, y1_scaled - 15, "🔺 Polygone")
                if hasattr(self.smart_result, 'vertex_count'):
                    painter.drawText(x1_scaled + 2, y1_scaled - 5, f"{self.smart_result.vertex_count} pts")
                    
            # Affichage comparatif si SAM appliqué (mais sans polygone)
            elif self.smart_result.refinement_applied and hasattr(self.smart_result, 'bbox_original'):
                # Bbox originale YOLO (rouge, pointillés)
                orig_x1, orig_y1, orig_x2, orig_y2 = self.smart_result.bbox_original
                orig_x1_scaled = int(orig_x1 * scale_x)
                orig_y1_scaled = int(orig_y1 * scale_y)
                orig_x2_scaled = int(orig_x2 * scale_x)
                orig_y2_scaled = int(orig_y2 * scale_y)
                
                pen_original = QPen(QColor(244, 67, 54), 1, Qt.DashLine)
                painter.setPen(pen_original)
                painter.drawRect(orig_x1_scaled, orig_y1_scaled, 
                               orig_x2_scaled - orig_x1_scaled, orig_y2_scaled - orig_y1_scaled)
                
                # Bbox optimisée SAM (vert, solide)
                pen_sam = QPen(QColor(76, 175, 80), 2, Qt.SolidLine)
                painter.setPen(pen_sam)
                painter.drawRect(x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled)
                
                # Légende
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawText(x1_scaled + 2, y1_scaled - 15, "YOLO →")
                painter.setPen(QPen(QColor(76, 175, 80), 1))
                painter.drawText(x1_scaled + 2, y1_scaled - 5, "SAM ✓")
            else:
                print(f"📦 DEBUG APERÇU: Pas de polygone SAM - Affichage bbox YOLO seulement")
                # YOLO uniquement - bleu
                pen = QPen(QColor(33, 150, 243), 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled)
            
            # Label confiance
            font = QFont()
            font.setPointSize(8)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            
            confidence_text = f"{self.smart_result.confidence_yolo:.0%}"
            painter.drawText(x1_scaled + 2, y1_scaled + 12, confidence_text)
            
            painter.end()
            
            return pixmap
            
        except Exception as e:
            print(f"❌ Erreur création aperçu: {e}")
            return None
    
    def _start_auto_close_animation(self):
        """Démarre l'animation de progression pour auto-fermeture"""
        self.auto_close_progress.setValue(0)
        
        # Timer d'animation
        self.animation_timer = QTimer()
        self.animation_step = 0
        
        def update_progress():
            self.animation_step += 2  # 2% par step
            self.auto_close_progress.setValue(self.animation_step)
            
            if self.animation_step >= 100:
                self.animation_timer.stop()
        
        self.animation_timer.timeout.connect(update_progress)
        self.animation_timer.start(40)  # 50 FPS pour animation fluide
    
    def _auto_accept(self):
        """Auto-acceptation après délai"""
        self.auto_close_timer.stop()
        self._on_accept()
    
    def _on_accept(self):
        """Gestion acceptation détection"""
        # Apprentissage du mapping si applicable
        if hasattr(self.smart_result, 'original_coco_class') and self.smart_result.original_coco_class:
            from ..core.class_mapping import learn_mapping_from_user
            learn_mapping_from_user(
                self.smart_result.class_name,
                [self.smart_result.original_coco_class],
                accepted=True
            )
            print(f"🧠 APPRENTISSAGE: Mapping '{self.smart_result.class_name}' ← '{self.smart_result.original_coco_class}' accepté")
        
        self.detection_accepted.emit(self.smart_result.__dict__)
        self.accept()
    
    def _on_reject(self):
        """Gestion rejet détection"""
        # Apprentissage négatif du mapping si applicable
        if hasattr(self.smart_result, 'original_coco_class') and self.smart_result.original_coco_class:
            from ..core.class_mapping import learn_mapping_from_user
            learn_mapping_from_user(
                self.smart_result.class_name,
                [self.smart_result.original_coco_class],
                accepted=False
            )
            print(f"🧠 APPRENTISSAGE: Mapping '{self.smart_result.class_name}' ← '{self.smart_result.original_coco_class}' rejeté")
        
        self.detection_rejected.emit()
        self.reject()
    
    def _on_edit(self):
        """Gestion demande édition manuelle"""
        self.detection_manual_edit.emit()
        self.reject()
    
    def keyPressEvent(self, event):
        """Gestion des raccourcis clavier"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self._on_accept()
        elif event.key() == Qt.Key_Escape:
            self._on_reject()
        elif event.key() == Qt.Key_E:
            self._on_edit()
        else:
            super().keyPressEvent(event)
    
    def show_at_cursor(self):
        """Affiche le dialog près du curseur"""
        from qgis.PyQt.QtGui import QCursor
        cursor_pos = QCursor.pos()
        
        # Décalage pour éviter d'être sous le curseur
        self.move(cursor_pos.x() + 10, cursor_pos.y() + 10)
        self.show()
        self.raise_()
        self.activateWindow()