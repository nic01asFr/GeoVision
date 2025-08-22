"""
Interface d'Annotation Rapide Optimisée

Cette interface fournit :
- Mode annotation rapide avec raccourcis clavier
- Validation par lot pour zones denses
- Interface minimaliste pendant annotation
- Suivi de productivité en temps réel
"""

from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal, QRect
from qgis.PyQt.QtGui import QFont, QColor, QPalette, QKeySequence
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QProgressBar, QFrame,
    QShortcut, QToolBar, QAction, QGroupBox,
    QCheckBox, QSlider, QSpinBox, QTextEdit,
    QScrollArea, QSplitter, QMessageBox
)

from qgis.gui import QgisInterface
from datetime import datetime
from typing import Dict, List, Optional, Callable
import time


class ProductivityIndicator(QFrame):
    """Indicateur de productivité en temps réel"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Titre
        title = QLabel("📊 Productivité")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        title.setStyleSheet("color: #00ff88; margin-bottom: 5px;")
        layout.addWidget(title)
        
        # Métriques
        self.annotations_count = QLabel("Annotations: 0")
        self.avg_time = QLabel("Temps moyen: --")
        self.session_time = QLabel("Session: 0:00")
        self.efficiency_rating = QLabel("Efficacité: --")
        
        for label in [self.annotations_count, self.avg_time, self.session_time, self.efficiency_rating]:
            label.setStyleSheet("color: white; font-size: 9px; margin: 1px;")
            layout.addWidget(label)
        
        # Timer pour mise à jour
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(1000)  # Mise à jour chaque seconde
        
        # Données de session
        self.session_start = datetime.now()
        self.annotation_times = []
    
    def add_annotation_time(self, processing_time: float):
        """Ajoute le temps d'une annotation"""
        self.annotation_times.append(processing_time)
        
    def update_metrics(self):
        """Met à jour les métriques affichées"""
        # Nombre d'annotations
        count = len(self.annotation_times)
        self.annotations_count.setText(f"Annotations: {count}")
        
        # Temps moyen
        if self.annotation_times:
            avg = sum(self.annotation_times) / len(self.annotation_times)
            self.avg_time.setText(f"Temps moyen: {avg:.1f}s")
            
            # Évaluation efficacité
            if avg < 15:
                rating = "Excellente ⭐⭐⭐"
                color = "#00ff88"
            elif avg < 25:
                rating = "Bonne ⭐⭐"
                color = "#ffaa00"
            else:
                rating = "À améliorer ⭐"
                color = "#ff6666"
            
            self.efficiency_rating.setText(f"Efficacité: {rating}")
            self.efficiency_rating.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: bold;")
        
        # Temps de session
        elapsed = datetime.now() - self.session_start
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)
        self.session_time.setText(f"Session: {minutes}:{seconds:02d}")


class QuickValidationGrid(QScrollArea):
    """Grille de validation rapide pour annotations multiples"""
    
    annotation_validated = pyqtSignal(str, bool)  # annotation_id, accepted
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumHeight(200)
        self.setMaximumHeight(400)
        
        # Widget contenu
        content_widget = QWidget()
        self.setWidget(content_widget)
        
        # Layout en grille
        self.grid_layout = QGridLayout(content_widget)
        self.grid_layout.setSpacing(5)
        
        # Données
        self.validation_items = {}
    
    def add_validation_item(self, detection_data: Dict, thumbnail_image=None):
        """Ajoute un élément à valider dans la grille"""
        item_id = detection_data['id']
        row = len(self.validation_items) // 3
        col = len(self.validation_items) % 3
        
        # Frame container
        item_frame = QFrame()
        item_frame.setFrameStyle(QFrame.StyledPanel)
        item_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
            QFrame:hover {
                border: 2px solid #4CAF50;
            }
        """)
        
        item_layout = QVBoxLayout(item_frame)
        item_layout.setContentsMargins(5, 5, 5, 5)
        
        # Thumbnail (placeholder pour l'instant)
        thumbnail = QLabel("🖼️ Image")
        thumbnail.setAlignment(Qt.AlignCenter)
        thumbnail.setStyleSheet("background-color: #e0e0e0; border: 1px solid #ccc; min-height: 80px;")
        item_layout.addWidget(thumbnail)
        
        # Informations
        info_text = f"Classe: {detection_data.get('class_name', 'Unknown')}\n"
        info_text += f"Confiance: {detection_data.get('confidence', 0):.2f}"
        info_label = QLabel(info_text)
        info_label.setFont(QFont("Arial", 8))
        info_label.setAlignment(Qt.AlignCenter)
        item_layout.addWidget(info_label)
        
        # Boutons de validation
        buttons_layout = QHBoxLayout()
        
        accept_btn = QPushButton("✅")
        accept_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        accept_btn.clicked.connect(lambda: self.validate_item(item_id, True))
        
        reject_btn = QPushButton("❌")
        reject_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        reject_btn.clicked.connect(lambda: self.validate_item(item_id, False))
        
        buttons_layout.addWidget(accept_btn)
        buttons_layout.addWidget(reject_btn)
        item_layout.addLayout(buttons_layout)
        
        # Ajout à la grille
        self.grid_layout.addWidget(item_frame, row, col)
        self.validation_items[item_id] = {
            'frame': item_frame,
            'data': detection_data,
            'validated': False
        }
    
    def validate_item(self, item_id: str, accepted: bool):
        """Valide un élément et met à jour l'interface"""
        if item_id in self.validation_items:
            item = self.validation_items[item_id]
            item['validated'] = True
            
            # Mise à jour visuelle
            if accepted:
                item['frame'].setStyleSheet("""
                    QFrame {
                        background-color: #e8f5e8;
                        border: 2px solid #4CAF50;
                        border-radius: 5px;
                        padding: 5px;
                    }
                """)
            else:
                item['frame'].setStyleSheet("""
                    QFrame {
                        background-color: #ffeaea;
                        border: 2px solid #f44336;
                        border-radius: 5px;
                        padding: 5px;
                    }
                """)
            
            # Émission du signal
            self.annotation_validated.emit(item_id, accepted)
    
    def get_validation_results(self) -> Dict:
        """Retourne les résultats de validation"""
        results = {
            'total': len(self.validation_items),
            'validated': sum(1 for item in self.validation_items.values() if item['validated']),
            'accepted': [],
            'rejected': []
        }
        
        for item_id, item in self.validation_items.items():
            if item['validated']:
                # Déterminer si accepté ou rejeté selon le style
                frame_style = item['frame'].styleSheet()
                if '#4CAF50' in frame_style:
                    results['accepted'].append(item_id)
                elif '#f44336' in frame_style:
                    results['rejected'].append(item_id)
        
        return results
    
    def clear_grid(self):
        """Vide la grille de validation"""
        for item in self.validation_items.values():
            item['frame'].deleteLater()
        self.validation_items.clear()


class RapidAnnotationInterface(QWidget):
    """
    Interface d'annotation rapide avec optimisations de productivité
    """
    
    # Signaux
    annotation_session_started = pyqtSignal(str)  # class_name
    annotation_completed = pyqtSignal(dict)       # annotation_data
    session_completed = pyqtSignal(dict)          # session_report
    batch_validation_requested = pyqtSignal(list) # detections_list
    
    def __init__(self, iface: QgisInterface, parent=None):
        super().__init__(parent)
        self.iface = iface
        
        # Configuration générale
        self.setWindowTitle("🚀 Annotation Rapide - Mode Productivité")
        self.setMinimumSize(600, 800)
        
        # État de la session
        self.current_session = None
        self.rapid_mode_active = False
        self.batch_mode_active = False
        
        # Gestionnaires
        self.quality_manager = None
        
        # Interface
        self.setup_ui()
        self.setup_shortcuts()
        
        # Timers
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_session_progress)
    
    def setup_ui(self):
        """Configuration de l'interface utilisateur"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # === HEADER - Contrôles Session ===
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_frame.setStyleSheet("background-color: #2b2b2b; border-radius: 5px; padding: 5px;")
        header_layout = QHBoxLayout(header_frame)
        
        # Informations session
        self.session_info = QLabel("Aucune session active")
        self.session_info.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.session_info)
        
        header_layout.addStretch()
        
        # Boutons de contrôle
        self.start_session_btn = QPushButton("🚀 Démarrer Session")
        self.start_session_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 5px 10px; }")
        self.start_session_btn.clicked.connect(self.start_rapid_session)
        
        self.end_session_btn = QPushButton("⏹️ Terminer")
        self.end_session_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 5px 10px; }")
        self.end_session_btn.clicked.connect(self.end_rapid_session)
        self.end_session_btn.setEnabled(False)
        
        header_layout.addWidget(self.start_session_btn)
        header_layout.addWidget(self.end_session_btn)
        
        main_layout.addWidget(header_frame)
        
        # === CONTENT - Zone principale ===
        content_splitter = QSplitter(Qt.Horizontal)
        
        # === LEFT PANEL - Contrôles et Métriques ===
        left_panel = QWidget()
        left_panel.setMaximumWidth(250)
        left_layout = QVBoxLayout(left_panel)
        
        # Indicateur de productivité
        self.productivity_indicator = ProductivityIndicator()
        left_layout.addWidget(self.productivity_indicator)
        
        # Configuration rapide
        config_group = QGroupBox("⚙️ Configuration Rapide")
        config_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        config_layout = QVBoxLayout(config_group)
        
        # Mode batch
        self.batch_mode_cb = QCheckBox("Mode Batch (zones denses)")
        self.batch_mode_cb.toggled.connect(self.toggle_batch_mode)
        config_layout.addWidget(self.batch_mode_cb)
        
        # Seuil de confiance
        config_layout.addWidget(QLabel("Seuil de confiance:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 90)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence_display)
        config_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("50%")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        config_layout.addWidget(self.confidence_label)
        
        # Target annotations
        config_layout.addWidget(QLabel("Objectif annotations:"))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(5, 200)
        self.target_spin.setValue(50)
        config_layout.addWidget(self.target_spin)
        
        left_layout.addWidget(config_group)
        
        # Raccourcis clavier
        shortcuts_group = QGroupBox("⌨️ Raccourcis")
        shortcuts_layout = QVBoxLayout(shortcuts_group)
        
        shortcuts_text = QTextEdit()
        shortcuts_text.setMaximumHeight(150)
        shortcuts_text.setReadOnly(True)
        shortcuts_text.setText("""
• Enter: Accepter suggestion IA
• Escape: Rejeter et annoter manuellement  
• Space: Suggestion suivante
• R: Activer raffinement SAM
• D: Dupliquer dernière classe
• Tab: Passer au suivant
• Ctrl+B: Mode batch
        """)
        shortcuts_text.setStyleSheet("font-size: 10px; background-color: #f9f9f9;")
        shortcuts_layout.addWidget(shortcuts_text)
        
        left_layout.addWidget(shortcuts_group)
        left_layout.addStretch()
        
        content_splitter.addWidget(left_panel)
        
        # === RIGHT PANEL - Zone de validation ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Titre zone validation
        validation_title = QLabel("🔍 Validation Rapide")
        validation_title.setFont(QFont("Arial", 12, QFont.Bold))
        validation_title.setStyleSheet("color: #333; margin-bottom: 10px;")
        right_layout.addWidget(validation_title)
        
        # Grille de validation
        self.validation_grid = QuickValidationGrid()
        self.validation_grid.annotation_validated.connect(self.on_annotation_validated)
        right_layout.addWidget(self.validation_grid)
        
        # Boutons de validation par lot
        batch_buttons = QHBoxLayout()
        
        accept_all_btn = QPushButton("✅ Accepter Tout")
        accept_all_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        accept_all_btn.clicked.connect(self.accept_all_validations)
        
        reject_all_btn = QPushButton("❌ Rejeter Tout")
        reject_all_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        reject_all_btn.clicked.connect(self.reject_all_validations)
        
        process_batch_btn = QPushButton("🚀 Traiter le Lot")
        process_batch_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        process_batch_btn.clicked.connect(self.process_validation_batch)
        
        batch_buttons.addWidget(accept_all_btn)
        batch_buttons.addWidget(reject_all_btn)
        batch_buttons.addWidget(process_batch_btn)
        right_layout.addLayout(batch_buttons)
        
        content_splitter.addWidget(right_panel)
        
        # Proportions splitter
        content_splitter.setSizes([250, 400])
        main_layout.addWidget(content_splitter)
        
        # === FOOTER - Barre de progression ===
        footer_frame = QFrame()
        footer_frame.setFrameStyle(QFrame.StyledPanel)
        footer_layout = QHBoxLayout(footer_frame)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Progression: 0/0 annotations")
        footer_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(footer_frame)
    
    def setup_shortcuts(self):
        """Configuration des raccourcis clavier"""
        # Raccourcis globaux
        QShortcut(QKeySequence("Return"), self, self.accept_current_suggestion)
        QShortcut(QKeySequence("Escape"), self, self.reject_current_suggestion)
        QShortcut(QKeySequence("Space"), self, self.next_suggestion)
        QShortcut(QKeySequence("R"), self, self.refine_with_sam)
        QShortcut(QKeySequence("D"), self, self.duplicate_last_class)
        QShortcut(QKeySequence("Tab"), self, self.next_annotation)
        QShortcut(QKeySequence("Ctrl+B"), self, self.toggle_batch_mode)
    
    def start_rapid_session(self):
        """Démarre une session d'annotation rapide"""
        from qgis.PyQt.QtWidgets import QInputDialog
        
        # Demande de la classe à annoter
        class_name, ok = QInputDialog.getText(
            self, 
            "Nouvelle Session",
            "Nom de la classe à annoter:",
            text="ma_classe"
        )
        
        if not ok or not class_name.strip():
            return
        
        # Configuration session
        self.current_session = {
            'class_name': class_name.strip(),
            'target_count': self.target_spin.value(),
            'start_time': datetime.now(),
            'annotations_completed': 0,
            'annotations_data': []
        }
        
        # Mise à jour interface
        self.session_info.setText(f"📝 Session: {class_name} (Objectif: {self.current_session['target_count']})")
        self.start_session_btn.setEnabled(False)
        self.end_session_btn.setEnabled(True)
        self.rapid_mode_active = True
        
        # Démarrage du timer
        self.session_timer.start(1000)
        
        # Mise à jour barre de progression
        self.progress_bar.setMaximum(self.current_session['target_count'])
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Progression: 0/{self.current_session['target_count']} annotations")
        
        # Initialisation gestionnaire qualité
        try:
            from ..core.annotation_quality_manager import get_quality_manager
            self.quality_manager = get_quality_manager()
            self.quality_manager.start_annotation_session(
                class_name, self.current_session['target_count']
            )
        except ImportError:
            print("⚠️ Gestionnaire de qualité non disponible")
        
        # Signal
        self.annotation_session_started.emit(class_name)
        
        print(f"🚀 Session rapide démarrée: {class_name}")
    
    def end_rapid_session(self):
        """Termine la session d'annotation rapide"""
        if not self.current_session:
            return
        
        # Arrêt du timer
        self.session_timer.stop()
        
        # Calcul métriques finales
        session_duration = (datetime.now() - self.current_session['start_time']).total_seconds()
        annotations_count = self.current_session['annotations_completed']
        
        # Rapport de session
        session_report = {
            'class_name': self.current_session['class_name'],
            'target_count': self.current_session['target_count'],
            'completed_count': annotations_count,
            'duration_minutes': session_duration / 60,
            'productivity_rating': self.get_productivity_rating(),
            'completion_rate': annotations_count / self.current_session['target_count'] * 100
        }
        
        # Finalisation gestionnaire qualité
        if self.quality_manager:
            quality_report = self.quality_manager.complete_annotation_session()
            session_report.update(quality_report)
        
        # Affichage rapport
        self.show_session_report(session_report)
        
        # Reset interface
        self.reset_interface()
        
        # Signal
        self.session_completed.emit(session_report)
    
    def update_session_progress(self):
        """Met à jour la progression de la session"""
        if not self.current_session:
            return
        
        # Mise à jour barre de progression
        completed = self.current_session['annotations_completed']
        target = self.current_session['target_count']
        
        self.progress_bar.setValue(completed)
        self.progress_bar.setFormat(f"Progression: {completed}/{target} annotations ({completed/target*100:.1f}%)")
        
        # Vérification objectif atteint
        if completed >= target:
            QMessageBox.information(
                self,
                "🎉 Objectif Atteint !",
                f"Félicitations ! Vous avez atteint votre objectif de {target} annotations.\n\n"
                f"Voulez-vous terminer la session maintenant ?"
            )
    
    def on_annotation_validated(self, annotation_id: str, accepted: bool):
        """Traite la validation d'une annotation"""
        if self.current_session:
            self.current_session['annotations_completed'] += 1
            
            # Données d'annotation pour tracking qualité
            annotation_data = {
                'id': annotation_id,
                'accepted': accepted,
                'timestamp': datetime.now(),
                'processing_time': 15.0,  # Estimation pour validation rapide
                'smart_mode_used': True,
                'user_accepted': accepted
            }
            
            self.current_session['annotations_data'].append(annotation_data)
            
            # Tracking qualité
            if self.quality_manager:
                self.quality_manager.track_annotation_event(annotation_data)
            
            # Mise à jour productivité
            self.productivity_indicator.add_annotation_time(annotation_data['processing_time'])
            
            # Signal
            self.annotation_completed.emit(annotation_data)
    
    def toggle_batch_mode(self):
        """Active/désactive le mode batch"""
        self.batch_mode_active = self.batch_mode_cb.isChecked()
        print(f"Mode batch: {'Activé' if self.batch_mode_active else 'Désactivé'}")
    
    def update_confidence_display(self):
        """Met à jour l'affichage du seuil de confiance"""
        value = self.confidence_slider.value()
        self.confidence_label.setText(f"{value}%")
    
    def get_productivity_rating(self) -> str:
        """Évalue la productivité de la session"""
        if not self.current_session:
            return "Non évaluée"
        
        duration_minutes = (datetime.now() - self.current_session['start_time']).total_seconds() / 60
        annotations_count = self.current_session['annotations_completed']
        
        if duration_minutes > 0:
            rate = annotations_count / duration_minutes  # annotations par minute
            
            if rate > 3:
                return "Excellente"
            elif rate > 2:
                return "Bonne"  
            elif rate > 1:
                return "Moyenne"
            else:
                return "À améliorer"
        
        return "Non évaluée"
    
    def show_session_report(self, report: Dict):
        """Affiche le rapport de session"""
        message = f"""
🎯 Session Terminée: {report['class_name']}

📊 Résultats:
• Annotations créées: {report['completed_count']}/{report['target_count']}
• Durée: {report['duration_minutes']:.1f} minutes
• Taux de completion: {report['completion_rate']:.1f}%
• Productivité: {report['productivity_rating']}

{self._format_quality_recommendations(report)}
        """
        
        QMessageBox.information(self, "📋 Rapport de Session", message.strip())
    
    def _format_quality_recommendations(self, report: Dict) -> str:
        """Formate les recommandations qualité pour affichage"""
        if 'recommendations' not in report or not report['recommendations']:
            return "✅ Aucune recommandation d'amélioration"
        
        text = "💡 Recommandations:\n"
        for rec in report['recommendations']:
            text += f"• {rec.message}\n"
        
        return text
    
    def reset_interface(self):
        """Remet l'interface à zéro"""
        self.current_session = None
        self.rapid_mode_active = False
        
        self.session_info.setText("Aucune session active")
        self.start_session_btn.setEnabled(True)
        self.end_session_btn.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Progression: 0/0 annotations")
        
        self.validation_grid.clear_grid()
    
    # === MÉTHODES RACCOURCIS CLAVIER ===
    
    def accept_current_suggestion(self):
        """Accepte la suggestion IA courante (Enter)"""
        print("✅ Suggestion acceptée (Enter)")
        # TODO: Implémenter logique acceptation
    
    def reject_current_suggestion(self):
        """Rejette la suggestion et passe en mode manuel (Escape)"""
        print("❌ Suggestion rejetée - Mode manuel (Escape)")
        # TODO: Implémenter logique rejet
    
    def next_suggestion(self):
        """Passe à la suggestion suivante (Space)"""
        print("⏭️ Suggestion suivante (Space)")
        # TODO: Implémenter navigation suggestions
    
    def refine_with_sam(self):
        """Active le raffinement SAM (R)"""
        print("🎨 Raffinement SAM activé (R)")
        # TODO: Implémenter raffinement SAM
    
    def duplicate_last_class(self):
        """Duplique la dernière classe utilisée (D)"""
        print("📋 Duplication dernière classe (D)")
        # TODO: Implémenter duplication classe
    
    def next_annotation(self):
        """Passe à l'annotation suivante (Tab)"""
        print("⏭️ Annotation suivante (Tab)")
        # TODO: Implémenter navigation annotations
    
    def accept_all_validations(self):
        """Accepte toutes les validations en attente"""
        print("✅ Acceptation de toutes les validations")
        # TODO: Implémenter acceptation en lot
    
    def reject_all_validations(self):
        """Rejette toutes les validations en attente"""
        print("❌ Rejet de toutes les validations")
        # TODO: Implémenter rejet en lot
    
    def process_validation_batch(self):
        """Traite le lot de validations"""
        results = self.validation_grid.get_validation_results()
        print(f"🚀 Traitement lot: {results['accepted']} acceptées, {results['rejected']} rejetées")
        
        # Émission signal pour traitement
        self.batch_validation_requested.emit(results)
        
        # Nettoyage grille
        self.validation_grid.clear_grid()