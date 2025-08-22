"""
Dialog Simple de Détails de Classe

Dialog simple et léger pour visualiser les détails d'une classe
sans complexité excessive.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QGroupBox, QTextEdit,
    QTabWidget, QWidget, QHeaderView
)
from datetime import datetime


class ClassDetailDialog(QDialog):
    """
    Dialog simple pour afficher les détails d'une classe
    
    Fonctionnalités simples :
    - Liste des exemples avec métadonnées
    - Statistiques de base
    - Actions simples (supprimer, exporter)
    """
    
    def __init__(self, class_name, examples, parent=None):
        super().__init__(parent)
        
        self.class_name = class_name
        self.examples = examples
        
        self.setWindowTitle(f"Détails - {class_name}")
        self.setMinimumSize(800, 600)
        
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        """Interface utilisateur simple"""
        layout = QVBoxLayout(self)
        
        # Header avec titre
        header = QLabel(f"📊 Détails de la classe '{self.class_name}'")
        header.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 4px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(header)
        
        # Onglets simples
        tabs = QTabWidget()
        
        # Onglet 1: Liste des exemples
        examples_tab = self.create_examples_tab()
        tabs.addTab(examples_tab, "📝 Exemples")
        
        # Onglet 2: Statistiques
        stats_tab = self.create_stats_tab()
        tabs.addTab(stats_tab, "📊 Statistiques")
        
        layout.addWidget(tabs)
        
        # Boutons
        buttons_layout = QHBoxLayout()
        
        export_btn = QPushButton("📤 Exporter Classe")
        export_btn.clicked.connect(self.export_class)
        buttons_layout.addWidget(export_btn)
        
        refresh_btn = QPushButton("🔄 Actualiser")
        refresh_btn.clicked.connect(self.refresh_data)
        buttons_layout.addWidget(refresh_btn)
        
        buttons_layout.addStretch()
        
        close_btn = QPushButton("✖️ Fermer")
        close_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(close_btn)
        
        layout.addLayout(buttons_layout)
    
    def create_examples_tab(self):
        """Onglet avec liste des exemples"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Table des exemples
        self.examples_table = QTableWidget()
        self.examples_table.setColumnCount(6)
        self.examples_table.setHorizontalHeaderLabels([
            "ID", "Date", "Confiance", "Polygone", "Source", "Dimensions"
        ])
        
        # Configuration table
        header = self.examples_table.horizontalHeader()
        header.setStretchLastSection(True)
        self.examples_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.examples_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.examples_table)
        
        # Actions sur exemples sélectionnés
        actions_layout = QHBoxLayout()
        
        delete_btn = QPushButton("🗑️ Supprimer Sélection")
        delete_btn.clicked.connect(self.delete_selected)
        actions_layout.addWidget(delete_btn)
        
        actions_layout.addStretch()
        
        layout.addLayout(actions_layout)
        
        return widget
    
    def create_stats_tab(self):
        """Onglet avec statistiques"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistiques générales
        stats_group = QGroupBox("📈 Statistiques Générales")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        
        # Informations qualité
        quality_group = QGroupBox("🎯 Qualité du Dataset")
        quality_layout = QVBoxLayout(quality_group)
        
        self.quality_text = QTextEdit()
        self.quality_text.setReadOnly(True)
        quality_layout.addWidget(self.quality_text)
        
        layout.addWidget(quality_group)
        
        return widget
    
    def load_data(self):
        """Charge les données dans l'interface"""
        self.load_examples_table()
        self.load_statistics()
    
    def load_examples_table(self):
        """Charge la table des exemples"""
        self.examples_table.setRowCount(len(self.examples))
        
        for row, example in enumerate(self.examples):
            # ID (tronqué)
            example_id = str(example.get('id', ''))[:12] + "..."
            self.examples_table.setItem(row, 0, QTableWidgetItem(example_id))
            
            # Date
            timestamp = example.get('timestamp', '')
            try:
                if timestamp:
                    date_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    date_str = date_obj.strftime('%Y-%m-%d %H:%M')
                else:
                    date_str = "N/A"
            except:
                date_str = timestamp
            
            self.examples_table.setItem(row, 1, QTableWidgetItem(date_str))
            
            # Confiance
            confidence = example.get('confidence', 1.0)
            conf_str = f"{confidence*100:.1f}%" if confidence else "N/A"
            self.examples_table.setItem(row, 2, QTableWidgetItem(conf_str))
            
            # Polygone
            has_polygon = "✅" if example.get('polygon_points') else "❌"
            self.examples_table.setItem(row, 3, QTableWidgetItem(has_polygon))
            
            # Source
            source = example.get('raster_name', example.get('layer_name', 'Unknown'))
            if len(source) > 20:
                source = source[:17] + "..."
            self.examples_table.setItem(row, 4, QTableWidgetItem(source))
            
            # Dimensions
            bbox = example.get('bbox_map', {})
            if isinstance(bbox, dict) and 'xmin' in bbox:
                width = bbox['xmax'] - bbox['xmin']
                height = bbox['ymax'] - bbox['ymin']
                dim_str = f"{width:.0f}×{height:.0f}"
            else:
                dim_str = "N/A"
            
            self.examples_table.setItem(row, 5, QTableWidgetItem(dim_str))
    
    def load_statistics(self):
        """Charge les statistiques"""
        total = len(self.examples)
        
        # Compter exemples avec polygones
        with_polygons = sum(1 for ex in self.examples if ex.get('polygon_points'))
        
        # Confiance moyenne
        confidences = [ex.get('confidence', 1.0) for ex in self.examples if ex.get('confidence')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Sources uniques
        sources = set(ex.get('raster_name', ex.get('layer_name', 'Unknown')) for ex in self.examples)
        
        # Dates
        dates = [ex.get('timestamp', '') for ex in self.examples if ex.get('timestamp')]
        dates = [d for d in dates if d]
        
        first_date = min(dates) if dates else "N/A"
        last_date = max(dates) if dates else "N/A"
        
        # Texte statistiques générales
        stats_text = f"""
📊 STATISTIQUES GÉNÉRALES

Total exemples: {total}
Avec polygones: {with_polygons} ({with_polygons/max(total,1)*100:.1f}%)
Confiance moyenne: {avg_confidence*100:.1f}%

Sources de données: {len(sources)}
Première annotation: {first_date[:16] if first_date != "N/A" else "N/A"}
Dernière annotation: {last_date[:16] if last_date != "N/A" else "N/A"}
        """
        
        self.stats_text.setPlainText(stats_text.strip())
        
        # Texte qualité
        quality_score = min(100, (total / 50) * 100)  # 50 exemples = 100%
        
        recommendations = []
        if total < 10:
            recommendations.append("❌ Nombre d'exemples insuffisant pour entraînement")
        elif total < 30:
            recommendations.append("⚠️ Augmenter le nombre d'exemples pour de meilleurs résultats")
        else:
            recommendations.append("✅ Nombre d'exemples suffisant")
        
        if with_polygons / max(total, 1) < 0.5:
            recommendations.append("⚠️ Peu d'exemples avec contours précis - activer Smart Mode")
        else:
            recommendations.append("✅ Bonne proportion d'exemples avec contours précis")
        
        if len(sources) < 2:
            recommendations.append("⚠️ Diversifier les sources de données pour robustesse")
        else:
            recommendations.append("✅ Bonne diversité des sources")
        
        quality_text = f"""
🎯 ÉVALUATION QUALITÉ

Score global: {quality_score:.0f}/100

RECOMMANDATIONS:
{chr(10).join(recommendations)}

PRÊT POUR ENTRAÎNEMENT: {'✅ OUI' if total >= 10 else '❌ NON'}
        """
        
        self.quality_text.setPlainText(quality_text.strip())
    
    def export_class(self):
        """Exporte les données de la classe"""
        # Placeholder pour export
        from qgis.PyQt.QtWidgets import QMessageBox
        QMessageBox.information(
            self, 
            "Export", 
            f"Export de la classe '{self.class_name}' avec {len(self.examples)} exemples\n\n"
            "(Fonctionnalité à implémenter)"
        )
    
    def refresh_data(self):
        """Actualise les données"""
        # Recharger depuis la base
        try:
            from ..core.annotation_manager import get_annotation_manager
            annotation_manager = get_annotation_manager()
            self.examples = annotation_manager.get_class_examples(self.class_name)
            self.load_data()
            
        except Exception as e:
            from qgis.PyQt.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Erreur", f"Erreur actualisation: {e}")
    
    def delete_selected(self):
        """Supprime les exemples sélectionnés"""
        selected_rows = set(item.row() for item in self.examples_table.selectedItems())
        
        if not selected_rows:
            from qgis.PyQt.QtWidgets import QMessageBox
            QMessageBox.information(self, "Information", "Aucun exemple sélectionné")
            return
        
        # Placeholder pour suppression
        from qgis.PyQt.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, 
            "Confirmation", 
            f"Supprimer {len(selected_rows)} exemple(s) sélectionné(s) ?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            QMessageBox.information(
                self, 
                "Suppression", 
                f"Suppression de {len(selected_rows)} exemples\n\n"
                "(Fonctionnalité à implémenter)"
            )