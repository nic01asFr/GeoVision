"""
Outil d'annotation interactive sur canvas QGIS

Cet outil permet à l'utilisateur de :
- Dessiner des rectangles/polygones directement sur le canvas
- Extraire des patches d'images géoréférencés
- Créer des exemples d'entraînement pour les modèles YOLO
- Fournir un feedback visuel en temps réel

Workflow :
1. L'utilisateur active l'outil depuis l'interface principale
2. Il dessine un rectangle autour d'un objet
3. Un popup s'ouvre pour caractériser l'objet
4. L'exemple est sauvegardé avec ses métadonnées spatiales
"""

from qgis.PyQt.QtCore import Qt, QRectF, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor
from qgis.PyQt.QtWidgets import QApplication, QMessageBox

from datetime import datetime
import os

from qgis.gui import QgsMapTool, QgsRubberBand, QgsMapCanvas
from qgis.core import (
    QgsWkbTypes, QgsPointXY, QgsRectangle, QgsGeometry,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsProject, QgsRasterLayer, QgsVectorLayer
)


class InteractiveAnnotationTool(QgsMapTool):
    """
    Outil d'annotation interactive pour créer des exemples d'entraînement
    
    Hérite de QgsMapTool pour intégration native avec QGIS
    PHASE 2: Support mode IA assistée révolutionnaire
    """
    
    # Signaux
    annotation_created = pyqtSignal(dict)  # Nouvel exemple créé
    tool_activated = pyqtSignal()          # Outil activé
    tool_deactivated = pyqtSignal()        # Outil désactivé
    
    def __init__(self, canvas: QgsMapCanvas, active_class: str = None):
        """
        Initialise l'outil d'annotation
        
        Args:
            canvas: Canvas QGIS
            active_class: Classe d'objet active pour l'annotation
        """
        super().__init__(canvas)
        
        self.canvas = canvas
        self.active_class = active_class
        self.annotation_mode = 'bbox'  # 'bbox' ou 'polygon'
        
        # NOUVEAU: Smart Mode
        self.smart_mode_enabled = False
        self.main_dialog = None  # Référence au dialog principal pour Smart Engine
        
        # État de l'outil
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        
        # Rubber band pour feedback visuel
        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.setup_rubber_band()
        
        # Curseur personnalisé
        self.setup_cursor()
        
        # Statistiques
        self.annotations_count = 0
        
    def setup_rubber_band(self):
        """Configure l'apparence du rubber band"""
        # Couleur et style selon la classe active
        color = QColor(255, 0, 0, 100)  # Rouge semi-transparent
        if self.active_class:
            # Couleur différente selon la classe (hash simple)
            hash_val = hash(self.active_class) % 360
            color = QColor.fromHsv(hash_val, 200, 255, 100)
        
        self.rubber_band.setColor(color)
        self.rubber_band.setFillColor(QColor(color.red(), color.green(), color.blue(), 50))
        self.rubber_band.setWidth(2)
        self.rubber_band.setLineStyle(Qt.DashLine)
    
    def setup_cursor(self):
        """Configure le curseur de l'outil"""
        # Curseur croix pour la précision
        self.setCursor(QCursor(Qt.CrossCursor))
    
    def set_active_class(self, class_name: str):
        """
        Définit la classe d'objet active
        
        Args:
            class_name: Nom de la classe à annoter
        """
        self.active_class = class_name
        self.setup_rubber_band()  # Mise à jour couleur
        
    def set_annotation_mode(self, mode: str):
        """
        Définit le mode d'annotation
        
        Args:
            mode: 'bbox' pour rectangle, 'polygon' pour polygone
        """
        if mode in ['bbox', 'polygon']:
            self.annotation_mode = mode
            # Nettoyage de l'annotation en cours si changement de mode
            if self.is_drawing:
                self.reset_annotation()
    
    def set_smart_mode(self, enabled: bool):
        """
        NOUVEAU: Active/désactive le mode Smart Assistant
        
        Args:
            enabled: True pour activer Smart Mode, False pour mode manuel
        """
        print(f"🔍 DEBUG ANNOTATION_TOOL: Réception set_smart_mode(enabled={enabled})")
        self.smart_mode_enabled = enabled
        print(f"🔍 DEBUG ANNOTATION_TOOL: self.smart_mode_enabled = {self.smart_mode_enabled}")
        
        # Mise à jour du curseur
        if enabled:
            self.setCursor(QCursor(Qt.CrossCursor))  # Curseur spécial Smart Mode
            print("🤖 Smart Mode ACTIVÉ dans annotation tool")
        else:
            self.setCursor(QCursor(Qt.CrossCursor))  # Curseur normal
            print("🖱️ Smart Mode DÉSACTIVÉ dans annotation tool")
        
        print(f"🎯 Outil annotation: {'Smart Mode' if enabled else 'Mode manuel'} activé")
    
    def set_main_dialog(self, dialog):
        """
        NOUVEAU: Définit la référence au dialog principal pour accès Smart Engine
        
        Args:
            dialog: Instance YOLOMainDialog
        """
        self.main_dialog = dialog
    
    def canvasPressEvent(self, event):
        """
        Gestion du clic initial (début d'annotation)
        
        Args:
            event: Événement souris QGIS
        """
        # Vérifications préalables
        if not self.active_class:
            QMessageBox.warning(
                self.canvas, 
                "Classe Manquante", 
                "Veuillez sélectionner une classe d'objet avant d'annoter."
            )
            return
        
        raster_layer = self.get_active_raster_layer()
        if not raster_layer:
            QMessageBox.warning(
                self.canvas,
                "Couche Manquante",
                "Veuillez charger et activer une couche raster pour l'annotation."
            )
            return
        
        # Début de l'annotation
        if event.button() == Qt.LeftButton:
            self.start_annotation(event)
        elif event.button() == Qt.RightButton:
            # Clic droit = annulation
            self.cancel_annotation()
    
    def start_annotation(self, event):
        """
        Démarre une nouvelle annotation
        
        Args:
            event: Événement souris
        """
        # Conversion en coordonnées carte
        self.start_point = self.toMapCoordinates(event.pos())
        self.is_drawing = True
        
        # Initialisation du rubber band
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        
        # Feedback utilisateur
        self.canvas.window().statusBar().showMessage(
            f"🎯 Annotation en cours - Classe: {self.active_class} | Mode: {self.annotation_mode.upper()}", 
            3000
        )
        
    def canvasMoveEvent(self, event):
        """
        Gestion du mouvement de souris (mise à jour visuelle)
        
        Args:
            event: Événement souris
        """
        if not self.is_drawing or not self.start_point:
            return
        
        # Point actuel
        current_point = self.toMapCoordinates(event.pos())
        
        # Mise à jour du rubber band selon le mode
        if self.annotation_mode == 'bbox':
            self.update_bbox_rubber_band(self.start_point, current_point)
        elif self.annotation_mode == 'polygon':
            self.update_polygon_rubber_band(current_point)
        
        # Affichage des dimensions en temps réel
        self.show_dimensions_info(self.start_point, current_point)
    
    def update_bbox_rubber_band(self, start_point, current_point):
        """
        Met à jour le rubber band pour un rectangle
        
        Args:
            start_point: Point de départ
            current_point: Point actuel
        """
        # Création du rectangle
        rect = QgsRectangle(start_point, current_point)
        
        # Points du rectangle pour le rubber band
        points = [
            QgsPointXY(rect.xMinimum(), rect.yMinimum()),
            QgsPointXY(rect.xMaximum(), rect.yMinimum()),
            QgsPointXY(rect.xMaximum(), rect.yMaximum()),
            QgsPointXY(rect.xMinimum(), rect.yMaximum()),
            QgsPointXY(rect.xMinimum(), rect.yMinimum())  # Fermeture
        ]
        
        # Mise à jour du rubber band
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        for point in points:
            self.rubber_band.addPoint(point)
    
    def update_polygon_rubber_band(self, current_point):
        """
        Met à jour le rubber band pour un polygone
        
        Args:
            current_point: Point actuel
        """
        # TODO: Implémentation polygone (pour version future)
        # Pour l'instant, utilise le mode rectangle
        self.update_bbox_rubber_band(self.start_point, current_point)
    
    def show_dimensions_info(self, start_point, current_point):
        """
        Affiche les dimensions de la sélection en cours
        
        Args:
            start_point: Point de départ
            current_point: Point actuel
        """
        # Calcul des dimensions
        width = abs(current_point.x() - start_point.x())
        height = abs(current_point.y() - start_point.y())
        
        # Conversion en mètres selon le CRS
        crs = self.canvas.mapSettings().destinationCrs()
        if crs.isGeographic():
            # CRS géographique - approximation grossière
            width_m = width * 111000  # ~111km par degré
            height_m = height * 111000
            unit = "m (approx.)"
        else:
            # CRS projeté - utilise les unités du CRS
            width_m = width
            height_m = height
            unit = crs.mapUnits().name if hasattr(crs.mapUnits(), 'name') else "unités"
        
        # Affichage dans la barre de statut
        if hasattr(self.canvas.window(), 'statusBar'):
            self.canvas.window().statusBar().showMessage(
                f"📏 Dimensions: {width_m:.1f} x {height_m:.1f} {unit} | "
                f"Classe: {self.active_class} | "
                f"Relâchez pour valider, Clic droit pour annuler"
            )
    
    def canvasReleaseEvent(self, event):
        """
        Gestion du relâchement (finalisation d'annotation)
        
        Args:
            event: Événement souris
        """
        if not self.is_drawing or not self.start_point:
            return
        
        if event.button() == Qt.LeftButton:
            self.end_point = self.toMapCoordinates(event.pos())
            self.finalize_annotation()
        
    def finalize_annotation(self):
        """Finalise l'annotation et crée l'exemple d'entraînement"""
        
        if not self.start_point or not self.end_point:
            return
        
        try:
            # Validation de la taille minimale
            min_size = 10  # pixels minimum
            screen_start = self.toCanvasCoordinates(self.start_point)
            screen_end = self.toCanvasCoordinates(self.end_point)
            
            width_px = abs(screen_end.x() - screen_start.x())
            height_px = abs(screen_end.y() - screen_start.y())
            
            if width_px < min_size or height_px < min_size:
                QMessageBox.information(
                    self.canvas,
                    "Sélection Trop Petite",
                    f"La sélection doit faire au moins {min_size}x{min_size} pixels.\n"
                    f"Taille actuelle: {width_px:.0f}x{height_px:.0f} pixels"
                )
                self.reset_annotation()
                return
            
            # Création du rectangle de sélection
            bbox_map = QgsRectangle(self.start_point, self.end_point)
            
            # Normalisation manuelle pour assurer la compatibilité
            if bbox_map.xMinimum() > bbox_map.xMaximum() or bbox_map.yMinimum() > bbox_map.yMaximum():
                bbox_map = QgsRectangle(
                    min(self.start_point.x(), self.end_point.x()),
                    min(self.start_point.y(), self.end_point.y()),
                    max(self.start_point.x(), self.end_point.x()),
                    max(self.start_point.y(), self.end_point.y())
                )
            
            # NOUVEAUTÉ PHASE 1: Extraction avec guidage sémantique optionnel
            example_data = self.create_enhanced_training_example(bbox_map)
            
            if example_data:
                # Conversion du format d'annotation pour le gestionnaire
                annotation_manager_data = self._convert_to_annotation_manager_format(example_data, bbox_map)
                
                # Stockage persistant via le gestionnaire d'annotations
                try:
                    from ..core.annotation_manager import get_annotation_manager
                    manager = get_annotation_manager()
                    annotation_id = manager.add_annotation(annotation_manager_data)
                    print(f"📝 Annotation {annotation_id} stockée avec succès")
                except Exception as e:
                    print(f"⚠️ Erreur stockage annotation : {e}")
                
                # Émission du signal avec les données
                self.annotation_created.emit(example_data)
                
                # Statistiques
                self.annotations_count += 1
                
                # Message de succès
                self.canvas.window().statusBar().showMessage(
                    f"✅ Exemple #{self.annotations_count} créé pour '{self.active_class}'", 
                    2000
                )
                
                # Feedback visuel temporaire
                self.show_success_feedback(bbox_map)
            
        except Exception as e:
            QMessageBox.critical(
                self.canvas,
                "Erreur d'Annotation",
                f"Erreur lors de la création de l'exemple:\n{str(e)}"
            )
        
        finally:
            self.reset_annotation()
    
    def create_training_example(self, bbox_map: QgsRectangle) -> dict:
        """
        Crée un exemple d'entraînement depuis une bbox
        
        Args:
            bbox_map: Rectangle en coordonnées carte
            
        Returns:
            dict: Données de l'exemple ou None si erreur
        """
        # Couche raster active
        raster_layer = self.get_active_raster_layer()
        if not raster_layer:
            return None
        
        try:
            # Extraction du patch d'image (simplifié pour l'instant)
            patch_data = self.extract_image_patch(bbox_map, raster_layer)
            
            if not patch_data:
                return None
            
            # Métadonnées de l'exemple
            example_data = {
                'class_name': self.active_class,
                'bbox_map': {
                    'xmin': bbox_map.xMinimum(),
                    'ymin': bbox_map.yMinimum(),
                    'xmax': bbox_map.xMaximum(),
                    'ymax': bbox_map.yMaximum()
                },
                'bbox_normalized': self.calculate_yolo_bbox(bbox_map, patch_data['image_bounds'], patch_data.get('metadata')),
                'image_patch': patch_data['image_array'],
                'crs': raster_layer.crs().authid(),
                'layer_name': raster_layer.name(),
                'pixel_size': self.get_layer_pixel_size(raster_layer),
                'timestamp': self.get_current_timestamp(),
                'annotation_mode': self.annotation_mode,
                'dimensions_m': {
                    'width': abs(bbox_map.width()),
                    'height': abs(bbox_map.height())
                }
            }
            
            return example_data
            
        except Exception as e:
            print(f"Erreur création exemple: {e}")
            return None
    
    def extract_image_patch(self, bbox_map: QgsRectangle, raster_layer: QgsRasterLayer) -> dict:
        """
        Extrait un patch d'image depuis une couche raster
        
        Args:
            bbox_map: Rectangle en coordonnées carte
            raster_layer: Couche raster source
            
        Returns:
            dict: Données du patch ou None
        """
        try:
            # Import de l'extracteur de patches
            from ..core.raster_extractor import RasterPatchExtractor
            
            # Création de l'extracteur (cache réutilisé)
            if not hasattr(self, '_extractor'):
                self._extractor = RasterPatchExtractor(target_size=(640, 640))
            
            # Extraction du patch avec métadonnées complètes
            patch_data = self._extractor.extract_patch(bbox_map, raster_layer)
            
            if patch_data:
                return {
                    'image_array': patch_data['image_array'],
                    'image_bounds': patch_data['extracted_bbox'],
                    'patch_size': patch_data['extraction_info']['target_size'],
                    'metadata': patch_data  # Métadonnées complètes pour le debug
                }
            else:
                print("Erreur: Extraction du patch échouée")
                return None
                
        except Exception as e:
            print(f"Erreur extraction patch: {e}")
            # Fallback vers données simulées si erreur
            import numpy as np
            patch_size = 640
            dummy_patch = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)
            
            return {
                'image_array': dummy_patch,
                'image_bounds': bbox_map,
                'patch_size': (patch_size, patch_size)
            }
    
    def calculate_yolo_bbox(self, bbox_map: QgsRectangle, image_bounds, metadata=None) -> list:
        """
        Calcule les coordonnées YOLO normalisées (0-1)
        
        Args:
            bbox_map: Rectangle objet en coordonnées carte
            image_bounds: Rectangle image complète ou dict avec coordonnées
            metadata: Métadonnées optionnelles du raster extractor
            
        Returns:
            list: [centre_x, centre_y, largeur, hauteur] normalisés
        """
        try:
            # Si nous avons les métadonnées du raster extractor, utilisons-les
            if metadata and 'yolo_bbox' in metadata:
                return metadata['yolo_bbox']
                
            # Sinon, calcul manuel
            if isinstance(image_bounds, dict):
                # Format dict {'xmin': x, 'ymin': y, 'xmax': x, 'ymax': y}
                img_xmin = image_bounds['xmin']
                img_ymin = image_bounds['ymin']
                img_width = image_bounds['xmax'] - image_bounds['xmin']
                img_height = image_bounds['ymax'] - image_bounds['ymin']
            else:
                # Format QgsRectangle
                img_xmin = image_bounds.xMinimum()
                img_ymin = image_bounds.yMinimum()
                img_width = image_bounds.width()
                img_height = image_bounds.height()
            
            # Centre de l'objet
            center_x = (bbox_map.center().x() - img_xmin) / img_width
            center_y = (bbox_map.center().y() - img_ymin) / img_height
            
            # Dimensions relatives
            width = bbox_map.width() / img_width
            height = bbox_map.height() / img_height
            
            # Clamp entre 0 et 1
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            return [center_x, center_y, width, height]
            
        except Exception as e:
            print(f"Erreur calcul YOLO bbox: {e}")
            # Coordonnées par défaut si erreur
            return [0.5, 0.5, 0.1, 0.1]
    
    def get_active_raster_layer(self) -> QgsRasterLayer:
        """
        Retourne la couche raster active
        
        Returns:
            QgsRasterLayer: Couche active ou None
        """
        # Couche active dans la légende
        active_layer = self.canvas.currentLayer()
        
        if isinstance(active_layer, QgsRasterLayer):
            return active_layer
        
        # Sinon, cherche la première couche raster visible
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsRasterLayer) and layer.isValid():
                return layer
        
        return None
    
    def get_layer_pixel_size(self, layer: QgsRasterLayer) -> dict:
        """
        Retourne la taille des pixels de la couche
        
        Args:
            layer: Couche raster
            
        Returns:
            dict: Informations sur la taille des pixels
        """
        if not layer or not layer.isValid():
            return {'x': 1.0, 'y': 1.0, 'unit': 'unknown'}
        
        extent = layer.extent()
        width = layer.width()
        height = layer.height()
        
        pixel_size_x = extent.width() / width if width > 0 else 1.0
        pixel_size_y = extent.height() / height if height > 0 else 1.0
        
        return {
            'x': pixel_size_x,
            'y': pixel_size_y,
            'unit': layer.crs().mapUnits().name if hasattr(layer.crs().mapUnits(), 'name') else 'unknown'
        }
    
    def get_current_timestamp(self) -> str:
        """Retourne le timestamp actuel"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def show_success_feedback(self, bbox_map: QgsRectangle):
        """
        Affiche un feedback visuel de succès
        
        Args:
            bbox_map: Rectangle de l'annotation
        """
        # Rubber band vert temporaire
        success_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        success_band.setColor(QColor(0, 255, 0, 150))  # Vert
        success_band.setWidth(3)
        
        # Points du rectangle
        points = [
            QgsPointXY(bbox_map.xMinimum(), bbox_map.yMinimum()),
            QgsPointXY(bbox_map.xMaximum(), bbox_map.yMinimum()),
            QgsPointXY(bbox_map.xMaximum(), bbox_map.yMaximum()),
            QgsPointXY(bbox_map.xMinimum(), bbox_map.yMaximum()),
            QgsPointXY(bbox_map.xMinimum(), bbox_map.yMinimum())
        ]
        
        for point in points:
            success_band.addPoint(point)
        
        # Suppression automatique après 1 seconde
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(1000, lambda: success_band.reset())
    
    def cancel_annotation(self):
        """Annule l'annotation en cours"""
        if self.is_drawing:
            self.canvas.window().statusBar().showMessage("❌ Annotation annulée", 1000)
        
        self.reset_annotation()
    
    def reset_annotation(self):
        """Remet à zéro l'état d'annotation"""
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.rubber_band.reset()
    
    def activate(self):
        """Active l'outil"""
        super().activate()
        self.tool_activated.emit()
        
        # Message d'activation
        if hasattr(self.canvas.window(), 'statusBar'):
            self.canvas.window().statusBar().showMessage(
                f"🎯 Outil d'annotation activé - Classe: {self.active_class or 'Non définie'}", 
                3000
            )
    
    def deactivate(self):
        """Désactive l'outil"""
        self.reset_annotation()
        
        # Nettoyage de l'extracteur
        if hasattr(self, '_extractor'):
            self._extractor.cleanup()
            del self._extractor
        
        super().deactivate()
        self.tool_deactivated.emit()
        
        # Message de désactivation
        if hasattr(self.canvas.window(), 'statusBar'):
            self.canvas.window().statusBar().showMessage("⏹️ Outil d'annotation désactivé", 2000)
    
    def get_statistics(self) -> dict:
        """
        Retourne les statistiques de l'outil
        
        Returns:
            dict: Statistiques d'utilisation
        """
        return {
            'annotations_created': self.annotations_count,
            'active_class': self.active_class,
            'annotation_mode': self.annotation_mode,
            'tool_active': self.canvas.mapTool() == self
        }
    
    def create_enhanced_training_example(self, bbox_map):
        """
        Crée un exemple d'entraînement enrichi (manuel ou Smart Mode)
        
        Args:
            bbox_map: Rectangle de sélection en coordonnées carte
            
        Returns:
            dict: Données d'annotation enrichies
        """
        try:
            # Extraction du patch raster
            raster_layer = self.get_active_raster_layer()
            if not raster_layer:
                QMessageBox.warning(self.canvas, "Erreur", "Aucune couche raster active")
                return None
            
            # NOUVEAU: Workflow Smart Mode si activé
            print(f"🔍 DEBUG: smart_mode_enabled={getattr(self, 'smart_mode_enabled', False)}, main_dialog={self.main_dialog is not None}")
            
            if getattr(self, 'smart_mode_enabled', False) and self.main_dialog:
                print(f"🎯 Smart Mode déclenché - Pipeline YOLO→SAM→Polygones")
                return self._create_smart_training_example(bbox_map, raster_layer)
            else:
                print(f"📝 Mode manuel - Annotation classique")
                return self._create_manual_training_example(bbox_map, raster_layer)
                
        except Exception as e:
            print(f"❌ Erreur création exemple d'entraînement: {e}")
            QMessageBox.critical(self.canvas, "Erreur", f"Erreur lors de la création de l'exemple:\n{str(e)}")
            return None
    
    def _create_manual_training_example(self, bbox_map, raster_layer):
        """Mode manuel classique"""
        try:
            from ..core.raster_extractor import RasterPatchExtractor
            
            extractor = RasterPatchExtractor()
            
            # Extraction du patch
            patch_data = extractor.extract_patch(bbox_map, raster_layer)
            if not patch_data:
                QMessageBox.warning(self.canvas, "Erreur", "Impossible d'extraire le patch raster")
                return None
            
            # Adaptation pour la nouvelle structure RasterPatchExtractor
            extracted_bbox = patch_data['extracted_bbox']
            patch_bounds = QgsRectangle(
                extracted_bbox['xmin'], extracted_bbox['ymin'],
                extracted_bbox['xmax'], extracted_bbox['ymax']
            )
            
            # Coordonnées relatives (utilise les coordonnées YOLO calculées)
            yolo_bbox = patch_data['yolo_bbox']  # [center_x, center_y, width, height] normalisé
            relative_bbox = {
                'x_min': yolo_bbox[0] - yolo_bbox[2]/2,  # center_x - width/2
                'y_min': yolo_bbox[1] - yolo_bbox[3]/2,  # center_y - height/2
                'x_max': yolo_bbox[0] + yolo_bbox[2]/2,  # center_x + width/2
                'y_max': yolo_bbox[1] + yolo_bbox[3]/2   # center_y + height/2
            }
            
            # Données d'annotation classiques
            annotation_data = {
                'class_name': self.active_class,
                'bbox': relative_bbox,
                'image_data': patch_data['image_array'],
                'bounds': patch_bounds,
                'crs': patch_data['crs'],
                'pixel_size': patch_data['layer_info']['pixel_size'],
                'extraction_method': 'manual_annotation'
            }
            
            return annotation_data
            
        except Exception as e:
            print(f"❌ Erreur mode manuel: {e}")
            return None
    
    def _create_smart_training_example(self, bbox_map, raster_layer):
        """NOUVEAU: Mode Smart avec détection IA assistée"""
        try:
            from ..core.raster_extractor import RasterPatchExtractor
            
            extractor = RasterPatchExtractor()
            
            # Extraction du patch
            patch_data = extractor.extract_patch(bbox_map, raster_layer)
            if not patch_data:
                QMessageBox.warning(self.canvas, "Erreur", "Impossible d'extraire le patch raster")
                return None
            
            # Adaptation pour la nouvelle structure RasterPatchExtractor
            extracted_bbox = patch_data['extracted_bbox']
            patch_bounds = QgsRectangle(
                extracted_bbox['xmin'], extracted_bbox['ymin'],
                extracted_bbox['xmax'], extracted_bbox['ymax']
            )
            
            # Calcul des coordonnées utilisateur (rectangle dessiné)
            user_rect_relative = self._map_bbox_to_patch_coordinates(bbox_map, patch_bounds, patch_data['image_array'].shape)
            
            # SMART MODE: Appel du SmartAnnotationEngine
            smart_result = self.main_dialog.get_smart_detection_result(
                user_rect_relative, 
                patch_data['image_array'], 
                self.active_class
            )
            
            if smart_result is None:
                # Fallback vers mode manuel si Smart Mode échoue
                print("⚠️ Smart Mode indisponible, fallback mode manuel")
                return self._create_manual_training_example(bbox_map, raster_layer)
            
            # Validation de l'utilisateur pour détections non auto-acceptées
            if smart_result.confidence_yolo < 0.9:  # Pas d'auto-acceptation
                validation_result = self._show_smart_validation_dialog(smart_result, patch_data['image_array'])
                
                if not validation_result:
                    # Utilisateur a rejeté la détection
                    self.reset_annotation()
                    return None
            
            # Données d'annotation enrichies Smart Mode
            annotation_data = {
                'class_name': smart_result.class_name,
                'bbox': {
                    'x_min': smart_result.bbox[0] / patch_data['image_array'].shape[1],
                    'y_min': smart_result.bbox[1] / patch_data['image_array'].shape[0], 
                    'x_max': smart_result.bbox[2] / patch_data['image_array'].shape[1],
                    'y_max': smart_result.bbox[3] / patch_data['image_array'].shape[0]
                },
                'image_data': patch_data['image_array'],
                'bounds': patch_bounds,
                'crs': patch_data['crs'],
                'pixel_size': patch_data['layer_info']['pixel_size'],
                
                # NOUVELLES MÉTADONNÉES SMART MODE
                'extraction_method': 'smart_annotation',
                'confidence_yolo': smart_result.confidence_yolo,
                'confidence_sam': smart_result.confidence_sam,
                'refinement_applied': smart_result.refinement_applied,
                'processing_time_ms': smart_result.processing_time,
                'improvement_ratio': smart_result.improvement_ratio,
                'bbox_original_user': user_rect_relative,
                'bbox_optimized_ai': smart_result.bbox,
                
                # POLYGONES SAM - NOUVEAU
                'polygon_points': getattr(smart_result, 'polygon_points', None),
                'polygon_available': getattr(smart_result, 'polygon_available', False)
            }
            
            return annotation_data
            
        except Exception as e:
            print(f"❌ Erreur Smart Mode: {e}")
            # Fallback vers mode manuel
            return self._create_manual_training_example(bbox_map, raster_layer)
    
    def _convert_to_annotation_manager_format(self, example_data, bbox_map):
        """
        Convertit les données d'annotation vers le format attendu par AnnotationManager
        
        Args:
            example_data: Données depuis _create_smart_training_example ou _create_manual_training_example
            bbox_map: Rectangle original en coordonnées carte
            
        Returns:
            dict: Format attendu par AnnotationManager.add_annotation()
        """
        try:
            # Calcul de bbox_normalized au format YOLO [center_x, center_y, width, height]
            bbox = example_data['bbox']
            center_x = (bbox['x_min'] + bbox['x_max']) / 2
            center_y = (bbox['y_min'] + bbox['y_max']) / 2
            width = bbox['x_max'] - bbox['x_min']
            height = bbox['y_max'] - bbox['y_min']
            
            # Données dans le format attendu par AnnotationManager
            manager_data = {
                'class_name': example_data['class_name'],
                'bbox_map': {
                    'xmin': bbox_map.xMinimum(),
                    'ymin': bbox_map.yMinimum(),
                    'xmax': bbox_map.xMaximum(),
                    'ymax': bbox_map.yMaximum()
                },
                'bbox_normalized': [center_x, center_y, width, height],
                'image_patch': example_data['image_data'],  # Numpy array pour sauvegarde
                'crs': example_data['crs'],
                'layer_name': self.get_active_raster_layer().name() if self.get_active_raster_layer() else "unknown",
                'pixel_size': example_data['pixel_size'],
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'extraction_method': example_data.get('extraction_method', 'unknown'),
                    'bounds': {
                        'xmin': example_data['bounds'].xMinimum(),
                        'ymin': example_data['bounds'].yMinimum(),
                        'xmax': example_data['bounds'].xMaximum(),
                        'ymax': example_data['bounds'].yMaximum()
                    }
                }
            }
            
            # Ajout des métadonnées Smart Mode si disponibles
            if 'confidence_yolo' in example_data:
                manager_data['metadata'].update({
                    'smart_mode': {
                        'confidence_yolo': example_data['confidence_yolo'],
                        'confidence_sam': example_data.get('confidence_sam'),
                        'refinement_applied': example_data.get('refinement_applied', False),
                        'processing_time_ms': example_data.get('processing_time_ms'),
                        'improvement_ratio': example_data.get('improvement_ratio'),
                        'bbox_original_user': example_data.get('bbox_original_user'),
                        'bbox_optimized_ai': example_data.get('bbox_optimized_ai')
                    }
                })
            
            return manager_data
            
        except Exception as e:
            print(f"❌ Erreur conversion données annotation: {e}")
            raise
    
    def _map_bbox_to_patch_coordinates(self, bbox_map, patch_bounds, image_shape):
        """Convertit bbox carte vers coordonnées patch image"""
        # Calcul de transformation
        patch_width = patch_bounds.xMaximum() - patch_bounds.xMinimum()
        patch_height = patch_bounds.yMaximum() - patch_bounds.yMinimum()
        
        # Coordonnées relatives dans le patch
        rel_x1 = max(0, (bbox_map.xMinimum() - patch_bounds.xMinimum()) / patch_width)
        rel_y1 = max(0, (bbox_map.yMinimum() - patch_bounds.yMinimum()) / patch_height)
        rel_x2 = min(1, (bbox_map.xMaximum() - patch_bounds.xMinimum()) / patch_width)
        rel_y2 = min(1, (bbox_map.yMaximum() - patch_bounds.yMinimum()) / patch_height)
        
        # Conversion en pixels
        img_h, img_w = image_shape[:2]
        return (
            rel_x1 * img_w,  # x1
            rel_y1 * img_h,  # y1  
            rel_x2 * img_w,  # x2
            rel_y2 * img_h   # y2
        )
    
    def _show_smart_validation_dialog(self, smart_result, image_patch):
        """Affiche le dialog de validation Smart Mode"""
        try:
            from .smart_validation_dialog import SmartValidationDialog
            
            dialog = SmartValidationDialog(smart_result, image_patch, self.canvas)
            dialog.show_at_cursor()
            
            # Dialog modal pour cette validation
            result = dialog.exec_()
            
            return result == SmartValidationDialog.Accepted
            
        except Exception as e:
            print(f"⚠️ Erreur dialog validation: {e}")
            # Fallback: demander validation simple
            from qgis.PyQt.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self.canvas,
                "Validation Smart Detection",
                f"Détection {smart_result.class_name} trouvée\n"
                f"Confiance: {smart_result.confidence_yolo:.1%}\n\n"
                f"Accepter cette détection ?",
                QMessageBox.Yes | QMessageBox.No
            )
            return reply == QMessageBox.Yes
    
    def _get_trained_model_for_class(self, class_name):
        """
        Trouve le meilleur modèle entraîné pour une classe donnée
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            str: Chemin vers le modèle ou None
        """
        try:
            from ..core.annotation_manager import get_annotation_manager
            manager = get_annotation_manager()
            trained_models = manager.get_trained_models()
            
            # Chercher un modèle pour cette classe (prendre le plus récent)
            best_model = None
            best_timestamp = None
            
            for model_info in trained_models:
                if class_name in model_info.get('classes', []):
                    model_path = model_info.get('model_path')
                    if model_path and os.path.exists(model_path):
                        # Extraire timestamp du nom du modèle
                        model_name = os.path.basename(model_path)
                        if '_Model_' in model_name:
                            timestamp_part = model_name.split('_Model_')[1].split('.')[0]
                            if best_timestamp is None or timestamp_part > best_timestamp:
                                best_model = model_path
                                best_timestamp = timestamp_part
            
            return best_model
            
        except Exception as e:
            print(f"⚠️ Erreur recherche modèle spécialisé: {e}")
            return None
    
    # ============================================================================
    # NOUVEAU: Smart Mode Auto-Detection Workflow
    # ============================================================================
    
    def start_smart_auto_detection(self):
        """
        NOUVEAU: Lance la détection automatique Smart Mode sur la zone visible
        
        Au lieu d'attendre que l'utilisateur dessine des rectangles,
        YOLO scanne automatiquement la zone visible et propose des bbox
        """
        if not self.smart_mode_enabled or not self.main_dialog:
            QMessageBox.warning(self.canvas, "Smart Mode", "Smart Mode non disponible")
            return
        
        if not self.active_class:
            QMessageBox.warning(self.canvas, "Classe Manquante", "Sélectionnez d'abord une classe à détecter")
            return
        
        try:
            # Zone visible du canvas
            extent = self.canvas.extent()
            raster_layer = self.get_active_raster_layer()
            
            if not raster_layer:
                QMessageBox.warning(self.canvas, "Couche Manquante", "Aucune couche raster active")
                return
            
            # Message d'information
            if hasattr(self.canvas.window(), 'statusBar'):
                self.canvas.window().statusBar().showMessage("🤖 Détection automatique Smart Mode en cours...", 5000)
            
            # Extraction de la zone visible
            from ..core.raster_extractor import RasterPatchExtractor
            extractor = RasterPatchExtractor()
            
            patch_data = extractor.extract_patch(extent, raster_layer)
            if not patch_data:
                QMessageBox.warning(self.canvas, "Erreur", "Impossible d'extraire la zone visible")
                return
            
            # Stratégie optimisée: Modèle générique YOLO d'abord pour orthophotos
            if self.main_dialog.smart_engine:
                print(f"🌍 Détection automatique avec modèle générique optimisé pour orthophotos")
                
                # Utilisation du modèle générique par défaut (YOLO optimisé géospatial)
                detections = self.main_dialog.smart_engine.yolo_engine.detect_objects(
                    patch_data['image_array'],
                    confidence_threshold=0.15,  # Seuil équilibré pour modèle générique sur orthophotos
                    target_classes=None  # Classes COCO standard adaptées orthophotos
                )
                
                # Fallback vers modèle spécialisé uniquement si pas de détections génériques
                if not detections:
                    specialized_model_path = self._get_trained_model_for_class(self.active_class)
                    
                    if specialized_model_path:
                        print(f"🎯 Fallback: Essai modèle spécialisé {specialized_model_path}")
                        # Chargement temporaire du modèle spécialisé
                        original_model_path = self.main_dialog.smart_engine.yolo_engine.current_model_path
                        self.main_dialog.smart_engine.yolo_engine.load_model(specialized_model_path)
                        
                        detections = self.main_dialog.smart_engine.yolo_engine.detect_objects(
                            patch_data['image_array'],
                            confidence_threshold=0.10,  # Seuil bas pour modèle spécialisé
                            target_classes=None  # Classe spécifique du modèle
                        )
                        
                        # Restauration du modèle original
                        if original_model_path:
                            self.main_dialog.smart_engine.yolo_engine.load_model(original_model_path)
                    else:
                        print(f"💡 Conseil: Créez d'abord quelques annotations manuelles pour '{self.active_class}' puis entraînez un modèle spécialisé")
                
                if detections:
                    self._process_auto_detections(detections, patch_data, extent)
                else:
                    QMessageBox.information(
                        self.canvas, 
                        "Smart Detection", 
                        f"Aucun objet de type '{self.active_class}' détecté dans la zone visible.\n\n"
                        f"Essayez :\n"
                        f"• Réduire le zoom pour inclure plus d'objets\n"
                        f"• Ajuster le seuil de confiance YOLO\n"
                        f"• Utiliser l'annotation manuelle pour créer des exemples"
                    )
            else:
                QMessageBox.warning(self.canvas, "Smart Mode", "Smart Engine non initialisé")
                
        except Exception as e:
            print(f"❌ Erreur détection automatique: {e}")
            QMessageBox.critical(self.canvas, "Erreur", f"Erreur détection automatique:\n{str(e)}")
    
    def _process_auto_detections(self, detections, patch_data, canvas_extent):
        """Traite les détections automatiques et présente à l'utilisateur"""
        try:
            from .smart_auto_detection_dialog import SmartAutoDetectionDialog
            
            # Dialog de sélection des détections
            dialog = SmartAutoDetectionDialog(
                detections, 
                patch_data['image_array'], 
                self.active_class,
                self.canvas
            )
            
            # Connexion pour traitement des détections validées
            dialog.detections_validated.connect(
                lambda validated_detections: self._save_validated_detections(
                    validated_detections, patch_data, canvas_extent
                )
            )
            
            dialog.show()
            
        except ImportError:
            # Fallback simple si dialog avancé non disponible
            self._simple_auto_detection_fallback(detections, patch_data)
    
    def _simple_auto_detection_fallback(self, detections, patch_data):
        """Fallback simple pour traitement auto-détections"""
        QMessageBox.information(
            self.canvas,
            "Détections Trouvées",
            f"🎯 {len(detections)} objets détectés automatiquement!\n\n"
            f"Fonctionnalité de validation avancée en développement.\n"
            f"Pour l'instant, utilisez l'annotation manuelle pour valider."
        )
    
    def _save_validated_detections(self, validated_detections, patch_data, canvas_extent):
        """Sauvegarde les détections validées par l'utilisateur"""
        try:
            from ..core.annotation_manager import get_annotation_manager
            manager = get_annotation_manager()
            
            saved_count = 0
            for detection in validated_detections:
                # Conversion détection → annotation (format compatible annotation_manager)
                annotation_data = {
                    'class_name': self.active_class,
                    'bbox_map': {
                        'xmin': patch_data['extracted_bbox']['xmin'] + (detection['bbox'][0] / patch_data['image_array'].shape[1]) * (patch_data['extracted_bbox']['xmax'] - patch_data['extracted_bbox']['xmin']),
                        'ymin': patch_data['extracted_bbox']['ymin'] + (detection['bbox'][1] / patch_data['image_array'].shape[0]) * (patch_data['extracted_bbox']['ymax'] - patch_data['extracted_bbox']['ymin']),
                        'xmax': patch_data['extracted_bbox']['xmin'] + (detection['bbox'][2] / patch_data['image_array'].shape[1]) * (patch_data['extracted_bbox']['xmax'] - patch_data['extracted_bbox']['xmin']),
                        'ymax': patch_data['extracted_bbox']['ymin'] + (detection['bbox'][3] / patch_data['image_array'].shape[0]) * (patch_data['extracted_bbox']['ymax'] - patch_data['extracted_bbox']['ymin'])
                    },
                    'bbox_normalized': [
                        (detection['bbox'][0] + detection['bbox'][2]) / (2 * patch_data['image_array'].shape[1]),  # center_x
                        (detection['bbox'][1] + detection['bbox'][3]) / (2 * patch_data['image_array'].shape[0]),  # center_y
                        (detection['bbox'][2] - detection['bbox'][0]) / patch_data['image_array'].shape[1],        # width
                        (detection['bbox'][3] - detection['bbox'][1]) / patch_data['image_array'].shape[0]         # height
                    ],
                    'image_patch': patch_data['image_array'],
                    'crs': patch_data['crs'],
                    'layer_name': patch_data.get('layer_name', 'Auto-Detection'),
                    'pixel_size': patch_data['layer_info']['pixel_size'],
                    'timestamp': self.get_current_timestamp(),
                    'annotation_mode': 'smart_auto_detection',
                    'metadata': {
                        'confidence_yolo': detection['confidence'],
                        'auto_detected': True,
                        'extraction_method': 'smart_auto_detection'
                    }
                }
                
                annotation_id = manager.add_annotation(annotation_data)
                saved_count += 1
                print(f"📝 Auto-détection {annotation_id} sauvegardée")
            
            QMessageBox.information(
                self.canvas,
                "Succès",
                f"✅ {saved_count} détections automatiques sauvegardées!\n\n"
                f"Ces exemples sont maintenant disponibles pour l'entraînement."
            )
            
            # Mise à jour des statistiques
            self.annotations_count += saved_count
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde auto-détections: {e}")
            QMessageBox.critical(self.canvas, "Erreur", f"Erreur sauvegarde:\n{str(e)}")