"""
Extracteur de patches d'images géoréférencés

Ce module gère l'extraction précise de portions d'images depuis les couches raster QGIS
en préservant le géoréférencement et en optimisant pour l'entraînement YOLO.

Fonctionnalités :
- Extraction de patches depuis couches raster QGIS
- Gestion des transformations de coordonnées (CRS)
- Redimensionnement intelligent pour YOLO (640x640)
- Préservation des métadonnées géospatiales
- Support multi-formats (GeoTIFF, ECW, JPEG, etc.)
- Optimisation mémoire pour gros rasters
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
import cv2

from qgis.core import (
    QgsRasterLayer, QgsRectangle, QgsPointXY,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsProject, QgsRasterDataProvider, QgsRasterBlock,
    QgsMapSettings, QgsMapRendererParallelJob,
    QgsRasterRenderer, QgsContrastEnhancement
)

from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QImage, QPainter


class RasterPatchExtractor:
    """
    Extracteur optimisé de patches d'images géoréférencés
    
    Gère l'extraction de portions d'images depuis les couches raster QGIS
    avec préservation du géoréférencement et optimisation pour YOLO.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialise l'extracteur
        
        Args:
            target_size: Taille cible des patches (largeur, hauteur)
        """
        self.target_size = target_size
        self.temp_dir = tempfile.mkdtemp(prefix="yolo_patches_")
        
        # Paramètres d'extraction
        self.default_padding = 0.1  # 10% de padding autour de l'objet
        self.max_memory_mb = 512    # Limite mémoire pour les patches
        
        # Cache pour optimiser les accès répétés
        self.cache = {}
        self.cache_max_size = 10
        
    def extract_patch(self, bbox_map: QgsRectangle, raster_layer: QgsRasterLayer, 
                     padding_percent: float = None, 
                     preserve_aspect: bool = True) -> Optional[Dict[str, Any]]:
        """
        Extrait un patch d'image depuis une couche raster
        
        Args:
            bbox_map: Rectangle en coordonnées carte (bbox de l'objet)
            raster_layer: Couche raster source
            padding_percent: Pourcentage de padding (défaut: 0.1 = 10%)
            preserve_aspect: Préserver le ratio d'aspect original
            
        Returns:
            dict: Données du patch avec métadonnées ou None si erreur
        """
        if not raster_layer or not raster_layer.isValid():
            print("Erreur: Couche raster invalide")
            return None
            
        if not bbox_map or bbox_map.isEmpty():
            print("Erreur: Rectangle de sélection vide")
            return None
        
        try:
            # Ajout du padding
            padding = padding_percent if padding_percent is not None else self.default_padding
            padded_bbox = self._add_padding(bbox_map, padding)
            
            # Gestion plus souple de l'emprise de la couche
            layer_extent = raster_layer.extent()
            if not layer_extent.intersects(padded_bbox):
                # Essayer avec la bbox originale sans padding
                if not layer_extent.intersects(bbox_map):
                    print("Erreur: La sélection est en dehors de l'emprise de la couche")
                    return None
                else:
                    print("⚠️ Padding réduit pour rester dans l'emprise de la couche")
                    padded_bbox = bbox_map
            
            # Intersection avec l'emprise de la couche (toujours faire l'intersection)
            clipped_bbox = padded_bbox.intersect(layer_extent)
            
            # Vérification que la bbox clippée n'est pas vide
            if clipped_bbox.isEmpty() or clipped_bbox.width() <= 0 or clipped_bbox.height() <= 0:
                print("Erreur: La zone d'intersection est vide ou trop petite")
                return None
            
            # Extraction du patch
            patch_data = self._extract_raster_data(clipped_bbox, raster_layer)
            
            if patch_data is None:
                return None
            
            # Redimensionnement vers la taille cible
            resized_patch = self._resize_patch(
                patch_data['image_array'], 
                preserve_aspect=preserve_aspect
            )
            
            # Calcul des coordonnées YOLO normalisées
            yolo_bbox = self._calculate_yolo_coordinates(bbox_map, clipped_bbox)
            
            # Métadonnées complètes
            result = {
                'image_array': resized_patch,
                'original_bbox': {
                    'xmin': bbox_map.xMinimum(),
                    'ymin': bbox_map.yMinimum(),
                    'xmax': bbox_map.xMaximum(),
                    'ymax': bbox_map.yMaximum()
                },
                'extracted_bbox': {
                    'xmin': clipped_bbox.xMinimum(),
                    'ymin': clipped_bbox.yMinimum(),
                    'xmax': clipped_bbox.xMaximum(),
                    'ymax': clipped_bbox.yMaximum()
                },
                'yolo_bbox': yolo_bbox,  # [center_x, center_y, width, height] normalisé
                'crs': raster_layer.crs().authid(),
                'layer_info': {
                    'name': raster_layer.name(),
                    'source': raster_layer.source(),
                    'pixel_size': self._get_pixel_size(raster_layer),
                    'data_type': raster_layer.dataProvider().dataType(1)
                },
                'extraction_info': {
                    'target_size': self.target_size,
                    'original_size': patch_data['original_size'],
                    'padding_used': padding,
                    'aspect_preserved': preserve_aspect,
                    'extraction_method': patch_data['method']
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Erreur lors de l'extraction du patch: {e}")
            return None
    
    def _add_padding(self, bbox: QgsRectangle, padding_percent: float) -> QgsRectangle:
        """
        Ajoute du padding autour d'une bbox
        
        Args:
            bbox: Rectangle original
            padding_percent: Pourcentage de padding (0.1 = 10%)
            
        Returns:
            QgsRectangle: Rectangle avec padding
        """
        width = bbox.width()
        height = bbox.height()
        
        padding_x = width * padding_percent
        padding_y = height * padding_percent
        
        return QgsRectangle(
            bbox.xMinimum() - padding_x,
            bbox.yMinimum() - padding_y,
            bbox.xMaximum() + padding_x,
            bbox.yMaximum() + padding_y
        )
    
    def _extract_raster_data(self, bbox: QgsRectangle, 
                           raster_layer: QgsRasterLayer) -> Optional[Dict[str, Any]]:
        """
        Extrait les données raster pour une bbox donnée
        
        Args:
            bbox: Rectangle d'extraction
            raster_layer: Couche raster
            
        Returns:
            dict: Données extraites ou None
        """
        try:
            # Méthode 1: Utilisation du rendu QGIS (recommandée)
            result = self._extract_via_qgs_renderer(bbox, raster_layer)
            if result:
                result['method'] = 'qgs_renderer'
                return result
            
            # Méthode 2: Accès direct aux données (fallback)
            print("Fallback vers l'accès direct aux données...")
            result = self._extract_via_data_provider(bbox, raster_layer)
            if result:
                result['method'] = 'data_provider'
                return result
            
            print("Échec de toutes les méthodes d'extraction")
            return None
            
        except Exception as e:
            print(f"Erreur extraction données raster: {e}")
            return None
    
    def _extract_via_qgs_renderer(self, bbox: QgsRectangle, 
                                 raster_layer: QgsRasterLayer) -> Optional[Dict[str, Any]]:
        """
        Extraction via le système de rendu QGIS (recommandé)
        
        Args:
            bbox: Rectangle d'extraction
            raster_layer: Couche raster
            
        Returns:
            dict: Données extraites ou None
        """
        try:
            # Configuration du rendu
            map_settings = QgsMapSettings()
            map_settings.setLayers([raster_layer])
            map_settings.setExtent(bbox)
            map_settings.setDestinationCrs(raster_layer.crs())
            
            # Calcul de la taille optimale
            pixel_size = self._get_pixel_size(raster_layer)
            width_pixels = max(int(bbox.width() / pixel_size['x']), 100)
            height_pixels = max(int(bbox.height() / pixel_size['y']), 100)
            
            # Limitation pour éviter les images trop grandes
            max_pixels = 2048
            if width_pixels > max_pixels or height_pixels > max_pixels:
                ratio = min(max_pixels / width_pixels, max_pixels / height_pixels)
                width_pixels = int(width_pixels * ratio)
                height_pixels = int(height_pixels * ratio)
            
            map_settings.setOutputSize(QSize(width_pixels, height_pixels))
            
            # Rendu de l'image
            image = QImage(width_pixels, height_pixels, QImage.Format_RGB32)
            image.fill(0)  # Fond noir
            
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Rendu de la couche
            job = QgsMapRendererParallelJob(map_settings)
            job.start()
            job.waitForFinished()
            
            if job.isActive():
                job.cancel()
                painter.end()
                return None
            
            # Récupération de l'image rendue
            rendered_image = job.renderedImage()
            painter.end()
            
            if rendered_image.isNull():
                return None
            
            # Conversion en array numpy
            image_array = self._qimage_to_numpy(rendered_image)
            
            return {
                'image_array': image_array,
                'original_size': (width_pixels, height_pixels)
            }
            
        except Exception as e:
            print(f"Erreur rendu QGIS: {e}")
            return None
    
    def _extract_via_data_provider(self, bbox: QgsRectangle, 
                                  raster_layer: QgsRasterLayer) -> Optional[Dict[str, Any]]:
        """
        Extraction via accès direct aux données (fallback)
        
        Args:
            bbox: Rectangle d'extraction
            raster_layer: Couche raster
            
        Returns:
            dict: Données extraites ou None
        """
        try:
            provider = raster_layer.dataProvider()
            if not provider or not provider.isValid():
                return None
            
            # Transformation des coordonnées si nécessaire
            layer_crs = raster_layer.crs()
            project_crs = QgsProject.instance().crs()
            transform = None
            
            if layer_crs != project_crs:
                transform = QgsCoordinateTransform(project_crs, layer_crs, QgsProject.instance())
                bbox = transform.transform(bbox)
            
            # Calcul des indices de pixels
            extent = raster_layer.extent()
            width = raster_layer.width()
            height = raster_layer.height()
            
            pixel_size_x = extent.width() / width
            pixel_size_y = extent.height() / height
            
            # Conversion bbox -> indices pixels
            left_px = max(0, int((bbox.xMinimum() - extent.xMinimum()) / pixel_size_x))
            right_px = min(width, int((bbox.xMaximum() - extent.xMinimum()) / pixel_size_x))
            top_px = max(0, int((extent.yMaximum() - bbox.yMaximum()) / pixel_size_y))
            bottom_px = min(height, int((extent.yMaximum() - bbox.yMinimum()) / pixel_size_y))
            
            # Vérification de la validité
            if left_px >= right_px or top_px >= bottom_px:
                return None
            
            patch_width = right_px - left_px
            patch_height = bottom_px - top_px
            
            # Extraction des données pour chaque bande
            band_count = raster_layer.bandCount()
            bands_data = []
            
            for band_num in range(1, min(band_count + 1, 4)):  # Max 3 bandes RGB + alpha
                block = provider.block(band_num, bbox, patch_width, patch_height)
                if block.isValid():
                    # Conversion en array numpy
                    band_array = np.frombuffer(
                        block.data(), 
                        dtype=np.uint8 if block.dataType() <= 1 else np.float32
                    ).reshape((patch_height, patch_width))
                    bands_data.append(band_array)
            
            if not bands_data:
                return None
            
            # Composition de l'image
            if len(bands_data) == 1:
                # Grayscale -> RGB
                image_array = np.stack([bands_data[0]] * 3, axis=2)
            elif len(bands_data) >= 3:
                # RGB
                image_array = np.stack(bands_data[:3], axis=2)
            else:
                return None
            
            # Normalisation si nécessaire
            if image_array.dtype != np.uint8:
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            return {
                'image_array': image_array,
                'original_size': (patch_width, patch_height)
            }
            
        except Exception as e:
            print(f"Erreur accès direct données: {e}")
            return None
    
    def _qimage_to_numpy(self, qimage: QImage) -> np.ndarray:
        """
        Convertit une QImage en array numpy
        
        Args:
            qimage: Image Qt
            
        Returns:
            np.ndarray: Array numpy RGB
        """
        # Conversion au format RGB32 si nécessaire
        if qimage.format() != QImage.Format_RGB32:
            qimage = qimage.convertToFormat(QImage.Format_RGB32)
        
        width = qimage.width()
        height = qimage.height()
        
        # Extraction des données
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)  # 4 bytes par pixel (RGBA)
        
        # Conversion en array numpy
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        
        # Conversion BGRA -> RGB
        rgb_arr = arr[:, :, [2, 1, 0]]  # Inversion B et R
        
        return rgb_arr
    
    def _resize_patch(self, image_array: np.ndarray, 
                     preserve_aspect: bool = True) -> np.ndarray:
        """
        Redimensionne un patch vers la taille cible
        
        Args:
            image_array: Array d'image original
            preserve_aspect: Préserver le ratio d'aspect
            
        Returns:
            np.ndarray: Image redimensionnée
        """
        if image_array is None or image_array.size == 0:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        target_w, target_h = self.target_size
        current_h, current_w = image_array.shape[:2]
        
        if preserve_aspect:
            # Calcul du ratio pour préserver l'aspect
            ratio = min(target_w / current_w, target_h / current_h)
            new_w = int(current_w * ratio)
            new_h = int(current_h * ratio)
            
            # Redimensionnement
            resized = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Padding pour atteindre la taille cible
            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Centrage de l'image
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            result[start_y:start_y + new_h, start_x:start_x + new_w] = resized
            
            return result
        else:
            # Redimensionnement direct (peut déformer)
            return cv2.resize(image_array, self.target_size, interpolation=cv2.INTER_AREA)
    
    def _calculate_yolo_coordinates(self, object_bbox: QgsRectangle, 
                                  image_bbox: QgsRectangle) -> list:
        """
        Calcule les coordonnées YOLO normalisées (0-1)
        
        Args:
            object_bbox: Rectangle de l'objet
            image_bbox: Rectangle de l'image complète
            
        Returns:
            list: [center_x, center_y, width, height] normalisés
        """
        # Dimensions de l'image
        img_width = image_bbox.width()
        img_height = image_bbox.height()
        
        # Position relative de l'objet dans l'image
        rel_x = (object_bbox.center().x() - image_bbox.xMinimum()) / img_width
        rel_y = (object_bbox.center().y() - image_bbox.yMinimum()) / img_height
        
        # Taille relative de l'objet
        rel_width = object_bbox.width() / img_width
        rel_height = object_bbox.height() / img_height
        
        # Clamp entre 0 et 1
        rel_x = max(0, min(1, rel_x))
        rel_y = max(0, min(1, rel_y))
        rel_width = max(0, min(1, rel_width))
        rel_height = max(0, min(1, rel_height))
        
        return [rel_x, rel_y, rel_width, rel_height]
    
    def _get_pixel_size(self, raster_layer: QgsRasterLayer) -> Dict[str, float]:
        """
        Calcule la taille des pixels d'une couche raster
        
        Args:
            raster_layer: Couche raster
            
        Returns:
            dict: Taille des pixels en X et Y
        """
        if not raster_layer or not raster_layer.isValid():
            return {'x': 1.0, 'y': 1.0}
        
        extent = raster_layer.extent()
        width = raster_layer.width()
        height = raster_layer.height()
        
        pixel_size_x = extent.width() / width if width > 0 else 1.0
        pixel_size_y = extent.height() / height if height > 0 else 1.0
        
        return {'x': pixel_size_x, 'y': pixel_size_y}
    
    def save_patch(self, patch_data: Dict[str, Any], 
                   output_path: str, format: str = 'PNG') -> bool:
        """
        Sauvegarde un patch sur disque
        
        Args:
            patch_data: Données du patch
            output_path: Chemin de sortie
            format: Format d'image ('PNG', 'JPEG', etc.)
            
        Returns:
            bool: Succès de la sauvegarde
        """
        try:
            image_array = patch_data.get('image_array')
            if image_array is None:
                return False
            
            # Conversion en PIL Image
            if image_array.dtype != np.uint8:
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_array)
            pil_image.save(output_path, format=format)
            
            return os.path.exists(output_path)
            
        except Exception as e:
            print(f"Erreur sauvegarde patch: {e}")
            return False
    
    def cleanup(self):
        """Nettoie les ressources temporaires"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Erreur nettoyage: {e}")
    
    def __del__(self):
        """Destructeur - nettoyage automatique"""
        self.cleanup()