"""
Optimiseur d'Entraînement Géospatial - Pipeline Spécialisé pour Imagerie Géospatiale

Ce module fournit :
- Configuration optimisée pour modèles géospatiaux
- Sélection intelligente de modèles de base
- Augmentation de données géo-intelligente
- Validation spécialisée télédétection
- Métriques de performance géospatiales
"""

import os
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math
import statistics

# Conditional imports for dependencies that might not be available in all QGIS environments
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class GeospatialTrainingConfig:
    """Configuration d'entraînement optimisée pour géospatial"""
    # Paramètres de base
    epochs: int = 100
    batch_size: str = 'auto'  # Détection automatique
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    
    # Augmentations géospatiales
    mosaic: float = 0.8      # Mosaic pour contexte géospatial étendu
    mixup: float = 0.1       # Mixup léger pour diversité
    copy_paste: float = 0.1   # Copy-paste pour objets rares
    
    # Transformations géométriques (conservatrices pour géospatial)
    degrees: float = 5.0      # Rotation limitée (objets orientation-dépendants)
    translate: float = 0.1    # Translation modérée
    scale: float = 0.2        # Variation d'échelle réaliste ±20%
    shear: float = 2.0        # Cisaillement minimal
    perspective: float = 0.0  # Pas de perspective (vue aérienne)
    
    # Flips (attention aux orientations géographiques)
    fliplr: float = 0.5       # Flip horizontal OK
    flipud: float = 0.0       # Pas de flip vertical (gravité, ombres)
    
    # Augmentations colorimétriques (importantes pour conditions d'éclairage)
    hsv_h: float = 0.015      # Variation teinte légère (conditions météo)
    hsv_s: float = 0.7        # Variation saturation importante (saisonnalité)
    hsv_v: float = 0.4        # Variation luminosité modérée (heures de prise de vue)
    
    # Paramètres spécifiques géospatial
    preserve_aspect_ratio: bool = True     # Préserver ratios objets géospatiaux
    altitude_aware_scaling: bool = True    # Échelle selon altitude de prise de vue
    seasonal_color_variation: bool = True  # Variations saisonnières
    shadow_augmentation: bool = True       # Augmentation ombres/éclairage


@dataclass
class ModelCompatibilityScore:
    """Score de compatibilité d'un modèle pour une tâche géospatiale"""
    model_name: str
    overall_score: float
    resolution_match: float    # Compatibilité résolution
    domain_match: float       # Compatibilité domaine (aérien/terrestre)
    object_scale_match: float # Compatibilité échelle objets
    training_efficiency: float # Efficacité d'entraînement estimée
    recommended: bool


class GeospatialModelSelector:
    """Sélecteur intelligent de modèles de base pour géospatial"""
    
    # Base de données modèles avec caractéristiques géospatiales
    GEOSPATIAL_MODEL_DATABASE = {
        'yolo11n.pt': {
            'size_mb': 5.4,
            'optimal_resolution_range': (320, 640),
            'optimal_object_size_pixels': (20, 200),
            'specializations': ['vehicles', 'small_objects'],
            'inference_speed_ms': 2.8,
            'training_speed_factor': 1.0,
            'geospatial_adaptation': 0.6
        },
        'yolo11s.pt': {
            'size_mb': 19.0,
            'optimal_resolution_range': (416, 832),
            'optimal_object_size_pixels': (30, 400),
            'specializations': ['buildings', 'infrastructure', 'vehicles'],
            'inference_speed_ms': 6.2,
            'training_speed_factor': 0.8,
            'geospatial_adaptation': 0.75
        },
        'yolo11m.pt': {
            'size_mb': 39.0,
            'optimal_resolution_range': (512, 1024),
            'optimal_object_size_pixels': (40, 600),
            'specializations': ['complex_shapes', 'detailed_objects'],
            'inference_speed_ms': 12.1,
            'training_speed_factor': 0.6,
            'geospatial_adaptation': 0.8
        }
    }
    
    @classmethod
    def select_optimal_model(cls, training_context: Dict) -> ModelCompatibilityScore:
        """
        Sélectionne le modèle optimal selon le contexte géospatial
        
        Args:
            training_context: Contexte d'entraînement avec métadonnées
            
        Returns:
            ModelCompatibilityScore: Meilleur modèle avec score détaillé
        """
        # Extraction des caractéristiques du contexte
        avg_pixel_size = training_context.get('avg_pixel_size', 0.5)  # m/pixel
        typical_object_size = training_context.get('typical_object_size_pixels', 100)
        image_resolution = training_context.get('image_resolution', 640)
        domain_type = training_context.get('domain_type', 'aerial')  # aerial/satellite/drone
        hardware_constraints = training_context.get('hardware_constraints', 'medium')
        
        best_model = None
        best_score = 0.0
        
        for model_name, model_info in cls.GEOSPATIAL_MODEL_DATABASE.items():
            # 1. Compatibilité résolution
            res_min, res_max = model_info['optimal_resolution_range']
            if res_min <= image_resolution <= res_max:
                resolution_score = 1.0
            else:
                # Pénalité selon l'écart
                if image_resolution < res_min:
                    resolution_score = image_resolution / res_min
                else:
                    resolution_score = res_max / image_resolution
                resolution_score = max(0.3, resolution_score)
            
            # 2. Compatibilité taille objets
            obj_min, obj_max = model_info['optimal_object_size_pixels']
            if obj_min <= typical_object_size <= obj_max:
                object_scale_score = 1.0
            else:
                if typical_object_size < obj_min:
                    object_scale_score = typical_object_size / obj_min
                else:
                    object_scale_score = obj_max / typical_object_size
                object_scale_score = max(0.2, object_scale_score)
            
            # 3. Compatibilité domaine (basée sur spécialisations)
            domain_score = model_info['geospatial_adaptation']
            
            # 4. Efficacité d'entraînement (selon contraintes hardware)
            if hardware_constraints == 'low':
                efficiency_score = model_info['training_speed_factor']
            elif hardware_constraints == 'high':
                efficiency_score = 1.0  # Tous les modèles OK
            else:  # medium
                efficiency_score = (model_info['training_speed_factor'] + 1.0) / 2
            
            # Score composite
            overall_score = (
                resolution_score * 0.3 +
                object_scale_score * 0.3 +
                domain_score * 0.25 +
                efficiency_score * 0.15
            )
            
            # Création du score de compatibilité
            compatibility = ModelCompatibilityScore(
                model_name=model_name,
                overall_score=overall_score,
                resolution_match=resolution_score,
                domain_match=domain_score,
                object_scale_match=object_scale_score,
                training_efficiency=efficiency_score,
                recommended=overall_score > 0.7
            )
            
            if overall_score > best_score:
                best_score = overall_score
                best_model = compatibility
        
        return best_model or ModelCompatibilityScore(
            model_name='yolo11s.pt',  # Fallback sûr
            overall_score=0.7,
            resolution_match=0.8,
            domain_match=0.7,
            object_scale_match=0.7,
            training_efficiency=0.8,
            recommended=True
        )


class GeospatialAugmentationEngine:
    """Moteur d'augmentation spécialisé pour données géospatiales"""
    
    def __init__(self, config: GeospatialTrainingConfig):
        self.config = config
    
    def generate_geospatial_augmentations(self, image: np.ndarray, 
                                        bbox_list: List[List[float]],
                                        metadata: Dict = None) -> List[Tuple[np.ndarray, List[List[float]]]]:
        """
        Génère des augmentations géo-intelligentes
        
        Args:
            image: Image originale
            bbox_list: Liste bounding boxes format YOLO [center_x, center_y, width, height]
            metadata: Métadonnées géospatiales (altitude, saison, heure, etc.)
            
        Returns:
            List[Tuple]: Liste (image_augmentée, bboxes_augmentées)
        """
        augmented_samples = [(image, bbox_list)]  # Original toujours inclus
        
        # 1. Rotations géographiques intelligentes (multiples de 90° préférés)
        if self.config.degrees > 0:
            for angle in [90, 180, 270]:  # Orientations cardinales
                if np.random.random() < 0.3:  # 30% de chance chacune
                    aug_img, aug_bboxes = self._rotate_geospatial(image, bbox_list, angle)
                    augmented_samples.append((aug_img, aug_bboxes))
        
        # 2. Variations d'éclairage saisonnières
        if self.config.seasonal_color_variation:
            seasonal_variants = self._generate_seasonal_variants(image, metadata)
            for variant_img in seasonal_variants:
                augmented_samples.append((variant_img, bbox_list))  # Bboxes inchangées
        
        # 3. Simulation conditions météorologiques
        weather_variants = self._simulate_weather_conditions(image)
        for weather_img in weather_variants:
            augmented_samples.append((weather_img, bbox_list))
        
        # 4. Variations d'altitude (simulation échelle)
        if self.config.altitude_aware_scaling:
            scale_variants = self._simulate_altitude_variations(image, bbox_list, metadata)
            augmented_samples.extend(scale_variants)
        
        # 5. Augmentation ombres/éclairage directionnel
        if self.config.shadow_augmentation:
            shadow_variants = self._generate_shadow_variations(image)
            for shadow_img in shadow_variants:
                augmented_samples.append((shadow_img, bbox_list))
        
        return augmented_samples
    
    def _rotate_geospatial(self, image: np.ndarray, bboxes: List[List[float]], 
                          angle: float) -> Tuple[np.ndarray, List[List[float]]]:
        """Rotation géospatiale préservant l'intégrité des objets"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Matrice de rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotation image
        rotated_img = cv2.warpAffine(image, M, (w, h), borderValue=(114, 114, 114))
        
        # Rotation bboxes (transformation des centres puis recalcul)
        rotated_bboxes = []
        for bbox in bboxes:
            center_x, center_y, width, height = bbox
            
            # Conversion en coordonnées pixel absolues
            abs_x = center_x * w
            abs_y = center_y * h
            
            # Application rotation au centre
            rotated_center = np.dot(M, np.array([abs_x, abs_y, 1]))
            
            # Adaptation dimensions selon rotation
            if angle in [90, 270]:
                # Échange largeur/hauteur pour rotations 90°
                new_width, new_height = height, width
            else:
                new_width, new_height = width, height
            
            # Reconversion en coordonnées normalisées
            rotated_bboxes.append([
                rotated_center[0] / w,
                rotated_center[1] / h,
                new_width,
                new_height
            ])
        
        return rotated_img, rotated_bboxes
    
    def _generate_seasonal_variants(self, image: np.ndarray, 
                                  metadata: Dict = None) -> List[np.ndarray]:
        """Génère des variantes saisonnières"""
        variants = []
        
        # Printemps - Verdissement
        spring_img = image.copy()
        spring_img = self._adjust_vegetation_color(spring_img, hue_shift=10, saturation_boost=1.2)
        variants.append(spring_img)
        
        # Été - Haute saturation, luminosité
        summer_img = image.copy()
        enhancer = ImageEnhance.Color(Image.fromarray(summer_img))
        summer_pil = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Brightness(summer_pil)
        summer_pil = enhancer.enhance(1.1)
        variants.append(np.array(summer_pil))
        
        # Automne - Tons chauds
        autumn_img = image.copy() 
        autumn_img = self._adjust_seasonal_colors(autumn_img, season='autumn')
        variants.append(autumn_img)
        
        # Hiver - Désaturation, luminosité réduite
        winter_img = image.copy()
        enhancer = ImageEnhance.Color(Image.fromarray(winter_img))
        winter_pil = enhancer.enhance(0.7)
        enhancer = ImageEnhance.Brightness(winter_pil)
        winter_pil = enhancer.enhance(0.9)
        variants.append(np.array(winter_pil))
        
        return variants
    
    def _simulate_weather_conditions(self, image: np.ndarray) -> List[np.ndarray]:
        """Simule différentes conditions météorologiques"""
        variants = []
        
        # Brouillard/Brume - Réduction contraste
        foggy_img = image.copy().astype(np.float32)
        foggy_img = foggy_img * 0.8 + 50  # Voile blanc
        foggy_img = np.clip(foggy_img, 0, 255).astype(np.uint8)
        variants.append(foggy_img)
        
        # Temps couvert - Réduction luminosité uniforme
        cloudy_img = image.copy()
        enhancer = ImageEnhance.Brightness(Image.fromarray(cloudy_img))
        cloudy_pil = enhancer.enhance(0.85)
        variants.append(np.array(cloudy_pil))
        
        # Soleil intense - Augmentation contraste/luminosité
        sunny_img = image.copy()
        enhancer = ImageEnhance.Contrast(Image.fromarray(sunny_img))
        sunny_pil = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Brightness(sunny_pil)
        sunny_pil = enhancer.enhance(1.15)
        variants.append(np.array(sunny_pil))
        
        return variants
    
    def _simulate_altitude_variations(self, image: np.ndarray, bboxes: List[List[float]], 
                                    metadata: Dict = None) -> List[Tuple[np.ndarray, List[List[float]]]]:
        """Simule variations d'altitude de prise de vue"""
        variants = []
        
        # Altitude plus haute = objets plus petits, plus de contexte
        for scale_factor in [0.8, 1.2]:  # ±20% d'altitude
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Redimensionnement image
            if scale_factor < 1.0:
                # Zoom out - Ajouter contexte (padding)
                resized_img = cv2.resize(image, (new_w, new_h))
                # Padding centré
                pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
                padded_img = cv2.copyMakeBorder(
                    resized_img, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
                
                # Ajustement bboxes (objets plus petits, centrés)
                adjusted_bboxes = []
                for bbox in bboxes:
                    center_x, center_y, width, height = bbox
                    # Réduction taille + ajustement position
                    new_center_x = (center_x * scale_factor + pad_w / w)
                    new_center_y = (center_y * scale_factor + pad_h / h)
                    adjusted_bboxes.append([
                        new_center_x, new_center_y,
                        width * scale_factor, height * scale_factor
                    ])
                
                variants.append((padded_img, adjusted_bboxes))
            
            else:
                # Zoom in - Crop centré
                crop_h, crop_w = int(h / scale_factor), int(w / scale_factor)
                start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
                
                cropped = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
                zoomed_img = cv2.resize(cropped, (w, h))
                
                # Ajustement bboxes (objets plus grands, recadrés)
                adjusted_bboxes = []
                for bbox in bboxes:
                    center_x, center_y, width, height = bbox
                    # Recalcul position après crop + zoom
                    new_center_x = (center_x - start_w / w) * scale_factor
                    new_center_y = (center_y - start_h / h) * scale_factor
                    
                    # Vérifier si bbox reste dans l'image
                    if (0 < new_center_x < 1 and 0 < new_center_y < 1):
                        adjusted_bboxes.append([
                            new_center_x, new_center_y,
                            width * scale_factor, height * scale_factor
                        ])
                
                if adjusted_bboxes:  # Seulement si objets visibles
                    variants.append((zoomed_img, adjusted_bboxes))
        
        return variants
    
    def _generate_shadow_variations(self, image: np.ndarray) -> List[np.ndarray]:
        """Génère des variations d'ombres directionnelles"""
        variants = []
        
        # Ombres multidirectionnelles (simulation différentes heures de journée)
        for direction in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
            shadow_img = self._apply_directional_shadow(image, direction)
            variants.append(shadow_img)
        
        return variants
    
    def _apply_directional_shadow(self, image: np.ndarray, direction: str) -> np.ndarray:
        """Applique un effet d'ombre directionnelle"""
        h, w = image.shape[:2]
        shadow_img = image.copy().astype(np.float32)
        
        # Création masque gradient selon direction
        if direction == 'top-left':
            x_gradient = np.linspace(0.7, 1.0, w)
            y_gradient = np.linspace(0.7, 1.0, h)
        elif direction == 'top-right':
            x_gradient = np.linspace(1.0, 0.7, w)
            y_gradient = np.linspace(0.7, 1.0, h)
        elif direction == 'bottom-left':
            x_gradient = np.linspace(0.7, 1.0, w)
            y_gradient = np.linspace(1.0, 0.7, h)
        else:  # bottom-right
            x_gradient = np.linspace(1.0, 0.7, w)
            y_gradient = np.linspace(1.0, 0.7, h)
        
        # Application gradient
        X, Y = np.meshgrid(x_gradient, y_gradient)
        shadow_mask = (X * Y)[:, :, np.newaxis]
        
        shadow_img *= shadow_mask
        shadow_img = np.clip(shadow_img, 0, 255).astype(np.uint8)
        
        return shadow_img
    
    def _adjust_vegetation_color(self, image: np.ndarray, 
                               hue_shift: int = 0, saturation_boost: float = 1.0) -> np.ndarray:
        """Ajuste les couleurs de végétation (zones vertes)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Masque végétation (tons verts)
        green_mask = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85)  # Teinte verte
        green_mask &= (hsv[:, :, 1] > 50)  # Saturation minimale
        
        # Application ajustements sur végétation uniquement
        hsv[green_mask, 0] = np.clip(hsv[green_mask, 0] + hue_shift, 0, 179)
        hsv[green_mask, 1] = np.clip(hsv[green_mask, 1] * saturation_boost, 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def _adjust_seasonal_colors(self, image: np.ndarray, season: str) -> np.ndarray:
        """Ajuste les couleurs selon la saison"""
        if season == 'autumn':
            # Tons chauds pour l'automne
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Décalage vers tons orangés/rouges pour végétation
            vegetation_mask = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85)
            hsv[vegetation_mask, 0] = np.clip(hsv[vegetation_mask, 0] - 20, 0, 179)  # Vers rouge/orange
            hsv[vegetation_mask, 1] = np.clip(hsv[vegetation_mask, 1] * 1.3, 0, 255)  # Plus saturé
            
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return image


class GeospatialTrainingOptimizer:
    """
    Optimiseur d'entraînement spécialisé pour données géospatiales
    """
    
    def __init__(self, annotation_manager=None, yolo_engine=None):
        self.annotation_manager = annotation_manager
        self.yolo_engine = yolo_engine
        self.model_selector = GeospatialModelSelector()
        
        # Check dependencies at initialization
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        missing_deps = []
        
        if not NUMPY_AVAILABLE:
            missing_deps.append("numpy")
        if not CV2_AVAILABLE:
            missing_deps.append("opencv-python")
        if not PIL_AVAILABLE:
            missing_deps.append("Pillow")
        
        if missing_deps:
            print(f"⚠️ GeospatialTrainingOptimizer: Dépendances manquantes: {', '.join(missing_deps)}")
            print("📚 Fonctionnalités limitées - augmentations géospatiales désactivées")
            self.dependencies_available = False
        else:
            self.dependencies_available = True
        
    def prepare_optimized_training(self, class_name: str, 
                                 custom_config: Dict = None) -> Dict:
        """
        Prépare un entraînement optimisé pour données géospatiales
        
        Args:
            class_name: Nom de la classe à entraîner
            custom_config: Configuration personnalisée
            
        Returns:
            Dict: Configuration d'entraînement optimisée
        """
        # Analyse du contexte de données
        training_context = self._analyze_training_context(class_name)
        
        # Sélection du modèle optimal
        optimal_model = self.model_selector.select_optimal_model(training_context)
        
        # Configuration géospatiale
        geo_config = GeospatialTrainingConfig()
        if custom_config:
            # Mise à jour avec paramètres personnalisés
            for key, value in custom_config.items():
                if hasattr(geo_config, key):
                    setattr(geo_config, key, value)
        
        # Génération dataset avec augmentations géospatiales
        dataset_config = self._prepare_geospatial_dataset(class_name, geo_config)
        
        # Configuration d'entraînement finale
        training_config = {
            'base_model': optimal_model.model_name,
            'model_compatibility_score': optimal_model.overall_score,
            'dataset_config': dataset_config,
            'training_params': {
                'epochs': geo_config.epochs,
                'batch_size': geo_config.batch_size,
                'lr0': geo_config.learning_rate,
                'weight_decay': geo_config.weight_decay,
                
                # Augmentations géospatiales
                'mosaic': geo_config.mosaic,
                'mixup': geo_config.mixup,
                'copy_paste': geo_config.copy_paste,
                
                # Transformations
                'degrees': geo_config.degrees,
                'translate': geo_config.translate,
                'scale': geo_config.scale,
                'shear': geo_config.shear,
                'perspective': geo_config.perspective,
                'fliplr': geo_config.fliplr,
                'flipud': geo_config.flipud,
                
                # Couleurs
                'hsv_h': geo_config.hsv_h,
                'hsv_s': geo_config.hsv_s,
                'hsv_v': geo_config.hsv_v
            },
            'optimization_notes': self._generate_optimization_notes(optimal_model, training_context)
        }
        
        return training_config
    
    def _analyze_training_context(self, class_name: str) -> Dict:
        """Analyse le contexte des données d'entraînement"""
        if not self.annotation_manager:
            return self._default_training_context()
        
        # Récupération des annotations
        annotations = self.annotation_manager.get_class_annotations(class_name)
        if not annotations:
            return self._default_training_context()
        
        # Analyse des métadonnées géospatiales
        pixel_sizes = []
        object_sizes = []
        crs_distribution = {}
        
        for annotation in annotations:
            # Taille de pixel
            if annotation.pixel_size and 'x' in annotation.pixel_size:
                avg_pixel_size = (annotation.pixel_size['x'] + annotation.pixel_size['y']) / 2
                pixel_sizes.append(avg_pixel_size)
            
            # Taille d'objet (estimation depuis bbox)
            bbox = annotation.bbox_map
            width_m = (bbox['xmax'] - bbox['xmin'])
            height_m = (bbox['ymax'] - bbox['ymin'])
            object_area = width_m * height_m
            object_sizes.append(math.sqrt(object_area))  # Dimension caractéristique
            
            # Distribution CRS
            crs = annotation.crs
            crs_distribution[crs] = crs_distribution.get(crs, 0) + 1
        
        # Calculs statistiques
        avg_pixel_size = statistics.mean(pixel_sizes) if pixel_sizes else 0.5
        avg_object_size_m = statistics.mean(object_sizes) if object_sizes else 10.0
        
        # Estimation taille objet en pixels (approximation)
        avg_object_size_pixels = avg_object_size_m / avg_pixel_size if avg_pixel_size > 0 else 100
        
        # Détermination type de domaine
        if avg_pixel_size < 0.1:  # < 10cm/pixel
            domain_type = 'drone'
        elif avg_pixel_size < 1.0:  # < 1m/pixel
            domain_type = 'aerial'
        else:
            domain_type = 'satellite'
        
        # Contraintes hardware (estimation basée sur nombre d'annotations)
        if len(annotations) > 100:
            hardware_constraints = 'high'  # Dataset large = besoin ressources
        elif len(annotations) > 30:
            hardware_constraints = 'medium'
        else:
            hardware_constraints = 'low'
        
        return {
            'avg_pixel_size': avg_pixel_size,
            'typical_object_size_pixels': avg_object_size_pixels,
            'image_resolution': 640,  # Standard YOLO
            'domain_type': domain_type,
            'hardware_constraints': hardware_constraints,
            'sample_count': len(annotations),
            'crs_diversity': len(crs_distribution),
            'geographic_diversity_score': min(1.0, len(crs_distribution) / 5.0)
        }
    
    def _default_training_context(self) -> Dict:
        """Contexte par défaut si pas de données disponibles"""
        return {
            'avg_pixel_size': 0.5,  # 50cm/pixel (orthophoto typique)
            'typical_object_size_pixels': 100,
            'image_resolution': 640,
            'domain_type': 'aerial',
            'hardware_constraints': 'medium',
            'sample_count': 20,
            'crs_diversity': 1,
            'geographic_diversity_score': 0.2
        }
    
    def _prepare_geospatial_dataset(self, class_name: str, 
                                  config: GeospatialTrainingConfig) -> Dict:
        """Prépare le dataset avec augmentations géospatiales"""
        if not self.annotation_manager:
            raise ValueError("AnnotationManager requis pour génération dataset")
        
        # Utilisation du générateur de dataset existant avec optimisations
        try:
            from .yolo_dataset_generator import YOLODatasetGenerator
            
            dataset_generator = YOLODatasetGenerator(self.annotation_manager)
            
            # Configuration spécialisée géospatial
            dataset_config = dataset_generator.generate_yolo_dataset(
                [class_name],
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.1,
                augment_factor=3,  # Plus d'augmentations pour géospatial
                min_examples_per_class=10
            )
            
            # Ajout métadonnées géospatiales
            dataset_config['geospatial_optimized'] = True
            dataset_config['augmentation_config'] = config
            
            return dataset_config
            
        except ImportError:
            raise ImportError("YOLODatasetGenerator non disponible")
    
    def _generate_optimization_notes(self, model_score: ModelCompatibilityScore, 
                                   context: Dict) -> List[str]:
        """Génère des notes d'optimisation pour l'utilisateur"""
        notes = []
        
        # Notes sur le modèle sélectionné
        notes.append(f"Modèle sélectionné: {model_score.model_name} (score: {model_score.overall_score:.2f})")
        
        if model_score.resolution_match < 0.8:
            notes.append(f"⚠️ Résolution non optimale pour ce modèle. Considérer redimensionnement images.")
        
        if model_score.object_scale_match < 0.7:
            notes.append(f"⚠️ Taille d'objets non optimale. Ajuster le niveau de zoom ou modèle.")
        
        # Notes sur le contexte
        if context['sample_count'] < 30:
            notes.append(f"ℹ️ Dataset petit ({context['sample_count']} exemples). Augmentation intensive recommandée.")
        
        if context['geographic_diversity_score'] < 0.5:
            notes.append(f"ℹ️ Diversité géographique limitée. Varier les zones d'annotation si possible.")
        
        # Recommandations domaine
        domain_tips = {
            'drone': "Drone détecté: Privilégier détails fins, attention aux ombres portées",
            'aerial': "Aérien détecté: Équilibrer contexte global et précision locale",
            'satellite': "Satellite détecté: Privilégier robustesse aux variations atmosphériques"
        }
        
        if context['domain_type'] in domain_tips:
            notes.append(f"💡 {domain_tips[context['domain_type']]}")
        
        return notes
    
    def validate_trained_model(self, model_path: str, test_data: List) -> Dict:
        """
        Valide un modèle entraîné avec métriques géospatiales
        
        Args:
            model_path: Chemin vers le modèle entraîné
            test_data: Données de test
            
        Returns:
            Dict: Métriques de validation géospatiales
        """
        if not self.yolo_engine:
            raise ValueError("YOLOEngine requis pour validation")
        
        # Chargement du modèle
        self.yolo_engine.load_model(model_path)
        
        # Métriques standard
        validation_results = {
            'model_path': model_path,
            'test_samples': len(test_data),
            'geospatial_metrics': {}
        }
        
        # Métriques géospatiales spécialisées
        scale_errors = []
        positional_errors = []
        confidence_distribution = []
        
        for test_sample in test_data:
            # Détection sur échantillon test
            detections = self.yolo_engine.detect_objects(
                test_sample['image'],
                confidence_threshold=0.3
            )
            
            # Comparaison avec vérité terrain
            for detection in detections:
                confidence_distribution.append(detection['confidence'])
                
                # Analyse erreurs géospatiales si vérité terrain disponible
                if 'ground_truth' in test_sample:
                    # Calcul erreurs de positionnement géospatial
                    positional_error = self._calculate_geospatial_error(
                        detection, test_sample['ground_truth']
                    )
                    positional_errors.append(positional_error)
        
        # Agrégation métriques géospatiales
        if positional_errors:
            validation_results['geospatial_metrics'].update({
                'avg_positional_error_m': statistics.mean(positional_errors),
                'max_positional_error_m': max(positional_errors),
                'positional_accuracy_90_percentile': self._percentile(positional_errors, 90)
            })
        
        if confidence_distribution:
            validation_results['geospatial_metrics'].update({
                'avg_confidence': statistics.mean(confidence_distribution),
                'confidence_std': statistics.stdev(confidence_distribution) if len(confidence_distribution) > 1 else 0,
                'high_confidence_rate': sum(1 for c in confidence_distribution if c > 0.8) / len(confidence_distribution)
            })
        
        # Recommandations finales
        validation_results['recommendations'] = self._generate_model_recommendations(
            validation_results['geospatial_metrics']
        )
        
        return validation_results
    
    def _calculate_geospatial_error(self, detection: Dict, ground_truth: Dict) -> float:
        """Calcule l'erreur de positionnement géospatial en mètres"""
        # Simulation - À implémenter avec vraies coordonnées géospatiales
        bbox_det = detection['bbox']
        bbox_gt = ground_truth['bbox']
        
        # Calcul distance centre à centre (en pixels puis conversion mètres)
        center_det = [(bbox_det[0] + bbox_det[2]) / 2, (bbox_det[1] + bbox_det[3]) / 2]
        center_gt = [(bbox_gt[0] + bbox_gt[2]) / 2, (bbox_gt[1] + bbox_gt[3]) / 2]
        
        pixel_distance = math.sqrt(
            (center_det[0] - center_gt[0]) ** 2 + 
            (center_det[1] - center_gt[1]) ** 2
        )
        
        # Conversion approximative en mètres (nécessite métadonnées pixel_size)
        pixel_size_m = 0.5  # Approximation
        return pixel_distance * pixel_size_m
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcule un percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile / 100.0 * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _generate_model_recommendations(self, metrics: Dict) -> List[str]:
        """Génère des recommandations basées sur les métriques"""
        recommendations = []
        
        if 'avg_positional_error_m' in metrics:
            error = metrics['avg_positional_error_m']
            if error > 2.0:
                recommendations.append("Précision géospatiale faible. Augmenter la qualité des annotations de référence.")
            elif error > 1.0:
                recommendations.append("Précision géospatiale moyenne. Considérer plus d'exemples d'entraînement.")
            else:
                recommendations.append("Excellente précision géospatiale !")
        
        if 'avg_confidence' in metrics:
            confidence = metrics['avg_confidence']
            if confidence < 0.6:
                recommendations.append("Confiance faible. Modèle sous-entraîné ou données insuffisantes.")
            elif confidence > 0.9:
                recommendations.append("Confiance très élevée. Risque de sur-entraînement à vérifier.")
        
        return recommendations or ["Modèle prêt pour utilisation opérationnelle"]