"""
Générateur de Datasets YOLO pour l'entraînement

Ce module transforme les annotations géospatiales stockées en datasets YOLO standards :
- Conversion des métadonnées en format YOLO (images + labels .txt)
- Augmentation de données pour améliorer la robustesse
- Division train/validation/test avec stratification
- Génération des fichiers de configuration YOLO
- Validation et contrôle qualité des datasets

Workflow :
1. Extraction des annotations depuis AnnotationManager
2. Conversion au format YOLO standard
3. Augmentation de données (rotation, contraste, etc.)
4. Division stratifiée des données
5. Génération des fichiers YAML de configuration
6. Validation finale du dataset
"""

import os
import yaml
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Imports conditionnels pour compatibilité QGIS
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ NumPy non disponible")

try:
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL non disponible")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV non disponible")

from qgis.PyQt.QtCore import QObject, pyqtSignal

from .annotation_manager import get_annotation_manager, AnnotationExample


@dataclass
class DatasetSplit:
    """Configuration de division du dataset"""
    train_ratio: float = 0.7      # 70% entraînement
    val_ratio: float = 0.2        # 20% validation  
    test_ratio: float = 0.1       # 10% test
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Les ratios doivent sommer à 1.0, obtenu: {total}")


@dataclass  
class AugmentationConfig:
    """Configuration d'augmentation des données"""
    enabled: bool = True
    rotation_range: int = 15          # Rotation ±15°
    brightness_range: float = 0.2     # Luminosité ±20%
    contrast_range: float = 0.2       # Contraste ±20%
    saturation_range: float = 0.15    # Saturation ±15%
    horizontal_flip: bool = True      # Miroir horizontal
    vertical_flip: bool = False       # Miroir vertical (rare pour géospatial)
    noise_probability: float = 0.1    # 10% chance de bruit
    blur_probability: float = 0.05    # 5% chance de flou
    augmentation_factor: int = 3      # Multiplicateur d'exemples


@dataclass
class YOLODatasetInfo:
    """Informations sur un dataset YOLO généré"""
    name: str
    classes: List[str]
    class_mapping: Dict[str, int]  # nom_classe -> index
    total_images: int
    train_images: int
    val_images: int
    test_images: int
    dataset_path: Path
    config_path: Path
    created_at: datetime
    augmentation_used: bool
    quality_metrics: Dict[str, float]


class YOLODatasetGenerator(QObject):
    """
    Générateur de datasets YOLO depuis les annotations géospatiales
    
    Transforme les données du AnnotationManager en datasets YOLO standards
    avec augmentation optionnelle et division train/val/test.
    """
    
    # Signaux
    dataset_generation_started = pyqtSignal(str)  # Début de génération
    dataset_generation_progress = pyqtSignal(str, int, int)  # Classe, actuel, total
    dataset_generation_completed = pyqtSignal(str, dict)  # Dataset créé
    dataset_generation_error = pyqtSignal(str, str)  # Erreur
    
    def __init__(self, annotation_manager=None):
        """
        Initialise le générateur
        
        Args:
            annotation_manager: Gestionnaire d'annotations (None = auto)
        """
        super().__init__()
        
        # Vérification des dépendances
        missing_deps = []
        if not NUMPY_AVAILABLE:
            missing_deps.append('numpy')
        if not PIL_AVAILABLE:
            missing_deps.append('PIL')
        
        if missing_deps:
            raise ImportError(f"Dépendances manquantes pour YOLODatasetGenerator: {', '.join(missing_deps)}")
        
        self.annotation_manager = annotation_manager or get_annotation_manager()
        self.datasets_dir = self.annotation_manager.exports_dir
        
        # Configuration par défaut
        self.split_config = DatasetSplit()
        self.augmentation_config = AugmentationConfig()
        
        # Désactivation de l'augmentation si OpenCV manque
        if not CV2_AVAILABLE:
            self.augmentation_config.enabled = False
            print("⚠️ OpenCV manquant - augmentation désactivée")
        
        print(f"📊 YOLODatasetGenerator initialisé : {self.datasets_dir}")
    
    def generate_dataset(self, 
                        dataset_name: str,
                        selected_classes: List[str] = None,
                        split_config: DatasetSplit = None,
                        augmentation_config: AugmentationConfig = None) -> YOLODatasetInfo:
        """
        Génère un dataset YOLO complet
        
        Args:
            dataset_name: Nom du dataset
            selected_classes: Classes à inclure (None = toutes)
            split_config: Configuration de division (None = défaut)
            augmentation_config: Configuration d'augmentation (None = défaut)
            
        Returns:
            YOLODatasetInfo: Informations sur le dataset généré
        """
        try:
            self.dataset_generation_started.emit(dataset_name)
            
            # Configuration par défaut si nécessaire
            split_config = split_config or self.split_config
            augmentation_config = augmentation_config or self.augmentation_config
            
            # Création du répertoire dataset
            dataset_path = self.datasets_dir / dataset_name
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
            dataset_path.mkdir(parents=True)
            
            # Récupération des annotations
            all_classes = selected_classes or self.annotation_manager.get_all_classes()
            annotations_by_class = {}
            total_annotations = 0
            
            for class_name in all_classes:
                annotations = self.annotation_manager.get_class_annotations(class_name)
                if annotations:  # Ignore les classes vides
                    annotations_by_class[class_name] = annotations
                    total_annotations += len(annotations)
            
            if not annotations_by_class:
                raise ValueError("Aucune annotation trouvée pour les classes sélectionnées")
            
            print(f"📊 Génération dataset '{dataset_name}' avec {total_annotations} annotations")
            
            # Création du mapping des classes
            class_names = sorted(annotations_by_class.keys())
            class_mapping = {name: idx for idx, name in enumerate(class_names)}
            
            # Structure du dataset YOLO
            self._create_dataset_structure(dataset_path)
            
            # Génération des images et labels
            dataset_info = self._generate_images_and_labels(
                dataset_path, annotations_by_class, class_mapping,
                split_config, augmentation_config
            )
            
            # NOUVEAU: Détection présence de polygones pour configuration YOLO
            has_polygons = self._detect_polygons_in_annotations(annotations_by_class)
            
            # Génération du fichier de configuration YOLO
            config_path = self._generate_yolo_config(
                dataset_path, class_names, dataset_name, has_polygons
            )
            
            # Validation du dataset
            quality_metrics = self._validate_dataset(dataset_path)
            
            # Informations finales
            final_info = YOLODatasetInfo(
                name=dataset_name,
                classes=class_names,
                class_mapping=class_mapping,
                total_images=dataset_info['total_images'],
                train_images=dataset_info['train_images'],
                val_images=dataset_info['val_images'], 
                test_images=dataset_info['test_images'],
                dataset_path=dataset_path,
                config_path=config_path,
                created_at=datetime.now(),
                augmentation_used=augmentation_config.enabled,
                quality_metrics=quality_metrics
            )
            
            # Sauvegarde des métadonnées
            self._save_dataset_metadata(dataset_path, final_info)
            
            self.dataset_generation_completed.emit(dataset_name, final_info.__dict__)
            
            print(f"✅ Dataset '{dataset_name}' généré avec succès : {final_info.total_images} images")
            return final_info
            
        except Exception as e:
            error_msg = f"Erreur génération dataset : {e}"
            print(f"❌ {error_msg}")
            self.dataset_generation_error.emit(dataset_name, error_msg)
            raise
    
    def _create_dataset_structure(self, dataset_path: Path):
        """Crée la structure de répertoires YOLO standard"""
        for split in ['train', 'val', 'test']:
            (dataset_path / split / 'images').mkdir(parents=True)
            (dataset_path / split / 'labels').mkdir(parents=True)
    
    def _generate_images_and_labels(self, 
                                   dataset_path: Path,
                                   annotations_by_class: Dict[str, List[AnnotationExample]],
                                   class_mapping: Dict[str, int],
                                   split_config: DatasetSplit,
                                   augmentation_config: AugmentationConfig) -> Dict[str, int]:
        """
        Génère les images et labels pour chaque split
        
        Returns:
            Dict avec les comptes par split
        """
        image_counter = 0
        split_counts = {'train_images': 0, 'val_images': 0, 'test_images': 0}
        
        for class_name, annotations in annotations_by_class.items():
            class_idx = class_mapping[class_name]
            
            print(f"📝 Traitement classe '{class_name}' : {len(annotations)} annotations")
            
            # Division stratifiée par classe
            train_annotations, val_annotations, test_annotations = self._split_annotations(
                annotations, split_config
            )
            
            splits = [
                ('train', train_annotations),
                ('val', val_annotations), 
                ('test', test_annotations)
            ]
            
            for split_name, split_annotations in splits:
                if not split_annotations:
                    continue
                    
                for annotation in split_annotations:
                    # Image originale
                    image_path, label_path = self._process_annotation(
                        annotation, class_idx, split_name, dataset_path, image_counter
                    )
                    image_counter += 1
                    split_counts[f'{split_name}_images'] += 1
                    
                    # Augmentation pour l'entraînement seulement
                    if (split_name == 'train' and augmentation_config.enabled and 
                        augmentation_config.augmentation_factor > 1):
                        
                        for aug_idx in range(augmentation_config.augmentation_factor - 1):
                            aug_image_path, aug_label_path = self._process_annotation_with_augmentation(
                                annotation, class_idx, split_name, dataset_path, 
                                image_counter, aug_idx, augmentation_config
                            )
                            image_counter += 1
                            split_counts[f'{split_name}_images'] += 1
                
                self.dataset_generation_progress.emit(
                    class_name, 
                    list(annotations_by_class.keys()).index(class_name) + 1,
                    len(annotations_by_class)
                )
        
        split_counts['total_images'] = image_counter
        return split_counts
    
    def _split_annotations(self, annotations: List[AnnotationExample], 
                          split_config: DatasetSplit) -> Tuple[List, List, List]:
        """Division stratifiée des annotations"""
        # Mélange aléatoire avec seed fixe pour reproductibilité
        shuffled = annotations.copy()
        random.seed(42)
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * split_config.train_ratio)
        n_val = int(n_total * split_config.val_ratio)
        
        train_annotations = shuffled[:n_train]
        val_annotations = shuffled[n_train:n_train + n_val]
        test_annotations = shuffled[n_train + n_val:]
        
        return train_annotations, val_annotations, test_annotations
    
    def _process_annotation(self, annotation: AnnotationExample, class_idx: int,
                           split_name: str, dataset_path: Path, image_counter: int) -> Tuple[Path, Path]:
        """Traite une annotation individuelle"""
        # Noms des fichiers
        image_filename = f"{image_counter:06d}.jpg"  
        label_filename = f"{image_counter:06d}.txt"
        
        # Chemins de destination
        image_path = dataset_path / split_name / 'images' / image_filename
        label_path = dataset_path / split_name / 'labels' / label_filename
        
        # Copie et conversion de l'image
        self._copy_and_process_image(annotation.image_patch, image_path)
        
        # Génération du label YOLO
        self._generate_yolo_label(annotation, class_idx, label_path)
        
        return image_path, label_path
    
    def _process_annotation_with_augmentation(self, annotation: AnnotationExample, class_idx: int,
                                            split_name: str, dataset_path: Path, 
                                            image_counter: int, aug_idx: int,
                                            augmentation_config: AugmentationConfig) -> Tuple[Path, Path]:
        """Traite une annotation avec augmentation"""
        # Noms des fichiers avec suffixe d'augmentation
        image_filename = f"{image_counter:06d}_aug{aug_idx}.jpg"
        label_filename = f"{image_counter:06d}_aug{aug_idx}.txt"
        
        # Chemins de destination
        image_path = dataset_path / split_name / 'images' / image_filename
        label_path = dataset_path / split_name / 'labels' / label_filename
        
        # Chargement de l'image originale
        original_image = Image.open(annotation.image_patch)
        
        # Application des augmentations
        augmented_image, bbox_adjusted = self._apply_augmentations(
            original_image, annotation.bbox_normalized, augmentation_config
        )
        
        # Sauvegarde de l'image augmentée
        augmented_image.save(str(image_path), format='JPEG', quality=90)
        
        # Génération du label avec bbox ajustée
        adjusted_annotation = annotation
        adjusted_annotation.bbox_normalized = bbox_adjusted
        self._generate_yolo_label(adjusted_annotation, class_idx, label_path)
        
        return image_path, label_path
    
    def _copy_and_process_image(self, source_path: str, dest_path: Path):
        """Copie et traite une image"""
        try:
            # Chargement et conversion
            image = Image.open(source_path)
            
            # Conversion RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionnement si nécessaire (YOLO fonctionne mieux avec 640x640)
            if image.size != (640, 640):
                image = image.resize((640, 640), Image.Resampling.LANCZOS)
            
            # Sauvegarde
            image.save(str(dest_path), format='JPEG', quality=90)
            
        except Exception as e:
            print(f"⚠️ Erreur traitement image {source_path}: {e}")
            # Fallback : copie directe
            shutil.copy2(source_path, dest_path)
    
    def _generate_yolo_label(self, annotation: AnnotationExample, class_idx: int, label_path: Path):
        """Génère un fichier label YOLO avec support contours précis"""
        
        # NOUVEAU: Utiliser polygone précis si disponible (YOLO v8 segmentation format)
        if annotation.polygon_available and annotation.polygon_points:
            print(f"🔺 Génération label polygonal: {annotation.vertex_count} vertices")
            self._generate_polygon_yolo_label(annotation, class_idx, label_path)
        else:
            # Fallback: format bbox traditionnel
            print("📦 Génération label bbox traditionnel")
            self._generate_bbox_yolo_label(annotation, class_idx, label_path)
    
    def _generate_polygon_yolo_label(self, annotation: AnnotationExample, class_idx: int, label_path: Path):
        """Génère un label YOLO au format polygone (segmentation)"""
        points = annotation.polygon_points
        
        # Conversion points pixels vers coordonnées normalisées [0-1]
        # Les points sont actuellement en pixels dans le patch, il faut normaliser
        # en fonction de la taille de l'image patch
        
        # Récupération taille image patch
        image_path = Path(annotation.image_patch)
        if image_path.exists() and PIL_AVAILABLE:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        else:
            # Fallback vers bbox si image inaccessible
            print("⚠️ Image patch inaccessible, fallback vers bbox")
            self._generate_bbox_yolo_label(annotation, class_idx, label_path)
            return
        
        # Normalisation des points polygonaux
        normalized_points = []
        for point in points[:-1]:  # Exclure le dernier point (fermeture)
            norm_x = point[0] / img_width
            norm_y = point[1] / img_height
            
            # Clamp dans [0, 1]
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            normalized_points.extend([norm_x, norm_y])
        
        # Format YOLO segmentation : class_id x1 y1 x2 y2 x3 y3 ... 
        points_str = ' '.join([f"{p:.6f}" for p in normalized_points])
        label_line = f"{class_idx} {points_str}\n"
        
        with open(label_path, 'w') as f:
            f.write(label_line)
            
        print(f"✅ Label polygonal généré: {len(normalized_points)//2} points, classe {class_idx}")
    
    def _generate_bbox_yolo_label(self, annotation: AnnotationExample, class_idx: int, label_path: Path):
        """Génère un label YOLO au format bbox traditionnel"""
        bbox = annotation.bbox_normalized  # [center_x, center_y, width, height]
        
        # Format YOLO : class_id center_x center_y width height
        label_line = f"{class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
        
        with open(label_path, 'w') as f:
            f.write(label_line)
    
    def _detect_polygons_in_annotations(self, annotations_by_class: Dict[str, List]) -> bool:
        """Détecte si au moins une annotation contient des polygones précis"""
        polygon_count = 0
        total_count = 0
        
        for class_name, annotations in annotations_by_class.items():
            for annotation in annotations:
                total_count += 1
                if annotation.polygon_available and annotation.polygon_points:
                    polygon_count += 1
        
        percentage = (polygon_count / total_count * 100) if total_count > 0 else 0
        
        print(f"🔺 Analyse dataset: {polygon_count}/{total_count} annotations avec contours précis ({percentage:.1f}%)")
        
        # Utiliser mode segmentation si >50% des annotations ont des polygones
        return polygon_count > 0 and percentage >= 50.0
    
    def _apply_augmentations(self, image: Image.Image, bbox: List[float], 
                           config: AugmentationConfig) -> Tuple[Image.Image, List[float]]:
        """
        Applique les augmentations à une image et ajuste la bbox
        
        Returns:
            Tuple[Image augmentée, bbox ajustée]
        """
        augmented_image = image.copy()
        adjusted_bbox = bbox.copy()
        
        # Rotation (avec ajustement de bbox complexe, simplifié ici)
        if config.rotation_range > 0:
            angle = random.uniform(-config.rotation_range, config.rotation_range)
            if abs(angle) > 1:  # Rotation significative
                augmented_image = augmented_image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
                # Note: Ajustement exact de bbox après rotation nécessiterait des calculs trigonométriques
        
        # Miroir horizontal
        if config.horizontal_flip and random.random() > 0.5:
            augmented_image = ImageOps.mirror(augmented_image)
            # Ajustement bbox : inversion de center_x
            adjusted_bbox[0] = 1.0 - adjusted_bbox[0]
        
        # Miroir vertical (rare pour géospatial)
        if config.vertical_flip and random.random() > 0.5:
            augmented_image = ImageOps.flip(augmented_image)
            # Ajustement bbox : inversion de center_y  
            adjusted_bbox[1] = 1.0 - adjusted_bbox[1]
        
        # Ajustements de couleur/luminosité
        if config.brightness_range > 0:
            factor = 1.0 + random.uniform(-config.brightness_range, config.brightness_range)
            enhancer = ImageEnhance.Brightness(augmented_image)
            augmented_image = enhancer.enhance(factor)
        
        if config.contrast_range > 0:
            factor = 1.0 + random.uniform(-config.contrast_range, config.contrast_range)
            enhancer = ImageEnhance.Contrast(augmented_image)
            augmented_image = enhancer.enhance(factor)
        
        if config.saturation_range > 0:
            factor = 1.0 + random.uniform(-config.saturation_range, config.saturation_range)
            enhancer = ImageEnhance.Color(augmented_image)
            augmented_image = enhancer.enhance(factor)
        
        # Bruit gaussien
        if random.random() < config.noise_probability:
            np_image = np.array(augmented_image)
            noise = np.random.normal(0, 5, np_image.shape).astype(np.uint8)
            np_image = np.clip(np_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            augmented_image = Image.fromarray(np_image)
        
        # Flou léger
        if random.random() < config.blur_probability:
            augmented_image = augmented_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return augmented_image, adjusted_bbox
    
    def _generate_yolo_config(self, dataset_path: Path, class_names: List[str], 
                             dataset_name: str, has_polygons: bool = False) -> Path:
        """Génère le fichier de configuration YOLO (.yaml) avec support segmentation"""
        config_path = dataset_path / f"{dataset_name}.yaml"
        
        # Configuration YOLO avec détection du type de dataset
        config = {
            'path': str(dataset_path),  # Chemin racine du dataset
            'train': str(dataset_path / 'train' / 'images'),
            'val': str(dataset_path / 'val' / 'images'),
            'test': str(dataset_path / 'test' / 'images'),
            'nc': len(class_names),  # Nombre de classes
            'names': class_names     # Noms des classes
        }
        
        # NOUVEAU: Indication du type de tâche selon les données
        if has_polygons:
            config['task'] = 'segment'  # Segmentation d'instances
            print("🔺 Configuration YOLO: Mode SEGMENTATION activé")
        else:
            config['task'] = 'detect'   # Détection d'objets classique
            print("📦 Configuration YOLO: Mode DÉTECTION classique")
        
        # Sauvegarde
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"📄 Configuration YOLO sauvegardée : {config_path}")
        return config_path
    
    def _validate_dataset(self, dataset_path: Path) -> Dict[str, float]:
        """
        Valide un dataset et calcule des métriques qualité
        
        Returns:
            Dict des métriques de qualité
        """
        metrics = {
            'images_with_labels_ratio': 0.0,
            'avg_objects_per_image': 0.0,
            'class_balance_score': 0.0,
            'corrupted_files_ratio': 0.0
        }
        
        try:
            total_images = 0
            total_labels = 0
            total_objects = 0
            corrupted_files = 0
            class_counts = {}
            
            for split in ['train', 'val', 'test']:
                images_dir = dataset_path / split / 'images'
                labels_dir = dataset_path / split / 'labels'
                
                if not images_dir.exists():
                    continue
                
                for image_file in images_dir.glob('*.jpg'):
                    total_images += 1
                    label_file = labels_dir / f"{image_file.stem}.txt"
                    
                    # Vérification intégrité image
                    try:
                        with Image.open(image_file) as img:
                            img.verify()
                    except:
                        corrupted_files += 1
                        continue
                    
                    # Vérification label
                    if label_file.exists():
                        total_labels += 1
                        try:
                            with open(label_file, 'r') as f:
                                lines = f.readlines()
                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                        total_objects += 1
                        except:
                            corrupted_files += 1
            
            # Calcul des métriques
            if total_images > 0:
                metrics['images_with_labels_ratio'] = total_labels / total_images
                metrics['corrupted_files_ratio'] = corrupted_files / total_images
                
            if total_labels > 0:
                metrics['avg_objects_per_image'] = total_objects / total_labels
                
            # Score d'équilibrage des classes (entropie normalisée)
            if class_counts:
                class_probs = np.array(list(class_counts.values())) / sum(class_counts.values())
                entropy = -np.sum(class_probs * np.log(class_probs + 1e-8))
                max_entropy = np.log(len(class_counts))
                metrics['class_balance_score'] = entropy / max_entropy if max_entropy > 0 else 1.0
                
        except Exception as e:
            print(f"⚠️ Erreur validation dataset : {e}")
        
        return metrics
    
    def _save_dataset_metadata(self, dataset_path: Path, dataset_info: YOLODatasetInfo):
        """Sauvegarde les métadonnées du dataset"""
        metadata_path = dataset_path / 'dataset_info.json'
        
        # Conversion en dict sérialisable
        metadata = {
            'name': dataset_info.name,
            'classes': dataset_info.classes,
            'class_mapping': dataset_info.class_mapping,
            'total_images': dataset_info.total_images,
            'train_images': dataset_info.train_images,
            'val_images': dataset_info.val_images,
            'test_images': dataset_info.test_images,
            'dataset_path': str(dataset_info.dataset_path),
            'config_path': str(dataset_info.config_path),
            'created_at': dataset_info.created_at.isoformat(),
            'augmentation_used': dataset_info.augmentation_used,
            'quality_metrics': dataset_info.quality_metrics
        }
        
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """Liste les datasets disponibles"""
        datasets = []
        
        for dataset_dir in self.datasets_dir.glob('*'):
            if dataset_dir.is_dir():
                metadata_path = dataset_dir / 'dataset_info.json'
                if metadata_path.exists():
                    try:
                        import json
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        datasets.append(metadata)
                    except Exception as e:
                        print(f"⚠️ Erreur lecture métadonnées {dataset_dir}: {e}")
        
        return sorted(datasets, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """Supprime un dataset"""
        try:
            dataset_path = self.datasets_dir / dataset_name
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                print(f"🗑️ Dataset '{dataset_name}' supprimé")
                return True
            return False
        except Exception as e:
            print(f"❌ Erreur suppression dataset '{dataset_name}': {e}")
            return False


# Fonction d'accès global
def create_yolo_dataset(dataset_name: str, 
                       selected_classes: List[str] = None,
                       augmentation_enabled: bool = True) -> YOLODatasetInfo:
    """
    Fonction simplifiée pour créer un dataset YOLO
    
    Args:
        dataset_name: Nom du dataset
        selected_classes: Classes à inclure (None = toutes)
        augmentation_enabled: Activer l'augmentation
        
    Returns:
        YOLODatasetInfo: Informations sur le dataset créé
    """
    generator = YOLODatasetGenerator()
    
    # Configuration d'augmentation
    aug_config = AugmentationConfig(enabled=augmentation_enabled)
    
    return generator.generate_dataset(
        dataset_name=dataset_name,
        selected_classes=selected_classes,
        augmentation_config=aug_config
    )