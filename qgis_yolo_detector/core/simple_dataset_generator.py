"""
Générateur de Datasets YOLO Simplifié (sans dépendances externes)

Version allégée qui fonctionne avec les bibliothèques disponibles dans QGIS
pour créer des datasets YOLO basiques sans augmentation avancée.

Fonctionnalités :
- Conversion annotations → format YOLO
- Division train/val/test simple
- Génération des fichiers YAML
- Validation basique des datasets
"""

import os
import json
import yaml
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from qgis.PyQt.QtCore import QObject, pyqtSignal

from .annotation_manager import get_annotation_manager, AnnotationExample


@dataclass
class SimpleDatasetSplit:
    """Configuration de division du dataset"""
    train_ratio: float = 0.7      # 70% entraînement
    val_ratio: float = 0.2        # 20% validation  
    test_ratio: float = 0.1       # 10% test
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Les ratios doivent sommer à 1.0, obtenu: {total}")


@dataclass
class SimpleDatasetInfo:
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


class SimpleYOLODatasetGenerator(QObject):
    """
    Générateur simplifié de datasets YOLO
    
    Fonctionne sans dépendances externes (PIL, OpenCV, NumPy)
    en utilisant seulement les capacités QGIS natives.
    """
    
    # Signaux
    dataset_generation_started = pyqtSignal(str)  # Début de génération
    dataset_generation_progress = pyqtSignal(str, int, int)  # Classe, actuel, total
    dataset_generation_completed = pyqtSignal(str, dict)  # Dataset créé
    dataset_generation_error = pyqtSignal(str, str)  # Erreur
    
    def __init__(self, annotation_manager=None):
        """
        Initialise le générateur simplifié
        
        Args:
            annotation_manager: Gestionnaire d'annotations (None = auto)
        """
        super().__init__()
        
        self.annotation_manager = annotation_manager or get_annotation_manager()
        self.datasets_dir = self.annotation_manager.exports_dir
        
        # Configuration par défaut
        self.split_config = SimpleDatasetSplit()
        
        print(f"📊 SimpleYOLODatasetGenerator initialisé : {self.datasets_dir}")
    
    def generate_dataset(self, 
                        dataset_name: str,
                        selected_classes: List[str] = None,
                        split_config: SimpleDatasetSplit = None) -> SimpleDatasetInfo:
        """
        Génère un dataset YOLO simple
        
        Args:
            dataset_name: Nom du dataset
            selected_classes: Classes à inclure (None = toutes)
            split_config: Configuration de division (None = défaut)
            
        Returns:
            SimpleDatasetInfo: Informations sur le dataset généré
        """
        try:
            self.dataset_generation_started.emit(dataset_name)
            
            # Configuration par défaut si nécessaire
            split_config = split_config or self.split_config
            
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
                dataset_path, annotations_by_class, class_mapping, split_config
            )
            
            # Génération du fichier de configuration YOLO
            config_path = self._generate_yolo_config(
                dataset_path, class_names, dataset_name
            )
            
            # Informations finales
            final_info = SimpleDatasetInfo(
                name=dataset_name,
                classes=class_names,
                class_mapping=class_mapping,
                total_images=dataset_info['total_images'],
                train_images=dataset_info['train_images'],
                val_images=dataset_info['val_images'], 
                test_images=dataset_info['test_images'],
                dataset_path=dataset_path,
                config_path=config_path,
                created_at=datetime.now()
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
                                   split_config: SimpleDatasetSplit) -> Dict[str, int]:
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
                    # Traitement de l'annotation
                    image_path, label_path = self._process_annotation(
                        annotation, class_idx, split_name, dataset_path, image_counter
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
                          split_config: SimpleDatasetSplit) -> Tuple[List, List, List]:
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
        
        # Copie de l'image
        self._copy_image(annotation.image_patch, image_path)
        
        # Génération du label YOLO
        self._generate_yolo_label(annotation, class_idx, label_path)
        
        return image_path, label_path
    
    def _copy_image(self, source_path: str, dest_path: Path):
        """Copie une image"""
        try:
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            print(f"⚠️ Erreur copie image {source_path}: {e}")
    
    def _generate_yolo_label(self, annotation: AnnotationExample, class_idx: int, label_path: Path):
        """Génère un fichier label YOLO"""
        bbox = annotation.bbox_normalized  # [center_x, center_y, width, height]
        
        # Format YOLO : class_id center_x center_y width height
        label_line = f"{class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
        
        try:
            with open(label_path, 'w') as f:
                f.write(label_line)
        except Exception as e:
            print(f"⚠️ Erreur écriture label {label_path}: {e}")
    
    def _generate_yolo_config(self, dataset_path: Path, class_names: List[str], 
                             dataset_name: str) -> Path:
        """Génère le fichier de configuration YOLO (.yaml)"""
        config_path = dataset_path / f"{dataset_name}.yaml"
        
        # Configuration YOLO standard
        config = {
            'path': str(dataset_path),  # Chemin racine du dataset
            'train': str(dataset_path / 'train' / 'images'),
            'val': str(dataset_path / 'val' / 'images'),
            'test': str(dataset_path / 'test' / 'images'),
            'nc': len(class_names),  # Nombre de classes
            'names': class_names     # Noms des classes
        }
        
        # Sauvegarde
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"📄 Configuration YOLO sauvegardée : {config_path}")
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde config {config_path}: {e}")
        
        return config_path
    
    def _save_dataset_metadata(self, dataset_path: Path, dataset_info: SimpleDatasetInfo):
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
            'generator_type': 'simple'
        }
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde métadonnées {metadata_path}: {e}")
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """Liste les datasets disponibles"""
        datasets = []
        
        try:
            for dataset_dir in self.datasets_dir.glob('*'):
                if dataset_dir.is_dir():
                    metadata_path = dataset_dir / 'dataset_info.json'
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            datasets.append(metadata)
                        except Exception as e:
                            print(f"⚠️ Erreur lecture métadonnées {dataset_dir}: {e}")
        except Exception as e:
            print(f"⚠️ Erreur liste datasets: {e}")
        
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


# Fonction d'accès global simplifiée
def create_simple_yolo_dataset(dataset_name: str, 
                              selected_classes: List[str] = None) -> SimpleDatasetInfo:
    """
    Fonction simplifiée pour créer un dataset YOLO basique
    
    Args:
        dataset_name: Nom du dataset
        selected_classes: Classes à inclure (None = toutes)
        
    Returns:
        SimpleDatasetInfo: Informations sur le dataset créé
    """
    generator = SimpleYOLODatasetGenerator()
    
    return generator.generate_dataset(
        dataset_name=dataset_name,
        selected_classes=selected_classes
    )