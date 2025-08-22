"""
Gestionnaire d'annotations pour YOLO Interactive Object Detector

Ce module gère :
- Stockage persistant des annotations dans SQLite
- Organisation des exemples d'entraînement par classe
- Génération de datasets YOLO
- Métadonnées géospatiales et validation des données

Workflow :
1. Réception des annotations depuis l'outil canvas
2. Stockage avec métadonnées complètes (CRS, coordonnées, patches)
3. Organisation par classe et validation qualité
4. Export vers format YOLO standard (images + labels)
"""

import os
import sqlite3
import json
import shutil
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

from qgis.core import QgsProject, QgsCoordinateReferenceSystem
from qgis.PyQt.QtCore import QObject, pyqtSignal


@dataclass
class AnnotationExample:
    """Structure d'un exemple d'annotation"""
    id: str
    class_name: str
    bbox_map: Dict[str, float]  # Coordonnées carte {xmin, ymin, xmax, ymax}
    bbox_normalized: List[float]  # YOLO format [center_x, center_y, width, height]
    image_patch: str  # Chemin vers l'image patch
    crs: str  # Système de coordonnées
    layer_name: str
    pixel_size: Dict[str, float]
    timestamp: str
    metadata: Dict[str, Any]  # Métadonnées additionnelles
    
    # NOUVEAUX CHAMPS POUR CONTOURS PRÉCIS
    polygon_points: Optional[List[List[float]]] = None  # Points polygone [[x,y], [x,y], ...]
    polygon_available: bool = False  # True si polygone précis disponible
    vertex_count: Optional[int] = None  # Nombre de vertices du polygone
    area_pixels: Optional[float] = None  # Aire en pixels du polygone
    refinement_applied: bool = False  # True si raffinement SAM appliqué
    
    def to_dict(self) -> Dict:
        """Conversion en dictionnaire pour stockage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnnotationExample':
        """Création depuis dictionnaire"""
        return cls(**data)


@dataclass 
class ClassStatistics:
    """Statistiques d'une classe d'objets"""
    name: str
    example_count: int
    total_area_coverage: float  # Aire totale couverte (m²)
    avg_object_size: Tuple[float, float]  # Taille moyenne (largeur, hauteur) en mètres
    crs_distribution: Dict[str, int]  # Distribution des CRS utilisés
    quality_score: float  # Score qualité basé sur diversité et quantité
    ready_for_training: bool
    # NOUVEAUX: Métadonnées d'échelle
    pixel_size_range: Tuple[float, float]  # Min/Max résolution (m/pixel)
    optimal_pixel_size: float  # Résolution optimale pour détection
    scale_range: Tuple[int, int]  # Plage d'échelles de carte (1:1000 → 1000)
    zoom_level_range: Tuple[int, int]  # Niveaux de zoom QGIS
    

class AnnotationManager(QObject):
    """
    Gestionnaire central des annotations d'entraînement
    
    Responsabilités :
    - Stockage persistant en SQLite
    - Validation et organisation des données
    - Génération de statistiques
    - Export vers format YOLO
    - Nettoyage et maintenance des données
    """
    
    # Signaux
    annotation_added = pyqtSignal(str, dict)  # Nouvelle annotation ajoutée
    statistics_updated = pyqtSignal(str, dict)  # Statistiques mises à jour
    dataset_exported = pyqtSignal(str, str)  # Dataset exporté (classe, chemin)
    
    def __init__(self, project_dir: str = None):
        """
        Initialise le gestionnaire d'annotations
        
        Args:
            project_dir: Répertoire racine du projet (None = auto-détection)
        """
        super().__init__()
        
        # Configuration des chemins
        self.project_dir = Path(project_dir) if project_dir else self._get_project_directory()
        self.data_dir = self.project_dir / "annotations"
        self.patches_dir = self.data_dir / "patches"
        self.exports_dir = self.data_dir / "yolo_datasets"
        self.db_path = self.data_dir / "annotations.db"
        
        # Création des répertoires
        self._ensure_directories()
        
        # Initialisation de la base de données
        self._init_database()
        
        # CORRECTION: Vérifier et créer nouvelles tables si manquantes
        self._ensure_all_tables()
        
        # Cache des statistiques
        self._stats_cache = {}
        self._cache_timestamp = None
        
        print(f"📁 AnnotationManager initialisé : {self.data_dir}")
    
    def _get_project_directory(self) -> Path:
        """Détermine le répertoire de projet - SOLUTION: utiliser répertoire de travail actuel"""
        project = QgsProject.instance()
        if project.fileName():
            # Utiliser le répertoire du projet QGIS
            project_path = Path(project.fileName()).parent / 'yolo_detector_data'
            print(f"📁 Répertoire projet QGIS: {project_path}")
            return project_path
        else:
            # Utiliser le répertoire de travail actuel (plus sûr que Documents)
            current_dir = Path.cwd() / 'qgis_yolo_detector' / 'project_data'
            print(f"📁 Répertoire de travail: {current_dir}")
            return current_dir
    
    def _ensure_directories(self):
        """Crée les répertoires nécessaires"""
        for directory in [self.data_dir, self.patches_dir, self.exports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    id TEXT PRIMARY KEY,
                    class_name TEXT NOT NULL,
                    bbox_map_json TEXT NOT NULL,
                    bbox_normalized_json TEXT NOT NULL,
                    image_patch_path TEXT NOT NULL,
                    crs TEXT NOT NULL,
                    layer_name TEXT NOT NULL,
                    pixel_size_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- NOUVEAUX CHAMPS POUR CONTOURS PRÉCIS
                    polygon_points_json TEXT,              -- Points du polygone format JSON
                    polygon_available BOOLEAN DEFAULT 0,   -- True si polygone disponible
                    vertex_count INTEGER,                   -- Nombre de vertices
                    area_pixels REAL,                       -- Aire en pixels
                    refinement_applied BOOLEAN DEFAULT 0,   -- True si SAM appliqué
                    
                    FOREIGN KEY (class_name) REFERENCES classes (name)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS classes (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    color TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # NOUVEAUTÉ: Table pour stocker les datasets générés
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    class_names_json TEXT NOT NULL,
                    image_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config_json TEXT,
                    status TEXT DEFAULT 'ready'
                )
            """)
            
            # NOUVEAUTÉ: Table pour stocker les modèles entraînés
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trained_models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    class_names_json TEXT NOT NULL,
                    metrics_json TEXT,
                    config_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'ready',
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_annotations_class 
                ON annotations(class_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_annotations_timestamp 
                ON annotations(timestamp)
            """)
            
            # Migration pour ajouter nouvelles colonnes polygones
            self._migrate_database_for_polygons(conn)

    def _ensure_all_tables(self):
        """S'assure que toutes les tables existent (pour upgrade des bases existantes)"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Vérification et création table datasets
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='datasets'
                """)
                if not cursor.fetchone():
                    print("📊 Création table 'datasets'")
                    conn.execute("""
                        CREATE TABLE datasets (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            path TEXT NOT NULL,
                            class_names_json TEXT NOT NULL,
                            image_count INTEGER NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            config_json TEXT,
                            status TEXT DEFAULT 'ready'
                        )
                    """)
                
                # Vérification et création table trained_models
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='trained_models'
                """)
                if not cursor.fetchone():
                    print("🤖 Création table 'trained_models'")
                    conn.execute("""
                        CREATE TABLE trained_models (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            path TEXT NOT NULL,
                            dataset_id TEXT NOT NULL,
                            class_names_json TEXT NOT NULL,
                            metrics_json TEXT,
                            config_json TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            status TEXT DEFAULT 'ready',
                            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
                        )
                    """)
                
                conn.commit()
                print("✅ Toutes les tables sont présentes")
                
        except Exception as e:
            print(f"❌ Erreur création tables: {e}")
            import traceback
            traceback.print_exc()
    
    def _migrate_database_for_polygons(self, conn):
        """Ajoute les colonnes polygones aux bases existantes"""
        try:
            # Vérification si colonnes polygones existent déjà
            cursor = conn.execute("PRAGMA table_info(annotations)")
            columns = [row[1] for row in cursor.fetchall()]
            
            polygon_columns = [
                'polygon_points_json', 'polygon_available', 'vertex_count', 
                'area_pixels', 'refinement_applied'
            ]
            
            for column in polygon_columns:
                if column not in columns:
                    column_type = {
                        'polygon_points_json': 'TEXT',
                        'polygon_available': 'BOOLEAN DEFAULT 0',
                        'vertex_count': 'INTEGER', 
                        'area_pixels': 'REAL',
                        'refinement_applied': 'BOOLEAN DEFAULT 0'
                    }[column]
                    
                    try:
                        conn.execute(f"ALTER TABLE annotations ADD COLUMN {column} {column_type}")
                        print(f"✅ Colonne '{column}' ajoutée à la table annotations")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e):
                            print(f"⚠️ Erreur ajout colonne '{column}': {e}")
            
            conn.commit()
            
        except Exception as e:
            print(f"❌ Erreur migration base données pour polygones: {e}")
    
    def create_class(self, class_name: str, description: str = "", color: str = "#FF0000") -> bool:
        """
        Crée une nouvelle classe d'objets dans la base de données
        
        Args:
            class_name: Nom de la classe
            description: Description optionnelle
            color: Couleur pour l'affichage (format hex)
            
        Returns:
            bool: Succès de la création
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO classes (name, description, color, updated_at)
                    VALUES (?, ?, ?, datetime('now'))
                """, (class_name, description, color))
                
            print(f"✅ Classe '{class_name}' créée/mise à jour dans la base")
            return True
            
        except Exception as e:
            print(f"❌ Erreur création classe '{class_name}': {e}")
            return False
    
    def add_annotation(self, example_data: Dict[str, Any]) -> str:
        """
        Ajoute une nouvelle annotation
        
        Args:
            example_data: Données de l'exemple depuis l'outil d'annotation
            
        Returns:
            str: ID unique de l'annotation
        """
        try:
            # S'assurer que la classe existe dans la base
            class_name = example_data['class_name']
            self.create_class(class_name, f"Classe créée automatiquement: {class_name}")
            
            # Génération d'un ID unique
            annotation_id = self._generate_annotation_id(class_name)
            
            # Sauvegarde de l'image patch
            patch_path = self._save_image_patch(
                example_data['image_patch'], 
                annotation_id, 
                example_data['class_name']
            )
            
            # Création de l'objet annotation avec nouvelles données polygonales
            annotation = AnnotationExample(
                id=annotation_id,
                class_name=example_data['class_name'],
                bbox_map=example_data['bbox_map'],
                bbox_normalized=example_data['bbox_normalized'],
                image_patch=str(patch_path),
                crs=example_data['crs'],
                layer_name=example_data['layer_name'],
                pixel_size=example_data['pixel_size'],
                timestamp=example_data['timestamp'],
                metadata=example_data.get('metadata', {}),
                
                # NOUVEAUX CHAMPS POLYGONAUX depuis Smart Mode
                polygon_points=example_data.get('polygon_points'),
                polygon_available=example_data.get('polygon_available', False),
                vertex_count=example_data.get('vertex_count'),
                area_pixels=example_data.get('area_pixels'),
                refinement_applied=example_data.get('refinement_applied', False)
            )
            
            # Stockage en base
            self._store_annotation(annotation)
            
            # Invalidation du cache statistiques
            self._invalidate_cache()
            
            # Émission des signaux
            self.annotation_added.emit(annotation.class_name, annotation.to_dict())
            self._emit_updated_statistics(annotation.class_name)
            
            print(f"✅ Annotation {annotation_id} ajoutée pour '{annotation.class_name}'")
            return annotation_id
            
        except Exception as e:
            print(f"❌ Erreur ajout annotation : {e}")
            raise
    
    def _generate_annotation_id(self, class_name: str) -> str:
        """Génère un ID unique pour l'annotation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        clean_class = "".join(c for c in class_name if c.isalnum() or c in "_-")
        return f"{clean_class}_{timestamp}"
    
    def _save_image_patch(self, image_array: np.ndarray, annotation_id: str, class_name: str) -> Path:
        """
        Sauvegarde un patch d'image sur disque
        
        Args:
            image_array: Array numpy de l'image
            annotation_id: ID de l'annotation
            class_name: Nome de la classe
            
        Returns:
            Path: Chemin vers l'image sauvegardée
        """
        # Répertoire de classe
        class_dir = self.patches_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Chemin de l'image
        image_path = class_dir / f"{annotation_id}.png"
        
        # Conversion et sauvegarde
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_array)
        pil_image.save(str(image_path), format='PNG', optimize=True)
        
        return image_path
    
    def _store_annotation(self, annotation: AnnotationExample):
        """Stocke une annotation en base de données"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Insertion de la classe si nécessaire
            conn.execute("""
                INSERT OR IGNORE INTO classes (name, description) 
                VALUES (?, ?)
            """, (annotation.class_name, f"Classe {annotation.class_name}"))
            
            # Insertion de l'annotation avec nouvelles colonnes polygones
            conn.execute("""
                INSERT INTO annotations (
                    id, class_name, bbox_map_json, bbox_normalized_json,
                    image_patch_path, crs, layer_name, pixel_size_json,
                    timestamp, metadata_json, polygon_points_json, polygon_available,
                    vertex_count, area_pixels, refinement_applied
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                annotation.id,
                annotation.class_name,
                json.dumps(annotation.bbox_map),
                json.dumps(annotation.bbox_normalized),
                annotation.image_patch,
                annotation.crs,
                annotation.layer_name,
                json.dumps(annotation.pixel_size),
                annotation.timestamp,
                json.dumps(annotation.metadata),
                # NOUVEAUX CHAMPS POLYGONAUX
                json.dumps(annotation.polygon_points) if annotation.polygon_points else None,
                annotation.polygon_available,
                annotation.vertex_count,
                annotation.area_pixels,
                annotation.refinement_applied
            ))
    
    def get_class_annotations(self, class_name: str) -> List[AnnotationExample]:
        """
        Récupère toutes les annotations d'une classe
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            List[AnnotationExample]: Liste des annotations
        """
        annotations = []
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM annotations 
                WHERE class_name = ? 
                ORDER BY timestamp DESC
            """, (class_name,))
            
            for row in cursor:
                annotation = AnnotationExample(
                    id=row['id'],
                    class_name=row['class_name'],
                    bbox_map=json.loads(row['bbox_map_json']),
                    bbox_normalized=json.loads(row['bbox_normalized_json']),
                    image_patch=row['image_patch_path'],
                    crs=row['crs'],
                    layer_name=row['layer_name'],
                    pixel_size=json.loads(row['pixel_size_json']),
                    timestamp=row['timestamp'],
                    metadata=json.loads(row['metadata_json'] or '{}'),
                    
                    # NOUVEAUX CHAMPS POLYGONAUX avec compatibilité anciennes bases
                    polygon_points=json.loads(row['polygon_points_json']) if 'polygon_points_json' in row.keys() and row['polygon_points_json'] else None,
                    polygon_available=bool(row['polygon_available']) if 'polygon_available' in row.keys() else False,
                    vertex_count=row['vertex_count'] if 'vertex_count' in row.keys() else None,
                    area_pixels=row['area_pixels'] if 'area_pixels' in row.keys() else None,
                    refinement_applied=bool(row['refinement_applied']) if 'refinement_applied' in row.keys() else False
                )
                annotations.append(annotation)
        
        return annotations
    
    def get_class_statistics(self, class_name: str) -> ClassStatistics:
        """
        Calcule les statistiques d'une classe
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            ClassStatistics: Statistiques de la classe
        """
        # Vérification du cache
        if (self._cache_timestamp and 
            class_name in self._stats_cache and 
            (datetime.now() - self._cache_timestamp).seconds < 60):
            return self._stats_cache[class_name]
        
        annotations = self.get_class_annotations(class_name)
        
        if not annotations:
            return ClassStatistics(
                name=class_name,
                example_count=0,
                total_area_coverage=0.0,
                avg_object_size=(0.0, 0.0),
                crs_distribution={},
                quality_score=0.0,
                ready_for_training=False,
                # NOUVEAUX champs par défaut
                pixel_size_range=(0.0, 0.0),
                optimal_pixel_size=0.0,
                scale_range=(0, 0),
                zoom_level_range=(0, 0)
            )
        
        # Calculs statistiques
        total_area = 0.0
        sizes = []
        crs_dist = {}
        
        for annotation in annotations:
            # Calcul de l'aire (approximatif)
            bbox = annotation.bbox_map
            width = bbox['xmax'] - bbox['xmin']
            height = bbox['ymax'] - bbox['ymin']
            area = width * height
            total_area += area
            sizes.append((width, height))
            
            # Distribution des CRS
            crs_dist[annotation.crs] = crs_dist.get(annotation.crs, 0) + 1
        
        # Taille moyenne
        avg_size = (
            sum(s[0] for s in sizes) / len(sizes),
            sum(s[1] for s in sizes) / len(sizes)
        ) if sizes else (0.0, 0.0)
        
        # NOUVEAUTÉ: Calcul des métadonnées d'échelle
        pixel_sizes = []
        for annotation in annotations:
            if 'x' in annotation.pixel_size and 'y' in annotation.pixel_size:
                # Utiliser résolution moyenne entre X et Y
                avg_pixel_size = (annotation.pixel_size['x'] + annotation.pixel_size['y']) / 2
                pixel_sizes.append(avg_pixel_size)
        
        # Plages de résolution et échelles
        if pixel_sizes:
            min_pixel_size = min(pixel_sizes)
            max_pixel_size = max(pixel_sizes)
            optimal_pixel_size = sum(pixel_sizes) / len(pixel_sizes)  # Moyenne
            
            # Conversion résolution → échelle approximative (règle empirique)
            # 1 pixel = pixel_size mètres → échelle ≈ pixel_size * 2000
            min_scale = int(min_pixel_size * 2000)
            max_scale = int(max_pixel_size * 2000)
            
            # Niveaux de zoom QGIS approximatifs (basés sur résolution)
            min_zoom = max(1, int(18 - math.log2(max_pixel_size / 0.25)))
            max_zoom = min(22, int(18 - math.log2(min_pixel_size / 0.25)))
        else:
            min_pixel_size = max_pixel_size = optimal_pixel_size = 0.0
            min_scale = max_scale = 0
            min_zoom = max_zoom = 0
        
        # Score qualité (basé sur quantité et diversité + cohérence d'échelle)
        quantity_score = min(len(annotations) / 20.0, 1.0)  # 20 exemples = score max
        diversity_score = min(len(crs_dist) / 3.0, 1.0)  # Diversité CRS
        
        # NOUVEAUTÉ: Bonus cohérence d'échelle
        scale_consistency = 1.0
        if pixel_sizes and len(pixel_sizes) > 1:
            scale_variance = max(pixel_sizes) / min(pixel_sizes) if min(pixel_sizes) > 0 else float('inf')
            # Bonus si variance < 2x (objets à échelles similaires)
            scale_consistency = min(1.0, 2.0 / scale_variance) if scale_variance < float('inf') else 0.0
        
        quality_score = (quantity_score * 0.5 + diversity_score * 0.3 + scale_consistency * 0.2)
        
        # Prêt pour entraînement
        ready = len(annotations) >= 10 and quality_score > 0.3
        
        stats = ClassStatistics(
            name=class_name,
            example_count=len(annotations),
            total_area_coverage=total_area,
            avg_object_size=avg_size,
            crs_distribution=crs_dist,
            quality_score=quality_score,
            ready_for_training=ready,
            # NOUVEAUX champs
            pixel_size_range=(min_pixel_size, max_pixel_size),
            optimal_pixel_size=optimal_pixel_size,
            scale_range=(min_scale, max_scale),
            zoom_level_range=(min_zoom, max_zoom)
        )
        
        # Mise en cache
        self._stats_cache[class_name] = stats
        self._cache_timestamp = datetime.now()
        
        return stats
    
    def get_all_classes(self) -> List[str]:
        """Retourne la liste de toutes les classes (depuis la table classes)"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("SELECT name FROM classes ORDER BY name")
            return [row[0] for row in cursor]
    
    def get_all_classes_from_annotations(self) -> List[str]:
        """Retourne les classes qui ont des annotations"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("SELECT DISTINCT class_name FROM annotations ORDER BY class_name")
            return [row[0] for row in cursor]
    
    def delete_annotation(self, annotation_id: str) -> bool:
        """
        Supprime une annotation
        
        Args:
            annotation_id: ID de l'annotation
            
        Returns:
            bool: Succès de la suppression
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Récupération du chemin de l'image
                cursor = conn.execute(
                    "SELECT image_patch_path, class_name FROM annotations WHERE id = ?", 
                    (annotation_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return False
                
                image_path, class_name = row
                
                # Suppression de la base
                conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
                
                # Suppression du fichier image
                try:
                    Path(image_path).unlink(missing_ok=True)
                except Exception as e:
                    print(f"⚠️ Erreur suppression image {image_path}: {e}")
                
                # Invalidation du cache
                self._invalidate_cache()
                self._emit_updated_statistics(class_name)
                
                print(f"✅ Annotation {annotation_id} supprimée")
                return True
                
        except Exception as e:
            print(f"❌ Erreur suppression annotation : {e}")
            return False
    
    def _invalidate_cache(self):
        """Invalide le cache des statistiques"""
        self._stats_cache.clear()
        self._cache_timestamp = None
    
    def _emit_updated_statistics(self, class_name: str):
        """Émet le signal de mise à jour des statistiques"""
        stats = self.get_class_statistics(class_name)
        self.statistics_updated.emit(class_name, asdict(stats))
    
    # NOUVEAUTÉ: Gestion des datasets
    def save_dataset(self, dataset_id: str, name: str, path: str, class_names: List[str], 
                    image_count: int, config: Dict = None) -> bool:
        """Sauvegarde un dataset généré"""
        try:
            # Conversion des WindowsPath en string pour éviter l'erreur JSON
            safe_config = {}
            if config:
                for key, value in config.items():
                    if isinstance(value, Path):
                        safe_config[key] = str(value)
                    else:
                        safe_config[key] = value
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO datasets 
                    (id, name, path, class_names_json, image_count, config_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset_id, name, str(path), json.dumps(class_names), 
                    image_count, json.dumps(safe_config), datetime.now().isoformat()
                ))
                conn.commit()
                print(f"💾 Dataset sauvegardé: {name} ({image_count} images)")
                return True
        except Exception as e:
            print(f"❌ Erreur sauvegarde dataset: {e}")
            return False
    
    def get_datasets(self) -> List[Dict]:
        """Récupère tous les datasets"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, name, path, class_names_json, image_count, 
                           config_json, created_at, status
                    FROM datasets ORDER BY created_at DESC
                """)
                datasets = []
                for row in cursor.fetchall():
                    datasets.append({
                        'id': row[0],
                        'name': row[1], 
                        'path': row[2],
                        'class_names': json.loads(row[3]),
                        'image_count': row[4],
                        'config': json.loads(row[5]) if row[5] else {},
                        'created_at': row[6],
                        'status': row[7]
                    })
                return datasets
        except Exception as e:
            print(f"❌ Erreur lecture datasets: {e}")
            return []
    
    # NOUVEAUTÉ: Gestion des modèles entraînés  
    def save_trained_model(self, model_id: str, name: str, path: str, dataset_id: str,
                          class_names: List[str], metrics: Dict = None, config: Dict = None) -> bool:
        """Sauvegarde un modèle entraîné"""
        try:
            # Conversion des WindowsPath en string pour éviter l'erreur JSON
            safe_metrics = {}
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, Path):
                        safe_metrics[key] = str(value)
                    else:
                        safe_metrics[key] = value
            
            safe_config = {}
            if config:
                for key, value in config.items():
                    if isinstance(value, Path):
                        safe_config[key] = str(value)
                    else:
                        safe_config[key] = value
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trained_models 
                    (id, name, path, dataset_id, class_names_json, metrics_json, config_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id, name, str(path), dataset_id, json.dumps(class_names),
                    json.dumps(safe_metrics), json.dumps(safe_config), datetime.now().isoformat()
                ))
                conn.commit()
                print(f"💾 Modèle sauvegardé: {name} → {class_names}")
                return True
        except Exception as e:
            print(f"❌ Erreur sauvegarde modèle: {e}")
            return False
    
    def get_trained_models(self) -> List[Dict]:
        """Récupère tous les modèles entraînés"""
        try:
            print(f"🔍 Lecture modèles depuis: {self.db_path}")
            with sqlite3.connect(self.db_path) as conn:
                # Vérification d'abord si la table existe
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='trained_models'
                """)
                if not cursor.fetchone():
                    print("⚠️ Table 'trained_models' n'existe pas")
                    return []
                
                cursor = conn.execute("""
                    SELECT tm.id, tm.name, tm.path, tm.dataset_id, tm.class_names_json,
                           tm.metrics_json, tm.config_json, tm.created_at, tm.status,
                           d.name as dataset_name
                    FROM trained_models tm
                    LEFT JOIN datasets d ON tm.dataset_id = d.id
                    ORDER BY tm.created_at DESC
                """)
                models = []
                rows = cursor.fetchall()
                print(f"🔍 Trouvé {len(rows)} lignes dans trained_models")
                
                for row in rows:
                    models.append({
                        'id': row[0],
                        'name': row[1],
                        'path': row[2], 
                        'dataset_id': row[3],
                        'class_names': json.loads(row[4]),
                        'metrics': json.loads(row[5]) if row[5] else {},
                        'config': json.loads(row[6]) if row[6] else {},
                        'created_at': row[7],
                        'status': row[8],
                        'dataset_name': row[9]
                    })
                print(f"✅ Modèles parsés: {len(models)}")
                return models
        except Exception as e:
            print(f"❌ Erreur lecture modèles: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_models_for_class(self, class_name: str) -> List[Dict]:
        """Récupère les modèles disponibles pour une classe"""
        models = self.get_trained_models()
        return [m for m in models if class_name in m.get('class_names', [])]
    
    def cleanup_orphaned_files(self):
        """Nettoie les fichiers orphelins"""
        try:
            # Récupération des chemins en base
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT image_patch_path FROM annotations")
                db_paths = {row[0] for row in cursor}
            
            # Parcours des fichiers sur disque
            cleaned_count = 0
            for patch_file in self.patches_dir.rglob("*.png"):
                if str(patch_file) not in db_paths:
                    patch_file.unlink()
                    cleaned_count += 1
            
            print(f"🧹 {cleaned_count} fichiers orphelins nettoyés")
            
        except Exception as e:
            print(f"❌ Erreur nettoyage : {e}")
    
    def get_class_examples(self, class_name: str) -> List[Dict]:
        """
        Retourne tous les exemples d'une classe spécifique
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            List[Dict]: Liste des exemples avec métadonnées
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM annotations 
                WHERE class_name = ? 
                ORDER BY timestamp DESC
            """, (class_name,))
            
            examples = []
            for row in cursor.fetchall():
                # Conversion Row vers dict
                example = {
                    'id': row['id'] if 'id' in row.keys() else '',
                    'class_name': row['class_name'] if 'class_name' in row.keys() else '',
                    'bbox_map': json.loads(row['bbox_map']) if 'bbox_map' in row.keys() else {},
                    'bbox_normalized': json.loads(row['bbox_normalized']) if 'bbox_normalized' in row.keys() else [],
                    'image_patch': row['image_patch'] if 'image_patch' in row.keys() else '',
                    'crs': row['crs'] if 'crs' in row.keys() else '',
                    'layer_name': row['layer_name'] if 'layer_name' in row.keys() else '',
                    'pixel_size': json.loads(row['pixel_size']) if 'pixel_size' in row.keys() else {},
                    'timestamp': row['timestamp'] if 'timestamp' in row.keys() else '',
                    'metadata': json.loads(row['metadata']) if 'metadata' in row.keys() else {},
                    'confidence': row['confidence'] if 'confidence' in row.keys() else 1.0,
                    'polygon_points': json.loads(row['polygon_points']) if 'polygon_points' in row.keys() and row['polygon_points'] else None,
                    'polygon_available': bool(row['polygon_available']) if 'polygon_available' in row.keys() else False,
                    'raster_name': row['layer_name'] if 'layer_name' in row.keys() else 'Unknown'
                }
                examples.append(example)
            
            return examples
            
        except Exception as e:
            print(f"❌ Erreur récupération exemples classe '{class_name}': {e}")
            return []
    
    def get_all_classes(self) -> List[str]:
        """
        Retourne la liste de toutes les classes avec exemples
        
        Returns:
            List[str]: Noms des classes
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT class_name FROM annotations ORDER BY class_name")
            return [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            print(f"❌ Erreur récupération classes: {e}")
            return []
    
    def update_example(self, example: Dict):
        """
        Met à jour un exemple dans la base de données
        
        Args:
            example: Dictionnaire avec les données de l'exemple
        """
        try:
            cursor = self.conn.cursor()
            
            # Préparer les données pour la mise à jour
            update_data = {
                'class_name': example.get('class_name', ''),
                'confidence': float(example.get('confidence', 1.0)),
                'bbox_map': json.dumps(example.get('bbox_map', {})),
                'bbox_normalized': json.dumps(example.get('bbox_normalized', [])),
                'metadata': json.dumps(example.get('metadata', {})),
                'polygon_points': json.dumps(example.get('polygon_points')) if example.get('polygon_points') else None,
                'polygon_available': bool(example.get('polygon_points'))
            }
            
            cursor.execute("""
                UPDATE annotations SET 
                    class_name = ?,
                    confidence = ?,
                    bbox_map = ?,
                    bbox_normalized = ?,
                    metadata = ?,
                    polygon_points = ?,
                    polygon_available = ?
                WHERE id = ?
            """, (
                update_data['class_name'],
                update_data['confidence'],
                update_data['bbox_map'],
                update_data['bbox_normalized'],
                update_data['metadata'],
                update_data['polygon_points'],
                update_data['polygon_available'],
                example.get('id', '')
            ))
            
            self.conn.commit()
            print(f"✅ Exemple {example.get('id', '')[:8]}... mis à jour")
            
        except Exception as e:
            print(f"❌ Erreur mise à jour exemple: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Retourne les informations de stockage"""
        try:
            # Taille de la base de données
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # Taille des patches
            patches_size = sum(
                f.stat().st_size for f in self.patches_dir.rglob("*") if f.is_file()
            )
            
            # Nombre total d'annotations
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM annotations")
                total_annotations = cursor.fetchone()[0]
            
            return {
                'database_size_mb': db_size / (1024 * 1024),
                'patches_size_mb': patches_size / (1024 * 1024),
                'total_size_mb': (db_size + patches_size) / (1024 * 1024),
                'total_annotations': total_annotations,
                'data_directory': str(self.data_dir)
            }
            
        except Exception as e:
            print(f"❌ Erreur info stockage : {e}")
            return {}


# Instance globale (singleton pattern)
_annotation_manager = None

def get_annotation_manager(project_dir: str = None) -> AnnotationManager:
    """
    Retourne l'instance globale du gestionnaire d'annotations
    
    Args:
        project_dir: Répertoire de projet (None = auto-détection)
        
    Returns:
        AnnotationManager: Instance du gestionnaire
    """
    global _annotation_manager
    
    if _annotation_manager is None or (project_dir and str(_annotation_manager.project_dir) != project_dir):
        _annotation_manager = AnnotationManager(project_dir)
    
    return _annotation_manager