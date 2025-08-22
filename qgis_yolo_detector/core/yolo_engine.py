"""
Moteur YOLO - Interface principale pour la détection d'objets et l'entraînement

Ce module contient la classe YOLOEngine qui gère:
- Le chargement et la mise en cache des modèles YOLO
- La détection d'objets sur images individuelles et en batch
- L'entraînement de modèles personnalisés via transfer learning
- L'optimisation GPU/CPU avec fallback automatique
- La gestion de la mémoire pour les gros datasets
"""

import os
import sys
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import cv2
from PIL import Image

# Import conditionnel d'Ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics YOLO not available. Please install with: pip install ultralytics")


class LRUCache:
    """
    Cache LRU simple pour les modèles YOLO
    """
    def __init__(self, maxsize: int = 3):
        self.maxsize = maxsize
        self.cache = OrderedDict()
    
    def get(self, key):
        if key in self.cache:
            # Déplace vers la fin (plus récent)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.maxsize:
            # Supprime le plus ancien
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            # Force garbage collection pour libérer la mémoire GPU
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.cache[key] = value


class DeviceManager:
    """
    Gestionnaire intelligent des devices GPU/CPU
    """
    
    @staticmethod
    def detect_optimal_device():
        """
        Détecte le device optimal pour YOLO
        
        Returns:
            str: Device optimal ('cuda:0', 'mps', 'cpu')
        """
        # NVIDIA CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            # Sélectionne le GPU avec le plus de mémoire libre
            best_gpu = 0
            max_free_memory = 0
            
            for i in range(device_count):
                try:
                    free_memory = torch.cuda.mem_get_info(i)[0]
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_gpu = i
                except:
                    pass
                    
            return f'cuda:{best_gpu}'
        
        # Apple Metal Performance Shaders (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        
        # CPU comme fallback
        return 'cpu'
    
    @staticmethod
    def get_device_info(device: str) -> Dict:
        """
        Retourne les informations détaillées sur un device
        
        Args:
            device: Device string ('cuda:0', 'mps', 'cpu')
            
        Returns:
            dict: Informations sur le device
        """
        info = {'device': device, 'available': True}
        
        if device.startswith('cuda'):
            if torch.cuda.is_available():
                gpu_id = int(device.split(':')[1]) if ':' in device else 0
                info.update({
                    'name': torch.cuda.get_device_name(gpu_id),
                    'memory_total': torch.cuda.get_device_properties(gpu_id).total_memory,
                    'memory_free': torch.cuda.mem_get_info(gpu_id)[0],
                    'compute_capability': torch.cuda.get_device_properties(gpu_id).major
                })
            else:
                info['available'] = False
                
        elif device == 'mps':
            info.update({
                'name': 'Apple Metal GPU',
                'memory_total': -1,  # Non accessible
                'memory_free': -1
            })
            
        elif device == 'cpu':
            info.update({
                'name': f'CPU ({torch.get_num_threads()} threads)',
                'memory_total': psutil.virtual_memory().total,
                'memory_free': psutil.virtual_memory().available,
                'cores': os.cpu_count()
            })
            
        return info


class YOLOEngine:
    """
    Moteur principal pour la détection d'objets YOLO
    
    Cette classe fournit une interface haut niveau pour:
    - Charger et gérer des modèles YOLO
    - Effectuer des détections sur images individuelles ou par batch
    - Entraîner des modèles personnalisés
    - Optimiser les performances selon le hardware disponible
    """
    
    def __init__(self, max_cached_models: int = 3):
        """
        Initialise le moteur YOLO
        
        Args:
            max_cached_models: Nombre maximum de modèles en cache
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not available. Please install with: pip install ultralytics")
        
        # SÉCURITÉ: Configurer NumPy pour éviter les conflits 2.x
        self._configure_numpy_compatibility()
        
        # SÉCURITÉ: Configurer Ultralytics pour éviter l'accès aux Documents
        self._configure_ultralytics_paths()
            
        # Détection du device optimal
        self.device = DeviceManager.detect_optimal_device()
        self.device_info = DeviceManager.get_device_info(self.device)
        
        # Cache des modèles chargés
        self.model_cache = LRUCache(maxsize=max_cached_models)
        self.current_model = None
        self.current_model_path = None
        
        # Configuration par défaut optimisée pour modèles custom
        self.default_confidence = 0.1  # ✅ Beaucoup plus permissif pour modèles custom
        self.default_iou_threshold = 0.45
        self.batch_size = self._calculate_optimal_batch_size()
        
        print(f"YOLOEngine initialized with device: {self.device}")
        print(f"Device info: {self.device_info}")
        
        # Vérification des modèles intégrés
        self._verify_integrated_models()

    def _configure_numpy_compatibility(self):
        """Configure NumPy pour éviter les conflits de version"""
        try:
            import warnings
            import os
            
            # Désactiver les warnings NumPy 2.x (correction: utiliser Warning au lieu de UserWarning)
            try:
                warnings.filterwarnings("ignore", category=Warning, module="numpy")
                warnings.filterwarnings("ignore", message=".*NumPy.*")
                warnings.filterwarnings("ignore", category=Warning, message=".*category.*")
            except Exception as warn_error:
                print(f"⚠️ Erreur configuration warnings: {warn_error}")
            
            # Variables d'environnement pour forcer compatibilité
            os.environ['NUMPY_EXPERIMENTAL_DTYPE_API'] = '0'
            os.environ['NPY_DISABLE_OPTIMIZATION'] = '1'
            
            print("✅ Configuration NumPy 2.x pour compatibilité")
            
        except Exception as e:
            print(f"⚠️ Erreur configuration NumPy: {e}")

    def _configure_ultralytics_paths(self):
        """Configure les chemins Ultralytics pour éviter l'accès aux Documents"""
        try:
            # SÉCURITÉ: Utiliser répertoire du plugin au lieu de cwd()
            plugin_dir = Path(__file__).parent.parent
            secure_dir = plugin_dir.parent / 'qgis_yolo_detector_data'
            models_dir = secure_dir / 'models' / 'ultralytics'
            datasets_dir = secure_dir / 'datasets'
            runs_dir = secure_dir / 'runs'
            config_dir = secure_dir / 'config'
            
            # Créer les répertoires
            models_dir.mkdir(parents=True, exist_ok=True)
            datasets_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Configuration des variables d'environnement Ultralytics
            os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
            os.environ['ULTRALYTICS_CONFIG_DIR'] = str(config_dir)
            os.environ['ULTRALYTICS_RUNS_DIR'] = str(runs_dir)
            
            # Désactiver les téléchargements automatiques depuis le web pour éviter curl
            os.environ['YOLO_OFFLINE'] = '1'
            os.environ['ULTRALYTICS_OFFLINE'] = '1'
            os.environ['YOLO_NO_DOWNLOADS'] = '1'
            
            # SÉCURITÉ: Désactiver les téléchargements HuggingFace et autres
            os.environ['HF_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            # Désactiver le logging externe qui peut causer des problèmes
            os.environ['WANDB_DISABLED'] = 'true'
            os.environ['COMET_OFFLINE'] = 'true'
            
            # Forcer le mode local pour éviter les appels réseau
            os.environ['ULTRALYTICS_NO_HUB'] = '1'
            
            # Configuration du logging Ultralytics pour éviter les erreurs de stream
            import logging
            # Créer un handler qui fonctionne dans QGIS
            ultralytics_logger = logging.getLogger('ultralytics')
            ultralytics_logger.setLevel(logging.ERROR)
            
            # Supprimer les handlers par défaut qui peuvent causer des problèmes
            for handler in ultralytics_logger.handlers[:]:
                ultralytics_logger.removeHandler(handler)
            
            # Ajouter un handler simple qui écrit dans la console QGIS
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.ERROR)
            ultralytics_logger.addHandler(console_handler)
            
            print(f"✅ Chemins Ultralytics configurés localement")
            print(f"📁 Modèles: {models_dir}")
            print(f"📁 Runs: {runs_dir}")
            
        except Exception as e:
            print(f"⚠️ Erreur configuration Ultralytics: {e}")

    def _verify_integrated_models(self):
        """Vérifie la présence et l'intégrité des modèles intégrés"""
        plugin_dir = Path(__file__).parent.parent
        models_dir = plugin_dir / 'models' / 'pretrained'
        
        expected_models = {
            'yolo11n.pt': {'min_size': 5000000, 'max_size': 6000000},    # ~5.4 MB
            'yolo11s.pt': {'min_size': 18000000, 'max_size': 20000000},  # ~19 MB
            'yolo11m.pt': {'min_size': 38000000, 'max_size': 41000000}   # ~39 MB
        }
        
        available_models = []
        for model_name, constraints in expected_models.items():
            model_path = models_dir / model_name
            if model_path.exists():
                size = model_path.stat().st_size
                if constraints['min_size'] <= size <= constraints['max_size']:
                    available_models.append(model_name)
                    print(f"✅ Modèle intégré OK: {model_name} ({size/1024/1024:.1f} MB)")
                else:
                    print(f"⚠️ Modèle corrompu: {model_name} (taille: {size/1024/1024:.1f} MB)")
            else:
                print(f"❌ Modèle manquant: {model_name}")
        
        if available_models:
            print(f"🎯 {len(available_models)}/3 modèles YOLO intégrés disponibles")
            # Sélection automatique du meilleur modèle selon les performances
            self.recommended_model = self._select_optimal_model(available_models)
            print(f"💡 Modèle recommandé pour votre système: {self.recommended_model}")
        else:
            print(f"⚠️ Aucun modèle intégré disponible - mode dégradé")
            self.recommended_model = None

    def _select_optimal_model(self, available_models: list) -> str:
        """Sélectionne le modèle optimal selon les capacités du système"""
        # Sélection intelligente basée sur les performances du système
        if self.device.startswith('cuda'):
            # GPU disponible - peut utiliser des modèles plus lourds
            if 'yolo11m.pt' in available_models:
                return 'yolo11m.pt'  # Maximum performance
            elif 'yolo11s.pt' in available_models:
                return 'yolo11s.pt'  # Bon compromis
        
        # CPU ou GPU limité - privilégier la vitesse
        if 'yolo11s.pt' in available_models:
            return 'yolo11s.pt'  # Équilibré par défaut
        elif 'yolo11n.pt' in available_models:
            return 'yolo11n.pt'  # Léger
        elif 'yolo11m.pt' in available_models:
            return 'yolo11m.pt'  # Précis mais plus lent
        
        return available_models[0] if available_models else None

    def _load_base_model_safely(self, base_model: str):
        """Charge un modèle de base en évitant les téléchargements non autorisés"""
        try:
            # Priorité 1: Modèles intégrés dans l'extension
            plugin_dir = Path(__file__).parent.parent
            integrated_model_path = plugin_dir / 'models' / 'pretrained' / base_model
            
            if integrated_model_path.exists():
                print(f"🎯 Utilisation du modèle intégré: {integrated_model_path}")
                return YOLO(str(integrated_model_path))
            
            # Priorité 2: Modèles dans le répertoire de données utilisateur
            secure_dir = plugin_dir.parent / 'qgis_yolo_detector_data'
            local_model_path = secure_dir / 'models' / 'pretrained' / base_model
            
            if local_model_path.exists():
                print(f"📁 Utilisation du modèle local utilisateur: {local_model_path}")
                return YOLO(str(local_model_path))
            
            # Créer un modèle basique si le modèle standard n'est pas disponible
            print(f"⚠️ Modèle {base_model} non trouvé localement")
            print(f"💡 Création d'un modèle basique pour l'entraînement...")
            
            # Essayer de charger un modèle sans téléchargement
            import tempfile
            
            # Utiliser notre configuration locale
            local_config_path = plugin_dir / 'models' / 'yolov8n_local.yaml'
            
            if local_config_path.exists():
                print(f"🔧 Utilisation de la configuration locale: {local_config_path}")
                model = YOLO(str(local_config_path))
            else:
                print(f"⚠️ Configuration locale non trouvée")
                print(f"🛠️ Création d'un modèle minimal depuis scratch...")
                
                # Solution de contournement : créer un fichier de modèle vide local
                print(f"🛠️ Création d'un modèle vide local pour contourner les téléchargements...")
                
                local_models_dir = secure_dir / 'models' / 'pretrained'
                local_models_dir.mkdir(parents=True, exist_ok=True)
                
                # Utiliser notre configuration YAML locale à la place
                yaml_config_path = plugin_dir / 'models' / 'yolov8n_local.yaml'
                try:
                    if yaml_config_path.exists():
                        print(f"🔧 Utilisation de la configuration YAML locale: {yaml_config_path}")
                        model = YOLO(str(yaml_config_path))
                        return model
                    else:
                        raise FileNotFoundError("Configuration YAML locale non trouvée")
                except Exception as e_yaml:
                    print(f"❌ Erreur avec YAML local: {e_yaml}")
                    # Dernier recours : message d'erreur informatif
                    raise Exception(
                        f"❌ Impossible de créer un modèle YOLO sans téléchargement.\n"
                        f"Solutions possibles:\n"
                        f"1. Téléchargez manuellement yolo11n.pt depuis https://github.com/ultralytics/assets/releases/\n"
                        f"2. Placez-le dans: {local_models_dir}\n"
                        f"3. Ou autorisez temporairement les téléchargements"
                    )
            
            return model
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle de base: {e}")
            raise Exception(f"Impossible de charger le modèle de base {base_model}: {e}")

    def _calculate_optimal_batch_size(self) -> int:
        """
        Calcule la taille de batch optimale selon le hardware disponible
        
        Returns:
            int: Taille de batch recommandée
        """
        if self.device.startswith('cuda'):
            # Estimation basée sur la mémoire GPU
            try:
                free_memory_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                if free_memory_gb > 8:
                    return 32
                elif free_memory_gb > 4:
                    return 16
                elif free_memory_gb > 2:
                    return 8
                else:
                    return 4
            except:
                return 8
        else:
            # CPU ou MPS - batch size plus conservateur
            return 4

    def get_current_layer_pixel_size(self, layer):
        """Calcule la résolution actuelle de la couche"""
        try:
            extent = layer.extent()
            provider = layer.dataProvider()
            
            if hasattr(provider, 'xSize') and hasattr(provider, 'ySize'):
                # Résolution = étendue / nombre de pixels
                pixel_size_x = (extent.xMaximum() - extent.xMinimum()) / provider.xSize()
                pixel_size_y = (extent.yMaximum() - extent.yMinimum()) / provider.ySize()
                return {'x': pixel_size_x, 'y': pixel_size_y}
        except Exception as e:
            print(f"⚠️ Erreur calcul résolution: {e}")
        
        return {'x': 1.0, 'y': 1.0}  # Défaut
    
    def calculate_scale_compatibility(self, current_pixel_size: float, 
                                    training_pixel_size: float) -> float:
        """
        Calcule la compatibilité d'échelle entre résolution actuelle et d'entraînement
        
        Returns:
            float: Score 0-1 (1 = parfaitement compatible)
        """
        if training_pixel_size <= 0:
            return 0.0
        
        scale_ratio = current_pixel_size / training_pixel_size
        
        # Courbe de compatibilité : optimal à 1.0, décroît exponentiellement
        if scale_ratio <= 0:
            return 0.0
        elif scale_ratio <= 0.25 or scale_ratio >= 4.0:
            return 0.1  # Très incompatible
        elif scale_ratio <= 0.5 or scale_ratio >= 2.0:
            return 0.5  # Moyennement compatible
        else:
            # Fonction gaussienne centrée sur 1.0
            import math
            return math.exp(-0.5 * ((math.log(scale_ratio)) ** 2) / (0.3 ** 2))

    def load_model(self, model_path: str, force_reload: bool = False) -> bool:
        """
        Charge un modèle YOLO avec mise en cache
        
        Args:
            model_path: Chemin vers le modèle (.pt)
            force_reload: Force le rechargement même si en cache
            
        Returns:
            bool: True si le chargement a réussi
        """
        # Vérification de l'existence du fichier
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Utilisation du cache si disponible
        if not force_reload and self.current_model_path == model_path:
            return True
            
        cached_model = self.model_cache.get(model_path)
        if cached_model and not force_reload:
            self.current_model = cached_model
            self.current_model_path = model_path
            return True
        
        try:
            # Chargement du modèle avec gestion NumPy 2.x
            print(f"Loading YOLO model: {model_path}")
            
            # Workaround pour NumPy 2.x (correction: utiliser Warning)
            import warnings
            with warnings.catch_warnings():
                try:
                    warnings.filterwarnings("ignore", category=Warning, module="numpy")
                    warnings.filterwarnings("ignore", message=".*NumPy.*")
                    warnings.filterwarnings("ignore", category=Warning, message=".*category.*")
                except Exception:
                    pass  # Ignorer les erreurs de configuration des warnings
                
                model = YOLO(model_path)
            
            # Configuration du device
            model.to(self.device)
            
            # Mise en cache
            self.model_cache.put(model_path, model)
            self.current_model = model
            self.current_model_path = model_path
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model {model_path}: {str(e)}")
            return False

    def detect_objects(self, image_array: np.ndarray, 
                      confidence_threshold: Optional[float] = None,
                      iou_threshold: Optional[float] = None,
                      target_classes: Optional[List[int]] = None) -> List[Dict]:
        """
        Détecte des objets sur une image
        
        Args:
            image_array: Image en format numpy array (H, W, C)
            confidence_threshold: Seuil de confiance (0.0-1.0)
            iou_threshold: Seuil IoU pour NMS
            target_classes: Liste des classes cibles (None = toutes)
            
        Returns:
            List[Dict]: Liste des détections avec format:
                {
                    'bbox': [x1, y1, x2, y2],  # Coordonnées absolues
                    'bbox_normalized': [x, y, w, h],  # Format YOLO normalisé
                    'confidence': float,
                    'class_id': int,
                    'class_name': str
                }
        """
        if self.current_model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Paramètres par défaut
        conf_threshold = confidence_threshold or self.default_confidence
        iou_threshold = iou_threshold or self.default_iou_threshold
        
        print(f"🔍 DEBUG YOLO_ENGINE: detect_objects appelé")
        print(f"🔍 DEBUG YOLO_ENGINE: conf_threshold={conf_threshold}, iou_threshold={iou_threshold}")
        print(f"🔍 DEBUG YOLO_ENGINE: current_model={self.current_model_name}")
        print(f"🔍 DEBUG YOLO_ENGINE: image_array shape={image_array.shape}, dtype={image_array.dtype}")
        
        try:
            # Workaround pour NumPy 2.x - supprimer warnings (correction: utiliser Warning)
            import warnings
            with warnings.catch_warnings():
                try:
                    warnings.filterwarnings("ignore", category=Warning, module="numpy")
                    warnings.filterwarnings("ignore", message=".*NumPy.*")
                    warnings.filterwarnings("ignore", category=Warning, message=".*category.*")
                except Exception:
                    pass  # Ignorer les erreurs de configuration des warnings
                
                # Détection avec YOLO
                print(f"🔍 DEBUG YOLO_ENGINE: Appel model.predict...")
                results = self.current_model(
                    image_array,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                print(f"🔍 DEBUG YOLO_ENGINE: Résultats reçus: {len(results) if results else 0} résultat(s)")
            
            # Traitement des résultats
            detections = []
            if results and len(results) > 0:
                result = results[0]  # Premier résultat
                print(f"🔍 DEBUG YOLO_ENGINE: result.boxes présent: {hasattr(result, 'boxes')}")
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    print(f"🔍 DEBUG YOLO_ENGINE: Nombre de boxes: {len(boxes)}")
                    
                    for i in range(len(boxes)):
                        # Coordonnées de la bounding box
                        bbox_xyxy = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Filtrage par classe si spécifié
                        if target_classes and class_id not in target_classes:
                            continue
                        
                        # Conversion en format YOLO normalisé
                        h, w = image_array.shape[:2]
                        x1, y1, x2, y2 = bbox_xyxy
                        
                        # Format YOLO: centre_x, centre_y, largeur, hauteur (normalisés)
                        bbox_normalized = [
                            ((x1 + x2) / 2) / w,  # centre_x
                            ((y1 + y2) / 2) / h,  # centre_y
                            (x2 - x1) / w,        # largeur
                            (y2 - y1) / h         # hauteur
                        ]
                        
                        # Nom de la classe
                        class_name = self.current_model.names.get(class_id, f"class_{class_id}")
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'bbox_normalized': bbox_normalized,
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'image_shape': (h, w)
                        }
                        
                        detections.append(detection)
            
            print(f"🔍 DEBUG YOLO_ENGINE: Retour {len(detections)} détections finales")
            return detections
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []

    def batch_detect(self, image_arrays: List[np.ndarray],
                    confidence_threshold: Optional[float] = None,
                    iou_threshold: Optional[float] = None,
                    progress_callback: Optional[Callable] = None) -> List[List[Dict]]:
        """
        Détection par batch sur plusieurs images
        
        Args:
            image_arrays: Liste d'images numpy
            confidence_threshold: Seuil de confiance
            iou_threshold: Seuil IoU
            progress_callback: Callback de progression (progress: float 0-1)
            
        Returns:
            List[List[Dict]]: Détections pour chaque image
        """
        if not image_arrays:
            return []
            
        results = []
        total_images = len(image_arrays)
        
        # Traitement par batch selon la capacité du device
        for i in range(0, total_images, self.batch_size):
            batch = image_arrays[i:i + self.batch_size]
            batch_results = []
            
            for j, image in enumerate(batch):
                detections = self.detect_objects(
                    image, 
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold
                )
                batch_results.append(detections)
                
                # Callback de progression
                if progress_callback:
                    progress = (i + j + 1) / total_images
                    progress_callback(progress)
            
            results.extend(batch_results)
            
            # Nettoyage mémoire entre les batches
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            gc.collect()
        
        return results

    def train_custom_model(self, dataset_config_path: str,
                          base_model: str = 'yolov8n.pt',
                          epochs: int = 100,  # ✅ Plus d'époques pour petit dataset
                          batch_size: Optional[int] = None,
                          learning_rate: float = 0.01,  # ✅ LR plus élevé pour transfer learning
                          patience: int = 25,  # ✅ Plus de patience
                          progress_callback: Optional[Callable] = None) -> Dict:
        """
        Entraîne un modèle personnalisé via transfer learning
        
        Args:
            dataset_config_path: Chemin vers le fichier dataset.yaml
            base_model: Modèle de base pour transfer learning
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille de batch (None = auto)
            learning_rate: Taux d'apprentissage
            patience: Patience pour early stopping
            progress_callback: Callback de progression
            
        Returns:
            Dict: Résultats d'entraînement
        """
        if not os.path.exists(dataset_config_path):
            raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
        
        try:
            # Gestion sécurisée des modèles de base
            model = self._load_base_model_safely(base_model)
            
            # Configuration de l'entraînement
            train_batch_size = batch_size or self.batch_size
            
            # Callback personnalisé pour le suivi de progression
            def on_epoch_end(trainer):
                if progress_callback:
                    epoch = trainer.epoch
                    total_epochs = trainer.args.epochs
                    progress = (epoch + 1) / total_epochs
                    progress_callback(progress, {
                        'epoch': epoch + 1,
                        'total_epochs': total_epochs,
                        'loss': float(trainer.loss) if hasattr(trainer, 'loss') else 0.0
                    })
            
            # Configuration du répertoire de sortie pour éviter l'erreur "runs"
            plugin_dir = Path(__file__).parent.parent
            secure_dir = plugin_dir.parent / 'qgis_yolo_detector_data'
            runs_dir = secure_dir / 'runs'
            runs_dir.mkdir(parents=True, exist_ok=True)
            
            # Configuration des variables d'environnement pour Ultralytics
            os.environ['ULTRALYTICS_RUNS_DIR'] = str(runs_dir)
            
            # Désactiver les logs verbeux d'Ultralytics pour éviter les erreurs de stream
            import logging
            ultralytics_logger = logging.getLogger('ultralytics')
            ultralytics_logger.setLevel(logging.ERROR)
            
            # Entraînement
            print(f"Starting training on {self.device}")
            results = model.train(
                data=dataset_config_path,
                epochs=epochs,
                batch=train_batch_size,
                lr0=learning_rate,
                patience=patience,
                device=self.device,
                save=True,
                save_period=10,  # Sauvegarde intermédiaire
                val=True,
                plots=False,  # Désactiver les plots pour éviter les erreurs
                verbose=False,  # Réduire la verbosité
                exist_ok=True,  # Écrase les résultats précédents
                project=str(runs_dir),  # Répertoire projet explicite
                name='yolo_training'  # Nom du run
            )
            
            # Récupération des chemins de sortie
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            last_model_path = results.save_dir / 'weights' / 'last.pt'
            
            training_results = {
                'success': True,
                'best_model_path': str(best_model_path),
                'last_model_path': str(last_model_path),
                'save_dir': str(results.save_dir),
                'final_metrics': results.results_dict if hasattr(results, 'results_dict') else {},
                'training_time': getattr(results, 'training_time', 0),
                'device_used': self.device
            }
            
            print(f"Training completed successfully!")
            print(f"Best model saved to: {best_model_path}")
            
            return training_results
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'device_used': self.device
            }

    def get_model_info(self) -> Dict:
        """
        Retourne les informations sur le modèle actuellement chargé
        
        Returns:
            Dict: Informations sur le modèle
        """
        if self.current_model is None:
            return {'loaded': False}
        
        try:
            return {
                'loaded': True,
                'model_path': self.current_model_path,
                'classes': list(self.current_model.names.values()),
                'num_classes': len(self.current_model.names),
                'device': self.device,
                'model_type': 'YOLO'
            }
        except:
            return {'loaded': False, 'error': 'Could not retrieve model info'}

    def cleanup(self):
        """
        Nettoie les ressources (modèles, cache GPU)
        """
        self.model_cache.cache.clear()
        self.current_model = None
        self.current_model_path = None
        
        # Nettoyage GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        print("YOLOEngine cleaned up")


# TODO pour Claude Code:
# 1. Tester le chargement de modèles YOLO pré-entraînés
# 2. Implémenter des tests unitaires pour les détections
# 3. Optimiser la gestion mémoire pour les gros datasets
# 4. Ajouter support pour d'autres formats d'images
# 5. Implémenter le logging détaillé pour debugging
# 6. Ajouter la validation des paramètres d'entrée
# 7. Gérer les cas d'erreur de manière plus robuste
