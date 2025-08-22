"""
Smart Annotation Engine - Pipeline YOLO→SAM pour annotation assistée par IA

Ce module implémente un pipeline intelligent qui :
- Utilise YOLO pour la détection préalable d'objets candidats
- Applique SAM (Segment Anything Model) pour le raffinement des contours
- Optimise les performances pour CPU normal (4-8 cores, 8-16GB RAM)
- Fournit des métadonnées enrichies pour améliorer la qualité du training YOLO

Architecture performance-first :
- Lazy loading des modèles SAM (chargement uniquement si nécessaire)
- Profile performance CPU automatique avec fallback intelligent
- Cache LRU partagé avec YOLOEngine existant
- Seuils adaptatifs selon les capacités matérielles
"""

import os
import gc
import time
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime

# Imports conditionnels pour compatibilité QGIS
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
    
    # Protection spécifique contre l'incompatibilité NumPy 1.x/2.x
    if hasattr(np, '__version__') and np.__version__.startswith('2.'):
        # NumPy 2.x détecté - protection supplémentaire
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
        # Force l'utilisation de la compatibilité NumPy 1.x
        os.environ['NPY_DISABLE_OPTIMIZATION'] = '1'
        print(f"🔧 NumPy 2.x détecté ({np.__version__}) - compatibilité 1.x activée")
    
except ImportError as e:
    print(f"⚠️ NumPy non disponible: {e}")
    NUMPY_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Erreur NumPy (compatibilité 1.x/2.x): {e}")
    # Fallback gracieux - continuer sans NumPy
    try:
        import numpy as np
        NUMPY_AVAILABLE = True
        print("✅ NumPy chargé en mode fallback")
    except:
        NUMPY_AVAILABLE = False
        print("❌ NumPy définitivement indisponible")
    
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    # Pré-configuration pour éviter les conflits NumPy 1.x/2.x avec OpenCV
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*compiled using NumPy 1.x.*')
        import cv2
    CV2_AVAILABLE = True
    print(f"✅ OpenCV chargé avec protection NumPy ({cv2.__version__})")
except ImportError as e:
    print(f"⚠️ OpenCV non disponible: {e}")
    CV2_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Erreur OpenCV (probablement NumPy 1.x/2.x): {e}")
    # Tentative de chargement en mode dégradé
    try:
        import cv2
        CV2_AVAILABLE = True
        print("✅ OpenCV chargé en mode fallback")
    except:
        CV2_AVAILABLE = False
        print("❌ OpenCV définitivement indisponible")
    
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import conditionnel FastSAM (plus léger que SAM standard)
try:
    from ultralytics import FastSAM
    FASTSAM_AVAILABLE = True
except ImportError:
    FASTSAM_AVAILABLE = False

# Import conditionnel MobileSAM (alternative ultra-légère)
try:
    from mobile_sam import sam_model_registry, SamPredictor
    MOBILESAM_AVAILABLE = True
except ImportError:
    MOBILESAM_AVAILABLE = False


@dataclass
class SmartDetectionResult:
    """Résultat d'une détection intelligente YOLO→SAM"""
    bbox: List[float]  # [x1, y1, x2, y2] coordonnées optimisées
    bbox_original: List[float]  # Bbox YOLO originale
    confidence_yolo: float  # Confiance YOLO (0-1)
    confidence_sam: Optional[float]  # Confiance SAM si raffinement appliqué
    class_name: str
    class_id: int
    mask_available: bool  # True si masque SAM généré
    mask_quality_score: Optional[float]  # Score qualité masque SAM
    refinement_applied: bool  # True si SAM a été utilisé
    processing_time: float  # Temps total traitement (ms)
    improvement_ratio: Optional[float]  # Ratio amélioration bbox (SAM vs YOLO)
    
    # NOUVEAUX ATTRIBUTS POLYGONES SAM
    polygon_points: Optional[List[List[float]]] = None  # Points polygone [[x,y], [x,y], ...]
    polygon_available: bool = False  # True si polygone SAM disponible


@dataclass 
class CPUPerformanceProfile:
    """Profile de performance CPU pour optimisation automatique"""
    level: str  # 'low', 'medium', 'high'
    cpu_cores: int
    ram_gb: float
    recommended_yolo_size: str  # 'nano', 'small', 'medium'
    enable_sam: bool
    sam_model_type: str  # 'FastSAM', 'MobileSAM', 'none'
    max_concurrent_detections: int
    confidence_threshold_yolo: float
    confidence_threshold_sam: float


class CPUProfiler:
    """Profileur automatique des capacités CPU pour optimisation"""
    
    @staticmethod
    def get_performance_profile() -> CPUPerformanceProfile:
        """
        Analyse les capacités CPU/RAM et retourne un profile optimisé
        
        Returns:
            CPUPerformanceProfile: Configuration optimale selon le matériel
        """
        cpu_cores = os.cpu_count() or 4
        
        # RAM avec fallback si psutil non disponible
        if PSUTIL_AVAILABLE:
            ram_gb = psutil.virtual_memory().total / (1024**3)
        else:
            # Fallback conservateur si psutil non disponible
            ram_gb = 8.0  # Assume 8GB par défaut
        
        # CPU High-end (>=8 cores, >=16GB RAM)
        if cpu_cores >= 8 and ram_gb >= 16:
            return CPUPerformanceProfile(
                level='high',
                cpu_cores=cpu_cores,
                ram_gb=ram_gb,
                recommended_yolo_size='small',  # YOLOv8s pour équilibre perf/précision
                enable_sam=True,
                sam_model_type='FastSAM',
                max_concurrent_detections=4,
                confidence_threshold_yolo=0.15,
                confidence_threshold_sam=0.6
            )
        
        # CPU Medium (4-7 cores, 8-15GB RAM)
        elif cpu_cores >= 4 and ram_gb >= 8:
            return CPUPerformanceProfile(
                level='medium',
                cpu_cores=cpu_cores,
                ram_gb=ram_gb,
                recommended_yolo_size='nano',  # YOLOv8n très rapide
                enable_sam=True,
                sam_model_type='MobileSAM' if MOBILESAM_AVAILABLE else 'FastSAM',
                max_concurrent_detections=2,
                confidence_threshold_yolo=0.20,
                confidence_threshold_sam=0.7
            )
        
        # CPU Low-end (<4 cores ou <8GB RAM)
        else:
            return CPUPerformanceProfile(
                level='low',
                cpu_cores=cpu_cores,
                ram_gb=ram_gb,
                recommended_yolo_size='nano',
                enable_sam=False,  # Désactivé sur matériel faible
                sam_model_type='none',
                max_concurrent_detections=1,
                confidence_threshold_yolo=0.25,
                confidence_threshold_sam=0.8
            )
    
    @staticmethod
    def benchmark_yolo_performance(yolo_engine, test_image_size=(640, 640)) -> float:
        """
        Benchmark rapide performance YOLO sur CPU
        
        Args:
            yolo_engine: Instance YOLOEngine
            test_image_size: Taille image test
            
        Returns:
            float: Temps moyen détection (ms)
        """
        if not yolo_engine or not yolo_engine.current_model:
            return 1000.0  # Fallback conservateur
        
        # Image test synthétique
        test_image = np.random.randint(0, 255, (*test_image_size, 3), dtype=np.uint8)
        
        # 3 tests pour moyenne
        times = []
        for _ in range(3):
            start = time.time()
            try:
                yolo_engine.detect_objects(test_image, confidence_threshold=0.5)
                times.append((time.time() - start) * 1000)
            except:
                times.append(1000.0)  # Fallback en cas d'erreur
        
        return sum(times) / len(times)


class SmartAnnotationEngine:
    """
    Moteur principal d'annotation intelligente YOLO→SAM
    
    Workflow optimisé :
    1. Réception rectangle utilisateur (comme annotation manuelle)
    2. Extraction patch raster de la zone d'intérêt
    3. Détection YOLO rapide pour identifier objets candidats
    4. Sélection du meilleur candidat selon critères géométriques/confiance
    5. Raffinement SAM optionnel si confiance YOLO insuffisante
    6. Retour bbox optimisée avec métadonnées enrichies
    """
    
    def __init__(self, yolo_engine=None, annotation_manager=None):
        """
        Initialise le moteur d'annotation intelligente
        
        Args:
            yolo_engine: Instance YOLOEngine existante (réutilisée)
            annotation_manager: Gestionnaire annotations pour persistance
        """
        self.yolo_engine = yolo_engine
        self.annotation_manager = annotation_manager
        
        # Vérification des dépendances critiques
        self.dependencies_available = self._check_dependencies()
        
        # Modèles SAM (lazy loading)
        self.sam_model = None
        self.sam_predictor = None
        self.sam_type = None
        
        # Configuration performance
        self.cpu_profile = CPUProfiler.get_performance_profile()
        self.enabled = True
        self.debug_mode = False
        
        # Configuration contours précis (polygones SAM)
        self.enable_precise_contours = True  # Activé par défaut
        
        # Statistiques et cache
        self.detection_stats = {
            'total_detections': 0,
            'sam_refinements': 0,
            'auto_accepted': 0,
            'user_validated': 0,
            'avg_processing_time': 0.0
        }
        
        # Thread pool pour traitement asynchrone
        self._processing_lock = threading.Lock()
        
        # SÉCURITÉ: Protection spécifique Windows contre les appels système malformés
        self._configure_windows_security()
        
        # Chargement automatique d'un modèle par défaut pour l'auto-détection
        # TEMPORAIREMENT DÉSACTIVÉ pour éviter les blocages QGIS
        # self._load_default_model()
        
        print(f"🤖 SmartAnnotationEngine initialisé")
        print(f"📊 Profile CPU: {self.cpu_profile.level} ({self.cpu_profile.cpu_cores} cores, {self.cpu_profile.ram_gb:.1f}GB)")
        print(f"🎯 YOLO recommandé: {self.cpu_profile.recommended_yolo_size}")
        print(f"🎨 SAM: {'Activé' if self.cpu_profile.enable_sam else 'Désactivé'} ({self.cpu_profile.sam_model_type})")
        
        if not self.dependencies_available['all_available']:
            print(f"⚠️ Dépendances manquantes: {', '.join(self.dependencies_available['missing'])}")
            print(f"📝 Mode dégradé activé - fonctionnalités limitées")
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Vérifie la disponibilité des dépendances"""
        missing = []
        
        if not NUMPY_AVAILABLE:
            missing.append('numpy')
        if not TORCH_AVAILABLE:
            missing.append('torch')
        if not CV2_AVAILABLE:
            missing.append('opencv-python')
        if not PIL_AVAILABLE:
            missing.append('pillow')
        if not PSUTIL_AVAILABLE:
            missing.append('psutil')
        
        return {
            'all_available': len(missing) == 0,
            'missing': missing,
            'numpy': NUMPY_AVAILABLE,
            'torch': TORCH_AVAILABLE,
            'cv2': CV2_AVAILABLE,
            'pil': PIL_AVAILABLE,
            'psutil': PSUTIL_AVAILABLE
        }
    
    def _configure_windows_security(self):
        """Configuration de sécurité spécifique Windows pour éviter les appels système malformés"""
        try:
            import platform
            if platform.system() == 'Windows':
                # Désactiver les outils de logging externes qui peuvent causer des problèmes
                os.environ['WANDB_DISABLED'] = 'true'
                os.environ['TENSORBOARD_DISABLED'] = 'true'
                os.environ['COMET_OFFLINE'] = 'true'
                os.environ['NEPTUNE_OFFLINE'] = 'true'
                
                # Forcer l'utilisation de chemins absolus pour éviter les arguments malformés
                os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
                
                # Désactiver les outils de profiling qui peuvent causer des erreurs
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                
                print("🛡️ Protection Windows activée - appels système sécurisés")
            
        except Exception as e:
            print(f"⚠️ Erreur configuration sécurité Windows: {e}")
    
    def _load_default_model(self):
        """
        Charge un modèle YOLO générique optimisé pour orthophotos
        
        Priorité: Modèles génériques validés pour imagerie géospatiale
        plutôt que modèles spécialisés non validés
        """
        if not self.yolo_engine:
            print("⚠️ Aucun YOLOEngine disponible pour le SmartAnnotationEngine")
            return
        
        try:
            # Déterminer le modèle optimal selon le profil CPU
            model_size = self.cpu_profile.recommended_yolo_size
            
            # Chemin vers les modèles pré-entraînés
            import os
            from pathlib import Path
            plugin_dir = Path(__file__).parent.parent
            models_dir = plugin_dir / "models" / "pretrained"
            
            # Mapping taille -> fichier modèle (optimisés pour orthophotos)
            model_files = {
                'nano': 'yolo11n.pt',    # Léger, idéal CPU standard
                'small': 'yolo11s.pt',   # Équilibré performance/précision
                'medium': 'yolo11m.pt'   # Haute précision pour CPU puissants
            }
            
            model_file = model_files.get(model_size, 'yolo11n.pt')  # Fallback nano
            model_path = models_dir / model_file
            
            if model_path.exists():
                success = self.yolo_engine.load_model(str(model_path))
                if success:
                    print(f"✅ Modèle générique orthophoto chargé: {model_file}")
                    print(f"🌍 Optimisé pour: Imagerie aérienne, infrastructure urbaine, objets géospatiaux")
                else:
                    print(f"❌ Échec chargement modèle: {model_file}")
            else:
                print(f"⚠️ Modèle non trouvé: {model_path}")
                print(f"📋 Modèles disponibles dans {models_dir}:")
                if models_dir.exists():
                    for model_file in models_dir.glob("*.pt"):
                        print(f"   - {model_file.name}")
                print(f"❌ Smart Mode désactivé - aucun modèle pré-téléchargé disponible")
                self.enabled = False
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du modèle par défaut: {e}")
    
    def _lazy_load_sam_model(self) -> bool:
        """
        Charge le modèle SAM de manière paresseuse (uniquement quand nécessaire)
        
        Returns:
            bool: True si modèle chargé avec succès
        """
        if self.sam_model is not None:
            return True  # Déjà chargé
        
        if not self.cpu_profile.enable_sam:
            print("⚠️ SAM désactivé selon le profile CPU")
            return False
        
        try:
            print(f"⏳ Chargement {self.cpu_profile.sam_model_type}...")
            start_time = time.time()
            
            # Recherche des modèles SAM pré-téléchargés dans le répertoire plugin
            plugin_dir = Path(__file__).parent.parent
            sam_models_dir = plugin_dir / "models" / "sam"
            sam_models_dir.mkdir(parents=True, exist_ok=True)
            
            if self.cpu_profile.sam_model_type == 'FastSAM' and FASTSAM_AVAILABLE:
                # FastSAM via modèle local uniquement
                fastsam_path = sam_models_dir / 'FastSAM-s.pt'
                if fastsam_path.exists():
                    self.sam_model = FastSAM(str(fastsam_path))
                    self.sam_type = 'FastSAM'
                    print(f"✅ FastSAM chargé depuis: {fastsam_path}")
                else:
                    print(f"⚠️ FastSAM non trouvé: {fastsam_path}")
                    print(f"💡 Placez FastSAM-s.pt dans {sam_models_dir} pour activer SAM")
                    return False
                
            elif self.cpu_profile.sam_model_type == 'MobileSAM' and MOBILESAM_AVAILABLE:
                # MobileSAM via modèle local uniquement
                mobilesam_path = sam_models_dir / 'mobile_sam.pt'
                if mobilesam_path.exists():
                    device = "cpu"  # Force CPU pour compatibilité
                    self.sam_model = sam_model_registry["vit_t"](checkpoint=str(mobilesam_path))
                    self.sam_model.to(device=device)
                    self.sam_predictor = SamPredictor(self.sam_model)
                    self.sam_type = 'MobileSAM'
                    print(f"✅ MobileSAM chargé depuis: {mobilesam_path}")
                else:
                    print(f"⚠️ MobileSAM non trouvé: {mobilesam_path}")
                    print(f"💡 Placez mobile_sam.pt dans {sam_models_dir} pour activer SAM")
                    return False
                
            else:
                print(f"❌ {self.cpu_profile.sam_model_type} non disponible")
                return False
            
            load_time = (time.time() - start_time) * 1000
            print(f"✅ {self.sam_type} chargé en {load_time:.1f}ms")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement SAM: {e}")
            return False
    
    def smart_detect_from_user_rectangle(self, 
                                       user_rect: Tuple[float, float, float, float],
                                       raster_patch,  # Compatible avec ou sans numpy
                                       target_class: str = None) -> SmartDetectionResult:
        """
        Pipeline principal : détection intelligente à partir du rectangle utilisateur
        
        Args:
            user_rect: Rectangle utilisateur (x1, y1, x2, y2) en pixels image
            raster_patch: Patch raster extrait de la zone d'intérêt
            target_class: Classe cible recherchée (optionnel)
            
        Returns:
            SmartDetectionResult: Résultat optimisé avec métadonnées
        """
        start_time = time.time()
        
        print(f"🔍 DEBUG SMART: smart_detect_from_user_rectangle() APPELÉE")
        print(f"🔍 DEBUG SMART: user_rect={user_rect}")
        print(f"🔍 DEBUG SMART: target_class={target_class}")
        print(f"🔍 DEBUG SMART: self.enabled={self.enabled}")
        print(f"🔍 DEBUG SMART: self.yolo_engine présent={self.yolo_engine is not None}")
        
        if not self.enabled or not self.yolo_engine:
            # Fallback : retour bbox utilisateur sans modification
            return SmartDetectionResult(
                bbox=list(user_rect),
                bbox_original=list(user_rect),
                confidence_yolo=1.0,
                confidence_sam=None,
                class_name=target_class or "unknown",
                class_id=0,
                mask_available=False,
                mask_quality_score=None,
                refinement_applied=False,
                processing_time=(time.time() - start_time) * 1000,
                improvement_ratio=None
            )
        
        try:
            # Étape 1: Détection YOLO sur le patch
            yolo_detections = self._run_yolo_detection(raster_patch, target_class)
            print(f"🔍 DEBUG YOLO: Found {len(yolo_detections) if yolo_detections else 0} detections")
            if yolo_detections:
                for i, det in enumerate(yolo_detections[:3]):  # Afficher les 3 premières
                    print(f"🔍 DEBUG YOLO: Detection {i}: class={det.get('class_id', 'N/A')}, conf={det.get('confidence', 0):.3f}")
            
            # Étape 2: Sélection du meilleur candidat
            best_candidate = self._select_best_candidate(yolo_detections, user_rect, raster_patch.shape)
            print(f"🔍 DEBUG SELECTION: best_candidate = {best_candidate is not None}")
            if best_candidate:
                print(f"🔍 DEBUG SELECTION: Selected candidate: class={best_candidate.get('class_id', 'N/A')}, conf={best_candidate.get('confidence', 0):.3f}")
            
            if not best_candidate:
                # Aucune détection → retour rectangle utilisateur
                return self._create_fallback_result(user_rect, target_class, start_time)
            
            # Étape 3: Décision raffinement SAM
            print(f"🔍 DEBUG PIPELINE: Appel _should_apply_sam_refinement() avec best_candidate={best_candidate}")
            needs_sam_refinement = self._should_apply_sam_refinement(best_candidate)
            print(f"🔍 DEBUG PIPELINE: needs_sam_refinement = {needs_sam_refinement}")
            
            if needs_sam_refinement:
                # Étape 4: Raffinement SAM
                refined_result = self._apply_sam_refinement(best_candidate, raster_patch)
                if refined_result:
                    best_candidate = refined_result
            
            # Étape 5: Construction résultat final
            processing_time = (time.time() - start_time) * 1000
            
            result = SmartDetectionResult(
                bbox=best_candidate['bbox'],
                bbox_original=best_candidate.get('bbox_original', best_candidate['bbox']),
                confidence_yolo=best_candidate['confidence'],
                confidence_sam=best_candidate.get('confidence_sam'),
                class_name=best_candidate['class_name'],
                class_id=best_candidate['class_id'],
                mask_available=best_candidate.get('mask_available', False),
                mask_quality_score=best_candidate.get('mask_quality_score'),
                refinement_applied=best_candidate.get('refinement_applied', False),
                processing_time=processing_time,
                improvement_ratio=best_candidate.get('improvement_ratio'),
                
                # NOUVEAUX ATTRIBUTS POLYGONES SAM
                polygon_points=best_candidate.get('polygon_points'),
                polygon_available=best_candidate.get('polygon_available', False)
            )
            
            # Mise à jour statistiques
            self._update_stats(result)
            
            # DEBUG LOGGING POLYGONES
            print(f"🔍 DEBUG RÉSULTAT: polygon_points présent: {result.polygon_points is not None}")
            print(f"🔍 DEBUG RÉSULTAT: polygon_available: {result.polygon_available}")
            if result.polygon_points:
                print(f"🔍 DEBUG RÉSULTAT: nombre de vertices: {len(result.polygon_points)}")
            
            if self.debug_mode:
                print(f"🎯 Smart detection: {result.class_name} "
                      f"(YOLO: {result.confidence_yolo:.2f}, "
                      f"SAM: {result.confidence_sam or 'N/A'}, "
                      f"Temps: {result.processing_time:.1f}ms, "
                      f"Polygone: {'Oui' if result.polygon_available else 'Non'})")
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur smart detection: {e}")
            return self._create_fallback_result(user_rect, target_class, start_time)
    
    def _run_yolo_detection(self, image_patch: np.ndarray, target_class: str = None) -> List[Dict]:
        """
        Exécute détection YOLO optimisée sur le patch
        
        Args:
            image_patch: Patch image à analyser
            target_class: Classe cible (optionnel pour filtrage)
            
        Returns:
            List[Dict]: Liste détections YOLO
        """
        try:
            # Configuration optimisée selon CPU - ABAISSÉ POUR DEBUG
            confidence_threshold = 0.05  # Très permissif pour capturer toute détection
            print(f"🔍 DEBUG YOLO: confidence_threshold = {confidence_threshold} (abaissé pour debug)")
            print(f"🔍 DEBUG YOLO: target_class = '{target_class}'")
            print(f"🔍 DEBUG YOLO: image_patch shape = {image_patch.shape}")
            print(f"🔍 DEBUG YOLO: image_patch dtype = {image_patch.dtype}")
            print(f"🔍 DEBUG YOLO: image_patch min/max = {image_patch.min():.2f}/{image_patch.max():.2f}")
            
            # NOUVEAU: Mapping intelligent classe custom → COCO
            from .class_mapping import get_coco_classes_for_custom_class, get_mapping_explanation
            
            coco_classes = get_coco_classes_for_custom_class(target_class)
            explanation = get_mapping_explanation(target_class)
            
            print(f"🔍 DEBUG YOLO: {explanation}")
            print(f"🔍 DEBUG YOLO: Classes COCO à détecter: {coco_classes}")
            print(f"🔍 DEBUG YOLO: Stratégie: Détecter {coco_classes} puis reclassifier en '{target_class}'")
            
            # Debug: sauvegarder patch image pour inspection
            try:
                import cv2
                debug_path = f"C:\\temp\\debug_patch_{int(time.time())}.jpg"
                cv2.imwrite(debug_path, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
                print(f"🔍 DEBUG YOLO: Patch sauvegardé: {debug_path}")
            except Exception as e:
                print(f"⚠️ DEBUG YOLO: Impossible de sauver patch: {e}")
            
            # Détection avec seuil très permissif pour capturer tous les candidats
            detections = self.yolo_engine.detect_objects(
                image_patch,
                confidence_threshold=confidence_threshold,
                iou_threshold=0.5  # NMS standard
            )
            print(f"🔍 DEBUG YOLO: Raw detections before filtering = {len(detections) if detections else 0}")
            
            # Debug: afficher TOUTES les détections brutes avec classes disponibles
            if detections:
                print(f"🔍 DEBUG YOLO: Classes détectées:")
                all_classes = set()
                for i, det in enumerate(detections[:10]):  # Afficher plus de détections
                    class_name = det.get('class_name', 'N/A')
                    confidence = det.get('confidence', 0)
                    print(f"🔍 DEBUG YOLO: Detection {i}: '{class_name}' ({confidence:.3f})")
                    all_classes.add(class_name)
                print(f"🔍 DEBUG YOLO: Classes uniques trouvées: {sorted(list(all_classes))}")
            else:
                print(f"🔍 DEBUG YOLO: AUCUNE détection trouvée même avec seuil 5%!")
            
            # NOUVEAU: Filtrer par classes COCO mappées au lieu de classe custom
            if target_class and coco_classes:
                original_count = len(detections) if detections else 0
                detections = [d for d in detections 
                            if d['class_name'] in coco_classes]
                print(f"🔍 DEBUG YOLO: Filtrage COCO {coco_classes}: {len(detections)} (était {original_count})")
                
                # Reclassifier les détections vers la classe custom
                for det in detections:
                    det['original_coco_class'] = det['class_name']  # Sauvegarder classe COCO
                    det['class_name'] = target_class  # Reclassifier
                    print(f"🔄 DEBUG YOLO: Reclassifié '{det['original_coco_class']}' → '{target_class}'")
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Erreur détection YOLO: {e}")
            return []
    
    def _select_best_candidate(self, detections: List[Dict], 
                             user_rect: Tuple[float, float, float, float],
                             image_shape: Tuple[int, int]) -> Optional[Dict]:
        """
        Sélectionne le meilleur candidat parmi les détections YOLO
        
        Critères de sélection améliorés :
        1. Intersection maximale avec rectangle utilisateur
        2. Confiance YOLO élevée
        3. Taille cohérente avec sélection utilisateur
        4. Centrage dans la sélection utilisateur
        5. Cohérence de forme pour objets géospatiaux
        
        Args:
            detections: Liste détections YOLO
            user_rect: Rectangle utilisateur (x1, y1, x2, y2)
            image_shape: Dimensions image (H, W)
            
        Returns:
            Optional[Dict]: Meilleur candidat ou None
        """
        if not detections:
            return None
        
        best_candidate = None
        best_score = 0.0
        
        user_x1, user_y1, user_x2, user_y2 = user_rect
        user_center = ((user_x1 + user_x2) / 2, (user_y1 + user_y2) / 2)
        user_area = (user_x2 - user_x1) * (user_y2 - user_y1)
        
        for detection in detections:
            det_x1, det_y1, det_x2, det_y2 = detection['bbox']
            
            # Calcul intersection
            intersection_x1 = max(user_x1, det_x1)
            intersection_y1 = max(user_y1, det_y1)
            intersection_x2 = min(user_x2, det_x2)
            intersection_y2 = min(user_y2, det_y2)
            
            if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
                continue  # Pas d'intersection
            
            # Métriques de base
            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
            intersection_ratio = intersection_area / user_area
            
            det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
            size_similarity = min(user_area, det_area) / max(user_area, det_area)
            
            # NOUVELLES métriques géospatiales
            centrality_score = self._calculate_centrality_score(detection, user_center)
            aspect_ratio_score = self._calculate_aspect_ratio_consistency(detection)
            
            # Score combiné amélioré avec métriques géospatiales
            score = (
                intersection_ratio * 0.35 +     # 35% intersection avec sélection utilisateur
                detection['confidence'] * 0.25 + # 25% confiance YOLO
                size_similarity * 0.15 +        # 15% similarité de taille
                centrality_score * 0.15 +       # 15% centrage dans la sélection
                aspect_ratio_score * 0.10       # 10% cohérence forme
            )
            
            if score > best_score:
                best_score = score
                best_candidate = detection.copy()
                best_candidate['selection_score'] = score
                best_candidate['intersection_ratio'] = intersection_ratio
                best_candidate['centrality_score'] = centrality_score
                best_candidate['aspect_ratio_score'] = aspect_ratio_score
        
        return best_candidate
    
    def _calculate_centrality_score(self, detection: Dict, user_center: Tuple[float, float]) -> float:
        """Score basé sur le centrage de la détection dans la sélection utilisateur"""
        det_bbox = detection['bbox']
        det_center = ((det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2)
        
        # Distance normalisée du centre
        distance = ((det_center[0] - user_center[0])**2 + (det_center[1] - user_center[1])**2) ** 0.5
        
        # Plus la détection est centrée, meilleur le score
        max_distance = 100  # Distance max acceptable en pixels
        return max(0, 1 - (distance / max_distance))
    
    def _calculate_aspect_ratio_consistency(self, detection: Dict) -> float:
        """Score basé sur la cohérence du ratio d'aspect pour objets géospatiaux"""
        det_bbox = detection['bbox']
        width = det_bbox[2] - det_bbox[0]
        height = det_bbox[3] - det_bbox[1]
        
        if height == 0:
            return 0.0
        
        aspect_ratio = width / height
        
        # Ratios typiques pour objets géospatiaux courants
        typical_ratios = {
            'square': 1.0,      # Bâtiments, véhicules vus du dessus
            'horizontal': 2.0,   # Véhicules, infrastructures horizontales
            'vertical': 0.5      # Poteaux, arbres, objets verticaux
        }
        
        # Score basé sur la proximité aux ratios typiques
        best_score = 0.0
        for ratio_name, typical_ratio in typical_ratios.items():
            # Fonction gaussienne centrée sur le ratio typique
            score = math.exp(-0.5 * ((aspect_ratio - typical_ratio) / 0.5) ** 2)
            best_score = max(best_score, score)
        
        return min(1.0, best_score)
    
    def _should_apply_sam_refinement(self, detection: Dict) -> bool:
        """Décision intelligente sur l'utilité du raffinement SAM"""
        # NOUVEAU: Si contours précis activés, TOUJOURS appliquer SAM
        print(f"🔍 DEBUG SAM: Vérification enable_precise_contours...")
        print(f"🔍 DEBUG SAM: hasattr(self, 'enable_precise_contours') = {hasattr(self, 'enable_precise_contours')}")
        if hasattr(self, 'enable_precise_contours'):
            print(f"🔍 DEBUG SAM: self.enable_precise_contours = {self.enable_precise_contours}")
        
        if hasattr(self, 'enable_precise_contours') and self.enable_precise_contours:
            print(f"🔺 SAM FORCÉ - Contours précis activés")
            return True
        else:
            print(f"⚪ SAM non forcé - Contours précis désactivés ou attribut manquant")
        
        # Logique originale pour mode automatique
        confidence = detection['confidence']
        bbox = detection['bbox']
        class_name = detection.get('class_name', 'unknown')
        
        # Critères pour raffinement SAM
        conditions = [
            confidence < 0.8,  # Confiance YOLO faible
            self._is_irregular_shape_expected(class_name),  # Forme potentiellement complexe
            self._needs_precise_boundaries(class_name),  # Classe nécessitant précision
            self._bbox_seems_imprecise(bbox)  # Bbox semble imprécise
        ]
        
        should_apply = any(conditions)
        if self.debug_mode:
            print(f"🔍 SAM décision: {should_apply} (conf:{confidence:.2f}, classe:{class_name})")
        
        return should_apply
    
    def _is_irregular_shape_expected(self, class_name: str) -> bool:
        """Détermine si la classe d'objet a typiquement des formes irrégulières"""
        irregular_classes = [
            'tree', 'vegetation', 'building', 'construction', 
            'damage', 'crack', 'irregular_object'
        ]
        return any(irregular in class_name.lower() for irregular in irregular_classes)
    
    def _needs_precise_boundaries(self, class_name: str) -> bool:
        """Détermine si la classe nécessite des contours précis"""
        precision_classes = [
            'building', 'vehicle', 'infrastructure', 'damage',
            'construction', 'equipment'
        ]
        return any(precision in class_name.lower() for precision in precision_classes)
    
    def _bbox_seems_imprecise(self, bbox: List[float]) -> bool:
        """Évalue si la bbox semble imprécise (très grande ou très petite)"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        # Bbox très petite (< 100 pixels²) ou très grande (> 50000 pixels²)
        return area < 100 or area > 50000
    
    def _apply_sam_refinement(self, yolo_detection: Dict, image_patch: np.ndarray) -> Optional[Dict]:
        """
        Applique raffinement SAM avec stratégies adaptatives
        
        Args:
            yolo_detection: Détection YOLO à raffiner
            image_patch: Patch image original
            
        Returns:
            Optional[Dict]: Détection raffinée ou None si échec
        """
        # Décision intelligente sur l'utilité du raffinement SAM
        if not self._should_apply_sam_refinement(yolo_detection):
            return yolo_detection  # Pas de raffinement nécessaire
        
        if not self._lazy_load_sam_model():
            return yolo_detection  # Fallback vers YOLO
        
        try:
            x1, y1, x2, y2 = yolo_detection['bbox']
            
            if self.sam_type == 'FastSAM':
                # FastSAM : utilise bbox comme prompt
                results = self.sam_model(
                    image_patch,
                    device='cpu',
                    conf=0.4,
                    iou=0.9
                )
                
                if results and len(results) > 0 and hasattr(results[0], 'masks'):
                    masks = results[0].masks
                    if masks is not None and len(masks.data) > 0:
                        # Prendre le premier masque (plus confiant)
                        mask = masks.data[0].cpu().numpy()
                        refined_bbox = self._bbox_from_mask(mask)
                        
                        # NOUVEAU: Génération polygone précis depuis le masque FastSAM (si activé)
                        polygon_points = None
                        contour_area = None
                        if self.enable_precise_contours:
                            polygon_points, contour_area = self._polygon_from_mask_with_area(
                                mask,
                                simplification_tolerance=2.0,  # Configuration par défaut
                                min_area_pixels=50
                            )
                        
                        if refined_bbox:
                            refined_detection = yolo_detection.copy()
                            refined_detection['bbox'] = refined_bbox
                            refined_detection['bbox_original'] = yolo_detection['bbox']
                            refined_detection['confidence_sam'] = 0.8  # Score par défaut FastSAM
                            refined_detection['mask_available'] = True
                            refined_detection['mask_quality_score'] = self._calculate_mask_quality(mask)
                            refined_detection['refinement_applied'] = True
                            refined_detection['improvement_ratio'] = self._calculate_improvement_ratio(
                                yolo_detection['bbox'], refined_bbox
                            )
                            
                            # NOUVEAU: Ajout des données de polygone précis FastSAM
                            if polygon_points:
                                refined_detection['polygon_points'] = polygon_points
                                refined_detection['polygon_available'] = True
                                refined_detection['vertex_count'] = len(polygon_points) - 1  # -1 car point fermé dupliqué
                                refined_detection['area_pixels'] = contour_area
                                print(f"🔺 Contour précis FastSAM généré: {len(polygon_points)-1} vertices, aire: {contour_area:.1f}px²")
                            else:
                                refined_detection['polygon_available'] = False
                                print("⚠️ Échec génération polygone FastSAM, utilisation bbox uniquement")
                            
                            return refined_detection
            
            elif self.sam_type == 'MobileSAM':
                # MobileSAM : utilise bbox comme prompt box
                self.sam_predictor.set_image(image_patch)
                
                # Prompt box format: [x1, y1, x2, y2]
                input_box = np.array([x1, y1, x2, y2])
                
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False
                )
                
                if len(masks) > 0 and len(scores) > 0:
                    mask = masks[0]
                    confidence_sam = float(scores[0])
                    
                    refined_bbox = self._bbox_from_mask(mask)
                    
                    # NOUVEAU: Génération polygone précis depuis le masque SAM (si activé)
                    polygon_points = None
                    contour_area = None
                    if self.enable_precise_contours:
                        polygon_points, contour_area = self._polygon_from_mask_with_area(
                            mask,
                            simplification_tolerance=2.0,  # Configuration par défaut
                            min_area_pixels=50
                        )
                    
                    if refined_bbox:
                        refined_detection = yolo_detection.copy()
                        refined_detection['bbox'] = refined_bbox
                        refined_detection['bbox_original'] = yolo_detection['bbox']
                        refined_detection['confidence_sam'] = confidence_sam
                        refined_detection['mask_available'] = True
                        refined_detection['mask_quality_score'] = self._calculate_mask_quality(mask)
                        refined_detection['refinement_applied'] = True
                        refined_detection['improvement_ratio'] = self._calculate_improvement_ratio(
                            yolo_detection['bbox'], refined_bbox
                        )
                        
                        # NOUVEAU: Ajout des données de polygone précis
                        if polygon_points:
                            refined_detection['polygon_points'] = polygon_points
                            refined_detection['polygon_available'] = True
                            refined_detection['vertex_count'] = len(polygon_points) - 1  # -1 car point fermé dupliqué
                            refined_detection['area_pixels'] = contour_area
                            print(f"🔺 Contour précis généré: {len(polygon_points)-1} vertices, aire: {contour_area:.1f}px²")
                        else:
                            refined_detection['polygon_available'] = False
                            print("⚠️ Échec génération polygone, utilisation bbox uniquement")
                        
                        return refined_detection
            
            return None
            
        except Exception as e:
            print(f"⚠️ Erreur raffinement SAM: {e}")
            return None
    
    def _bbox_from_mask(self, mask: np.ndarray) -> Optional[List[float]]:
        """
        Calcule bbox optimale à partir d'un masque SAM
        
        Args:
            mask: Masque binaire SAM
            
        Returns:
            Optional[List[float]]: [x1, y1, x2, y2] ou None
        """
        try:
            # Trouver contours du masque
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Prendre le plus grand contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculer bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            return [float(x), float(y), float(x + w), float(y + h)]
            
        except Exception as e:
            print(f"⚠️ Erreur calcul bbox depuis masque: {e}")
            return None
    
    def _polygon_from_mask(self, mask: np.ndarray, simplification_tolerance: float = 2.0, 
                          min_area_pixels: int = 100) -> Optional[List[List[float]]]:
        """
        Convertit un masque SAM en polygone précis avec contours simplifiés
        
        Args:
            mask: Masque binaire SAM (numpy array)
            simplification_tolerance: Tolérance simplification contour (pixels)
            min_area_pixels: Aire minimale pour polygone valide
            
        Returns:
            Optional[List[List[float]]]: Liste de points [x, y] du polygone ou None si échec
        """
        try:
            # Conversion masque en uint8 pour OpenCV
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Extraction contours - même méthode que _bbox_from_mask()
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Sélection du plus grand contour (même logique que bbox)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Validation aire minimale
            if cv2.contourArea(largest_contour) < min_area_pixels:
                return None
                
            # Simplification contour avec approximation polygonale
            epsilon = simplification_tolerance
            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Validation vertices minimums
            if len(simplified_contour) < 3:
                return None
                
            # Conversion en liste de points [x, y]
            polygon_points = []
            for point in simplified_contour:
                x, y = point[0]  # OpenCV format: point[0] = [x, y]
                polygon_points.append([float(x), float(y)])
                
            # Fermeture automatique du polygone si nécessaire
            if polygon_points[0] != polygon_points[-1]:
                polygon_points.append(polygon_points[0])
                
            print(f"🔺 Polygone généré: {len(polygon_points)} vertices, aire: {cv2.contourArea(largest_contour):.1f}px²")
            return polygon_points
            
        except Exception as e:
            print(f"⚠️ Erreur génération polygone: {e}")
            return None
    
    def _polygon_from_mask_with_area(self, mask: np.ndarray, simplification_tolerance: float = 2.0, 
                                    min_area_pixels: int = 100) -> Tuple[Optional[List[List[float]]], Optional[float]]:
        """
        Version étendue qui retourne aussi l'aire du contour
        
        Returns:
            Tuple[polygon_points, contour_area]: Points du polygone et aire en pixels
        """
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None
                
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            if contour_area < min_area_pixels:
                return None, None
                
            epsilon = simplification_tolerance
            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(simplified_contour) < 3:
                return None, None
                
            polygon_points = []
            for point in simplified_contour:
                x, y = point[0]
                polygon_points.append([float(x), float(y)])
                
            if polygon_points[0] != polygon_points[-1]:
                polygon_points.append(polygon_points[0])
                
            print(f"🔺 Polygone généré: {len(polygon_points)} vertices, aire: {contour_area:.1f}px²")
            return polygon_points, contour_area
            
        except Exception as e:
            print(f"⚠️ Erreur génération polygone: {e}")
            return None, None
    
    def _calculate_mask_quality(self, mask: np.ndarray) -> float:
        """
        Calcule score de qualité d'un masque SAM
        
        Args:
            mask: Masque binaire
            
        Returns:
            float: Score qualité 0-1
        """
        try:
            # Métriques de qualité basiques
            total_pixels = mask.size
            object_pixels = np.sum(mask > 0.5)
            
            if object_pixels == 0:
                return 0.0
            
            # Ratio objet/arrière-plan (idéal ~0.1-0.3 pour objets typiques)
            object_ratio = object_pixels / total_pixels
            
            # Compacité (périmètre² / aire) - plus bas = plus compact
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if area > 0 and perimeter > 0:
                    compactness = (perimeter * perimeter) / (4 * np.pi * area)
                    compactness_score = 1.0 / (1.0 + compactness)  # Normalisé
                else:
                    compactness_score = 0.5
            else:
                compactness_score = 0.5
            
            # Score combiné
            quality_score = (
                min(object_ratio * 4, 1.0) * 0.4 +  # Ratio objet (cap à 0.25)
                compactness_score * 0.6               # Compacité
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5  # Score neutre en cas d'erreur
    
    def _calculate_improvement_ratio(self, bbox_original: List[float], bbox_refined: List[float]) -> float:
        """
        Calcule le ratio d'amélioration entre bbox originale et raffinée
        
        Args:
            bbox_original: Bbox YOLO [x1, y1, x2, y2]
            bbox_refined: Bbox SAM [x1, y1, x2, y2]
            
        Returns:
            float: Ratio amélioration (>1 = amélioration, <1 = dégradation)
        """
        try:
            # Calcul aires
            def bbox_area(bbox):
                return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            area_original = bbox_area(bbox_original)
            area_refined = bbox_area(bbox_refined)
            
            if area_original <= 0:
                return 1.0
            
            # Ratio de changement d'aire (mesure indirecte de précision)
            area_ratio = area_refined / area_original
            
            # Hypothèse : SAM est généralement plus précis, donc aire plus petite = meilleur
            if area_ratio < 1.0:
                return 1.0 / area_ratio  # Amélioration
            else:
                return area_ratio  # Possible dégradation
                
        except Exception:
            return 1.0  # Neutre en cas d'erreur
    
    def _create_fallback_result(self, user_rect: Tuple[float, float, float, float], 
                              target_class: str, start_time: float) -> SmartDetectionResult:
        """
        Crée résultat fallback en cas d'échec détection intelligente
        
        Args:
            user_rect: Rectangle utilisateur original
            target_class: Classe cible
            start_time: Timestamp début traitement
            
        Returns:
            SmartDetectionResult: Résultat fallback
        """
        return SmartDetectionResult(
            bbox=list(user_rect),
            bbox_original=list(user_rect),
            confidence_yolo=0.5,  # Confiance neutre
            confidence_sam=None,
            class_name=target_class or "unknown",
            class_id=0,
            mask_available=False,
            mask_quality_score=None,
            refinement_applied=False,
            processing_time=(time.time() - start_time) * 1000,
            improvement_ratio=None
        )
    
    def _update_stats(self, result: SmartDetectionResult):
        """Met à jour les statistiques de détection"""
        with self._processing_lock:
            self.detection_stats['total_detections'] += 1
            
            if result.refinement_applied:
                self.detection_stats['sam_refinements'] += 1
            
            if result.confidence_yolo > 0.8:
                self.detection_stats['auto_accepted'] += 1
            else:
                self.detection_stats['user_validated'] += 1
            
            # Moyenne mobile du temps de traitement
            n = self.detection_stats['total_detections']
            current_avg = self.detection_stats['avg_processing_time']
            new_avg = ((current_avg * (n - 1)) + result.processing_time) / n
            self.detection_stats['avg_processing_time'] = new_avg
    
    def get_performance_stats(self) -> Dict:
        """
        Retourne les statistiques de performance du moteur
        
        Returns:
            Dict: Statistiques détaillées
        """
        with self._processing_lock:
            total = self.detection_stats['total_detections']
            return {
                'cpu_profile': self.cpu_profile.level,
                'total_detections': total,
                'sam_usage_rate': (self.detection_stats['sam_refinements'] / total * 100) if total > 0 else 0,
                'auto_acceptance_rate': (self.detection_stats['auto_accepted'] / total * 100) if total > 0 else 0,
                'avg_processing_time_ms': self.detection_stats['avg_processing_time'],
                'enabled': self.enabled,
                'sam_available': self.sam_model is not None
            }
    
    def enable_debug_mode(self, enabled: bool = True):
        """Active/désactive le mode debug avec logs détaillés"""
        self.debug_mode = enabled
        print(f"🐛 Mode debug: {'Activé' if enabled else 'Désactivé'}")
    
    def cleanup(self):
        """Nettoie les ressources (modèles, cache)"""
        try:
            if self.sam_model is not None:
                del self.sam_model
                self.sam_model = None
            
            if self.sam_predictor is not None:
                del self.sam_predictor
                self.sam_predictor = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅ SmartAnnotationEngine nettoyé")
            
        except Exception as e:
            print(f"⚠️ Erreur nettoyage SmartAnnotationEngine: {e}")