"""
Gestionnaire de Qualité d'Annotation - Optimisation Continue du Workflow

Ce module fournit :
- Suivi des métriques de qualité d'annotation
- Analyse des patterns d'usage utilisateur
- Recommandations d'optimisation automatiques
- Validation de la préparation à l'entraînement
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

# Conditional imports for dependencies that might not be available in all QGIS environments
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class AnnotationSessionMetrics:
    """Métriques d'une session d'annotation"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    class_name: str
    total_annotations: int
    smart_suggestions_count: int
    accepted_suggestions_count: int
    manual_corrections: int
    avg_time_per_annotation: float
    quality_score: float
    user_satisfaction: Optional[float]


@dataclass
class QualityRecommendation:
    """Recommandation d'amélioration qualité"""
    category: str  # 'speed', 'accuracy', 'ai_tuning', 'training'
    priority: str  # 'high', 'medium', 'low'
    message: str
    action_items: List[str]
    expected_improvement: str


class AnnotationQualityManager:
    """
    Gestionnaire de qualité pour optimisation continue du workflow d'annotation
    """
    
    def __init__(self, project_dir: str = None):
        """
        Initialise le gestionnaire de qualité
        
        Args:
            project_dir: Répertoire de projet (None = auto-détection)
        """
        self.project_dir = Path(project_dir) if project_dir else self._get_project_directory()
        self.data_dir = self.project_dir / "quality_metrics"
        self.db_path = self.data_dir / "quality_metrics.db"
        
        # Création des répertoires et base de données
        self._ensure_directories()
        self._init_database()
        
        # Métriques en cours de session
        self.current_session = None
        self.session_start_time = None
        
        print(f"📊 AnnotationQualityManager initialisé : {self.data_dir}")
    
    def _get_project_directory(self) -> Path:
        """Détermine le répertoire de projet"""
        from qgis.core import QgsProject
        
        project = QgsProject.instance()
        if project.fileName():
            return Path(project.fileName()).parent / 'yolo_detector_data'
        else:
            return Path.cwd() / 'qgis_yolo_detector' / 'project_data'
    
    def _ensure_directories(self):
        """Crée les répertoires nécessaires"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialise la base de données des métriques"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Table des sessions d'annotation
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotation_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    class_name TEXT NOT NULL,
                    total_annotations INTEGER DEFAULT 0,
                    smart_suggestions_count INTEGER DEFAULT 0,
                    accepted_suggestions_count INTEGER DEFAULT 0,
                    manual_corrections INTEGER DEFAULT 0,
                    avg_time_per_annotation REAL DEFAULT 0.0,
                    quality_score REAL DEFAULT 0.0,
                    user_satisfaction REAL,
                    session_notes TEXT
                )
            """)
            
            # Table des métriques détaillées par annotation
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotation_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    annotation_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    smart_mode_used BOOLEAN DEFAULT FALSE,
                    sam_refinement_applied BOOLEAN DEFAULT FALSE,
                    confidence_yolo REAL,
                    confidence_sam REAL,
                    user_accepted BOOLEAN DEFAULT TRUE,
                    correction_type TEXT,
                    FOREIGN KEY (session_id) REFERENCES annotation_sessions(session_id)
                )
            """)
            
            # Table des recommandations générées
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generated_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    message TEXT NOT NULL,
                    action_items_json TEXT NOT NULL,
                    applied BOOLEAN DEFAULT FALSE,
                    effectiveness_score REAL
                )
            """)
    
    def start_annotation_session(self, class_name: str, target_annotations: int = 50) -> str:
        """
        Démarre une nouvelle session d'annotation avec suivi qualité
        
        Args:
            class_name: Nom de la classe à annoter
            target_annotations: Nombre cible d'annotations
            
        Returns:
            str: ID de la session
        """
        session_id = f"{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            'session_id': session_id,
            'class_name': class_name,
            'target_annotations': target_annotations,
            'start_time': datetime.now(),
            'annotations': [],
            'smart_usage': {
                'suggestions_offered': 0,
                'suggestions_accepted': 0,
                'sam_refinements': 0
            }
        }
        
        # Enregistrement en base
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO annotation_sessions 
                (session_id, start_time, class_name)
                VALUES (?, ?, ?)
            """, (session_id, self.current_session['start_time'].isoformat(), class_name))
        
        print(f"📝 Session d'annotation démarrée: {session_id}")
        return session_id
    
    def track_annotation_event(self, annotation_data: Dict):
        """
        Enregistre un événement d'annotation pour suivi qualité
        
        Args:
            annotation_data: Données de l'annotation créée
        """
        if not self.current_session:
            print("⚠️ Aucune session active pour tracking")
            return
        
        # Extraction des métriques
        annotation_metrics = {
            'annotation_id': annotation_data.get('id', 'unknown'),
            'timestamp': datetime.now(),
            'processing_time': annotation_data.get('processing_time', 0.0),
            'smart_mode_used': annotation_data.get('smart_mode_used', False),
            'sam_refinement_applied': annotation_data.get('refinement_applied', False),
            'confidence_yolo': annotation_data.get('confidence_yolo', 0.0),
            'confidence_sam': annotation_data.get('confidence_sam'),
            'user_accepted': annotation_data.get('user_accepted', True),
            'correction_type': annotation_data.get('correction_type', 'none')
        }
        
        # Ajout à la session courante
        self.current_session['annotations'].append(annotation_metrics)
        
        # Mise à jour des statistiques Smart Mode
        if annotation_metrics['smart_mode_used']:
            self.current_session['smart_usage']['suggestions_offered'] += 1
            if annotation_metrics['user_accepted']:
                self.current_session['smart_usage']['suggestions_accepted'] += 1
        
        if annotation_metrics['sam_refinement_applied']:
            self.current_session['smart_usage']['sam_refinements'] += 1
        
        # Enregistrement détaillé en base
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO annotation_details 
                (session_id, annotation_id, timestamp, processing_time, 
                 smart_mode_used, sam_refinement_applied, confidence_yolo, 
                 confidence_sam, user_accepted, correction_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.current_session['session_id'],
                annotation_metrics['annotation_id'],
                annotation_metrics['timestamp'].isoformat(),
                annotation_metrics['processing_time'],
                annotation_metrics['smart_mode_used'],
                annotation_metrics['sam_refinement_applied'],
                annotation_metrics['confidence_yolo'],
                annotation_metrics['confidence_sam'],
                annotation_metrics['user_accepted'],
                annotation_metrics['correction_type']
            ))
    
    def complete_annotation_session(self, user_satisfaction: float = None) -> Dict:
        """
        Finalise la session d'annotation et génère le rapport qualité
        
        Args:
            user_satisfaction: Score de satisfaction utilisateur (1-5)
            
        Returns:
            Dict: Rapport de session avec recommandations
        """
        if not self.current_session:
            return {'error': 'Aucune session active'}
        
        session = self.current_session
        session['end_time'] = datetime.now()
        session['duration'] = (session['end_time'] - session['start_time']).total_seconds()
        
        # Calcul des métriques de session
        annotations_count = len(session['annotations'])
        if annotations_count > 0:
            avg_time = sum(a['processing_time'] for a in session['annotations']) / annotations_count
            smart_acceptance_rate = (session['smart_usage']['suggestions_accepted'] / 
                                   max(1, session['smart_usage']['suggestions_offered']))
        else:
            avg_time = 0.0
            smart_acceptance_rate = 0.0
        
        # Score de qualité composite
        quality_score = self._calculate_session_quality_score(session)
        
        # Mise à jour en base
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                UPDATE annotation_sessions 
                SET end_time = ?, total_annotations = ?, 
                    smart_suggestions_count = ?, accepted_suggestions_count = ?,
                    avg_time_per_annotation = ?, quality_score = ?, user_satisfaction = ?
                WHERE session_id = ?
            """, (
                session['end_time'].isoformat(),
                annotations_count,
                session['smart_usage']['suggestions_offered'],
                session['smart_usage']['suggestions_accepted'],
                avg_time,
                quality_score,
                user_satisfaction,
                session['session_id']
            ))
        
        # Génération des recommandations
        recommendations = self.generate_session_recommendations(session)
        
        # Rapport final
        session_report = {
            'session_id': session['session_id'],
            'class_name': session['class_name'],
            'duration_minutes': session['duration'] / 60,
            'annotations_created': annotations_count,
            'avg_time_per_annotation': avg_time,
            'smart_acceptance_rate': smart_acceptance_rate,
            'quality_score': quality_score,
            'recommendations': recommendations,
            'productivity_rating': self._rate_productivity(avg_time, annotations_count)
        }
        
        # Reset session courante
        self.current_session = None
        
        print(f"✅ Session terminée: {annotations_count} annotations en {session_report['duration_minutes']:.1f}min")
        return session_report
    
    def _calculate_session_quality_score(self, session: Dict) -> float:
        """Calcule un score de qualité composite pour la session"""
        if not session['annotations']:
            return 0.0
        
        # Métriques de qualité
        annotations = session['annotations']
        
        # 1. Consistance des temps (moins de variation = mieux)
        times = [a['processing_time'] for a in annotations if a['processing_time'] > 0]
        if times:
            time_consistency = 1.0 - (statistics.stdev(times) / statistics.mean(times))
            time_consistency = max(0, min(1, time_consistency))
        else:
            time_consistency = 0.5
        
        # 2. Efficacité Smart Mode
        smart_annotations = [a for a in annotations if a['smart_mode_used']]
        if smart_annotations:
            smart_efficiency = sum(1 for a in smart_annotations if a['user_accepted']) / len(smart_annotations)
        else:
            smart_efficiency = 0.5
        
        # 3. Qualité des confidences
        confidences = [a['confidence_yolo'] for a in annotations if a['confidence_yolo'] > 0]
        if confidences:
            avg_confidence = statistics.mean(confidences)
        else:
            avg_confidence = 0.5
        
        # Score composite
        quality_score = (
            time_consistency * 0.3 +
            smart_efficiency * 0.4 +
            avg_confidence * 0.3
        )
        
        return min(1.0, max(0.0, quality_score))
    
    def generate_session_recommendations(self, session: Dict) -> List[QualityRecommendation]:
        """Génère des recommandations basées sur la session"""
        recommendations = []
        
        if not session['annotations']:
            return recommendations
        
        annotations_count = len(session['annotations'])
        avg_time = sum(a['processing_time'] for a in session['annotations']) / annotations_count
        smart_acceptance_rate = (session['smart_usage']['suggestions_accepted'] / 
                               max(1, session['smart_usage']['suggestions_offered']))
        
        # Recommandations vitesse
        if avg_time > 30:  # Plus de 30s par annotation
            recommendations.append(QualityRecommendation(
                category='speed',
                priority='high',
                message="Temps d'annotation élevé détecté",
                action_items=[
                    "Activer le mode Smart pour assistance IA",
                    "Utiliser les raccourcis clavier (Enter/Escape)",
                    "Considérer le mode batch pour zones denses"
                ],
                expected_improvement="Réduction 40-60% du temps d'annotation"
            ))
        
        # Recommandations Smart Mode
        if smart_acceptance_rate < 0.6:  # Moins de 60% d'acceptation
            recommendations.append(QualityRecommendation(
                category='ai_tuning',
                priority='medium', 
                message="Faible taux d'acceptation des suggestions IA",
                action_items=[
                    "Ajuster les seuils de confiance YOLO",
                    "Améliorer la précision des rectangles utilisateur",
                    "Vérifier la qualité de l'imagerie de base"
                ],
                expected_improvement="Amélioration 20-30% acceptation IA"
            ))
        
        # Recommandations entraînement
        if annotations_count >= 20:
            recommendations.append(QualityRecommendation(
                category='training',
                priority='high',
                message="Données suffisantes pour entraînement disponibles",
                action_items=[
                    "Lancer l'entraînement d'un modèle spécialisé",
                    "Valider la qualité du dataset avant entraînement",
                    "Configurer les paramètres d'entraînement optimaux"
                ],
                expected_improvement="Modèle spécialisé avec >80% précision"
            ))
        
        return recommendations
    
    def _rate_productivity(self, avg_time: float, annotations_count: int) -> str:
        """Évaluation de la productivité de la session"""
        if avg_time < 15 and annotations_count > 10:
            return "Excellente"
        elif avg_time < 25 and annotations_count > 5:
            return "Bonne"
        elif avg_time < 40:
            return "Moyenne"
        else:
            return "À améliorer"
    
    def get_historical_performance(self, class_name: str = None, 
                                 days_back: int = 30) -> Dict:
        """
        Récupère les performances historiques
        
        Args:
            class_name: Classe spécifique (None = toutes)
            days_back: Nombre de jours d'historique
            
        Returns:
            Dict: Métriques de performance historiques
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            # Requête sessions récentes
            query = """
                SELECT * FROM annotation_sessions 
                WHERE start_time > ? 
            """
            params = [cutoff_date.isoformat()]
            
            if class_name:
                query += " AND class_name = ?"
                params.append(class_name)
            
            cursor = conn.execute(query + " ORDER BY start_time DESC", params)
            sessions = cursor.fetchall()
        
        if not sessions:
            return {'error': 'Aucune donnée historique trouvée'}
        
        # Calcul des métriques agrégées
        total_annotations = sum(s[4] for s in sessions if s[4])  # total_annotations
        avg_time_overall = statistics.mean([s[7] for s in sessions if s[7] > 0])  # avg_time_per_annotation
        avg_quality = statistics.mean([s[8] for s in sessions if s[8] > 0])  # quality_score
        
        # Tendances temporelles
        sessions_by_date = defaultdict(list)
        for session in sessions:
            date = datetime.fromisoformat(session[1]).date()  # start_time
            sessions_by_date[date].append(session)
        
        return {
            'period_days': days_back,
            'total_sessions': len(sessions),
            'total_annotations': total_annotations,
            'avg_time_per_annotation': avg_time_overall,
            'avg_quality_score': avg_quality,
            'productivity_trend': self._calculate_productivity_trend(sessions_by_date),
            'most_productive_class': self._find_most_productive_class(sessions),
            'improvement_areas': self._identify_improvement_areas(sessions)
        }
    
    def _calculate_productivity_trend(self, sessions_by_date: Dict) -> str:
        """Calcule la tendance de productivité"""
        if len(sessions_by_date) < 3:
            return "Données insuffisantes"
        
        dates = sorted(sessions_by_date.keys())
        recent_avg = statistics.mean([
            statistics.mean([s[7] for s in sessions_by_date[date] if s[7] > 0])
            for date in dates[-3:]  # 3 derniers jours
            if any(s[7] > 0 for s in sessions_by_date[date])
        ])
        
        early_avg = statistics.mean([
            statistics.mean([s[7] for s in sessions_by_date[date] if s[7] > 0])
            for date in dates[:3]  # 3 premiers jours
            if any(s[7] > 0 for s in sessions_by_date[date])
        ])
        
        if recent_avg < early_avg * 0.8:
            return "En amélioration"
        elif recent_avg > early_avg * 1.2:
            return "En dégradation"
        else:
            return "Stable"
    
    def _find_most_productive_class(self, sessions: List) -> str:
        """Trouve la classe la plus productive"""
        class_performance = defaultdict(list)
        
        for session in sessions:
            if session[7] > 0:  # avg_time_per_annotation
                class_performance[session[3]].append(session[7])  # class_name
        
        if not class_performance:
            return "Aucune donnée"
        
        # Classe avec le temps moyen le plus faible
        best_class = min(class_performance.keys(), 
                        key=lambda x: statistics.mean(class_performance[x]))
        
        return best_class
    
    def _identify_improvement_areas(self, sessions: List) -> List[str]:
        """Identifie les domaines d'amélioration"""
        areas = []
        
        # Analyse des temps moyens
        times = [s[7] for s in sessions if s[7] > 0]
        if times and statistics.mean(times) > 25:
            areas.append("Réduction du temps d'annotation")
        
        # Analyse qualité
        qualities = [s[8] for s in sessions if s[8] > 0]
        if qualities and statistics.mean(qualities) < 0.7:
            areas.append("Amélioration de la qualité d'annotation")
        
        # Analyse consistance
        if times and len(times) > 1 and statistics.stdev(times) > statistics.mean(times) * 0.5:
            areas.append("Amélioration de la consistance")
        
        return areas or ["Performance satisfaisante"]


# Instance globale (singleton pattern)
_quality_manager = None

def get_quality_manager(project_dir: str = None) -> AnnotationQualityManager:
    """
    Retourne l'instance globale du gestionnaire de qualité
    
    Args:
        project_dir: Répertoire de projet (None = auto-détection)
        
    Returns:
        AnnotationQualityManager: Instance du gestionnaire
    """
    global _quality_manager
    
    if _quality_manager is None or (project_dir and str(_quality_manager.project_dir) != project_dir):
        _quality_manager = AnnotationQualityManager(project_dir)
    
    return _quality_manager