"""
Système de mapping intelligent entre classes COCO et classes custom

Ce module permet de résoudre le problème logique circulaire :
- Smart Mode utilise des modèles génériques COCO pour détecter des objets proches
- Les détections sont reclassifiées vers les classes custom définies par l'utilisateur
- Cela permet d'utiliser l'IA pour aider à créer le dataset custom
"""

from typing import List, Dict, Set, Optional
import difflib


class ClassMappingManager:
    """
    Gestionnaire de mapping entre classes COCO et classes custom
    
    Utilise une combinaison de :
    - Mappings prédéfinis (pour cas courants)
    - Similarité linguistique (pour nouveaux cas)
    - Apprentissage automatique des mappings utilisateur
    """
    
    # Classes COCO standard (YOLO11)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    # Mappings prédéfinis pour cas courants géospatiaux/urbains
    PREDEFINED_MAPPINGS = {
        # Bâtiments et structures
        "Batiment": ["person", "car", "truck"],  # Objets souvent proches des bâtiments
        "Bâtiment": ["person", "car", "truck"],
        "Building": ["person", "car", "truck"],
        "Maison": ["person", "car"],
        "House": ["person", "car"],
        
        # Infrastructure
        "Poteau": ["traffic light", "stop sign", "fire hydrant"],
        "Pole": ["traffic light", "stop sign", "fire hydrant"],
        "Panneau": ["stop sign", "traffic light"],
        "Sign": ["stop sign", "traffic light"],
        
        # Transport
        "Véhicule": ["car", "truck", "bus", "motorcycle"],
        "Vehicle": ["car", "truck", "bus", "motorcycle"],
        "Voiture": ["car"],
        "Car": ["car"],
        "Camion": ["truck"],
        "Truck": ["truck"],
        
        # Personnes
        "Personne": ["person"],
        "Person": ["person"],
        "People": ["person"],
        "Piéton": ["person"],
        "Pedestrian": ["person"],
        
        # Nature/Végétation
        "Arbre": ["potted plant"],
        "Tree": ["potted plant"],
        "Plante": ["potted plant"],
        "Plant": ["potted plant"],
        
        # Mobilier urbain
        "Banc": ["bench"],
        "Bench": ["bench"],
        "Poubelle": ["fire hydrant"],  # Approximation
        "Trash": ["fire hydrant"],
        
        # Infrastructure routière
        "Route": ["car", "truck", "bus"],  # Détecte via véhicules
        "Road": ["car", "truck", "bus"],
        "Rue": ["car", "truck", "person"],
        "Street": ["car", "truck", "person"],
    }
    
    # Mots-clés pour améliorer la correspondance
    SEMANTIC_KEYWORDS = {
        "building": ["bâtiment", "batiment", "building", "maison", "house", "structure", "construction"],
        "vehicle": ["véhicule", "vehicle", "voiture", "car", "camion", "truck", "transport"],
        "person": ["personne", "person", "people", "piéton", "pedestrian", "humain", "human"],
        "infrastructure": ["poteau", "pole", "panneau", "sign", "infrastructure", "équipement"],
        "nature": ["arbre", "tree", "plante", "plant", "végétation", "nature"],
        "street": ["route", "road", "rue", "street", "voie", "chaussée"],
    }
    
    def __init__(self):
        self.user_mappings = {}  # Mappings appris depuis les actions utilisateur
        self.usage_stats = {}    # Statistiques d'utilisation pour optimisation
        
    def get_coco_classes_for_custom(self, custom_class: str) -> List[str]:
        """
        Retourne les classes COCO à détecter pour une classe custom
        
        Args:
            custom_class: Nom de la classe custom (ex: "Batiment")
            
        Returns:
            List[str]: Liste des classes COCO correspondantes
        """
        # 1. Vérifier mappings utilisateur appris
        if custom_class in self.user_mappings:
            return self.user_mappings[custom_class]
        
        # 2. Vérifier mappings prédéfinis
        if custom_class in self.PREDEFINED_MAPPINGS:
            return self.PREDEFINED_MAPPINGS[custom_class]
        
        # 3. Utiliser similarité linguistique
        similar_classes = self._find_similar_coco_classes(custom_class)
        if similar_classes:
            return similar_classes
        
        # 4. Utiliser mots-clés sémantiques
        semantic_classes = self._find_semantic_matches(custom_class)
        if semantic_classes:
            return semantic_classes
        
        # 5. Fallback : retourner les plus courantes pour annotation générale
        return ["person", "car", "truck"]
    
    def _find_similar_coco_classes(self, custom_class: str) -> List[str]:
        """
        Trouve des classes COCO similaires via similarité textuelle
        """
        custom_lower = custom_class.lower()
        matches = []
        
        # Recherche directe (traduction)
        translations = {
            "batiment": "person",  # Bâtiments souvent associés à des personnes
            "maison": "person",
            "voiture": "car",
            "camion": "truck",
            "personne": "person",
            "arbre": "potted plant",
        }
        
        if custom_lower in translations:
            return [translations[custom_lower]]
        
        # Recherche par similarité
        for coco_class in self.COCO_CLASSES:
            similarity = difflib.SequenceMatcher(None, custom_lower, coco_class).ratio()
            if similarity > 0.6:  # Seuil de similarité
                matches.append(coco_class)
        
        return matches[:3] if matches else []  # Max 3 correspondances
    
    def _find_semantic_matches(self, custom_class: str) -> List[str]:
        """
        Trouve des correspondances via mots-clés sémantiques
        """
        custom_lower = custom_class.lower()
        matches = []
        
        for coco_category, keywords in self.SEMANTIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in custom_lower or custom_lower in keyword:
                    # Retourner classes COCO correspondantes
                    if coco_category == "building":
                        matches.extend(["person", "car"])  # Objets près des bâtiments
                    elif coco_category == "vehicle":
                        matches.extend(["car", "truck", "bus"])
                    elif coco_category == "person":
                        matches.append("person")
                    elif coco_category == "infrastructure":
                        matches.extend(["traffic light", "stop sign"])
                    elif coco_category == "nature":
                        matches.append("potted plant")
                    elif coco_category == "street":
                        matches.extend(["car", "truck", "person"])
        
        return list(set(matches))  # Supprimer doublons
    
    def learn_from_user_action(self, custom_class: str, detected_coco_classes: List[str], user_accepted: bool):
        """
        Apprend depuis les actions utilisateur pour améliorer les mappings
        
        Args:
            custom_class: Classe custom utilisée
            detected_coco_classes: Classes COCO qui ont été détectées
            user_accepted: True si l'utilisateur a accepté la suggestion
        """
        if user_accepted:
            # Renforcer ce mapping
            if custom_class not in self.user_mappings:
                self.user_mappings[custom_class] = []
            
            for coco_class in detected_coco_classes:
                if coco_class not in self.user_mappings[custom_class]:
                    self.user_mappings[custom_class].append(coco_class)
            
            # Statistiques d'usage
            if custom_class not in self.usage_stats:
                self.usage_stats[custom_class] = {"accepted": 0, "rejected": 0}
            self.usage_stats[custom_class]["accepted"] += 1
            
            print(f"🧠 MAPPING APPRIS: '{custom_class}' → {detected_coco_classes}")
        else:
            # Marquer comme rejeté
            if custom_class not in self.usage_stats:
                self.usage_stats[custom_class] = {"accepted": 0, "rejected": 0}
            self.usage_stats[custom_class]["rejected"] += 1
    
    def get_confidence_score(self, custom_class: str) -> float:
        """
        Retourne un score de confiance pour le mapping d'une classe
        
        Returns:
            float: Score entre 0.0 et 1.0
        """
        # Mappings appris ont plus de confiance
        if custom_class in self.user_mappings:
            stats = self.usage_stats.get(custom_class, {"accepted": 1, "rejected": 0})
            total = stats["accepted"] + stats["rejected"]
            return stats["accepted"] / max(total, 1)
        
        # Mappings prédéfinis ont confiance moyenne
        if custom_class in self.PREDEFINED_MAPPINGS:
            return 0.7
        
        # Autres ont confiance faible
        return 0.3
    
    def get_mapping_explanation(self, custom_class: str) -> str:
        """
        Retourne une explication du mapping pour l'utilisateur
        """
        coco_classes = self.get_coco_classes_for_custom(custom_class)
        
        if custom_class in self.user_mappings:
            return f"🧠 Mapping appris: '{custom_class}' détecté via {coco_classes} (basé sur vos validations)"
        
        elif custom_class in self.PREDEFINED_MAPPINGS:
            return f"📖 Mapping prédéfini: '{custom_class}' détecté via {coco_classes}"
        
        else:
            return f"🔍 Mapping automatique: '{custom_class}' détecté via {coco_classes} (similarité linguistique)"
    
    def export_mappings(self) -> Dict:
        """Exporte les mappings pour sauvegarde"""
        return {
            "user_mappings": self.user_mappings,
            "usage_stats": self.usage_stats
        }
    
    def import_mappings(self, data: Dict):
        """Importe des mappings sauvegardés"""
        self.user_mappings = data.get("user_mappings", {})
        self.usage_stats = data.get("usage_stats", {})


# Instance globale partagée
class_mapping_manager = ClassMappingManager()


def get_coco_classes_for_custom_class(custom_class: str) -> List[str]:
    """
    Fonction helper pour obtenir les classes COCO pour une classe custom
    
    Args:
        custom_class: Nom de la classe custom
        
    Returns:
        List[str]: Classes COCO à détecter
    """
    return class_mapping_manager.get_coco_classes_for_custom(custom_class)


def learn_mapping_from_user(custom_class: str, coco_classes: List[str], accepted: bool):
    """
    Fonction helper pour apprentissage des mappings
    """
    class_mapping_manager.learn_from_user_action(custom_class, coco_classes, accepted)


def get_mapping_explanation(custom_class: str) -> str:
    """
    Fonction helper pour explication des mappings
    """
    return class_mapping_manager.get_mapping_explanation(custom_class)