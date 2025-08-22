# QGIS YOLO Interactive Object Detector

![BlockNote image](https://img.shields.io/badge/version-1.9.2-blue.svg)![BlockNote image](https://img.shields.io/badge/QGIS-3.28--3.99-green.svg)![BlockNote image](https://img.shields.io/badge/Status-POC%20Development-orange.svg)![BlockNote image](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

Plugin QGIS expérimental permettant la création de modèles de détection d'objets YOLO personnalisés à partir d'annotations interactives sur des données géospatiales.

> **⚠️ Statut : Proof of Concept en développement actif**\
> Ce plugin est un prototype expérimental. Les fonctionnalités peuvent être instables et l'API peut changer sans préavis.

![Capture d’écran 2025-08-22 125806.png](https://docs.numerique.gouv.fr/media/f89b1068-f36b-46c1-a568-ae84c4223d81/attachments/dfe798d6-0aad-4ce6-8daa-ed4da196e09d.png)![Capture d’écran 2025-08-22 132602.png](https://docs.numerique.gouv.fr/media/f89b1068-f36b-46c1-a568-ae84c4223d81/attachments/4e1e73a0-b859-4329-b032-0d8352baaca2.png)

## 📋 **Description du Projet**

Ce plugin QGIS propose une approche intégrée pour créer des modèles de détection d'objets spécialisés directement depuis l'interface QGIS. L'objectif est de simplifier le processus d'annotation, d'entraînement et de déploiement de modèles YOLO pour des cas d'usage géospatiaux spécifiques.

### **Principe de Fonctionnement**

1. **Annotation** : Interface interactive pour marquer les objets d'intérêt

2. **Assistance IA** : Suggestions automatiques basées sur des modèles pré-entraînés

3. **Entraînement** : Pipeline automatisé pour créer des modèles personnalisés

4. **Détection** : Application des modèles entraînés sur de nouvelles données

### **Cas d'Usage Expérimentés**

* Infrastructure : Détection d'équipements urbains

* Environnement : Monitoring d'éléments naturels

* Transport : Analyse d'infrastructures routières

* Urbanisme : Inventaire d'éléments bâtis

## 🔧 **État Actuel du Développement**

### **Fonctionnalités Implémentées**

**Interface Utilisateur**

* Interface 4 onglets : Classes, Annotation, Entraînement, Détection

* Gestion des classes d'objets avec descriptions sémantiques

* Outils d'annotation manuelle (rectangle, polygone)

**Smart Assistant (Expérimental)**

* Pipeline YOLO + SAM pour assistance à l'annotation

* Mapping automatique classes COCO vers classes personnalisées

* Génération de contours précis via segmentation SAM

* Validation interactive des propositions IA

**Système de Données**

* Base de données SQLite pour stockage des annotations

* Support des géométries polygonales (pas seulement bounding boxes)

* Métadonnées complètes : confiance, timestamps, méthode d'annotation

* Migration automatique entre versions

**Pipeline d'Entraînement**

* Génération automatique de datasets YOLO

* Support format détection (bbox) et segmentation (polygones)

* Optimiseur géospatial avec augmentations adaptées

* Entraînement avec transfert learning

**Détection**

* Application de modèles entraînés sur nouvelles images

* Export des résultats en couches vectorielles QGIS

* Support de différents seuils de confiance

### **Limitations Connues**

* **Stabilité** : Fonctionnalités expérimentales pouvant être instables

* **Performance** : Non optimisé pour images très haute résolution (>4K)

* **Mémoire** : Consommation importante lors de l'entraînement

* **Modèles** : Efficacité variable selon les types d'objets

* **Documentation** : En cours de développement

## ⚙️ **Installation**

### **Prérequis**

* QGIS 3.28 ou supérieur

* Python 3.8+

* 8GB RAM minimum (16GB recommandé)

* Espace disque : ~2GB pour modèles et données

### **Procédure d'Installation**

**Méthode 1 : Script automatique (Windows)**

```shellscript
# Télécharger le repository
git clone https://github.com/nic01asFr/qgis_yolo_detector.git
cd qgis_yolo_detector

# Installation automatique
install_plugin.bat
```

**Méthode 2 : Installation manuelle**

1. Télécharger l'archive depuis les releases GitHub

2. QGIS → Extensions → Installer depuis ZIP

3. Sélectionner le fichier `qgis_yolo_detector_v1.9.2.zip`

4. Activer l'extension dans le gestionnaire

### **Dépendances Python**

Le plugin installera automatiquement :

* `ultralytics` (YOLO)

* `torch` + `torchvision`

* `opencv-python`

* `fastsam` + `mobile-sam` (inclus)

## 📖 **Guide d'Utilisation**

### **1. Configuration Initiale**

1. Charger une couche raster dans QGIS

2. Ouvrir le plugin : Extensions → YOLO Interactive Object Detector

3. Configurer les paramètres de base (CPU/GPU)

### **2. Création de Classes d'Objets**

1. Onglet "Classes d'Objets"

2. Créer une nouvelle classe avec nom et description

3. Exemple : Classe "Panneau_Solaire", Description "Panneaux photovoltaïques sur toitures"

### **3. Annotation des Données**

**Mode Manuel**

* Sélectionner une classe

* Dessiner des rectangles ou polygones autour des objets

* Valider chaque annotation

**Mode Smart Assistant (Expérimental)**

* Activer le Smart Mode

* Dessiner un rectangle approximatif

* Le système propose une détection raffinée

* Valider ou corriger la proposition

### **4. Génération de Dataset**

1. Onglet "Entraînement"

2. Générer un dataset à partir des annotations

3. Le format (détection/segmentation) est automatiquement détecté

### **5. Entraînement de Modèle**

1. Configurer les paramètres d'entraînement

2. Lancer l'entraînement (durée variable selon dataset)

3. Le modèle est sauvegardé automatiquement

### **6. Application du Modèle**

1. Onglet "Détection"

2. Sélectionner le modèle entraîné

3. Configurer le seuil de confiance

4. Lancer la détection sur la zone d'intérêt

## 🏗️ **Architecture Technique**

### **Structure du Code**

```javascript
qgis_yolo_detector/
├── core/                    # Composants principaux
│   ├── annotation_manager.py          # Gestion base données
│   ├── smart_annotation_engine.py     # Pipeline YOLO+SAM
│   ├── yolo_engine.py                 # Détection et entraînement
│   ├── class_mapping.py               # Mapping classes
│   └── geospatial_training_optimizer.py # Optimisations géospatiales
├── ui/                      # Interface utilisateur
├── models/                  # Modèles pré-entraînés
└── utils/                   # Utilitaires
```

### **Pipeline de Données**

```javascript
Annotation Interactive → Stockage SQLite → Génération Dataset YOLO → 
Entraînement Modèle → Application Détection → Export Couches Vectorielles
```

### **Technologies Utilisées**

* **YOLO** : Ultralytics YOLOv8/v11 pour détection

* **SAM** : Segment Anything Model pour raffinement contours

* **OpenCV** : Traitement d'images et extraction contours

* **SQLite** : Stockage local des annotations

* **PyTorch** : Framework d'apprentissage automatique

## 🐛 **État de Développement et Problèmes Connus**

### **Fonctionnalités Stables**

* ✅ Interface 4 onglets

* ✅ Annotation manuelle

* ✅ Stockage SQLite

* ✅ Génération datasets YOLO

* ✅ Pipeline d'entraînement de base

### **Fonctionnalités Expérimentales**

* ⚠️ Smart Assistant (YOLO+SAM)

* ⚠️ Mapping intelligent classes

* ⚠️ Contours précis polygonaux

* ⚠️ Auto-détection zones

### **Problèmes Identifiés**

* Consommation mémoire élevée lors entraînement

* Instabilités occasionnelles du Smart Assistant

* Performance variable selon résolution images

* Messages d'erreur parfois peu explicites

* Documentation utilisateur incomplète

### **Limitations Actuelles**

* Support limité très hautes résolutions (>4K)

* Optimisé principalement pour objets >20px

* Workflow vectoriel-raster pas encore implémenté

* Interface peut être lente sur machines peu puissantes

## 🛣️ **Développements Futurs**

### **Priorités Court Terme**

* [ ] Stabilisation Smart Assistant
* [ ] Amélioration gestion mémoire
* [ ] Pipeline vectoriel-raster automatique
* [ ] Documentation utilisateur complète
* [ ] Tests sur différentes configurations

### **Objectifs Moyen Terme**

* [ ] Interface utilisateur plus intuitive
* [ ] Support formats raster additionnels
* [ ] Optimisations performance
* [ ] Workflow 100% automatique depuis couches vectorielles
* [ ] Métriques de qualité avancées

### **Vision Long Terme**

* [ ] API REST pour déploiement production
* [ ] Hub communautaire de modèles
* [ ] Intégration avec autres outils SIG
* [ ] Support entraînement distribué

## 🤝 **Contribution**

### **Comment Contribuer**

Ce projet étant en développement actif, les contributions sont les bienvenues :

1. **Signaler des bugs** : Issues GitHub avec descriptions détaillées

2. **Proposer améliorations** : Discussions GitHub

3. **Tester fonctionnalités** : Retours d'expérience sur différents cas d'usage

4. **Documentation** : Améliorer guides et tutoriels

5. **Code** : Pull requests après discussion préalable

### **Guidelines de Développement**

* Code style : PEP 8

* Tests : Validation sur QGIS 3.28+ minimum

* Documentation : Docstrings pour nouvelles fonctions

* Compatibilité : Maintenir rétrocompatibilité données

## 📞 **Support et Contact**

### **Resources**

* **Code source** : [GitHub Repository](https://github.com/nic01asFr/qgis_yolo_detector)

* **Issues** : [Bug Reports](https://github.com/nic01asFr/qgis_yolo_detector/issues)

* **Discussions** : [GitHub Discussions](https://github.com/nic01asFr/qgis_yolo_detector/discussions)

### **Contact Développeur**

* **GitHub** : [@nic01asFr](https://github.com/nic01asFr)

## 📜 **Licence**

Ce projet est distribué sous licence **GPL-3.0**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

### **Attributions**

* Ultralytics pour le framework YOLO

* Meta Research pour Segment Anything Model

* Équipe QGIS pour la plateforme SIG

* Communauté OpenCV pour les outils de vision par ordinateur

## ⚠️ **Avertissements**

* **Statut expérimental** : Ce plugin est un prototype en développement

* **Stabilité non garantie** : Fonctionnalités peuvent changer sans préavis

* **Usage recommandé** : Tests et évaluations, pas production critique

* **Sauvegarde données** : Toujours sauvegarder vos projets avant utilisation

* **Performance** : Résultats peuvent varier selon configuration matérielle

*Dernière mise à jour : Version 1.9.2 - Projet en développement actif*
