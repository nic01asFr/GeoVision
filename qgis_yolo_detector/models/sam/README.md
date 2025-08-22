# Modèles SAM pour Smart Annotation Engine

## Modèles Requis (à télécharger manuellement)

Pour activer le mode SAM dans l'extension, vous devez télécharger et placer les modèles suivants dans ce répertoire :

### FastSAM (Recommandé pour CPU moyens/puissants)
- **Fichier** : `FastSAM-s.pt`
- **Source** : https://github.com/CASIA-IVA-Lab/FastSAM
- **Taille** : ~23MB
- **Usage** : CPU 4+ cores, 8GB+ RAM

### MobileSAM (Recommandé pour CPU faibles)
- **Fichier** : `mobile_sam.pt`
- **Source** : https://github.com/ChaoningZhang/MobileSAM
- **Taille** : ~6MB
- **Usage** : CPU 2+ cores, 4GB+ RAM

## Installation

1. Téléchargez les modèles depuis les sources officielles
2. Placez les fichiers `.pt` dans ce répertoire
3. Redémarrez QGIS
4. Le Smart Mode activera automatiquement SAM si les modèles sont détectés

## Sécurité

⚠️ **IMPORTANT** : Ces modèles doivent être téléchargés manuellement et inclus dans l'extension. L'extension ne télécharge JAMAIS de modèles automatiquement pour des raisons de sécurité et de performance.

## Structure attendue
```
models/sam/
├── FastSAM-s.pt      # Modèle FastSAM (optionnel)
├── mobile_sam.pt     # Modèle MobileSAM (optionnel)
└── README.md         # Ce fichier
```