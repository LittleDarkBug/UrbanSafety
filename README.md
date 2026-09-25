# UrbanSafety

Détection d'anomalies en vidéosurveillance, avec une couche vision-langage pour retrouver une scène à partir d'une phrase et associer du texte à ce que montre une vidéo.

Le détecteur apprend sans horodatage précis : il ne connaît que l'étiquette de chaque vidéo (normale ou anormale) et apprend seul à localiser les segments suspects. Les embeddings viennent d'OpenCLIP, adapté aux scènes urbaines par LoRA sur des descriptions générées par un petit modèle de langage.

Projet étudiant, conçu et développé de bout en bout : génération des données textuelles, adaptation du modèle, détecteur, indexation et démonstration.

## Fonctionnalités

- **Détection temporelle** : un score d'anomalie pour chaque segment de 5 secondes d'une vidéo.
- **Recherche texte vers vidéo** : « car accident on the road » renvoie les vidéos les plus proches, via un index FAISS.
- **Description** : les phrases d'annotation les plus proches du contenu d'une vidéo.
- **Démonstration** : interface Gradio réunissant recherche, analyse et description.

## Architecture

| Étape | Rôle | Détail |
|---|---|---|
| Captions | Transformer les métadonnées BDD100K en phrases | Qwen2.5-Instruct résume météo, scène, heure, objets, feux et marquages en une phrase ; 69 863 captions d'entraînement, 10 000 de validation |
| Adaptation OpenCLIP | Spécialiser l'encodeur sur la conduite urbaine | ViT-L/14 (`laion400m_e32`), adaptateurs LoRA de rang 16 sur `c_fc` et `c_proj`, perte InfoNCE symétrique image-texte, bfloat16 |
| Embeddings | Représenter chaque vidéo | 5 images par segment de 5 secondes, encodées puis moyennées, normalisées L2 |
| Indexation | Recherche par similarité cosinus | Index FAISS des vidéos et de 29 201 phrases d'annotation |
| Détecteur | Scorer chaque segment | LSTM bidirectionnel (2 × 512) entraîné en Multiple Instance Learning par ranking hinge loss |
| Robustesse | Résister aux perturbations des embeddings | Entraînement adversarial PGD (epsilon 0,005, 3 pas) |

La ranking hinge loss compare les sacs de segments : le segment le plus suspect d'une vidéo anormale doit dépasser celui d'une vidéo normale d'une marge fixe. C'est ce qui permet d'apprendre sans annoter chaque instant.

## Données

| Jeu | Contenu | Usage |
|---|---|---|
| BDD100K | Images de conduite urbaine et leurs métadonnées | Captions, adaptation OpenCLIP |
| UCF-Crime (annotations UCA) | 1 854 vidéos de vidéosurveillance, crimes, incidents et scènes normales | Entraînement et évaluation du détecteur |
| CADP | 982 séquences routières, dont 455 accidents | Recherche, évaluation |
| DoTA | 4 677 vidéos d'anomalies de trafic | Recherche |

Le détecteur est entraîné sur UCF-Crime, seul jeu à structure temporelle continue : 53 814 segments d'entraînement, dont 12 318 anormaux.

## Résultats

Validation (UCF-Crime) :

| Modèle | AUC | F1 |
|---|---|---|
| BiLSTM, ranking hinge loss | **0,9941** | **0,9627** |
| BiLSTM, entraînement adversarial PGD | 0,9895 | 0,9190 |

Test sur 410 vidéos (UCF-Crime et CADP, 253 anomalies), seuil 0,4 :

| AUC | Précision moyenne | Précision | Rappel | F1 |
|---|---|---|---|---|
| 0,8994 | 0,9434 | 0,6941 | 0,9684 | 0,8086 |

245 anomalies détectées sur 253. Le réglage privilégie le rappel, au prix de fausses alarmes sur les scènes normales. C'est le premier point à améliorer.

## Installation

Un GPU NVIDIA est nécessaire pour les captions, l'adaptation d'OpenCLIP et l'extraction des embeddings ; 6 Go de mémoire vidéo suffisent.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[captions,data,demo]"
```

BDD100K et UCF-Crime sont à placer dans le répertoire de travail ; CADP et DoTA se téléchargent avec `urbansafety download`.

## Utilisation

```bash
urbansafety download     # CADP et DoTA
urbansafety captions     # captions BDD100K
urbansafety finetune     # adaptation LoRA d'OpenCLIP
urbansafety index        # embeddings et index FAISS
urbansafety train        # détecteur de base et détecteur PGD, le meilleur est conservé
urbansafety evaluate     # métriques et figures
urbansafety demo         # interface Gradio
```

`--mode kaggle` lit les jeux montés sous `/kaggle/input` ; `--work` choisit le répertoire des artefacts.

## Organisation

```
src/urbansafety/
  data/          téléchargement, annotations, pseudo-labels, BDD100K
  models/        OpenCLIP + LoRA, détecteur BiLSTM et attaque PGD
  captioning.py  captions par modèle de langage
  embeddings.py  embeddings par segment
  indexing.py    index FAISS
  training.py    entraînement et arrêt anticipé
  evaluation.py  métriques et figures
  retrieval.py   recherche, description, analyse
  app.py         interface Gradio
  cli.py         commandes
notebooks/       carnet d'expérimentation
tests/           tests unitaires (CPU, sans jeu de données)
```

```bash
pip install -e ".[dev]"
pytest
```

## Stack

PyTorch, OpenCLIP, PEFT (LoRA), Transformers, FAISS, OpenCV, scikit-learn, Gradio.
