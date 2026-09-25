# UrbanSafety

Détection d'anomalies en vidéosurveillance par apprentissage faiblement supervisé, avec une couche vision-langage pour chercher des vidéos en langage naturel et leur associer des descriptions textuelles.

Projet du cours SY23, réalisé par Didi Orlog SOSSOU. Le pipeline est un paquet Python installable (`src/urbansafety`) piloté en ligne de commande ; le notebook [`notebooks/UrbanSafety.ipynb`](notebooks/UrbanSafety.ipynb) garde la trace de l'expérimentation et de ses résultats.

## Ce que fait le système

| Fonction | Principe |
|---|---|
| Détection temporelle d'anomalies | LSTM bidirectionnel qui attribue un score d'anomalie à chaque segment de 5 secondes |
| Recherche texte vers vidéo | Requête encodée par OpenCLIP, recherche des vidéos les plus proches dans un index FAISS |
| Description d'une vidéo | Recherche, dans un index FAISS de descriptions, des phrases les plus proches de l'embedding de la vidéo |
| Démonstration | Interface Gradio à trois onglets : recherche, analyse, description |

## Pipeline

| Phase | Étape | Détail |
|---|---|---|
| 1 | Captions BDD100K | Un petit modèle de langage (Qwen2.5-Instruct) transforme les métadonnées BDD100K (météo, scène, heure, objets, feux, voies) en phrases descriptives. 69 863 captions d'entraînement et 10 000 de validation |
| 2 | Fine-tuning OpenCLIP | Adaptation contrastive d'OpenCLIP ViT-L/14 (`laion400m_e32`) par LoRA sur les paires image-caption BDD100K : adaptateurs de rang 16 (alpha 32, dropout 0,05) sur les couches `c_fc` et `c_proj`, perte InfoNCE symétrique image-texte, AdamW, précision mixte bfloat16 |
| 3 | Indexation | Embeddings OpenCLIP de segments de 5 secondes (5 images par segment, moyennées), index FAISS vidéo et index FAISS de 29 201 descriptions |
| 4 | Détecteur | LSTM bidirectionnel (512 unités par direction) entraîné en Multiple Instance Learning avec une ranking hinge loss, puis variante robuste entraînée sous attaque PGD |
| 5 | Évaluation | AUC-ROC, précision moyenne, précision, rappel, F1, matrice de confusion, courbe ROC |
| 6 | Démonstration | Interface Gradio |

Chaque phase correspond à une commande (`urbansafety download`, `captions`, `finetune`, `index`, `train`, `evaluate`, `demo`) et lit les artefacts de la précédente.

## Données

| Jeu | Contenu | Usage |
|---|---|---|
| BDD100K | Images de conduite urbaine avec métadonnées | Captions et fine-tuning OpenCLIP |
| UCF-Crime | 1 854 vidéos, crimes et incidents ainsi que vidéos normales, annotations UCA | Entraînement et évaluation du détecteur, index |
| CADP | 982 séquences de vidéosurveillance routière, dont 455 accidents | Index et évaluation |
| DoTA | 4 677 vidéos d'anomalies de trafic, 4 352 exploitées | Index |

Le LSTM n'est entraîné que sur UCF-Crime. DoTA et CADP n'ont pas de structure temporelle exploitable par un modèle séquentiel ; ils servent à la recherche texte vers vidéo.

Les annotations temporelles sont converties en pseudo-labels par segment de 5 secondes : 53 814 segments d'entraînement UCF-Crime, dont 12 318 anormaux.

## Résultats

Sélection du modèle sur le jeu de validation UCF-Crime :

| Modèle | AUC validation | F1 validation |
|---|---|---|
| LSTM bidirectionnel, ranking loss | **0,9941** | **0,9627** |
| Même modèle entraîné sous attaque PGD (epsilon 0,005, 3 itérations) | 0,9895 | 0,9190 |

Le modèle de base est retenu. L'entraînement adversarial coûte un peu de performance sur des données non perturbées.

Évaluation finale sur 410 vidéos de test (UCF-Crime et CADP, dont 253 anomalies), seuil 0,4 :

| AUC | Précision moyenne | Précision | Rappel | F1 |
|---|---|---|---|---|
| 0,8994 | 0,9434 | 0,6941 | 0,9684 | 0,8086 |

Matrice de confusion : 245 anomalies détectées sur 253, 108 fausses alarmes sur 157 vidéos normales. Le système privilégie le rappel.

## Limites

La description d'une vidéo est une recherche de phrases existantes par similarité, pas une génération de texte.

## Organisation du dépôt

```
src/urbansafety/
  config.py            chemins (local ou Kaggle), constantes, graine
  data/
    download.py        CADP via kagglehub, DoTA (annotations GitHub, images Google Drive)
    annotations.py     UCF-Crime, DoTA, CADP harmonisés ; pseudo-labels par segment
    bdd100k.py         métadonnées BDD100K et paires image-caption
    features.py        séquences d'embeddings pour le LSTM
  captioning.py        phase 1 : captions par Qwen2.5-Instruct
  models/
    clip_lora.py       phase 2 : OpenCLIP ViT-L/14 et adaptateurs LoRA
    detector.py        phase 4 : BiLSTM, ranking hinge loss (MIL), attaque PGD
  embeddings.py        phase 3 : embeddings par segment de 5 secondes
  indexing.py          phase 3 : index FAISS vidéo et texte
  training.py          phase 4 : entraînement de base ou adversarial, arrêt anticipé
  evaluation.py        phase 5 : métriques et figures
  retrieval.py         phase 6 : recherche, description, analyse temporelle
  app.py               phase 6 : interface Gradio
  cli.py               commandes du pipeline
notebooks/UrbanSafety.ipynb   expérimentation d'origine, avec ses sorties
tests/                        tests unitaires, exécutables sur CPU sans les jeux de données
```

## Utilisation

Un GPU NVIDIA est nécessaire pour les phases 1 à 3 ; l'expérimentation a tourné avec environ 6 Go de mémoire vidéo.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[captions,data,demo]"

urbansafety download                 # CADP et DoTA ; BDD100K et UCF-Crime sont à placer à la main
urbansafety captions                 # phase 1
urbansafety finetune --subset 1000   # phase 2
urbansafety index                    # phase 3
urbansafety train                    # phase 4 : modèle de base et modèle PGD, le meilleur est conservé
urbansafety evaluate                 # phase 5 : métriques JSON et evaluation.png
urbansafety demo                     # phase 6
```

`--mode kaggle` utilise les jeux montés sous `/kaggle/input`, `--work` fixe le répertoire des artefacts. Les jeux de données, captions, embeddings, index et poids ne sont pas versionnés.

Tests :

```bash
pip install -e ".[dev]"
pytest
```

## Différences avec le notebook

Le paquet reprend le notebook à l'identique, à trois corrections près :

- l'attaque PGD calcule le gradient par rapport à la seule perturbation ; dans le notebook, `loss.backward()` accumulait aussi des gradients dans les poids du détecteur, appliqués ensuite par l'optimiseur ;
- l'onglet Description de l'interface Gradio lit correctement les résultats de la recherche ;
- les deux boucles d'entraînement, de base et adversariale, ne forment plus qu'une fonction paramétrée.

Les résultats ci-dessus proviennent du notebook.

## Stack

PyTorch, OpenCLIP, PEFT (LoRA), Transformers, FAISS, OpenCV, scikit-learn, Gradio, kagglehub.
