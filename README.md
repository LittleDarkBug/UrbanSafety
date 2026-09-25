# UrbanSafety

Détection d'anomalies en vidéosurveillance par apprentissage faiblement supervisé, avec une couche vision-langage pour chercher des vidéos en langage naturel et leur associer des descriptions textuelles.

Projet du cours SY23, réalisé par Didi Orlog SOSSOU. Tout le pipeline tient dans le notebook [`UrbanSafety.ipynb`](UrbanSafety.ipynb).

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

## Exécution

Le notebook fonctionne en local ou sur Kaggle (variable `MODE` de la cellule de configuration). Un GPU NVIDIA est nécessaire ; l'exécution enregistrée disposait d'environ 6 Go de mémoire vidéo.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

CADP et les annotations DoTA se téléchargent automatiquement via `kagglehub`. BDD100K et UCF-Crime sont à placer dans les dossiers indiqués dans la cellule de configuration.

Les jeux de données, les captions générées, les embeddings, les index FAISS et les poids entraînés ne sont pas versionnés.

## Stack

PyTorch, OpenCLIP, PEFT (LoRA), Transformers, FAISS, OpenCV, scikit-learn, Gradio, kagglehub.
