# BCS Cattle Predictor

Nouvelle application, séparée de **BCS Cattle Acquisition**.

Elle utilise les photos et scores déjà recueillis dans Supabase pour entraîner un modèle de vision qui prédit un Body Condition Score (BCS) à partir d'une nouvelle photo.

## Architecture

1. `BCS Cattle Acquisition` continue à produire la base de données.
2. `train_model.py` :
   - sélectionne uniquement les acquisitions `accepted` ;
   - récupère les photos dans `bcs-photos` ;
   - rassemble les scores terrain (`bcs_operateur_1..3`) et les scores indépendants de la table `scores` ;
   - calcule comme cible la **médiane** des scores disponibles ;
   - exige par défaut au moins 2 scores de référence par photo ;
   - sépare apprentissage et validation **par animal**, pour éviter qu'une même vache soit présente dans les deux ensembles ;
   - fine-tune MobileNetV3-Small pré-entraîné sur ImageNet ;
   - mesure MAE, RMSE, proportion des prédictions à ±0,25 et ±0,50 point BCS ;
   - envoie le modèle final dans le bucket privé `bcs-models`.
3. `app.py` télécharge ce modèle et prédit le BCS d'une nouvelle photo.

## 1. Préparer Supabase

Exécuter `setup_model_storage.sql` dans **Supabase > SQL Editor**.

Le projet doit déjà contenir les tables `acquisitions`, `scores` et le bucket privé `bcs-photos` créés par BCS Cattle Acquisition.

## 2. Secrets

Créer `.streamlit/secrets.toml` en local :

```toml
SUPABASE_URL = "https://VOTRE_PROJET.supabase.co"
SUPABASE_SECRET_KEY = "sb_secret_..."
```

Dans Streamlit Community Cloud, mettre les mêmes valeurs dans **App > Settings > Secrets**.

La clé secrète est utilisée côté serveur uniquement. Ne jamais la placer dans GitHub.

## 3. Entraîner le modèle

L'entraînement est à exécuter sur une machine locale disposant de Python, idéalement avec GPU mais il fonctionne aussi sur CPU :

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python train_model.py
```

Par défaut :

- seules les photos `accepted` sont utilisées ;
- au moins 2 scores valides sont requis ;
- 20 % des données servent à la validation ;
- la séparation se fait par `animal_id` ;
- le modèle est envoyé vers `bcs-models/models/bcs_mobilenet_v3_small.pt`.

Variables optionnelles :

```text
BCS_MIN_REFERENCE_SCORES=2
BCS_VAL_FRACTION=0.20
BCS_FROZEN_EPOCHS=5
BCS_FINETUNE_EPOCHS=12
BCS_BATCH_SIZE=16
```

## 4. Déployer l'application de prédiction

Créer un **nouveau dépôt GitHub** pour cette application, distinct de BCS Cattle Acquisition, et y placer :

- `app.py`
- `requirements.txt`
- `.gitignore`
- `README.md`

Déployer ensuite ce dépôt sur Streamlit Community Cloud en utilisant `app.py` comme Main file.

Ajouter les deux secrets Supabase dans les paramètres de la nouvelle application.

Le fichier `.pt` n'a pas besoin d'être sur GitHub : l'application le charge depuis Supabase Storage.

## 5. Interprétation

Le modèle produit une prédiction continue entre 1 et 5, puis l'affichage principal est arrondi au quart de point le plus proche. L'intervalle affiché est dérivé du 95e percentile des erreurs absolues observées sur le jeu de validation ; il ne constitue pas un intervalle de confiance individuel au sens statistique strict.

## 6. Points méthodologiques importants

- Ne pas mélanger des photos du même animal entre apprentissage et validation.
- Maintenir le même protocole photographique lors de l'utilisation de l'application.
- Une base comportant peu d'animaux ou une distribution BCS étroite produira un modèle peu généralisable.
- Évaluer idéalement le modèle final sur un **jeu de test externe** provenant d'animaux qui n'ont jamais été utilisés pour l'entraînement ni pour le réglage du modèle.
- La médiane des évaluateurs réduit l'influence d'un score aberrant, mais ne remplace pas une étude formelle de la concordance inter-évaluateurs.
