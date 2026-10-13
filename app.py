import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from supabase import create_client

MODEL_BUCKET = "bcs-models"
MODEL_OBJECT = "models/bcs_mobilenet_v3_small.pt"

st.set_page_config(page_title="BCS Cattle Predictor", page_icon="🐄", layout="centered")


def _secret(*names):
    for name in names:
        value = st.secrets.get(name, "")
        if value:
            return value
    raise RuntimeError("Secret manquant : " + " ou ".join(names))


@st.cache_resource
def get_supabase():
    return create_client(
        _secret("SUPABASE_URL"),
        _secret("SUPABASE_SECRET_KEY", "SUPABASE_KEY"),
    )


def build_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


@st.cache_resource(show_spinner="Chargement du modèle BCS…")
def load_predictor():
    payload = get_supabase().storage.from_(MODEL_BUCKET).download(MODEL_OBJECT)
    bundle = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if bundle.get("architecture") != "mobilenet_v3_small":
        raise RuntimeError("Architecture du modèle non reconnue.")
    model = build_model()
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    weights = MobileNet_V3_Small_Weights.DEFAULT
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(int(bundle.get("input_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
    ])
    return model, tf, bundle


def round_quarter(x: float) -> float:
    return round(x * 4.0) / 4.0


def predict(image: Image.Image, model, tf, bundle):
    x = tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        y = float(model(x).squeeze().item())
    y = float(np.clip(y, bundle.get("score_min", 1.0), bundle.get("score_max", 5.0)))
    return y, round_quarter(y)


st.title("🐄 BCS Cattle Predictor")
st.caption("Estimation automatique du Body Condition Score à partir d’une photographie.")

try:
    model, tf, bundle = load_predictor()
except Exception as exc:
    st.error("Le modèle BCS ne peut pas être chargé.")
    st.code(str(exc))
    st.info(
        "Vérifiez SUPABASE_URL et SUPABASE_SECRET_KEY dans Streamlit Secrets, "
        "puis assurez-vous que le bucket privé 'bcs-models' contient "
        f"'{MODEL_OBJECT}'."
    )
    st.stop()

st.info(
    "Utilisez une photo conforme au protocole employé pour constituer la base d’apprentissage "
    "(vue, cadrage, distance et qualité comparables)."
)

camera = st.camera_input("Prendre une photo")
upload = st.file_uploader("ou importer une photo", type=["jpg", "jpeg", "png"])
photo = camera if camera is not None else upload

if photo is not None:
    image = Image.open(photo).convert("RGB")
    st.image(image, caption="Photo à évaluer", use_container_width=True)
    if st.button("Déterminer le BCS", type="primary", use_container_width=True):
        raw, rounded = predict(image, model, tf, bundle)
        st.metric("BCS estimé", f"{rounded:.2f}")
        st.caption(f"Prédiction continue du modèle : {raw:.2f}")

        q95 = bundle.get("validation_metrics", {}).get("residual_abs_q95")
        if q95 is not None:
            lo = max(float(bundle.get("score_min", 1.0)), raw - float(q95))
            hi = min(float(bundle.get("score_max", 5.0)), raw + float(q95))
            st.write(f"Intervalle empirique basé sur 95 % des erreurs de validation : **{lo:.2f} à {hi:.2f}**")

with st.expander("Informations sur le modèle"):
    vm = bundle.get("validation_metrics", {})
    st.write(f"Entraîné le : {bundle.get('trained_at_utc', '—')}")
    st.write(f"Images d’apprentissage : {bundle.get('n_images_train', '—')}")
    st.write(f"Images de validation : {bundle.get('n_images_validation', '—')}")
    st.write(f"Animaux distincts : {bundle.get('n_animals_total', '—')}")
    if vm:
        st.write(f"MAE de validation : {vm.get('mae', float('nan')):.3f} point BCS")
        st.write(f"RMSE de validation : {vm.get('rmse', float('nan')):.3f}")
        st.write(f"À ±0,25 : {100*vm.get('within_0_25', 0):.1f} %")
        st.write(f"À ±0,50 : {100*vm.get('within_0_50', 0):.1f} %")

st.warning(
    "Cette estimation est un résultat algorithmique. Sa validité dépend de la qualité et de la représentativité "
    "de la base d’apprentissage. Elle doit être évaluée sur un jeu de validation indépendant avant usage scientifique ou clinique."
)
