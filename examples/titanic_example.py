"""Titanic survival example for NeuronLens (tabular data).

Loads the Titanic dataset from Hugging Face (CSV download), trains a small
MLP, then generates a NeuronLens visualisation with precomputed filters for
survival, sex, and passenger class.

Usage:
    python examples/titanic_example.py

Requirements:
    pip install scikit-learn
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neuronlens import NeuronLens
from neuronlens.filters import eq, and_

# ── Data loading ──────────────────────────────────────────────────────────────

TITANIC_URL = (
    "https://huggingface.co/datasets/julien-c/"
    "titanic-survival/resolve/main/titanic.csv"
)

def load_titanic():
    """Load Titanic CSV from Hugging Face and return (X, metadata)."""
    df = pd.read_csv(TITANIC_URL)

    # Drop rows with missing Age or Fare
    df = df.dropna(subset=["Age", "Fare"]).reset_index(drop=True)

    # Encode Sex as 0/1
    df["sex_enc"] = (df["Sex"] == "female").astype(float)

    feature_cols = [
        "Pclass", "sex_enc", "Age",
        "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare",
    ]
    X = df[feature_cols].values.astype(np.float32)

    metadata = pd.DataFrame({
        "survived":  df["Survived"].map({0: "no", 1: "yes"}),
        "sex":       df["Sex"],
        "pclass":    df["Pclass"].astype(str),
        "adult":     (df["Age"] >= 18).map({True: "yes", False: "no"}),
    })

    return X, metadata, df["Survived"].values

# ── Model ─────────────────────────────────────────────────────────────────────

def make_model(n_in: int = 6) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

# ── Training ──────────────────────────────────────────────────────────────────

def train(model, X_tr, y_tr, n_epochs=100, lr=1e-3):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            with torch.no_grad():
                preds = (model(Xt) > 0).float()
                acc = (preds == yt).float().mean().item() * 100
            print(f"  Epoch {epoch:3d}  loss {loss.item():.4f}  train acc {acc:.1f}%")

def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        preds = (model(torch.tensor(X_te)) > 0).float().numpy().flatten()
    acc = (preds == y_te).mean() * 100
    print(f"  Test accuracy: {acc:.1f}%")
    return acc

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[titanic] Loading dataset…")
    X, metadata, y = load_titanic()
    print(f"  {len(X)} samples, {X.shape[1]} features")

    # Train/test split (stratified by survival)
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(X)), test_size=0.2, stratify=y, random_state=42
    )

    # Normalise features (fit on train, apply to all)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_sc = scaler.transform(X_te).astype(np.float32)
    X_all   = scaler.transform(X).astype(np.float32)

    print("[titanic] Training model…")
    model = make_model(n_in=X.shape[1])
    train(model, X_tr_sc, y_tr)
    evaluate(model, X_te_sc, y_te)
    model.eval()

    print("[titanic] Running NeuronLens…")
    viz = NeuronLens(
        model=model,
        dataset=X_all,
        metadata=metadata,
        max_display_units=200,
        n_reorder_passes=10,
        reorder_by="survived",
        precomputed_filters=[
            eq("survived", "yes"),
            eq("survived", "no"),
            eq("sex", "female"),
            eq("sex", "male"),
            eq("pclass", "1"),
            eq("pclass", "3"),
            and_(eq("survived", "yes"), eq("sex", "female")),
            and_(eq("survived", "no"),  eq("sex", "male")),
        ],
    )

    out_dir = os.path.join(os.path.dirname(__file__), "titanic_output")
    viz.show(output_dir=out_dir)
    print(f"[titanic] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
