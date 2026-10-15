# -*- coding: utf-8 -*-
"""Verificación SHAP-GLCD en escala de probabilidad para Ocupación de Menores.

Ejecutar desde examples/tabular/verify_child_occupation_probability.py.
NO entrena modelos, NO recalcula valores SHAP ni modifica artefactos.
"""
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# IMPORTANTE: el modelo fue guardado cuando esta clase vivía en __main__.
# Conservar el mismo nombre y constructor permite que joblib lo recupere.
class MyOptimizedRandomForestClassifier(RandomForestClassifier):
    def __init__(self, random_state=42):
        super().__init__(
            n_estimators=800, max_depth=30, min_samples_split=5,
            min_samples_leaf=5, max_features=0.5, class_weight=None,
            n_jobs=-1, random_state=random_state,
        )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "child_occupation"
OUTPUT_DIR = PROJECT_ROOT / "results" / "child_occupation" / "probability_validation"
SEED = 42
N_SAMPLE = 200  # usar 0 para comprobar las 12 250 instancias
ATOL = 1e-5
J_SEL, R_SEL = 0, 1


def main():
    print("\n=== VALIDACIÓN SHAP EN ESCALA DE PROBABILIDAD ===")
    print("Artefactos:", ARTIFACT_DIR)
    print("Par:", J_SEL, "vs", R_SEL, "| Semilla:", SEED)
    print("NOTA: no recalcula SHAP; reconstruye usando SHAP ya guardado.")

    files = ["X_test_final.csv", "class_names.json", "shap_values_3d.npy",
             "test_probabilities.csv", "random_forest_optimizado.joblib"]
    missing = [f for f in files if not (ARTIFACT_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError("Faltan artefactos:\n" + "\n".join(missing))

    with (ARTIFACT_DIR / "class_names.json").open(encoding="utf-8-sig") as f:
        class_names = {int(k): str(v) for k, v in json.load(f).items()}
    if class_names.get(0) != "asoTrabajo" or class_names.get(1) != "estudia":
        raise ValueError(f"Clases inesperadas: {class_names}")

    x_df = pd.read_csv(ARTIFACT_DIR / "X_test_final.csv", encoding="utf-8-sig")
    x = x_df.to_numpy(dtype=float)
    sv = np.load(ARTIFACT_DIR / "shap_values_3d.npy", mmap_mode="r", allow_pickle=False)
    p_saved_df = pd.read_csv(ARTIFACT_DIR / "test_probabilities.csv")
    p_saved = p_saved_df.to_numpy(dtype=float)

    print("X_test:", x.shape, "| SHAP:", sv.shape, "| P guardadas:", p_saved.shape)
    n, m = x.shape
    if sv.shape != (n, m, 4) or p_saved.shape != (n, 4):
        raise ValueError("Dimensiones inconsistentes: se espera SHAP (N, M, 4) y probabilidades (N, 4).")
    if not np.isfinite(p_saved).all() or not np.allclose(p_saved.sum(axis=1), 1, atol=ATOL):
        raise ValueError("Las probabilidades guardadas no son finitas o no suman 1.")
    if np.any(p_saved < -ATOL) or np.any(p_saved > 1 + ATOL):
        raise ValueError("Probabilidades fuera del intervalo [0, 1].")

    print("Cargando Random Forest guardado (sin entrenarlo)...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(ARTIFACT_DIR / "random_forest_optimizado.joblib")
    for w in caught:
        print("ADVERTENCIA al cargar modelo:", str(w.message))
    classes = [int(c) for c in model.classes_]
    if classes != [0, 1, 2, 3]:
        raise ValueError(f"Orden de clases inesperado: {classes}")

    rng = np.random.default_rng(SEED)
    idx = (np.arange(n) if N_SAMPLE == 0 or N_SAMPLE >= n
           else np.sort(rng.choice(n, size=N_SAMPLE, replace=False)))
    # El modelo fue entrenado con arrays, no con nombres de columnas.
    p_model = np.asarray(model.predict_proba(x[idx]), dtype=float)
    error_saved = np.abs(p_model - p_saved[idx]).max()
    print(f"Máxima diferencia predict_proba vs CSV guardado: {error_saved:.10g}")

    # TreeExplainer se usa ÚNICAMENTE para recuperar expected_value.
    # No se llama a shap_values() ni a explainer(x).
    import shap
    tree_explainer = shap.TreeExplainer(model)
    base = np.asarray(tree_explainer.expected_value, dtype=float).reshape(-1)
    if base.size != 4:
        raise ValueError(f"expected_value debe tener 4 componentes; recibido: {base.shape}")
    print("Valores base SHAP por clase:", base)
    print("Suma valores base:", base.sum())

    phi = np.asarray(sv[idx, :, :], dtype=float)
    p_reconstructed = base[np.newaxis, :] + phi.sum(axis=1)
    error = p_reconstructed - p_model
    abs_error = np.abs(error)

    # Comprobación diferencial: (base_0 - base_1) + suma (SHAP_0 - SHAP_1)
    d_model = p_model[:, J_SEL] - p_model[:, R_SEL]
    d_reconstructed = (base[J_SEL] - base[R_SEL]) + (
        phi[:, :, J_SEL] - phi[:, :, R_SEL]
    ).sum(axis=1)
    d_abs_error = np.abs(d_reconstructed - d_model)

    print("\n--- RESULTADOS ---")
    print("Instancias verificadas:", len(idx))
    print(f"Error máximo reconstrucción por clase: {abs_error.max():.10g}")
    print(f"Error medio reconstrucción por clase: {abs_error.mean():.10g}")
    print(f"Error máximo reconstrucción par 0 vs 1: {d_abs_error.max():.10g}")
    print(f"Error medio reconstrucción par 0 vs 1: {d_abs_error.mean():.10g}")
    ok_model = bool(error_saved <= ATOL)
    ok_class = bool(abs_error.max() <= ATOL)
    ok_pair = bool(d_abs_error.max() <= ATOL)
    print("Coincidencia de probabilidades guardadas:", "APROBADO" if ok_model else "NO APROBADO")
    print("Reconstrucción aditiva por clase:", "APROBADO" if ok_class else "NO APROBADO")
    print("Reconstrucción diferencial 0 vs 1:", "APROBADO" if ok_pair else "NO APROBADO")

    # Evidencia reproducible, una fila por instancia para el par principal.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "test_position": idx,
        "p_asoTrabajo_modelo": p_model[:, J_SEL],
        "p_estudia_modelo": p_model[:, R_SEL],
        "deltaP_modelo": d_model,
        "deltaP_reconstruido_SHAP": d_reconstructed,
        "error_abs_deltaP": d_abs_error,
    })
    output_path = OUTPUT_DIR / "validation_probabilities_pair_0_vs_1.csv"
    out.to_csv(output_path, index=False)
    print("CSV de verificación:", output_path)
    if not (ok_model and ok_class and ok_pair):
        print("ATENCIÓN: revisión no aprobada. No alterar los artefactos; investigar el motivo.")
        sys.exit(1)
    print("VALIDACIÓN NUMÉRICA APROBADA (tolerancia absoluta:", ATOL, ")")
    print("IMPORTANTE: el resultado no resuelve la advertencia de versiones de scikit-learn.")


if __name__ == "__main__":
    main()
