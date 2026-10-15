# ============================================================
# glcd_explainer_from_artifacts_CORREGIDA_PAIR_TRUE_BALANCED010926.py
# ============================================================
# Corrección metodológica SHAP-GLCD para datos tabulares cargados
# desde artefactos.
#
# Esta versión HEREDA la clase original:
#     glcd_explainer_from_artifacts.GLCDExplainerFromArtifacts
#
# y modifica únicamente las partes necesarias para que el análisis
# global pairwise sea metodológicamente consistente con los demás
# conjuntos tabulares del proyecto:
#
#   1) selección por ETIQUETA VERDADERA;
#   2) muestra balanceada entre Cj y Cr;
#   3) semilla 42 (o RANDOM_STATE indicado);
#   4) muestreo sin reemplazo;
#   5) NO se filtra por predicción:
#      se conservan aciertos y errores;
#   6) ΔS y todas las métricas globales pairwise usan exactamente
#      la misma muestra balanceada;
#   7) KDE, Cohen d/h, KS, t_opt, t_bnd y precision/recall usan esa
#      misma muestra;
#   8) las métricas predictivas del modelo continúan calculándose
#      sobre TODO el conjunto test guardado;
#   9) la KDE "all classes" continúa siendo contextual y usa todo
#      el test.
#
# No reentrena el modelo.
# No recalcula SHAP.
# No modifica los artefactos originales.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score
)

from scipy.stats import fisher_exact, ks_2samp

from .artifacts_base import (
    GLCDExplainerFromArtifacts as _GLCDExplainerFromArtifactsBase,
    safe_filename,
)

class GLCDExplainerFromArtifacts(_GLCDExplainerFromArtifactsBase):
    """
    Versión corregida de GLCDExplainerFromArtifacts.

    Mantiene la implementación original para:
    - carga de artefactos;
    - figuras Mean |ΔS|, Median |ΔS| y Signed Mean;
    - Conditional Mean;
    - Weighted Dominance;
    - métricas predictivas;
    - análisis extra de la característica top-1.

    Corrige la población usada en el análisis pairwise:
    únicamente se agregan instancias cuya ETIQUETA VERDADERA pertenece
    a Cj o Cr y se balancean ambas clases con la misma cantidad de casos.
    """

    # ========================================================
    # VALIDACIÓN REFORZADA
    # ========================================================

    def _validate_artifacts(self):
        """
        Conserva las validaciones de la clase original y añade:
        - correspondencia X_test / y_test / SHAP;
        - j_sel != r_sel;
        - presencia de ambas clases seleccionadas en test.
        """

        super()._validate_artifacts()

        if len(self.y_te) != self.X_te.shape[0]:
            raise ValueError(
                "Inconsistencia entre X_test_final y y_test_final:\n"
                f"X_test_final tiene {self.X_te.shape[0]} filas, "
                f"pero y_test_final tiene {len(self.y_te)} etiquetas."
            )

        if self.shap_vals.shape[0] != len(self.y_te):
            raise ValueError(
                "Inconsistencia entre SHAP y y_test_final:\n"
                f"SHAP tiene {self.shap_vals.shape[0]} instancias, "
                f"pero y_test_final tiene {len(self.y_te)} etiquetas."
            )

        if self.j_sel == self.r_sel:
            raise ValueError(
                "j_sel y r_sel deben representar clases diferentes."
            )

        n_j = int(np.sum(self.y_te == self.j_sel))
        n_r = int(np.sum(self.y_te == self.r_sel))

        if n_j == 0 or n_r == 0:
            raise ValueError(
                "No es posible construir el par seleccionado porque alguna "
                "de las clases no tiene instancias en el conjunto test.\n"
                f"Clase j={self.j_sel}: {n_j} instancias.\n"
                f"Clase r={self.r_sel}: {n_r} instancias."
            )

    # ========================================================
    # SELECCIÓN GLOBAL PAIRWISE
    # ========================================================

    def _seleccionar_muestra_balanceada_par(self):
        """
        Selecciona una muestra balanceada dentro del conjunto de prueba,
        formada exclusivamente por instancias cuya ETIQUETA VERDADERA
        pertenece al par (j_sel, r_sel).

        Criterios:
        - selección por etiqueta verdadera;
        - mismo número de instancias de Cj y Cr;
        - semilla RANDOM_STATE;
        - muestreo sin reemplazo;
        - NO se filtra por predicción;
        - se conservan aciertos y errores.

        La selección se realiza sobre POSICIONES del test para preservar
        la correspondencia exacta entre:
            X_te[i] <-> y_te[i] <-> shap_vals[i, :, :].
        """

        # Evita volver a seleccionar si la muestra ya fue creada.
        if hasattr(self, "pair_indices"):
            return self.pair_indices

        if not hasattr(self, "shap_vals"):
            raise AttributeError(
                "Primero deben cargarse los valores SHAP."
            )

        if len(self.shap_vals) != len(self.y_te):
            raise ValueError(
                "No existe correspondencia uno a uno entre SHAP y y_te: "
                f"len(shap_vals)={len(self.shap_vals)}, "
                f"len(y_te)={len(self.y_te)}."
            )

        if self.X_te.shape[0] != len(self.y_te):
            raise ValueError(
                "No existe correspondencia uno a uno entre X_te y y_te: "
                f"X_te.shape[0]={self.X_te.shape[0]}, "
                f"len(y_te)={len(self.y_te)}."
            )

        idx_j = np.flatnonzero(
            self.y_te == self.j_sel
        )

        idx_r = np.flatnonzero(
            self.y_te == self.r_sel
        )

        if len(idx_j) == 0 or len(idx_r) == 0:
            raise ValueError(
                "No hay suficientes instancias en test para construir "
                f"el par {self.j_sel} vs {self.r_sel}. "
                f"Disponibles: Cj={len(idx_j)}, Cr={len(idx_r)}."
            )

        n_per_class = min(
            len(idx_j),
            len(idx_r)
        )

        rng = np.random.RandomState(
            self.RANDOM_STATE
        )

        selected_j = rng.choice(
            idx_j,
            size=n_per_class,
            replace=False
        )

        selected_r = rng.choice(
            idx_r,
            size=n_per_class,
            replace=False
        )

        selected_idx = np.concatenate([
            selected_j,
            selected_r
        ])

        # El orden no afecta las métricas agregadas, pero se mezcla
        # reproduciblemente para no dejar las clases agrupadas por filas.
        rng.shuffle(selected_idx)

        self.pair_indices = selected_idx.copy()
        self.N_pair = int(len(selected_idx))

        self.X_te_pair = self.X_te.iloc[
            selected_idx
        ].copy().reset_index(drop=True)

        self.y_te_pair = self.y_te[
            selected_idx
        ].copy()

        self.shap_vals_pair = self.shap_vals[
            selected_idx,
            :,
            :
        ].copy()

        # ----------------------------------------------------
        # Metadatos de las instancias seleccionadas
        # ----------------------------------------------------

        metadata = pd.DataFrame({
            "test_position": selected_idx,
            "y_true": self.y_te_pair
        })

        # Si existe el archivo con índices originales del split,
        # se incorpora para trazabilidad. No participa en la selección.
        original_index_path = (
            self.artifact_dir / "test_original_indices.csv"
        )

        if original_index_path.exists():
            try:
                original_index_df = pd.read_csv(
                    original_index_path
                )

                if (
                    "original_index" in original_index_df.columns
                    and len(original_index_df) == len(self.y_te)
                ):
                    metadata["original_index"] = (
                        original_index_df["original_index"]
                        .to_numpy()[selected_idx]
                    )
            except Exception as e:
                print(
                    "Advertencia: no se pudieron incorporar los "
                    f"índices originales del test: {e}"
                )

        # ----------------------------------------------------
        # Predicción: SOLO diagnóstico, NUNCA criterio de selección
        # ----------------------------------------------------

        y_pred_pair = None

        if getattr(self, "modelo_final", None) is not None:
            try:
                y_pred_pair = self.modelo_final.predict(
                    self.X_te_pair
                )
            except Exception as e:
                print(
                    "Advertencia: no se pudo obtener la predicción "
                    f"del modelo para la muestra pairwise: {e}"
                )

        if (
            y_pred_pair is None
            and getattr(self, "test_predictions", None) is not None
        ):
            pred_df = self.test_predictions

            candidate_columns = [
                "y_pred",
                "prediction",
                "predicted_class",
                "pred"
            ]

            pred_col = next(
                (
                    col
                    for col in candidate_columns
                    if col in pred_df.columns
                ),
                None
            )

            if (
                pred_col is not None
                and len(pred_df) == len(self.y_te)
            ):
                y_pred_pair = (
                    pred_df[pred_col]
                    .to_numpy()
                    .astype(int)[selected_idx]
                )

        if y_pred_pair is not None:
            metadata["y_pred"] = y_pred_pair
            metadata["correct_prediction"] = (
                y_pred_pair == self.y_te_pair
            )

        metadata_path = self._outpath(
            f"selected_pair_metadata_{self.pair_tag}.csv"
        )

        metadata.to_csv(
            metadata_path,
            index=False
        )

        # ----------------------------------------------------
        # Resumen de la configuración metodológica
        # ----------------------------------------------------

        n_correct = np.nan
        n_incorrect = np.nan

        if "correct_prediction" in metadata.columns:
            n_correct = int(
                metadata["correct_prediction"].sum()
            )
            n_incorrect = int(
                self.N_pair - n_correct
            )

        summary = pd.DataFrame([{
            "class_j": self.j_sel,
            "class_j_name": self.class_mapping[self.j_sel],
            "class_r": self.r_sel,
            "class_r_name": self.class_mapping[self.r_sel],
            "available_j_in_test": int(len(idx_j)),
            "available_r_in_test": int(len(idx_r)),
            "selected_j": int(n_per_class),
            "selected_r": int(n_per_class),
            "N_test_full": int(len(self.y_te)),
            "N_selected": int(self.N_pair),
            "random_state": int(self.RANDOM_STATE),
            "selection_basis": "true_label",
            "balanced_pair": True,
            "sampling_without_replacement": True,
            "prediction_filter": False,
            "correct_and_incorrect_included": True,
            "n_correct_selected": n_correct,
            "n_incorrect_selected": n_incorrect
        }])

        summary_path = self._outpath(
            f"selected_pair_summary_{self.pair_tag}.csv"
        )

        summary.to_csv(
            summary_path,
            index=False
        )

        # ----------------------------------------------------
        # Comprobaciones finales de la muestra
        # ----------------------------------------------------

        count_j_selected = int(
            np.sum(self.y_te_pair == self.j_sel)
        )

        count_r_selected = int(
            np.sum(self.y_te_pair == self.r_sel)
        )

        if count_j_selected != n_per_class:
            raise RuntimeError(
                "La selección balanceada falló para la clase j."
            )

        if count_r_selected != n_per_class:
            raise RuntimeError(
                "La selección balanceada falló para la clase r."
            )

        if self.shap_vals_pair.shape[0] != self.N_pair:
            raise RuntimeError(
                "La cantidad de valores SHAP del par no coincide con "
                "N_pair."
            )

        # ----------------------------------------------------
        # Reporte en consola
        # ----------------------------------------------------

        print("\n============================================================")
        print(" CONFIGURACIÓN DE LA POBLACIÓN SHAP-GLCD")
        print("============================================================")
        print(
            f"Par                     : "
            f"{self.class_mapping[self.j_sel]} "
            f"vs {self.class_mapping[self.r_sel]}"
        )
        print("Selección               : etiqueta verdadera")
        print("Balanceo del par        : sí")
        print("Muestreo                 : sin reemplazo")
        print("Filtro por predicción   : NO")
        print("Aciertos + errores      : incluidos")
        print(f"Semilla                 : {self.RANDOM_STATE}")
        print(
            f"Disponibles en test     : "
            f"{self.class_mapping[self.j_sel]}={len(idx_j)}, "
            f"{self.class_mapping[self.r_sel]}={len(idx_r)}"
        )
        print(
            f"Seleccionadas por clase : "
            f"{n_per_class} + {n_per_class}"
        )
        print(f"N global balanceado     : {self.N_pair}")

        if y_pred_pair is not None:
            print(f"Predicciones correctas  : {n_correct}")
            print(f"Predicciones incorrectas: {n_incorrect}")

        print("============================================================")
        print(f"Metadata selección      : {metadata_path}")
        print(f"Resumen selección       : {summary_path}")
        print("============================================================\n")

        return self.pair_indices

    # ========================================================
    # ΔS SOBRE LA MUESTRA BALANCEADA
    # ========================================================

    def calcular_delta(self):
        """
        Calcula:

            ΔS = SHAP(clase j) - SHAP(clase r)

        exclusivamente sobre la muestra balanceada del par seleccionada
        por ETIQUETA VERDADERA.

        No se filtra por predicción del modelo.
        """

        if not hasattr(self, "pair_indices"):
            self._seleccionar_muestra_balanceada_par()

        self.delta = (
            self.shap_vals_pair[:, :, self.j_sel]
            - self.shap_vals_pair[:, :, self.r_sel]
        )

        print("\nΔSHAP calculado sobre el par balanceado:")
        print(f"delta shape: {self.delta.shape}")
        print(
            f"Interpretación: valores positivos favorecen "
            f"{self.class_mapping[self.j_sel]}; "
            f"valores negativos favorecen "
            f"{self.class_mapping[self.r_sel]}."
        )

        np.save(
            self._outpath(
                f"delta_shap_{self.pair_tag}.npy"
            ),
            self.delta
        )

    # ========================================================
    # KDE: MISMA MUESTRA PAIRWISE + CONTEXTO ALL-CLASSES
    # ========================================================

    def plot_kde_top_features(self, dpi=600, show=True):
        """
        KDE para las dos primeras características del ranking.

        Las KDE pairwise, t_opt y t_bnd utilizan EXACTAMENTE la misma
        muestra balanceada usada para calcular ΔS.

        La KDE all-classes se conserva como figura contextual y utiliza
        todo el conjunto test.
        """

        if not hasattr(self, "rank"):
            self.ranking_global()

        if not hasattr(self, "pair_indices"):
            self._seleccionar_muestra_balanceada_par()

        print(
            "Clases presentes en self.y_te:",
            set(self.y_te)
        )

        print(
            "Claves en self.class_mapping:",
            set(self.class_mapping.keys())
        )

        X_te_df = self.X_te.copy()
        X_pair_df = self.X_te_pair.copy()
        rank = self.rank

        # ----------------------------------------------------
        # KDE pairwise: MISMA muestra balanceada de SHAP-GLCD
        # ----------------------------------------------------

        for f in rank["Feature"].iloc[:2]:
            fig, ax = plt.subplots(
                figsize=(16, 10)
            )

            vals_j = X_pair_df.loc[
                self.y_te_pair == self.j_sel,
                f
            ].dropna().values

            vals_r = X_pair_df.loc[
                self.y_te_pair == self.r_sel,
                f
            ].dropna().values

            if len(vals_j) == 0 or len(vals_r) == 0:
                print(
                    f"No se puede construir KDE pairwise para {f}: "
                    "alguna clase no tiene valores."
                )
                plt.close(fig)
                continue

            # El escalador fue ajustado solo con train; por ello un valor
            # de test puede quedar ligeramente fuera de [0, 1].
            x_min = min(
                0.0,
                float(np.min(vals_j)),
                float(np.min(vals_r))
            )

            x_max = max(
                1.0,
                float(np.max(vals_j)),
                float(np.max(vals_r))
            )

            plotted = False

            for c in [
                self.j_sel,
                self.r_sel
            ]:
                vals_c = X_pair_df.loc[
                    self.y_te_pair == c,
                    f
                ].dropna()

                if (
                    len(vals_c) < 2
                    or vals_c.nunique() < 2
                ):
                    print(
                        f"Advertencia: no se grafica KDE para {f}, "
                        f"clase {c}, porque tiene muy pocos "
                        "valores o varianza nula."
                    )
                    continue

                try:
                    sns.kdeplot(
                        vals_c,
                        fill=True,
                        alpha=0.4,
                        color=(
                            "#56B4E9"
                            if c == self.j_sel
                            else "#ECE133"
                        ),
                        label=self.class_mapping[c],
                        cut=0,
                        clip=(x_min, x_max),
                        ax=ax
                    )

                    plotted = True

                except Exception as e:
                    print(
                        f"No se pudo graficar KDE para {f}, "
                        f"clase {c}: {e}"
                    )

            t_opt, ba_opt, direction = (
                self._best_balanced_threshold_bidir(
                    vals_j,
                    vals_r
                )
            )

            t_bnd = self._boundary_pairwise(
                vals_j,
                vals_r,
                direction
            )

            if t_opt is not None:
                ax.axvline(
                    t_opt,
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.95,
                    color="tab:orange",
                    label=(
                        rf"$t_{{opt}}$={t_opt:.3f} "
                        rf"(BA={ba_opt:.2f}, rule: $x{direction}t$)"
                    )
                )

            if t_bnd is not None:
                ax.axvline(
                    t_bnd,
                    linestyle=":",
                    linewidth=2.5,
                    alpha=0.95,
                    color="tab:green",
                    label=rf"$t_{{bnd}}$={t_bnd:.3f}"
                )

            ax.set_xlim(
                x_min,
                x_max
            )

            ax.tick_params(
                axis="x",
                labelsize=self._fs(30)
            )

            ax.tick_params(
                axis="y",
                labelsize=self._fs(30)
            )

            ax.set_xlabel(
                f"{f} (scaled)",
                fontsize=self._fs(36)
            )

            ax.set_ylabel(
                "Density",
                fontsize=self._fs(36)
            )

            ax.set_title(
                f"Density of {f} – "
                f"{self.class_mapping[self.j_sel]} and "
                f"{self.class_mapping[self.r_sel]}",
                fontsize=self._fs(36)
            )

            if plotted:
                ax.legend(
                    fontsize=self._fs(26),
                    loc="upper right"
                )

                fig.tight_layout()

                save_path = self._outpath(
                    f"density_{safe_filename(f)}_{self.pair_tag}.png"
                )

                fig.savefig(
                    save_path,
                    dpi=dpi,
                    bbox_inches="tight"
                )

                print(f"KDE guardado: {save_path}")

                if show:
                    plt.show()

            else:
                print(
                    f"No se guardó KDE para {f}, "
                    "porque ninguna de las dos clases "
                    "tenía variabilidad suficiente."
                )

            plt.close(fig)

        # ----------------------------------------------------
        # KDE top-1 para todas las clases: TODO EL TEST
        # ----------------------------------------------------

        top_feat = rank.iloc[0]["Feature"]

        fig, ax = plt.subplots(
            figsize=(16, 10)
        )

        vals_all = (
            X_te_df[top_feat]
            .dropna()
            .to_numpy()
        )

        if len(vals_all) == 0:
            print(
                f"No se puede construir KDE all-classes para "
                f"{top_feat}: no hay valores."
            )
            plt.close(fig)
            return

        x_min_all = min(
            0.0,
            float(np.min(vals_all))
        )

        x_max_all = max(
            1.0,
            float(np.max(vals_all))
        )

        plotted_any = False

        for c in self.class_ids:
            vals = X_te_df.loc[
                self.y_te == c,
                top_feat
            ].dropna()

            if (
                len(vals) < 2
                or vals.nunique() < 2
            ):
                print(
                    f"Advertencia: no se grafica KDE all-classes para "
                    f"{top_feat}, clase {c}, por pocos valores "
                    "o varianza nula."
                )
                continue

            try:
                sns.kdeplot(
                    vals,
                    fill=True,
                    alpha=0.25,
                    color=self.class_color[c],
                    label=self.class_mapping[c],
                    cut=0,
                    clip=(x_min_all, x_max_all),
                    ax=ax
                )

                plotted_any = True

            except Exception as e:
                print(
                    f"No se pudo graficar KDE all-classes para "
                    f"{top_feat}, clase {c}: {e}"
                )

        ax.set_title(
            f"Density of {top_feat} across all classes",
            fontsize=self._fs(36)
        )

        ax.set_xlabel(
            f"{top_feat} (scaled)",
            fontsize=self._fs(36)
        )

        ax.set_ylabel(
            "Density",
            fontsize=self._fs(36)
        )

        # Corrección: NO usar un límite hardcodeado como [0, 0.03].
        ax.set_xlim(
            x_min_all,
            x_max_all
        )

        ax.tick_params(
            axis="x",
            labelsize=self._fs(30)
        )

        ax.tick_params(
            axis="y",
            labelsize=self._fs(30)
        )

        if plotted_any:
            ax.legend(
                ncol=2,
                fontsize=self._fs(24)
            )

            fig.tight_layout()

            save_path = self._outpath(
                f"kde_all_classes_{safe_filename(top_feat)}_"
                f"{self.pair_tag}.png"
            )

            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight"
            )

            print(
                f"KDE all-classes guardado: {save_path}"
            )

            if show:
                plt.show()

        else:
            print(
                f"No se guardó KDE all-classes para {top_feat}, "
                "porque ninguna clase tenía variabilidad suficiente."
            )

        plt.close(fig)

    # ========================================================
    # ESTADÍSTICOS TOP-1 SOBRE LA MISMA MUESTRA BALANCEADA
    # ========================================================

    def estadisticos_top1(self):
        """
        Estadísticos complementarios de la característica top-1.

        Todos los cálculos pairwise usan EXACTAMENTE la misma muestra
        balanceada seleccionada por etiqueta verdadera que se utilizó
        para SHAP-GLCD.

        Se reportan:
        - Cohen d para características cuantitativas;
        - Cohen h para características binarias 0/1;
        - prueba de Kolmogorov-Smirnov para características cuantitativas;
        - prueba exacta de Fisher para características binarias 0/1;
        - t_opt bidireccional y su Balanced Accuracy;
        - t_bnd y su Balanced Accuracy;
        - tabla precision/recall.
        """

        if not hasattr(self, "rank"):
            self.ranking_global()

        if not hasattr(self, "pair_indices"):
            self._seleccionar_muestra_balanceada_par()

        top_feat = self.rank.iloc[0]["Feature"]
        top_sign = self.rank.iloc[0]["SignedMeanΔSHAP"]

        favours = (
            self.class_mapping[self.j_sel]
            if top_sign > 0
            else self.class_mapping[self.r_sel]
        )

        print(
            f"\n=== Top-1 feature: {top_feat} "
            f"(favorece {favours}) ==="
        )

        X_pair_df = self.X_te_pair.copy()

        vals_j = X_pair_df.loc[
            self.y_te_pair == self.j_sel,
            top_feat
        ].values

        vals_r = X_pair_df.loc[
            self.y_te_pair == self.r_sel,
            top_feat
        ].values

        def es_binaria_01(a, b, atol=1e-12):
            valores = np.concatenate([
                np.asarray(a, dtype=float),
                np.asarray(b, dtype=float)
            ])
            valores = valores[np.isfinite(valores)]

            if valores.size == 0:
                return False

            return bool(np.all(
                np.isclose(valores, 0.0, atol=atol, rtol=0.0)
                | np.isclose(valores, 1.0, atol=atol, rtol=0.0)
            ))

        def cohens_d(a, b):
            n1, n2 = len(a), len(b)

            if n1 < 2 or n2 < 2:
                return np.nan

            s1 = a.std(ddof=1)
            s2 = b.std(ddof=1)

            pooled_var = (
                (
                    (n1 - 1) * s1 ** 2
                    + (n2 - 1) * s2 ** 2
                )
                / max((n1 + n2 - 2), 1)
            )

            pooled_std = np.sqrt(
                pooled_var
            )

            if pooled_std == 0:
                return np.nan

            return (
                a.mean() - b.mean()
            ) / pooled_std

        def cohens_h_binaria(a, b):
            if len(a) == 0 or len(b) == 0:
                return np.nan, np.nan, np.nan

            p_j = float(np.mean(np.isclose(a, 1.0)))
            p_r = float(np.mean(np.isclose(b, 1.0)))

            h = (
                2.0 * np.arcsin(np.sqrt(p_j))
                - 2.0 * np.arcsin(np.sqrt(p_r))
            )
            return float(h), p_j, p_r

        is_binary = es_binaria_01(vals_j, vals_r)

        if is_binary:
            effect_metric = "Cohen h"
            effect_symbol = "h"
            effect_signed, proportion_j, proportion_r = (
                cohens_h_binaria(vals_j, vals_r)
            )
            feature_type = "binary"
        else:
            effect_metric = "Cohen d"
            effect_symbol = "d"
            effect_signed = cohens_d(vals_j, vals_r)
            proportion_j = np.nan
            proportion_r = np.nan
            feature_type = "quantitative"

        ks_statistic = np.nan
        p_ks = np.nan
        fisher_odds_ratio = np.nan
        p_fisher = np.nan
        count_j_1 = np.nan
        count_j_0 = np.nan
        count_r_1 = np.nan
        count_r_0 = np.nan

        if is_binary:
            count_j_1 = int(np.sum(np.isclose(vals_j, 1.0)))
            count_j_0 = int(len(vals_j) - count_j_1)
            count_r_1 = int(np.sum(np.isclose(vals_r, 1.0)))
            count_r_0 = int(len(vals_r) - count_r_1)

            try:
                fisher_result = fisher_exact([
                    [count_j_1, count_j_0],
                    [count_r_1, count_r_0]
                ], alternative="two-sided")
                fisher_odds_ratio = float(fisher_result.statistic)
                p_fisher = float(fisher_result.pvalue)
            except Exception:
                fisher_odds_ratio = np.nan
                p_fisher = np.nan
        else:
            try:
                ks_result = ks_2samp(vals_j, vals_r)
                ks_statistic = float(ks_result.statistic)
                p_ks = float(ks_result.pvalue)
            except Exception:
                ks_statistic = np.nan
                p_ks = np.nan

        thr_BA, ba, direction = (
            self._best_balanced_threshold_bidir(
                vals_j,
                vals_r
            )
        )

        print("\n=== Top-1 statistics ===")

        print(
            f"Feature type            : {feature_type}"
        )

        if np.isnan(effect_signed):
            print(
                f"{effect_metric:<24}: nan "
                "(no computable)"
            )
            print(
                f"Cohen |{effect_symbol}|"
                f"              : nan"
            )
        else:
            print(
                f"{effect_metric:<24}: {effect_signed:.3f}"
            )
            print(
                f"Cohen |{effect_symbol}|"
                f"              : {abs(effect_signed):.3f}"
            )

        if is_binary:
            print(
                f"Proportion {self.class_mapping[self.j_sel]} (1)"
                f" : {proportion_j:.3f}"
            )
            print(
                f"Proportion {self.class_mapping[self.r_sel]} (1)"
                f" : {proportion_r:.3f}"
            )

        if is_binary:
            if np.isnan(fisher_odds_ratio):
                print("Fisher odds ratio       : nan")
            else:
                print(
                    f"Fisher odds ratio       : "
                    f"{fisher_odds_ratio:.6f}"
                )

            if np.isnan(p_fisher):
                print("p-value Fisher          : nan")
            else:
                print(
                    f"p-value Fisher          : {p_fisher:.3e}"
                )
        elif np.isnan(ks_statistic):
            print("KS statistic D          : nan")
        else:
            print(
                f"KS statistic D          : {ks_statistic:.3f}"
            )

        if not is_binary:
            if np.isnan(p_ks):
                print("p-value KS              : nan")
            else:
                print(
                    f"p-value KS              : {p_ks:.3e}"
                )

        print(
            f"Best threshold (BA)     : {thr_BA}"
        )

        if ba is None or direction is None:
            print(
                "Balanced-Acc @thr       : None"
            )
            return

        print(
            f"Balanced-Acc @thr       : {ba:.3f}"
        )

        # ----------------------------------------------------
        # Precision/Recall: misma muestra balanceada
        # ----------------------------------------------------

        x_pair = X_pair_df[
            top_feat
        ].values

        y_true_pair = (
            self.y_te_pair == self.j_sel
        ).astype(int)

        thr_vals = np.sort(
            np.unique(x_pair)
        )

        records = []

        for t in thr_vals:
            if direction == "<=":
                pred = (
                    x_pair <= t
                ).astype(int)
            else:
                pred = (
                    x_pair >= t
                ).astype(int)

            prec = precision_score(
                y_true_pair,
                pred,
                zero_division=0
            )

            rec = recall_score(
                y_true_pair,
                pred,
                zero_division=0
            )

            records.append({
                "thr": t,
                "precision": prec,
                "recall": rec
            })

        pr_df = pd.DataFrame(
            records
        )

        pr_path = self._outpath(
            f"pr_thresholds_{safe_filename(top_feat)}_"
            f"{self.pair_tag}.csv"
        )

        pr_df.to_csv(
            pr_path,
            index=False
        )

        print(
            f"\nTabla P–R completa guardada en: {pr_path}"
        )

        PREC_REQ = 0.90

        meet = pr_df.loc[
            pr_df["precision"] >= PREC_REQ
        ]

        if not meet.empty:
            row = meet.iloc[0]

            print(
                f"\nFirst thr with precision ≥ {PREC_REQ:.0%}:"
            )

            print(
                f"thr={row['thr']:.6f} | "
                f"precision={row['precision']:.3f} | "
                f"recall={row['recall']:.3f}"
            )

        else:
            print(
                f"\nNo threshold reaches precision ≥ "
                f"{PREC_REQ:.0%}\n"
            )

        # ----------------------------------------------------
        # Boundary threshold pairwise
        # ----------------------------------------------------

        boundary = self._boundary_pairwise(
            vals_j,
            vals_r,
            direction
        )

        if boundary is None:
            print(
                "No se pudo calcular boundary threshold."
            )
            return

        if direction == "<=":
            pred_pair = (
                x_pair <= boundary
            ).astype(int)
        else:
            pred_pair = (
                x_pair >= boundary
            ).astype(int)

        bal_rule = balanced_accuracy_score(
            y_true_pair,
            pred_pair
        )

        print("\n=== Boundary threshold (pairwise) ===")

        # Reporte coherente con la dirección seleccionada.
        if direction == ">=":
            print(
                f"max {self.class_mapping[self.r_sel]}: "
                f"{vals_r.max():.6f}"
            )
            print(
                f"min {self.class_mapping[self.j_sel]}: "
                f"{vals_j.min():.6f}"
            )
        else:
            print(
                f"max {self.class_mapping[self.j_sel]}: "
                f"{vals_j.max():.6f}"
            )
            print(
                f"min {self.class_mapping[self.r_sel]}: "
                f"{vals_r.min():.6f}"
            )

        print(
            f"boundary thr      : {boundary:.6f}"
        )

        print(
            f"Balanced-Acc @bnd : {bal_rule:.3f}\n"
        )

        # ----------------------------------------------------
        # Guardar resumen top-1
        # ----------------------------------------------------

        top1_stats = pd.DataFrame([{
            "feature": top_feat,
            "class_j": self.j_sel,
            "class_j_name": self.class_mapping[self.j_sel],
            "class_r": self.r_sel,
            "class_r_name": self.class_mapping[self.r_sel],
            "N_pair": int(self.N_pair),
            "feature_type": feature_type,
            "effect_size_metric": effect_metric,
            "effect_size_symbol": effect_symbol,
            "effect_size_signed": (
                float(effect_signed)
                if not np.isnan(effect_signed)
                else np.nan
            ),
            "effect_size_absolute": (
                float(abs(effect_signed))
                if not np.isnan(effect_signed)
                else np.nan
            ),
            "proportion_class_j": proportion_j,
            "proportion_class_r": proportion_r,
            "count_class_j_value_1": count_j_1,
            "count_class_j_value_0": count_j_0,
            "count_class_r_value_1": count_r_1,
            "count_class_r_value_0": count_r_0,
            "ks_statistic_D": ks_statistic,
            "ks_pvalue": p_ks,
            "fisher_odds_ratio": fisher_odds_ratio,
            "fisher_pvalue": p_fisher,
            "t_opt": (
                float(thr_BA)
                if thr_BA is not None
                else np.nan
            ),
            "t_opt_direction": direction,
            "balanced_accuracy_t_opt": float(ba),
            "t_bnd": float(boundary),
            "balanced_accuracy_t_bnd": float(bal_rule)
        }])

        top1_path = self._outpath(
            f"top1_statistics_{self.pair_tag}.csv"
        )

        top1_stats.to_csv(
            top1_path,
            index=False
        )

        print(
            f"Estadísticos top-1 guardados en: {top1_path}"
        )
