# glcd_explainer_from_artifacts.py
# Versión actualizada:
# - Métricas F1-macro y MCC.
# - Tipografías ampliadas para publicación.
# - Leyenda externa, centrada debajo, en Conditional Mean ΔS y Dominance Plot.

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from scipy.stats import ks_2samp


# ============================================================
# HELPER: LIMPIAR NOMBRES PARA ARCHIVOS
# ============================================================

def safe_filename(text):
    text = str(text)
    keep = []

    for ch in text:
        if ch.isalnum() or ch in ["_", "-"]:
            keep.append(ch)
        else:
            keep.append("_")

    return "".join(keep).strip("_")


# ============================================================
# CLASE GLCD DESDE ARTEFACTOS
# ============================================================

class GLCDExplainerFromArtifacts:
    """
    Explicador SHAP-GLCD construido desde artefactos guardados previamente
    por el main de SHAP-LCD.

    Esta clase NO:
    - vuelve a leer el dataset original,
    - vuelve a procesar variables,
    - vuelve a hacer split train/test,
    - vuelve a entrenar el modelo,
    - vuelve a calcular SHAP.

    Esta clase SÍ:
    - carga X_test_final,
    - carga y_test_final,
    - carga feature_names,
    - carga class_names,
    - carga shap_values_3d,
    - carga el modelo entrenado si es posible,
    - calcula ΔSHAP entre dos clases,
    - genera rankings y gráficas SHAP-GLCD,
    - calcula métricas finales sobre el conjunto test guardado.
    """

    def __init__(
        self,
        artifact_dir,
        j_sel=0,
        r_sel=2,
        top=10,
        random_state=42,
        output_dir=None,
        load_model=True,
        figure_font_scale=1.25
    ):
        self.artifact_dir = Path(artifact_dir)
        self.j_sel = int(j_sel)
        self.r_sel = int(r_sel)
        self.TOP = int(top)
        self.RANDOM_STATE = random_state
        self.load_model = load_model

        # Solo modifica la presentación visual de las figuras.
        # No afecta los resultados, rankings ni valores SHAP.
        self.figure_font_scale = float(figure_font_scale)

        if not self.artifact_dir.exists():
            raise FileNotFoundError(
                f"No existe la carpeta de artefactos: {self.artifact_dir}"
            )

        self._load_artifacts()
        self._validate_artifacts()
        self._setup_plot_style()

        j_name = self.class_mapping[self.j_sel]
        r_name = self.class_mapping[self.r_sel]

        self.pair_tag = (
            f"{self.j_sel}_vs_{self.r_sel}_"
            f"{safe_filename(j_name)}_vs_{safe_filename(r_name)}"
        )

        if output_dir is None:
            output_dir = (
                Path("glcd_results")
                / f"pair_{self.j_sel}_vs_{self.r_sel}"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n==========================================")
        print(" GLCDExplainerFromArtifacts inicializado")
        print("==========================================")
        print(f"Carpeta de artefactos : {self.artifact_dir}")
        print(f"Carpeta de resultados : {self.output_dir}")
        print(f"Par seleccionado      : {self.j_sel} vs {self.r_sel}")
        print(f"Clases                : {j_name} vs {r_name}")
        print(f"X_test shape          : {self.X_te.shape}")
        print(f"SHAP shape            : {self.shap_vals.shape}")
        print(f"N características     : {len(self.features)}")
        print(f"N clases              : {self.K}")
        print(f"Escala de fuente      : {self.figure_font_scale}")
        print("==========================================\n")

    # ========================================================
    # CARGA Y VALIDACIÓN
    # ========================================================

    def _require_file(self, filename):
        path = self.artifact_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo requerido: {path}"
            )

        return path

    def _optional_file(self, filename):
        path = self.artifact_dir / filename

        if path.exists():
            return path

        return None

    def _load_artifacts(self):
        x_test_path = self._require_file("X_test_final.csv")
        y_test_path = self._require_file("y_test_final.csv")
        feature_path = self._require_file("feature_names.csv")
        class_path = self._require_file("class_names.json")
        shap_path = self._require_file("shap_values_3d.npy")

        # ----------------------------------------------------
        # Feature names
        # ----------------------------------------------------
        feature_df = pd.read_csv(feature_path)

        if "feature" not in feature_df.columns:
            raise ValueError(
                "feature_names.csv debe contener una columna llamada 'feature'."
            )

        self.features = feature_df["feature"].astype(str).tolist()

        # ----------------------------------------------------
        # X_test final
        # ----------------------------------------------------
        self.X_te = pd.read_csv(x_test_path)

        if list(self.X_te.columns) != self.features:
            print(
                "Advertencia: las columnas de X_test_final.csv no coinciden "
                "exactamente con feature_names.csv. Se reasignarán los nombres."
            )

            if self.X_te.shape[1] != len(self.features):
                raise ValueError(
                    "El número de columnas de X_test_final.csv no coincide "
                    "con el número de nombres de características."
                )

            self.X_te.columns = self.features

        # ----------------------------------------------------
        # y_test final
        # ----------------------------------------------------
        y_test_df = pd.read_csv(y_test_path)

        if "target" not in y_test_df.columns:
            raise ValueError(
                "y_test_final.csv debe contener una columna llamada 'target'."
            )

        self.y_te = y_test_df["target"].values.astype(int)

        # ----------------------------------------------------
        # Class names
        # ----------------------------------------------------
        with open(class_path, "r", encoding="utf-8") as f:
            class_mapping_raw = json.load(f)

        self.class_mapping = {
            int(k): str(v)
            for k, v in class_mapping_raw.items()
        }

        self.class_ids = sorted(self.class_mapping.keys())
        self.class_names = self.class_ids

        # ----------------------------------------------------
        # SHAP values
        # ----------------------------------------------------
        self.shap_vals = np.load(shap_path)

        # ----------------------------------------------------
        # Probabilidades guardadas
        # ----------------------------------------------------
        prob_path = self._optional_file("test_probabilities.csv")
        self.test_probabilities = None

        if prob_path is not None:
            self.test_probabilities = pd.read_csv(prob_path)

        # ----------------------------------------------------
        # Predicciones guardadas
        # ----------------------------------------------------
        pred_path = self._optional_file("test_predictions.csv")
        self.test_predictions = None

        if pred_path is not None:
            self.test_predictions = pd.read_csv(pred_path)

        # ----------------------------------------------------
        # Modelo entrenado
        # ----------------------------------------------------
        self.modelo_final = None
        model_path = self._optional_file("random_forest_optimizado.joblib")

        if self.load_model and model_path is not None:
            try:
                self.modelo_final = joblib.load(model_path)
                print(f"Modelo cargado correctamente desde: {model_path}")

            except Exception as e:
                print("\nAdvertencia: no se pudo cargar el modelo con joblib.")
                print(
                    "El análisis GLCD puede continuar usando "
                    "los valores SHAP guardados."
                )
                print("Motivo del error:")
                print(e)

                self.modelo_final = None

        self.N_te = self.X_te.shape[0]
        self.M = self.X_te.shape[1]
        self.K = self.shap_vals.shape[2]

    def _validate_artifacts(self):
        if self.shap_vals.ndim != 3:
            raise ValueError(
                f"shap_values_3d.npy debe tener 3 dimensiones. "
                f"Shape recibido: {self.shap_vals.shape}"
            )

        n_shap, m_shap, k_shap = self.shap_vals.shape

        if n_shap != self.X_te.shape[0]:
            raise ValueError(
                "Inconsistencia entre X_test_final y shap_values_3d:\n"
                f"X_test_final tiene {self.X_te.shape[0]} filas, "
                f"pero shap_values_3d tiene {n_shap} instancias."
            )

        if m_shap != self.X_te.shape[1]:
            raise ValueError(
                "Inconsistencia entre X_test_final y shap_values_3d:\n"
                f"X_test_final tiene {self.X_te.shape[1]} columnas, "
                f"pero shap_values_3d tiene {m_shap} características."
            )

        if len(self.features) != self.X_te.shape[1]:
            raise ValueError(
                "El número de feature_names no coincide "
                "con las columnas de X_test."
            )

        if k_shap != len(self.class_mapping):
            raise ValueError(
                "El número de clases en shap_values_3d no coincide "
                "con class_names.json:\n"
                f"SHAP tiene {k_shap} clases, "
                f"class_names tiene {len(self.class_mapping)}."
            )

        if self.j_sel not in self.class_mapping:
            raise ValueError(
                f"j_sel={self.j_sel} no existe en class_names.json."
            )

        if self.r_sel not in self.class_mapping:
            raise ValueError(
                f"r_sel={self.r_sel} no existe en class_names.json."
            )

    # ========================================================
    # CONFIGURACIÓN VISUAL
    # ========================================================

    def _fs(self, size):
        """
        Escala un tamaño de fuente para figuras de publicación.
        """
        return int(round(float(size) * self.figure_font_scale))

    def _setup_plot_style(self):
        """
        Configuración visual para figuras insertadas en artículos.

        figure_font_scale aumenta o reduce la tipografía sin modificar
        cálculos, rankings, métricas ni valores SHAP.
        """

        sns.set_context("paper", font_scale=1.6)
        sns.set_style("whitegrid")

        plt.rcParams.update({
            "axes.titlesize": self._fs(32),
            "axes.labelsize": self._fs(30),
            "xtick.labelsize": self._fs(26),
            "ytick.labelsize": self._fs(26),
            "legend.fontsize": self._fs(22)
        })

        self.palette = sns.color_palette(
            "colorblind",
            max(3, self.K, 10)
        )

        self.class_color = {
            c: self.palette[i % len(self.palette)]
            for i, c in enumerate(self.class_ids)
        }

    def _legend_below(
        self,
        ax,
        fig,
        ncol=2,
        fontsize=24,
        anchor_y=-0.20,
        bottom_margin=0.27
    ):
        """
        Coloca la leyenda fuera del área de representación, centrada debajo.

        Esta posición evita que la leyenda tape barras, valores numéricos
        o curvas, independientemente de las características seleccionadas.
        """

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, anchor_y),
            ncol=ncol,
            fontsize=self._fs(fontsize),
            frameon=True,
            framealpha=0.90,
            borderaxespad=0.0
        )

        fig.subplots_adjust(
            bottom=bottom_margin
        )

    def _outpath(self, filename):
        return self.output_dir / filename

    def _format_feature_label(self, label, max_chars=24):
        """
        Formatea nombres largos de características solo para visualización.

        Importante:
        - No cambia el nombre real de la variable.
        - Solo divide el texto en varias líneas dentro de la figura.
        - Conserva los nombres originales en CSV y cálculos.
        """

        label = str(label)

        if len(label) <= max_chars:
            return label

        parts = label.split("_")
        lines = []
        current = ""

        for part in parts:
            candidate = (
                part
                if current == ""
                else current + "_" + part
            )

            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)

                current = part

        if current:
            lines.append(current)

        return "\n".join(lines)

    # ========================================================
    # CÁLCULO SHAP-GLCD
    # ========================================================

    def calcular_delta(self):
        """
        Calcula ΔS = SHAP(clase j) - SHAP(clase r)
        para todas las instancias y características.
        """

        self.delta = (
            self.shap_vals[:, :, self.j_sel]
            - self.shap_vals[:, :, self.r_sel]
        )

        print("\nΔSHAP calculado:")
        print(f"delta shape: {self.delta.shape}")
        print(
            f"Interpretación: valores positivos favorecen "
            f"{self.class_mapping[self.j_sel]}; "
            f"valores negativos favorecen "
            f"{self.class_mapping[self.r_sel]}."
        )

        np.save(
            self._outpath(f"delta_shap_{self.pair_tag}.npy"),
            self.delta
        )

    def ranking_global(self):
        if not hasattr(self, "delta"):
            self.calcular_delta()

        mean_abs = np.mean(
            np.abs(self.delta),
            axis=0
        )

        median_abs = np.median(
            np.abs(self.delta),
            axis=0
        )

        signed_mean = np.mean(
            self.delta,
            axis=0
        )

        self.rank = pd.DataFrame({
            "Feature": self.features,
            "Mean_|ΔSHAP|": mean_abs,
            "Median_|ΔSHAP|": median_abs,
            "SignedMeanΔSHAP": signed_mean
        }).sort_values(
            "Mean_|ΔSHAP|",
            ascending=False
        ).reset_index(drop=True)

        fname = self._outpath(
            f"global_delta_shap_ranking_{self.pair_tag}.csv"
        )

        self.rank.to_csv(
            fname,
            index=False
        )

        print(f"\nRanking global guardado en: {fname}")
        print("\nTop características:")
        print(
            self.rank.head(self.TOP).to_string(index=False)
        )

        return self.rank

    # ========================================================
    # GRÁFICAS PRINCIPALES
    # ========================================================

    def plot_barras(self, dpi=600, show=True):
        """
        Gráficas de:
        - Signed Mean ΔS
        - Mean |ΔS|
        - Median |ΔS|

        Incluye formato visual para nombres largos.
        """

        if not hasattr(self, "rank"):
            self.ranking_global()

        def barh_plot(
            df,
            col,
            title,
            fname,
            show_values=False,
            title_fontsize=38,
            label_fontsize=38,
            tick_fontsize=36,
            value_fontsize=28
        ):
            top_df = (
                df.head(self.TOP)
                .iloc[::-1]
                .copy()
            )

            vals = top_df[col].values.astype(float)

            labs = [
                self._format_feature_label(f, max_chars=24)
                for f in top_df["Feature"].astype(str).values
            ]

            fig, ax = plt.subplots(
                figsize=(20, 12)
            )

            bars = ax.barh(
                labs,
                vals,
                color=self.palette[:len(labs)]
            )

            if show_values:
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))

                span = max(
                    abs(vmin),
                    abs(vmax),
                    1e-12
                )

                dx = 0.04 * span

                ax.set_xlim(
                    vmin - 6 * dx,
                    vmax + 6 * dx
                )

                bbox_kw = dict(
                    facecolor="white",
                    alpha=0.70,
                    edgecolor="none",
                    pad=1.0
                )

                for b in bars:
                    w = b.get_width()
                    y = (
                        b.get_y()
                        + b.get_height() / 2.0
                    )

                    if w >= 0:
                        ax.text(
                            w + dx,
                            y,
                            f"{w:.3f}",
                            va="center",
                            ha="left",
                            fontsize=self._fs(value_fontsize),
                            bbox=bbox_kw
                        )

                    else:
                        ax.text(
                            w - dx,
                            y,
                            f"{w:.3f}",
                            va="center",
                            ha="right",
                            fontsize=self._fs(value_fontsize),
                            bbox=bbox_kw
                        )

            ax.set_title(
                f"{title} – top {self.TOP}\n"
                f"{self.class_mapping[self.j_sel]} and "
                f"{self.class_mapping[self.r_sel]}",
                fontsize=self._fs(title_fontsize)
            )

            ax.set_xlabel(
                title,
                fontsize=self._fs(label_fontsize)
            )

            ax.set_ylabel(
                "Feature",
                fontsize=self._fs(label_fontsize)
            )

            ax.tick_params(
                axis="x",
                labelsize=self._fs(tick_fontsize)
            )

            ax.tick_params(
                axis="y",
                labelsize=self._fs(32)
            )

            fig.tight_layout()

            save_path = self._outpath(fname)

            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight"
            )

            if show:
                plt.show()

            plt.close(fig)

            print(f"Figura guardada: {save_path}")

        barh_plot(
            self.rank,
            "SignedMeanΔSHAP",
            "Signed Mean $\\Delta S$",
            f"signed_mean_delta_{self.pair_tag}.png",
            show_values=True,
            title_fontsize=38,
            label_fontsize=38,
            tick_fontsize=36,
            value_fontsize=28
        )

        barh_plot(
            self.rank,
            "Mean_|ΔSHAP|",
            "Mean $|\\Delta S|$",
            f"mean_abs_delta_{self.pair_tag}.png",
            show_values=False,
            title_fontsize=49,
            label_fontsize=48,
            tick_fontsize=43,
            value_fontsize=28
        )

        rank_med = self.rank.sort_values(
            "Median_|ΔSHAP|",
            ascending=False
        ).reset_index(drop=True)

        barh_plot(
            rank_med,
            "Median_|ΔSHAP|",
            "Median $|\\Delta S|$",
            f"median_abs_delta_{self.pair_tag}.png",
            show_values=False,
            title_fontsize=49,
            label_fontsize=48,
            tick_fontsize=43,
            value_fontsize=28
        )

        print("✓ Barras PNG generadas.")

    def plot_barras_alineadas(self, dpi=600, show=True):
        """
        Gráfico de medias condicionales:
        - media positiva de ΔS
        - media negativa de ΔS

        La leyenda se coloca fuera del área del gráfico,
        centrada debajo, para evitar superposición con los datos.
        """

        if not hasattr(self, "delta"):
            self.calcular_delta()

        delta = self.delta
        M = delta.shape[1]

        mean_pos = np.zeros(M)
        mean_neg = np.zeros(M)

        n_pos = np.zeros(
            M,
            dtype=int
        )

        n_neg = np.zeros(
            M,
            dtype=int
        )

        for j in range(M):
            v = delta[:, j]

            pos = v[v > 0]
            neg = v[v < 0]

            n_pos[j] = pos.size
            n_neg[j] = neg.size

            mean_pos[j] = (
                pos.mean()
                if pos.size
                else 0.0
            )

            mean_neg[j] = (
                neg.mean()
                if neg.size
                else 0.0
            )

        mean_abs = np.mean(
            np.abs(delta),
            axis=0
        )

        rank = pd.DataFrame({
            "Feature": self.features,
            "Mean_|ΔSHAP|": mean_abs,
            "MeanPosΔ": mean_pos,
            "MeanNegΔ": mean_neg,
            "n_pos": n_pos,
            "n_neg": n_neg
        })

        csv_path = self._outpath(
            f"conditional_mean_delta_{self.pair_tag}.csv"
        )

        rank.to_csv(
            csv_path,
            index=False
        )

        top = (
            rank
            .sort_values(
                "Mean_|ΔSHAP|",
                ascending=False
            )
            .head(self.TOP)
            .iloc[::-1]
            .reset_index(drop=True)
        )

        labels = [
            self._format_feature_label(f, max_chars=24)
            for f in top["Feature"].astype(str)
        ]

        pos_vals = top["MeanPosΔ"].values
        neg_vals = top["MeanNegΔ"].values
        y_pos = np.arange(len(labels))

        fig, ax = plt.subplots(
            figsize=(20, 14)
        )

        ax.barh(
            y_pos,
            pos_vals,
            color="#56B4E9",
            label=(
                f"$\\overline{{\\Delta S}}^{{+}}$ "
                f"(favours {self.class_mapping[self.j_sel]})"
            )
        )

        ax.barh(
            y_pos,
            neg_vals,
            color="#ECE133",
            label=(
                f"$\\overline{{\\Delta S}}^{{-}}$ "
                f"(favours {self.class_mapping[self.r_sel]})"
            )
        )

        ax.axvline(
            0,
            color="k",
            linewidth=1
        )

        ax.set_yticks(y_pos)

        ax.set_yticklabels(
            labels,
            fontsize=self._fs(32)
        )

        ax.tick_params(
            axis="x",
            labelsize=self._fs(34)
        )

        ax.set_xlabel(
            "Conditional Mean ΔS",
            fontsize=self._fs(34)
        )

        ax.set_ylabel(
            "Feature",
            fontsize=self._fs(34)
        )

        ax.set_title(
            f"Conditional Mean ΔS for top {len(labels)} Features\n"
            f"{self.class_mapping[self.j_sel]} and "
            f"{self.class_mapping[self.r_sel]}",
            fontsize=self._fs(36)
        )

        max_pos = (
            np.max(pos_vals)
            if len(pos_vals)
            else 0.0
        )

        min_neg = (
            np.min(neg_vals)
            if len(neg_vals)
            else 0.0
        )

        span = max(
            max_pos,
            abs(min_neg),
            1e-12
        )

        dx = 0.04 * span

        ax.set_xlim(
            min_neg - 6 * dx,
            max_pos + 6 * dx
        )

        bbox_kw = dict(
            facecolor="white",
            alpha=0.70,
            edgecolor="none",
            pad=1.0
        )

        fs_val = self._fs(30)

        for yi, pv, nv in zip(
            y_pos,
            pos_vals,
            neg_vals
        ):
            if pv != 0:
                ax.text(
                    pv + dx,
                    yi,
                    f"{pv:.2f}",
                    va="center",
                    ha="left",
                    fontsize=fs_val,
                    bbox=bbox_kw
                )

            if nv != 0:
                ax.text(
                    nv - dx,
                    yi,
                    f"{nv:.3f}",
                    va="center",
                    ha="right",
                    fontsize=fs_val,
                    bbox=bbox_kw
                )

        fig.tight_layout()

        self._legend_below(
            ax=ax,
            fig=fig,
            ncol=2,
            fontsize=26,
            anchor_y=-0.20,
            bottom_margin=0.28
        )

        save_path = self._outpath(
            f"cond_mean_delta_aligned_{self.pair_tag}.png"
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close(fig)

        print(f"Figura guardada: {save_path}")
        print(f"CSV guardado: {csv_path}")

    def plot_figura_B_100stack_weighted(
        self,
        dpi=600,
        fname=None,
        show=True
    ):
        """
        Figura B / Dominance Plot:

            w+ = p+ * |mean(ΔS | ΔS > 0)|
            w- = p- * |mean(ΔS | ΔS < 0)|
            W+ = w+ / (w+ + w-)
            W- = 1 - W+

        La leyenda se coloca fuera del área del gráfico,
        centrada debajo, para evitar superposición con los datos.
        """

        if not hasattr(self, "delta"):
            self.calcular_delta()

        delta = self.delta
        N, M = delta.shape

        mean_pos = np.zeros(M)
        mean_neg = np.zeros(M)

        n_pos = np.zeros(
            M,
            dtype=int
        )

        n_neg = np.zeros(
            M,
            dtype=int
        )

        for j in range(M):
            v = delta[:, j]

            pos = v[v > 0]
            neg = v[v < 0]

            n_pos[j] = pos.size
            n_neg[j] = neg.size

            mean_pos[j] = (
                pos.mean()
                if pos.size
                else 0.0
            )

            mean_neg[j] = (
                neg.mean()
                if neg.size
                else 0.0
            )

        p_pos = n_pos / N
        p_neg = n_neg / N

        w_pos = p_pos * np.abs(mean_pos)
        w_neg = p_neg * np.abs(mean_neg)
        w_tot = w_pos + w_neg

        eps = 1e-12

        W_pos = np.where(
            w_tot > 0,
            w_pos / (w_tot + eps),
            0.5
        )

        W_neg = 1.0 - W_pos

        df = pd.DataFrame({
            "Feature": self.features,
            "W_plus": W_pos,
            "W_minus": W_neg,
            "Mass_total": w_tot,
            "p_plus": p_pos,
            "p_minus": p_neg,
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "n_pos": n_pos,
            "n_neg": n_neg
        })

        csv_path = self._outpath(
            f"dominance_plot_weights_{self.pair_tag}.csv"
        )

        df.to_csv(
            csv_path,
            index=False
        )

        top = (
            df
            .sort_values(
                "Mass_total",
                ascending=False
            )
            .head(self.TOP)
            .iloc[::-1]
            .reset_index(drop=True)
        )

        labels = [
            self._format_feature_label(f, max_chars=24)
            for f in top["Feature"].astype(str).tolist()
        ]

        w_minus = top["W_minus"].values
        w_plus = top["W_plus"].values
        y = np.arange(len(labels))

        fig, ax = plt.subplots(
            figsize=(20, 14)
        )

        ax.barh(
            y,
            w_minus,
            color="#ECE133",
            label=(
                rf"$W^-_m$ "
                rf"(favours {self.class_mapping[self.r_sel]})"
            )
        )

        ax.barh(
            y,
            w_plus,
            left=w_minus,
            color="#56B4E9",
            label=(
                rf"$W^+_m$ "
                rf"(favours {self.class_mapping[self.j_sel]})"
            )
        )

        ax.set_yticks(y)

        ax.set_yticklabels(
            labels,
            fontsize=self._fs(32)
        )

        ax.set_xticks(
            [0, 0.25, 0.5, 0.75, 1.0]
        )

        ax.set_xticklabels(
            ["0%", "25%", "50%", "75%", "100%"],
            fontsize=self._fs(36)
        )

        ax.set_xlim(
            0,
            1
        )

        ax.axvline(
            0.5,
            color="k",
            linewidth=1,
            alpha=0.6
        )

        ax.set_xlabel(
            "Row-normalized weighted contribution",
            fontsize=self._fs(34)
        )

        ax.set_ylabel(
            "Feature",
            fontsize=self._fs(34)
        )

        ax.set_title(
            f"Dominance Plot (DP)\n"
            f"{self.class_mapping[self.j_sel]} and "
            f"{self.class_mapping[self.r_sel]}",
            fontsize=self._fs(38)
        )

        for yi, a, b in zip(
            y,
            w_minus,
            w_plus
        ):
            if a >= 0.12:
                ax.text(
                    a / 2,
                    yi,
                    f"{100 * a:.0f}%",
                    va="center",
                    ha="center",
                    fontsize=self._fs(28)
                )

            if b >= 0.12:
                ax.text(
                    a + b / 2,
                    yi,
                    f"{100 * b:.0f}%",
                    va="center",
                    ha="center",
                    fontsize=self._fs(28)
                )

        fig.tight_layout()

        self._legend_below(
            ax=ax,
            fig=fig,
            ncol=2,
            fontsize=26,
            anchor_y=-0.20,
            bottom_margin=0.28
        )

        if fname is None:
            fname = f"figB_weighted_100stack_{self.pair_tag}.png"

        save_path = self._outpath(fname)

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close(fig)

        print(f"Figura B guardada: {save_path}")
        print(f"CSV pesos DP guardado: {csv_path}")

        return df

    # ========================================================
    # UMBRALES Y ESTADÍSTICOS
    # ========================================================

    def _best_balanced_threshold_bidir(self, pos, neg):
        """
        Encuentra el umbral que maximiza balanced accuracy probando:
        - pred = 1 si x >= t
        - pred = 1 si x <= t
        """

        if len(pos) == 0 or len(neg) == 0:
            return None, None, None

        y_true = np.r_[
            np.ones(len(pos)),
            np.zeros(len(neg))
        ]

        vals = np.r_[
            pos,
            neg
        ]

        uniq = np.unique(vals)

        if uniq.size == 1:
            return float(uniq[0]), 0.5, ">="

        best_ba = -1.0
        best_t = None
        best_dir = None

        for t in uniq:
            pred_ge = (
                vals >= t
            ).astype(int)

            ba_ge = balanced_accuracy_score(
                y_true,
                pred_ge
            )

            pred_le = (
                vals <= t
            ).astype(int)

            ba_le = balanced_accuracy_score(
                y_true,
                pred_le
            )

            if ba_ge > best_ba:
                best_ba = ba_ge
                best_t = float(t)
                best_dir = ">="

            if ba_le > best_ba:
                best_ba = ba_le
                best_t = float(t)
                best_dir = "<="

        return best_t, float(best_ba), best_dir

    def _boundary_pairwise(self, pos, neg, direction):
        """
        Boundary coherente con la dirección elegida.
        """

        if (
            len(pos) == 0
            or len(neg) == 0
            or direction is None
        ):
            return None

        if direction == ">=":
            return 0.5 * (
                np.max(neg)
                + np.min(pos)
            )

        return 0.5 * (
            np.max(pos)
            + np.min(neg)
        )

    def plot_kde_top_features(self, dpi=600, show=True):
        """
        KDE para las dos primeras características del ranking.
        Si una variable tiene varianza nula o muy pocos valores,
        se omite la curva para evitar figuras vacías o errores.
        """

        if not hasattr(self, "rank"):
            self.ranking_global()

        print(
            "Clases presentes en self.y_te:",
            set(self.y_te)
        )

        print(
            "Claves en self.class_mapping:",
            set(self.class_mapping.keys())
        )

        X_te_df = self.X_te.copy()
        rank = self.rank

        # ----------------------------------------------------
        # KDE para las dos primeras características del ranking
        # ----------------------------------------------------
        for f in rank["Feature"].iloc[:2]:
            fig, ax = plt.subplots(
                figsize=(16, 10)
            )

            plotted = False

            for c in [
                self.j_sel,
                self.r_sel
            ]:
                vals = X_te_df.loc[
                    self.y_te == c,
                    f
                ].dropna()

                if (
                    len(vals) < 2
                    or vals.nunique() < 2
                ):
                    print(
                        f"Advertencia: no se grafica KDE para {f}, "
                        f"clase {c}, porque tiene muy pocos "
                        "valores o varianza nula."
                    )

                    continue

                try:
                    sns.kdeplot(
                        vals,
                        fill=True,
                        alpha=0.4,
                        color=(
                            "#56B4E9"
                            if c == self.j_sel
                            else "#ECE133"
                        ),
                        label=self.class_mapping[c],
                        cut=0,
                        clip=(0, 1),
                        ax=ax
                    )

                    plotted = True

                except Exception as e:
                    print(
                        f"No se pudo graficar KDE para {f}, "
                        f"clase {c}: {e}"
                    )

            vals_j = X_te_df.loc[
                self.y_te == self.j_sel,
                f
            ].values

            vals_r = X_te_df.loc[
                self.y_te == self.r_sel,
                f
            ].values

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
                0,
                1
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
        # KDE top-1 para todas las clases
        # ----------------------------------------------------
        top_feat = rank.iloc[0]["Feature"]

        fig, ax = plt.subplots(
            figsize=(16, 10)
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
                    clip=(0, 1),
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

        ax.set_xlim(
            0,
            0.03
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

    def estadisticos_top1(self):
        """
        Estadísticos de la característica top-1.
        Se calcula precision/recall solo en el par j vs r.
        """

        if not hasattr(self, "rank"):
            self.ranking_global()

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

        X_te_df = self.X_te.copy()

        vals_j = X_te_df.loc[
            self.y_te == self.j_sel,
            top_feat
        ].values

        vals_r = X_te_df.loc[
            self.y_te == self.r_sel,
            top_feat
        ].values

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

        d_abs = abs(
            cohens_d(
                vals_j,
                vals_r
            )
        )

        try:
            p_ks = ks_2samp(
                vals_j,
                vals_r
            ).pvalue

        except Exception:
            p_ks = np.nan

        thr_BA, ba, direction = (
            self._best_balanced_threshold_bidir(
                vals_j,
                vals_r
            )
        )

        print("\n=== Top-1 statistics ===")

        if np.isnan(d_abs):
            print(
                "Cohen |d|               : nan "
                "(no computable por varianza cero)"
            )
        else:
            print(
                f"Cohen |d|               : {d_abs:.3f}"
            )

        print(
            f"p-value KS              : {p_ks:.3e}"
        )

        print(
            f"Best threshold (BA)     : {thr_BA}"
        )

        if ba is None:
            print(
                "Balanced-Acc @thr       : None"
            )
        else:
            print(
                f"Balanced-Acc @thr       : {ba:.3f}"
            )

        if direction is None:
            print(
                "No se pudo calcular dirección de umbral."
            )

            return

        # ----------------------------------------------------
        # Cálculo P-R solo para el par j vs r
        # ----------------------------------------------------
        mask_pair = (
            (self.y_te == self.j_sel)
            | (self.y_te == self.r_sel)
        )

        x_pair = X_te_df.loc[
            mask_pair,
            top_feat
        ].values

        y_true_pair = (
            self.y_te[mask_pair] == self.j_sel
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

        pr_df = pd.DataFrame(records)

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

        print(
            f"max {self.class_mapping[self.r_sel]}: "
            f"{vals_r.max():.6f}"
        )

        print(
            f"min {self.class_mapping[self.j_sel]}: "
            f"{vals_j.min():.6f}"
        )

        print(
            f"boundary thr      : {boundary:.6f}"
        )

        print(
            f"Balanced-Acc @bnd : {bal_rule:.3f}\n"
        )

    # ========================================================
    # MÉTRICAS Y ANÁLISIS EXTRA
    # ========================================================

    def metricas_modelo(self):
        """
        Calcula las métricas finales del modelo sobre el conjunto de prueba
        guardado en los artefactos.

        Métricas:
        - Accuracy
        - Balanced Accuracy
        - F1-score macro
        - Matthews Correlation Coefficient (MCC)
        - Macro-AUC OvO, cuando existen probabilidades
        - Matriz de confusión

        Este método no reentrena el modelo, no recalcula SHAP y no modifica
        el análisis SHAP-GLCD.
        """

        y_pred = None
        y_proba = None

        # ----------------------------------------------------
        # Intentar predecir con el modelo cargado
        # ----------------------------------------------------
        if self.modelo_final is not None:
            try:
                y_pred = self.modelo_final.predict(
                    self.X_te
                )

                y_proba = self.modelo_final.predict_proba(
                    self.X_te
                )

            except Exception as e:
                print(
                    "\nAdvertencia: no se pudo predecir "
                    "con el modelo cargado."
                )

                print(
                    "Se intentará usar test_predictions.csv "
                    "y test_probabilities.csv."
                )

                print(e)

        # ----------------------------------------------------
        # Utilizar predicciones guardadas cuando corresponda
        # ----------------------------------------------------
        if (
            y_pred is None
            and self.test_predictions is not None
        ):
            if "y_pred" in self.test_predictions.columns:
                y_pred = (
                    self.test_predictions["y_pred"]
                    .values
                    .astype(int)
                )

        # ----------------------------------------------------
        # Utilizar probabilidades guardadas cuando corresponda
        # ----------------------------------------------------
        if (
            y_proba is None
            and self.test_probabilities is not None
        ):
            y_proba = self.test_probabilities.values

        if y_pred is None:
            print(
                "\nNo fue posible calcular métricas del modelo."
            )

            print(
                "Se requiere cargar correctamente el modelo o "
                "disponer de test_predictions.csv con una columna "
                "llamada 'y_pred'."
            )

            return None

        # ----------------------------------------------------
        # Métricas finales sobre test
        # ----------------------------------------------------
        acc = accuracy_score(
            self.y_te,
            y_pred
        )

        bal_acc = balanced_accuracy_score(
            self.y_te,
            y_pred
        )

        f1_macro = f1_score(
            self.y_te,
            y_pred,
            average="macro"
        )

        mcc = matthews_corrcoef(
            self.y_te,
            y_pred
        )

        cm = confusion_matrix(
            self.y_te,
            y_pred,
            labels=self.class_ids
        )

        auc_mac = None

        if y_proba is not None:
            try:
                auc_mac = roc_auc_score(
                    self.y_te,
                    y_proba,
                    multi_class="ovo",
                    average="macro"
                )

            except Exception as e:
                print(
                    "Advertencia: no se pudo calcular Macro-AUC OvO."
                )

                print(e)

        # ----------------------------------------------------
        # Mostrar resultados
        # ----------------------------------------------------
        print(
            "\n=== Model metrics on saved test set ==="
        )

        print(
            f"Accuracy           : {acc:.4f}"
        )

        print(
            f"Balanced accuracy  : {bal_acc:.4f}"
        )

        print(
            f"F1-score macro     : {f1_macro:.4f}"
        )

        print(
            f"MCC                : {mcc:.4f}"
        )

        if auc_mac is not None:
            print(
                f"Macro-AUC (OvO)    : {auc_mac:.4f}"
            )
        else:
            print(
                "Macro-AUC (OvO)    : No disponible"
            )

        print(
            "Confusion matrix:"
        )

        print(cm)

        # ----------------------------------------------------
        # Guardar resultados
        # ----------------------------------------------------
        metrics = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_macro": f1_macro,
            "mcc": mcc,
            "macro_auc_ovo": auc_mac
        }

        metrics_path = self._outpath(
            f"model_metrics_{self.pair_tag}.csv"
        )

        cm_path = self._outpath(
            f"confusion_matrix_{self.pair_tag}.csv"
        )

        pd.DataFrame([metrics]).to_csv(
            metrics_path,
            index=False
        )

        pd.DataFrame(
            cm,
            index=[
                f"true_{self.class_mapping[c]}"
                for c in self.class_ids
            ],
            columns=[
                f"pred_{self.class_mapping[c]}"
                for c in self.class_ids
            ]
        ).to_csv(
            cm_path,
            index=True
        )

        print(
            f"\nMétricas guardadas en: {metrics_path}"
        )

        print(
            f"Matriz de confusión guardada en: {cm_path}"
        )

        return metrics

    def extra_feature_analysis(self, feature_name):
        if not hasattr(self, "delta"):
            self.calcular_delta()

        if feature_name not in self.features:
            raise ValueError(
                f"La característica '{feature_name}' "
                "no está en feature_names."
            )

        ci = self.delta[
            :,
            self.features.index(feature_name)
        ]

        n_total = len(ci)

        n_pos = int(
            np.sum(ci > 0)
        )

        n_neg = int(
            np.sum(ci < 0)
        )

        n_zero = int(
            np.sum(ci == 0)
        )

        mean_pos = (
            ci[ci > 0].mean()
            if n_pos
            else 0.0
        )

        mean_neg = (
            ci[ci < 0].mean()
            if n_neg
            else 0.0
        )

        signed_mean = ci.mean()
        mean_abs = np.mean(np.abs(ci))
        median_abs = np.median(np.abs(ci))

        result = {
            "feature": feature_name,
            "n_total": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_zero": n_zero,
            "p_pos": n_pos / n_total,
            "p_neg": n_neg / n_total,
            "p_zero": n_zero / n_total,
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "signed_mean": signed_mean,
            "mean_abs": mean_abs,
            "median_abs": median_abs,
            "favours_positive_class": self.class_mapping[self.j_sel],
            "favours_negative_class": self.class_mapping[self.r_sel]
        }

        print(
            f"\n=== Extra feature analysis: {feature_name} ==="
        )

        print(
            f"n_+={n_pos}, n_-={n_neg}, "
            f"n_0={n_zero}, N={n_total}"
        )

        print(
            f"p_+={result['p_pos']:.2f}, "
            f"p_-={result['p_neg']:.2f}, "
            f"p_0={result['p_zero']:.2f}"
        )

        print(
            f"mean_+={mean_pos:.2f}, "
            f"mean_-={mean_neg:.2f}"
        )

        print(
            f"signed_mean={signed_mean:.2f}"
        )

        out = self._outpath(
            f"extra_feature_analysis_"
            f"{safe_filename(feature_name)}_"
            f"{self.pair_tag}.csv"
        )

        pd.DataFrame([result]).to_csv(
            out,
            index=False
        )

        print(
            f"Análisis extra guardado en: {out}"
        )

        return result

    # ========================================================
    # PIPELINE PRINCIPAL
    # ========================================================

    def run_glcd(
        self,
        make_plots=True,
        show=True,
        dpi=600
    ):
        """
        Ejecuta SHAP-GLCD usando artefactos ya guardados.

        No recalcula SHAP.
        No reentrena modelo.
        No cambia el conjunto de prueba.
        """

        print("\n==========================================")
        print(" EJECUTANDO SHAP-GLCD DESDE ARTEFACTOS")
        print("==========================================")

        self.calcular_delta()
        self.ranking_global()

        if make_plots:
            self.plot_barras(
                dpi=dpi,
                show=show
            )

            self.plot_barras_alineadas(
                dpi=dpi,
                show=show
            )

            self.plot_kde_top_features(
                dpi=dpi,
                show=show
            )

        self.estadisticos_top1()
        self.metricas_modelo()

        print("\n==========================================")
        print(" SHAP-GLCD FINALIZADO")
        print("==========================================\n")

        return self.rank