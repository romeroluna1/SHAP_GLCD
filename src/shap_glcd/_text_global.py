# ============================================================
# Clase_Global_Text_SHAP_GLCD.py
# ============================================================
# SHAP-GLCD global para texto.
#
# Visualización principal:
#   una figura tipo token-highlight para cada métrica:
#
#   1) Mean |ΔS|      -> escala neutra, solo magnitud
#   2) Median |ΔS|    -> escala neutra, solo magnitud
#   3) Signed mean ΔS -> magenta/verde, dirección de clase
#   4) CDM            -> magenta/verde, dirección de clase
#   5) DP             -> magenta/verde, dominancia relativa de clase
#
# Convención direccional:
#   Magenta → favorece la primera clase Cj
#   Verde   → favorece la segunda clase Cr
#
# Para Mean y Median:
#   Azul → mayor magnitud de la métrica, sin dirección de clase.
#
# Definición DP según la teoría:
#
#   w_m^+ = p_m^+ * |mean(ΔS_m^+)|
#   w_m^- = p_m^- * |mean(ΔS_m^-)|
#
#   W_m^+ = w_m^+ / (w_m^+ + w_m^-)
#   W_m^- = w_m^- / (w_m^+ + w_m^-)
#
#   dp_signed_value = W_m^+ - W_m^-
#   dp_strength     = |W_m^+ - W_m^-|
#
# Nota:
#   normalize_token_case=True unifica tokens como Oil/oil/OIL
#   durante la agregación global. No recalcula SHAP.
# ============================================================

import os
import html as html_lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SHAPGlobalTextGLCD:
    """
    SHAP-GLCD global para texto.

    En texto, la unidad explicativa global es el token.
    """

    def __init__(
        self,
        lcd_explainer,
        output_dir="results_shap_glcd_text",
        min_token_frequency=2,
        normalize_token_case=True
    ):
        self.lcd_explainer = lcd_explainer
        self.class_names = lcd_explainer.class_names
        self.output_dir = output_dir
        self.min_token_frequency = min_token_frequency

        # Si True:
        #   Oil, oil, OIL -> oil
        # Esto se hace solo para la agregación global.
        self.normalize_token_case = normalize_token_case

        os.makedirs(self.output_dir, exist_ok=True)

        if getattr(self.lcd_explainer, "shap_values", None) is None:
            raise ValueError(
                "El objeto lcd_explainer no tiene valores SHAP. "
                "Primero calcula o carga shap_values."
            )

    # ============================================================
    # Utilidades
    # ============================================================

    @staticmethod
    def safe_filename(text):
        return (
            str(text)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
            .replace(" ", "_")
        )

    def _class_name_to_index(self, class_name_or_idx):
        if isinstance(class_name_or_idx, int):
            return class_name_or_idx

        if self.class_names is None:
            raise ValueError("No hay class_names definido. Usa índices.")

        inverse = {v: k for k, v in self.class_names.items()}

        if class_name_or_idx not in inverse:
            raise ValueError(
                f"No se encontró la clase '{class_name_or_idx}'. "
                f"Clases disponibles: {list(inverse.keys())}"
            )

        return inverse[class_name_or_idx]

    def _class_label(self, class_idx):
        if self.class_names is None:
            return f"Class {class_idx}"

        return self.class_names[int(class_idx)]

    @staticmethod
    def _clean_token(token):
        token = str(token)

        token = token.replace("Ġ", " ")
        token = token.replace("▁", " ")
        token = token.replace("##", "")
        token = token.strip()

        token = html_lib.unescape(token)
        token = token.replace("&amp;", "&")
        token = token.replace("&quot;", '"')
        token = token.replace("&#39;", "'")
        token = token.replace("# 39 ;", "'")

        return token.strip()

    def _normalize_token_for_global(self, token):
        """
        Normaliza el token solo para el resumen global.

        Esto permite agrupar:
            Oil, oil, OIL -> oil

        Se aplica después de limpiar y validar el token.
        """
        token = str(token).strip()

        if self.normalize_token_case:
            token = token.lower()

        return token

    @staticmethod
    def _is_valid_token(token):
        if token is None:
            return False

        token = str(token).strip()
        token_lower = token.lower()

        if token == "":
            return False

        special_tokens = {
            "[CLS]", "[SEP]", "[PAD]", "[MASK]",
            "<s>", "</s>", "<pad>", "<mask>",
            "[cls]", "[sep]", "[pad]", "[mask]"
        }

        if token in special_tokens or token_lower in special_tokens:
            return False

        if len(token) == 1:
            return False

        punctuation = set(".,;:!?()[]{}\"'`´“”‘’-/\\|_+=*&^%$#@~")

        if all(ch in punctuation for ch in token):
            return False

        noisy_tokens = {
            "amp", "quot", "39", "#", "nbsp"
        }

        if token_lower in noisy_tokens:
            return False

        return True

    # ============================================================
    # Extracción de tokens y SHAP
    # ============================================================

    def _extract_tokens_from_instance(self, instance_idx):
        shap_values = self.lcd_explainer.shap_values

        try:
            data_i = shap_values[instance_idx].data
        except Exception:
            data_i = shap_values.data[instance_idx]

        if isinstance(data_i, str):
            encoded = self.lcd_explainer.tokenizer(
                data_i,
                add_special_tokens=False,
                truncation=True,
                max_length=self.lcd_explainer.max_length
            )

            tokens = self.lcd_explainer.tokenizer.convert_ids_to_tokens(
                encoded["input_ids"]
            )
        else:
            tokens = list(data_i)

        tokens = [self._clean_token(tok) for tok in tokens]

        return tokens

    def _extract_values_from_instance(self, instance_idx):
        shap_values = self.lcd_explainer.shap_values

        try:
            values = shap_values[instance_idx].values
        except Exception:
            values = shap_values.values[instance_idx]

        values = np.asarray(values)

        if values.dtype == object:
            try:
                values = np.asarray(values.tolist(), dtype=float)
            except Exception:
                values = np.asarray(values, dtype=float)

        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]

        if values.ndim != 2:
            raise ValueError(
                f"Forma SHAP no esperada para la instancia {instance_idx}: "
                f"{values.shape}. Se esperaba (n_tokens, n_classes)."
            )

        values = np.nan_to_num(values)

        return values

    def _get_instance_values_and_tokens(self, instance_idx):
        tokens = self._extract_tokens_from_instance(instance_idx)
        values = self._extract_values_from_instance(instance_idx)

        n_tokens_data = len(tokens)
        n_tokens_values = values.shape[0]

        if n_tokens_data != n_tokens_values:
            n = min(n_tokens_data, n_tokens_values)
            tokens = tokens[:n]
            values = values[:n, :]

        return tokens, values

    # ============================================================
    # Registros ΔS
    # ============================================================

    def get_delta_records_for_pair(self, class_j, class_r):
        idx_j = self._class_name_to_index(class_j)
        idx_r = self._class_name_to_index(class_r)

        label_j = self._class_label(idx_j)
        label_r = self._class_label(idx_r)

        records = []

        n_instances = len(self.lcd_explainer.texts_eval)

        original_indices = getattr(
            self.lcd_explainer,
            "original_indices",
            np.arange(n_instances)
        )

        for instance_idx in range(n_instances):

            tokens, values = self._get_instance_values_and_tokens(instance_idx)

            if values.shape[1] <= max(idx_j, idx_r):
                raise ValueError(
                    f"No hay suficientes clases en SHAP values. "
                    f"Forma recibida: {values.shape}. "
                    f"Clases requeridas: {idx_j}, {idx_r}."
                )

            shap_j = values[:, idx_j]
            shap_r = values[:, idx_r]
            delta = shap_j - shap_r

            for tok, sj, sr, ds in zip(tokens, shap_j, shap_r, delta):

                tok_clean = self._clean_token(tok)

                if not self._is_valid_token(tok_clean):
                    continue

                # Normalización para agregación global.
                # Aquí se unifica Oil/oil/OIL -> oil.
                tok_global = self._normalize_token_for_global(tok_clean)

                if not self._is_valid_token(tok_global):
                    continue

                if ds > 0:
                    direction = label_j
                elif ds < 0:
                    direction = label_r
                else:
                    direction = "neutral"

                records.append({
                    "instance_idx": int(instance_idx),
                    "original_test_index": int(original_indices[instance_idx])
                    if len(original_indices) > instance_idx else int(instance_idx),
                    "token": tok_global,
                    "class_j": label_j,
                    "class_r": label_r,
                    "shap_j": float(sj),
                    "shap_r": float(sr),
                    "deltaS": float(ds),
                    "abs_deltaS": float(abs(ds)),
                    "direction": direction
                })

        return pd.DataFrame(records)

    # ============================================================
    # Agregación global
    # ============================================================

    def aggregate_pair_global(self, class_j, class_r):
        df_records = self.get_delta_records_for_pair(class_j, class_r)

        if df_records.empty:
            raise ValueError("No se generaron registros de ΔS.")

        label_j = str(df_records["class_j"].iloc[0])
        label_r = str(df_records["class_r"].iloc[0])

        grouped_rows = []

        for token, group in df_records.groupby("token"):

            values = group["deltaS"].to_numpy(dtype=float)
            abs_values = np.abs(values)

            pos_values = values[values > 0]
            neg_values = values[values < 0]

            n_total = len(values)
            n_positive = len(pos_values)
            n_negative = len(neg_values)

            # ----------------------------------------------------
            # Métricas básicas
            # ----------------------------------------------------

            mean_abs_deltaS = float(np.mean(abs_values))
            median_abs_deltaS = float(np.median(abs_values))
            signed_mean = float(np.mean(values))

            if signed_mean > 0:
                signed_direction = label_j
            elif signed_mean < 0:
                signed_direction = label_r
            else:
                signed_direction = "neutral"

            # ----------------------------------------------------
            # Medias condicionadas de ΔS
            # ----------------------------------------------------

            positive_conditional_mean = (
                float(np.mean(pos_values)) if n_positive > 0 else 0.0
            )

            negative_conditional_mean = (
                float(np.mean(neg_values)) if n_negative > 0 else 0.0
            )

            # ----------------------------------------------------
            # CDM:
            # Se toma la media condicionada con mayor magnitud.
            # ----------------------------------------------------

            if abs(positive_conditional_mean) >= abs(negative_conditional_mean):
                cdm_value = positive_conditional_mean
                cdm_direction = label_j if positive_conditional_mean > 0 else "neutral"
            else:
                cdm_value = negative_conditional_mean
                cdm_direction = label_r if negative_conditional_mean < 0 else "neutral"

            cdm_strength = abs(cdm_value)

            # ----------------------------------------------------
            # Masas discriminativas ponderadas según la teoría:
            #
            #   w_m^+ = p_m^+ * |mean(ΔS_m^+)|
            #   w_m^- = p_m^- * |mean(ΔS_m^-)|
            #
            # Luego se normalizan:
            #
            #   W_m^+ = w_m^+ / (w_m^+ + w_m^-)
            #   W_m^- = w_m^- / (w_m^+ + w_m^-)
            #
            # DP se representa como:
            #
            #   dp_signed_value = W_m^+ - W_m^-
            #   dp_strength     = |W_m^+ - W_m^-|
            # ----------------------------------------------------

            p_positive = n_positive / n_total if n_total > 0 else 0.0
            p_negative = n_negative / n_total if n_total > 0 else 0.0

            mass_positive = p_positive * abs(positive_conditional_mean)
            mass_negative = p_negative * abs(negative_conditional_mean)

            total_mass = mass_positive + mass_negative

            if total_mass > 0:
                W_positive = mass_positive / total_mass
                W_negative = mass_negative / total_mass
            else:
                # Caso sin masa discriminativa:
                # se representa como una situación neutral y equilibrada.
                W_positive = 0.5
                W_negative = 0.5

            dp_signed_value = float(W_positive - W_negative)
            dominance_margin = float(abs(dp_signed_value))
            dp_strength = dominance_margin

            if dp_signed_value > 0:
                dominant_class = label_j
            elif dp_signed_value < 0:
                dominant_class = label_r
            else:
                dominant_class = "neutral"

            grouped_rows.append({
                "token": token,
                "class_j": label_j,
                "class_r": label_r,

                "frequency": int(n_total),
                "document_frequency": int(group["instance_idx"].nunique()),

                "n_positive": int(n_positive),
                "n_negative": int(n_negative),

                "mean_abs_deltaS": mean_abs_deltaS,
                "median_abs_deltaS": median_abs_deltaS,
                "signed_mean_deltaS": signed_mean,

                "positive_conditional_mean": positive_conditional_mean,
                "negative_conditional_mean": negative_conditional_mean,

                "cdm_value": float(cdm_value),
                "cdm_strength": float(cdm_strength),
                "cdm_direction": cdm_direction,

                "p_positive": float(p_positive),
                "p_negative": float(p_negative),

                "mass_positive": float(mass_positive),
                "mass_negative": float(mass_negative),

                "W_positive": float(W_positive),
                "W_negative": float(W_negative),

                "dominance_margin": float(dominance_margin),
                "dp_strength": float(dp_strength),
                "dp_signed_value": float(dp_signed_value),

                "dominant_class": dominant_class,
                "signed_direction": signed_direction
            })

        df_global = pd.DataFrame(grouped_rows)

        df_global = df_global[
            df_global["frequency"] >= self.min_token_frequency
        ].copy()

        df_global = df_global.sort_values(
            by=["mean_abs_deltaS", "frequency"],
            ascending=[False, False]
        ).reset_index(drop=True)

        return df_records, df_global

    # ============================================================
    # Preparación de datos para figuras tipo texto
    # ============================================================

    def _prepare_highlight_df(
        self,
        df_global,
        metric,
        class_j,
        class_r,
        top_k
    ):
        df = df_global.copy()

        if metric == "mean_abs":
            df["plot_strength"] = df["mean_abs_deltaS"]
            df["plot_signed"] = df["plot_strength"]
            df["plot_value_text"] = df["mean_abs_deltaS"].map(
                lambda x: f"mean={x:.3f}"
            )

            df = df.sort_values(
                ["plot_strength", "frequency"],
                ascending=[False, False]
            )

            title_metric = r"Mean $|\Delta S|$"

        elif metric == "median_abs":
            df["plot_strength"] = df["median_abs_deltaS"]
            df["plot_signed"] = df["plot_strength"]
            df["plot_value_text"] = df["median_abs_deltaS"].map(
                lambda x: f"median={x:.3f}"
            )

            df = df.sort_values(
                ["plot_strength", "frequency"],
                ascending=[False, False]
            )

            title_metric = r"Median $|\Delta S|$"

        elif metric == "signed_mean":
            df["plot_strength"] = df["signed_mean_deltaS"].abs()
            df["plot_signed"] = df["signed_mean_deltaS"]
            df["plot_value_text"] = df["signed_mean_deltaS"].map(
                lambda x: f"SDM={x:+.3f}"
            )

            df = df.sort_values(
                ["plot_strength", "frequency"],
                ascending=[False, False]
            )

            title_metric = r"Signed mean $\Delta S$"

        elif metric == "cdm":
            df["plot_strength"] = df["cdm_strength"]
            df["plot_signed"] = df["cdm_value"]
            df["plot_value_text"] = df["cdm_value"].map(
                lambda x: f"CDM={x:+.3f}"
            )

            df = df.sort_values(
                ["plot_strength", "frequency"],
                ascending=[False, False]
            )

            title_metric = "Conditional Difference Mean (CDM)"

        elif metric == "dp":
            df["plot_strength"] = df["dp_strength"]
            df["plot_signed"] = df["dp_signed_value"]

            df["plot_value_text"] = df.apply(
                lambda row: (
                    f"W+={row['W_positive']:.2f}"
                    if row["dp_signed_value"] >= 0
                    else f"W-={row['W_negative']:.2f}"
                ),
                axis=1
            )

            df = df.sort_values(
                ["plot_strength", "frequency"],
                ascending=[False, False]
            )

            title_metric = "Dominance Plot (DP)"

        else:
            raise ValueError(
                "metric debe ser uno de: "
                "'mean_abs', 'median_abs', 'signed_mean', 'cdm', 'dp'."
            )

        df = df.head(top_k).copy()

        max_strength = df["plot_strength"].max()

        if max_strength == 0 or np.isnan(max_strength):
            df["importance_norm"] = 0.5
        else:
            df["importance_norm"] = df["plot_strength"] / max_strength

        return df, title_metric

    # ============================================================
    # Figura tipo token-highlight para una métrica
    # ============================================================

    def plot_global_token_highlight_by_metric(
        self,
        df_global,
        class_j,
        class_r,
        metric,
        top_k=40,
        save_path=None,
        show=False
    ):
        df_top, title_metric = self._prepare_highlight_df(
            df_global=df_global,
            metric=metric,
            class_j=class_j,
            class_r=class_r,
            top_k=top_k
        )

        if df_top.empty:
            raise ValueError(
                "df_global está vacío después del filtrado. "
                "No se puede generar la figura."
            )

        # Colores
        magenta = np.array([214, 39, 140]) / 255.0
        green = np.array([44, 160, 44]) / 255.0
        blue = np.array([31, 119, 180]) / 255.0
        white = np.array([1.0, 1.0, 1.0])

        def blend_with_white(color, alpha):
            alpha = 0.25 + 0.75 * float(alpha)
            return tuple((1 - alpha) * white + alpha * color)

        fig, ax = plt.subplots(figsize=(13.8, 7.8), dpi=300)
        ax.axis("off")

        title = f"Global SHAP-GLCD token highlights ({title_metric})"

        if metric in ["mean_abs", "median_abs"]:
            subtitle = (
                f"{class_j} vs {class_r} | "
                "Blue intensity represents metric magnitude"
            )
        else:
            subtitle = (
                f"{class_j} vs {class_r} | "
                f"Magenta favors {class_j}; green favors {class_r}"
            )

        ax.text(
            0.5,
            1.045,
            title,
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
            transform=ax.transAxes
        )

        ax.text(
            0.5,
            1.000,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=12,
            transform=ax.transAxes
        )

        x = 0.03
        y = 0.90
        line_height = 0.115
        max_x = 0.96

        for _, row in df_top.iterrows():

            token = str(row["token"])
            importance = float(row["importance_norm"])
            signed_value = float(row["plot_signed"])

            if metric in ["mean_abs", "median_abs"]:
                facecolor = blend_with_white(blue, importance)
                edgecolor = tuple(blue)
            else:
                if signed_value >= 0:
                    facecolor = blend_with_white(magenta, importance)
                    edgecolor = tuple(magenta)
                else:
                    facecolor = blend_with_white(green, importance)
                    edgecolor = tuple(green)

            font_size = 12 + 8 * importance

            token_width = 0.0175 * len(token) + 0.048

            if x + token_width > max_x:
                x = 0.03
                y -= line_height

            if y < 0.13:
                break

            ax.text(
                x,
                y,
                token,
                fontsize=font_size,
                ha="left",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=0.8,
                    alpha=0.96
                ),
                transform=ax.transAxes
            )

            small_text = (
                f"f={int(row['frequency'])} | "
                f"{row['plot_value_text']}"
            )

            ax.text(
                x,
                y - 0.043,
                small_text,
                fontsize=8.5,
                ha="left",
                va="center",
                color="dimgray",
                transform=ax.transAxes
            )

            x += token_width + 0.018

        # Leyenda inferior
        if metric in ["mean_abs", "median_abs"]:

            ax.text(
                0.03,
                0.035,
                "Blue: higher metric magnitude",
                fontsize=10,
                ha="left",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor=tuple(blue),
                    edgecolor="none",
                    alpha=0.70
                ),
                transform=ax.transAxes
            )

            ax.text(
                0.42,
                0.035,
                "Size/intensity: metric strength",
                fontsize=10,
                ha="left",
                va="center",
                color="dimgray",
                transform=ax.transAxes
            )

        else:

            ax.text(
                0.03,
                0.035,
                f"Magenta: favors {class_j}",
                fontsize=10,
                ha="left",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor=tuple(magenta),
                    edgecolor="none",
                    alpha=0.70
                ),
                transform=ax.transAxes
            )

            ax.text(
                0.32,
                0.035,
                f"Green: favors {class_r}",
                fontsize=10,
                ha="left",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor=tuple(green),
                    edgecolor="none",
                    alpha=0.70
                ),
                transform=ax.transAxes
            )

            ax.text(
                0.66,
                0.035,
                "Size/intensity: metric strength",
                fontsize=10,
                ha="left",
                va="center",
                color="dimgray",
                transform=ax.transAxes
            )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")

            if save_path.lower().endswith(".png"):
                pdf_path = save_path[:-4] + ".pdf"
            else:
                pdf_path = save_path + ".pdf"

            plt.savefig(pdf_path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    # ============================================================
    # Generar las cinco figuras principales
    # ============================================================

    def plot_all_metric_highlights(
        self,
        df_global,
        class_j,
        class_r,
        pair_dir,
        safe_j,
        safe_r,
        top_k=45,
        show_plots=False
    ):
        metrics = [
            ("mean_abs", "01_global_highlight_mean_abs"),
            ("median_abs", "02_global_highlight_median_abs"),
            ("signed_mean", "03_global_highlight_signed_mean"),
            ("cdm", "04_global_highlight_cdm"),
            ("dp", "05_global_highlight_dp")
        ]

        saved_paths = []

        for metric_name, file_prefix in metrics:
            save_path = os.path.join(
                pair_dir,
                f"{file_prefix}_{safe_j}_vs_{safe_r}.png"
            )

            self.plot_global_token_highlight_by_metric(
                df_global=df_global,
                class_j=class_j,
                class_r=class_r,
                metric=metric_name,
                top_k=top_k,
                save_path=save_path,
                show=show_plots
            )

            saved_paths.append(save_path)

        return saved_paths

    # ============================================================
    # Proceso completo por par
    # ============================================================

    def run_pair_analysis(
        self,
        class_j,
        class_r,
        top_k=20,
        highlight_top_k=45,
        save_outputs=True,
        show_plots=False
    ):
        idx_j = self._class_name_to_index(class_j)
        idx_r = self._class_name_to_index(class_r)

        label_j = self._class_label(idx_j)
        label_r = self._class_label(idx_r)

        safe_j = self.safe_filename(label_j)
        safe_r = self.safe_filename(label_r)

        pair_dir = os.path.join(
            self.output_dir,
            f"pair_{safe_j}_vs_{safe_r}"
        )

        os.makedirs(pair_dir, exist_ok=True)

        print("\n==================================================")
        print(f"Running global SHAP-GLCD text analysis: {label_j} vs {label_r}")
        print("==================================================")

        df_records, df_global = self.aggregate_pair_global(
            class_j=idx_j,
            class_r=idx_r
        )

        if save_outputs:
            records_path = os.path.join(
                pair_dir,
                f"deltaS_records_{safe_j}_vs_{safe_r}.csv"
            )

            global_path = os.path.join(
                pair_dir,
                f"global_token_summary_{safe_j}_vs_{safe_r}.csv"
            )

            df_records.to_csv(records_path, index=False, encoding="utf-8")
            df_global.to_csv(global_path, index=False, encoding="utf-8")

            print(f"Token-level ΔS records saved to: {records_path}")
            print(f"Global token summary saved to: {global_path}")

        saved_figures = self.plot_all_metric_highlights(
            df_global=df_global,
            class_j=label_j,
            class_r=label_r,
            pair_dir=pair_dir,
            safe_j=safe_j,
            safe_r=safe_r,
            top_k=highlight_top_k,
            show_plots=show_plots
        )

        print("\nMain SHAP-GLCD text figures saved:")
        for path in saved_figures:
            print(path)

        cols_to_show = [
            "token",
            "frequency",
            "document_frequency",
            "mean_abs_deltaS",
            "median_abs_deltaS",
            "signed_mean_deltaS",
            "positive_conditional_mean",
            "negative_conditional_mean",
            "cdm_value",
            "p_positive",
            "p_negative",
            "mass_positive",
            "mass_negative",
            "W_positive",
            "W_negative",
            "dominance_margin",
            "dp_strength",
            "dp_signed_value",
            "dominant_class",
            "signed_direction"
        ]

        print("\nTop global tokens by mean |ΔS|:")
        print(
            df_global[cols_to_show]
            .head(top_k)
            .to_string(index=False)
        )

        return df_records, df_global