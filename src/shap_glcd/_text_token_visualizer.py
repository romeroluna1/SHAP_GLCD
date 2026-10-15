# -*- coding: utf-8 -*-

# ============================================================
# Clase_Global_Text_TokenSize_Article.py
# ============================================================
# Versión para artículo de las figuras tipo bloque de palabras.
#
# Esta clase NO modifica los cálculos SHAP ni GLCD.
# Solo cambia la presentación visual.
#
# Importante:
#   - Los textos completos se usan para el modelo y para SHAP.
#   - El filtro de stopwords se aplica SOLO para las figuras.
#   - df_global y los valores calculados no se modifican.
#
# Salidas:
#   26_article_word_block_mean_abs.png/pdf
#   27_article_word_block_median_abs.png/pdf
#   28_article_word_block_signed_mean.png/pdf
#   29_article_word_block_cdm.png/pdf
#   30_article_word_block_dp.png/pdf
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Stopwords y tokens poco informativos para VISUALIZACIÓN
# ============================================================

DEFAULT_STOPWORDS_FOR_PLOT = {
    # Artículos y determinantes
    "a", "an", "the",

    # Conectores
    "and", "or", "but", "nor", "so", "yet",

    # Preposiciones
    "of", "to", "in", "on", "for", "from", "with", "by", "at", "as",
    "into", "over", "under", "after", "before", "about", "between",
    "through", "during", "against", "without", "within", "among",

    # Verbos auxiliares / funcionales
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had",
    "do", "does", "did",
    "will", "would", "can", "could", "may", "might", "must", "shall", "should",

    # Pronombres
    "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "them", "his", "her", "their", "theirs",
    "we", "us", "our", "ours",
    "you", "your", "yours",
    "i", "me", "my", "mine",

    # Negaciones muy frecuentes
    "not", "no",

    # Palabras de estilo periodístico muy generales
    "said", "says", "say", "according",

    # Fragmentos frecuentes poco interpretables
    "lt", "gt", "br", "amp", "quot", "nbsp",
    "http", "https", "www",
    "com", "org", "net",
    "cm", "ct", "st", "ro", "ron", "ele", "ics",
    "en", "ny"
}


class SHAPGlobalTextTokenSizeArticleVisualizer:
    """
    Visualizador tipo bloque de palabras para versión final de artículo.

    Mean y Median:
        tamaño = importancia global
        color = azul fijo, sin dirección

    Signed Mean, CDM y DP:
        tamaño = magnitud/fuerza de la métrica
        color = clase favorecida

    El filtro de tokens se usa solo para las figuras.
    """

    def __init__(
        self,
        output_dir,
        token_col=None,
        min_font_size=12,
        max_font_size=38,
        title_font_size=18,
        legend_font_size=12,
        dpi=300,
        neutral_color="#31688E",
        cj_color="#C2185B",
        cr_color="#00897B",
        fig_width=12,
        fig_height=7.2,
        background_color="white",
        exclude_tokens_for_plot=None,
        min_token_length_for_plot=3,
        exclude_numeric_tokens_for_plot=True
    ):
        self.output_dir = output_dir
        self.token_col = token_col

        self.min_font_size = min_font_size
        self.max_font_size = max_font_size

        self.title_font_size = title_font_size
        self.legend_font_size = legend_font_size
        self.dpi = dpi

        self.neutral_color = neutral_color
        self.cj_color = cj_color
        self.cr_color = cr_color

        self.fig_width = fig_width
        self.fig_height = fig_height
        self.background_color = background_color

        self.exclude_tokens_for_plot = set(DEFAULT_STOPWORDS_FOR_PLOT)

        if exclude_tokens_for_plot is not None:
            self.exclude_tokens_for_plot.update(
                {str(t).strip().lower() for t in exclude_tokens_for_plot}
            )

        self.min_token_length_for_plot = int(min_token_length_for_plot)
        self.exclude_numeric_tokens_for_plot = bool(exclude_numeric_tokens_for_plot)

        os.makedirs(self.output_dir, exist_ok=True)

    # ========================================================
    # Utilidades internas
    # ========================================================

    def safe_filename(self, text):
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

    def _find_column(self, df, candidates, metric_name):
        for col in candidates:
            if col in df.columns:
                return col

        raise ValueError(
            f"\nNo se encontró columna para: {metric_name}\n"
            f"Candidatas buscadas: {candidates}\n"
            f"Columnas disponibles:\n{list(df.columns)}"
        )

    def _find_token_column(self, df):
        if self.token_col is not None:
            if self.token_col in df.columns:
                return self.token_col

            raise ValueError(
                f"\nNo se encontró la columna de tokens indicada: {self.token_col}\n"
                f"Columnas disponibles:\n{list(df.columns)}"
            )

        candidates = [
            "token",
            "word",
            "term",
            "feature",
            "feature_name",
            "token_text"
        ]

        return self._find_column(
            df=df,
            candidates=candidates,
            metric_name="token"
        )

    def _normalize_values(self, values):
        values = np.asarray(values, dtype=float)

        min_val = np.min(values)
        max_val = np.max(values)

        if np.isclose(min_val, max_val):
            return np.ones_like(values) * 0.5

        return (values - min_val) / (max_val - min_val)

    def _compute_font_sizes(self, values):
        """
        Escalado visual de tamaños.
        """

        values = np.asarray(values, dtype=float)

        normalized = self._normalize_values(values)
        normalized = np.sqrt(normalized)

        font_sizes = (
            self.min_font_size
            + normalized * (self.max_font_size - self.min_font_size)
        )

        return font_sizes

    def _estimate_token_width(self, token, font_size):
        """
        Estimación del ancho del token para evitar que palabras se peguen.
        """

        token = str(token)

        # Estimación más amplia que la versión inicial.
        base_width = 0.0102 * len(token) * (float(font_size) / 12.0)
        spacing = 0.022

        return base_width + spacing

    def _estimate_line_height(self, font_size):
        return font_size * 0.0032 + 0.035

    def _get_token_color(self, color_value=None, use_direction_colors=False):
        if not use_direction_colors:
            return self.neutral_color

        if color_value is None:
            return self.neutral_color

        if color_value > 0:
            return self.cj_color

        if color_value < 0:
            return self.cr_color

        return self.neutral_color

    def _is_valid_plot_token(self, token, extra_exclude_tokens=None):
        """
        Filtro usado solo para visualización.

        No modifica los cálculos SHAP ni df_global.
        Solo decide si un token aparece o no en las figuras.
        """

        token = str(token).strip().lower()

        if token == "":
            return False

        exclude_set = set(self.exclude_tokens_for_plot)

        if extra_exclude_tokens is not None:
            exclude_set.update(
                {str(t).strip().lower() for t in extra_exclude_tokens}
            )

        if token in exclude_set:
            return False

        if len(token) < self.min_token_length_for_plot:
            return False

        if self.exclude_numeric_tokens_for_plot:
            token_no_decimal = token.replace(".", "", 1).replace(",", "")
            if token_no_decimal.isdigit:
                if token_no_decimal.isdigit():
                    return False

        # Elimina tokens que no contienen letras.
        if not any(ch.isalpha() for ch in token):
            return False

        # Evita fragmentos muy ruidosos derivados de tokenización o HTML.
        noisy_tokens = {
            "lt", "gt", "br", "amp", "quot", "nbsp",
            "http", "https", "www",
            "com", "org", "net",
            "cm", "ct", "st", "ro", "ron", "ele", "ics"
        }

        if token in noisy_tokens:
            return False

        return True

    def _prepare_plot_df(
        self,
        df_global,
        size_metric_col,
        top_k,
        color_metric_col=None,
        use_abs_for_ranking=True,
        exclude_tokens=None
    ):
        """
        Prepara los tokens que se van a dibujar.

        exclude_tokens:
            Lista opcional de tokens que se excluyen solo de la figura.
            No modifica df_global ni los cálculos.
        """

        token_col = self._find_token_column(df_global)

        cols = [token_col, size_metric_col]

        if color_metric_col is not None and color_metric_col not in cols:
            cols.append(color_metric_col)

        if "frequency" in df_global.columns and "frequency" not in cols:
            cols.append("frequency")

        plot_df = df_global[cols].copy().dropna()

        plot_df[token_col] = plot_df[token_col].astype(str)

        # Filtro visual general:
        # conectores, pronombres, números, tokens cortos y tokens ruidosos.
        plot_df = plot_df[
            plot_df[token_col].apply(
                lambda tok: self._is_valid_plot_token(
                    tok,
                    extra_exclude_tokens=exclude_tokens
                )
            )
        ].copy()

        plot_df[size_metric_col] = pd.to_numeric(
            plot_df[size_metric_col],
            errors="coerce"
        )

        if color_metric_col is not None:
            plot_df[color_metric_col] = pd.to_numeric(
                plot_df[color_metric_col],
                errors="coerce"
            )

        if "frequency" in plot_df.columns:
            plot_df["frequency"] = pd.to_numeric(
                plot_df["frequency"],
                errors="coerce"
            )

        plot_df = plot_df.dropna()

        if use_abs_for_ranking:
            plot_df["ranking_value"] = plot_df[size_metric_col].abs()
        else:
            plot_df["ranking_value"] = plot_df[size_metric_col]

        if "frequency" in plot_df.columns:
            plot_df = plot_df.sort_values(
                ["ranking_value", "frequency"],
                ascending=[False, False]
            )
        else:
            plot_df = plot_df.sort_values(
                ["ranking_value"],
                ascending=[False]
            )

        plot_df = plot_df.head(top_k).reset_index(drop=True)

        return plot_df, token_col

    def _save_png_and_pdf(self, output_path):
        plt.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            facecolor=self.background_color
        )

        if output_path.lower().endswith(".png"):
            pdf_path = output_path[:-4] + ".pdf"
        else:
            pdf_path = output_path + ".pdf"

        plt.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor=self.background_color
        )

    # ========================================================
    # Render principal
    # ========================================================

    def _plot_token_block_article(
            self,
            df_global,
            size_metric_col,
            title,
            output_name,
            top_k=35,
            color_metric_col=None,
            use_abs_for_ranking=True,
            use_direction_colors=False,
            class_j=None,
            class_r=None,
            exclude_tokens=None
    ):
        plot_df, token_col = self._prepare_plot_df(
            df_global=df_global,
            size_metric_col=size_metric_col,
            top_k=top_k,
            color_metric_col=color_metric_col,
            use_abs_for_ranking=use_abs_for_ranking,
            exclude_tokens=exclude_tokens
        )

        if plot_df.empty:
            print(f"No hay datos para graficar: {title}")
            return

        font_sizes = self._compute_font_sizes(
            plot_df["ranking_value"].values
        )

        fig, ax = plt.subplots(
            figsize=(self.fig_width, self.fig_height),
            dpi=self.dpi,
            facecolor=self.background_color
        )

        ax.set_facecolor(self.background_color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # ====================================================
        # Preambulo mejorado
        # ====================================================

        ax.text(
            0.5,
            0.965,
            title,
            fontsize=self.title_font_size,
            fontweight="bold",
            ha="center",
            va="top",
            color="black",
            transform=ax.transAxes
        )

        if class_j is not None and class_r is not None:
            subtitle = (
                f"Aggregated over evaluation instances | "
                f"ΔS = SHAP({class_j}) - SHAP({class_r})"
            )
        else:
            subtitle = "Aggregated over evaluation instances"

        ax.text(
            0.5,
            0.905,
            subtitle,
            fontsize=max(self.legend_font_size, 11),
            ha="center",
            va="top",
            color="#333333",
            transform=ax.transAxes
        )

        # Leyenda según la métrica
        title_lower = title.lower()

        if "mean |δs|" in title_lower or "mean |Δs|" in title:
            legend_text = "Size: mean |ΔS|"

        elif "median |δs|" in title_lower or "median |Δs|" in title:
            legend_text = "Size: median |ΔS|"

        elif "signed mean" in title_lower:
            legend_text = (
                f"Magenta: {class_j} | Green: {class_r} | "
                "Size: |signed mean ΔS|"
            )

        elif "conditional difference mean" in title_lower:
            legend_text = (
                f"Magenta: {class_j} | Green: {class_r} | "
                "Size: CDM strength"
            )

        elif "weighted dominance" in title_lower:
            legend_text = (
                f"Magenta: {class_j} | Green: {class_r} | "
                "Size: DP strength"
            )

        else:
            if use_direction_colors and class_j is not None and class_r is not None:
                legend_text = f"Magenta: {class_j} | Green: {class_r}"
            else:
                legend_text = "Size: importance"

        ax.text(
            0.5,
            0.845,
            legend_text,
            fontsize=self.legend_font_size,
            ha="center",
            va="top",
            color="#333333",
            transform=ax.transAxes
        )

        # Línea horizontal separadora
        ax.plot(
            [0.04, 0.96],
            [0.785, 0.785],
            color="#DDDDDD",
            linewidth=1.0,
            transform=ax.transAxes
        )

        # ====================================================
        # Bloque de tokens
        # ====================================================

        x = 0.055
        y = 0.705

        x_max = 0.945
        y_min = 0.070

        current_line_height = 0.0

        for pos, (_, row) in enumerate(plot_df.iterrows()):

            token = str(row[token_col])
            font_size = float(font_sizes[pos])

            if color_metric_col is not None:
                color_value = float(row[color_metric_col])
            else:
                color_value = None

            token_width = self._estimate_token_width(
                token=token,
                font_size=font_size
            )

            token_height = self._estimate_line_height(font_size)

            if x + token_width > x_max:
                x = 0.055
                y = y - max(current_line_height, 0.060)
                current_line_height = 0.0

            if y < y_min:
                print(
                    "Advertencia: no todos los tokens caben en la figura. "
                    "Reduce top_k o aumenta fig_height."
                )
                break

            color = self._get_token_color(
                color_value=color_value,
                use_direction_colors=use_direction_colors
            )

            ax.text(
                x,
                y,
                token,
                fontsize=font_size,
                ha="left",
                va="center",
                color=color,
                transform=ax.transAxes
            )

            x = x + token_width
            current_line_height = max(current_line_height, token_height)

        output_path = os.path.join(self.output_dir, output_name)

        plt.tight_layout()
        self._save_png_and_pdf(output_path)
        plt.close()

        print(f"Figura para artículo guardada en: {output_path}")
    # ========================================================
    # Columnas candidatas por métrica
    # ========================================================

    def _get_mean_abs_col(self, df_global):
        return self._find_column(
            df=df_global,
            candidates=[
                "mean_abs_deltaS",
                "mean_abs",
                "mean_abs_delta",
                "mean_abs_delta_s",
                "mean_abs_delta_shap",
                "mean_abs_delta_shap_glcd",
                "mean_abs_glcd",
                "mean_abs_value",
                "mean_abs_ds",
                "mean_abs_DeltaS",
                "mean_abs_ΔS"
            ],
            metric_name="Mean |ΔS|"
        )

    def _get_median_abs_col(self, df_global):
        return self._find_column(
            df=df_global,
            candidates=[
                "median_abs_deltaS",
                "median_abs",
                "median_abs_delta",
                "median_abs_delta_s",
                "median_abs_delta_shap",
                "median_abs_delta_shap_glcd",
                "median_abs_glcd",
                "median_abs_value",
                "median_abs_ds",
                "median_abs_DeltaS",
                "median_abs_ΔS"
            ],
            metric_name="Median |ΔS|"
        )

    def _get_signed_mean_col(self, df_global):
        return self._find_column(
            df=df_global,
            candidates=[
                "signed_mean_deltaS",
                "signed_mean",
                "signed_mean_delta",
                "signed_mean_delta_s",
                "signed_mean_delta_shap",
                "signed_mean_delta_shap_glcd",
                "mean_delta",
                "mean_delta_s",
                "mean_delta_shap",
                "mean_DeltaS",
                "mean_ΔS"
            ],
            metric_name="Signed Mean ΔS"
        )

    def _get_cdm_strength_col(self, df_global):
        return self._find_column(
            df=df_global,
            candidates=[
                "cdm_strength",
                "cdm_abs",
                "abs_cdm",
                "cdm_magnitude"
            ],
            metric_name="CDM strength"
        )

    def _get_cdm_value_col(self, df_global):
        return self._find_column(
            df=df_global,
            candidates=[
                "cdm_value",
                "cdm",
                "signed_cdm"
            ],
            metric_name="CDM value"
        )

    def _get_dp_strength_col(self, df_global):
        return self._find_column(
            df=df_global,
            candidates=[
                "dp_strength",
                "dominance_margin",
                "dp",
                "DP"
            ],
            metric_name="DP strength"
        )

    def _get_dp_signed_col(self, df_global):
        return self._find_column(
            df=df_global,
            candidates=[
                "dp_signed_value",
                "dp_signed",
                "signed_dp",
                "dp_direction"
            ],
            metric_name="DP signed value"
        )

    # ========================================================
    # Métodos públicos
    # ========================================================

    def plot_mean_abs(
        self,
        df_global,
        class_j,
        class_r,
        top_k=35
    ):
        metric_col = self._get_mean_abs_col(df_global)

        title = f"Global Mean |ΔS| ({class_j}/{class_r})"

        self._plot_token_block_article(
            df_global=df_global,
            size_metric_col=metric_col,
            color_metric_col=None,
            title=title,
            output_name="26_article_word_block_mean_abs.png",
            top_k=top_k,
            use_abs_for_ranking=True,
            use_direction_colors=False
        )

    def plot_median_abs(
        self,
        df_global,
        class_j,
        class_r,
        top_k=35
    ):
        metric_col = self._get_median_abs_col(df_global)

        title = f"Global Median |ΔS| ({class_j}/{class_r})"

        self._plot_token_block_article(
            df_global=df_global,
            size_metric_col=metric_col,
            color_metric_col=None,
            title=title,
            output_name="27_article_word_block_median_abs.png",
            top_k=top_k,
            use_abs_for_ranking=True,
            use_direction_colors=False
        )

    def plot_signed_mean(
        self,
        df_global,
        class_j,
        class_r,
        top_k=35
    ):
        metric_col = self._get_signed_mean_col(df_global)

        title = f"Global Signed Mean ΔS ({class_j}/{class_r})"

        self._plot_token_block_article(
            df_global=df_global,
            size_metric_col=metric_col,
            color_metric_col=metric_col,
            title=title,
            output_name="28_article_word_block_signed_mean.png",
            top_k=top_k,
            use_abs_for_ranking=True,
            use_direction_colors=True,
            class_j=class_j,
            class_r=class_r
        )

    def plot_cdm(
        self,
        df_global,
        class_j,
        class_r,
        top_k=35
    ):
        cdm_strength_col = self._get_cdm_strength_col(df_global)
        cdm_value_col = self._get_cdm_value_col(df_global)

        title = f"Conditional Difference Mean ({class_j}/{class_r})"

        self._plot_token_block_article(
            df_global=df_global,
            size_metric_col=cdm_strength_col,
            color_metric_col=cdm_value_col,
            title=title,
            output_name="29_article_word_block_cdm.png",
            top_k=top_k,
            use_abs_for_ranking=True,
            use_direction_colors=True,
            class_j=class_j,
            class_r=class_r
        )

    def plot_dp(
        self,
        df_global,
        class_j,
        class_r,
        top_k=35
    ):
        """
        DP:
        tamaño = dp_strength
        color = signo de dp_signed_value
        """

        dp_strength_col = self._get_dp_strength_col(df_global)
        dp_signed_col = self._get_dp_signed_col(df_global)

        title = f"Weighted dominance map ({class_j}/{class_r})"

        self._plot_token_block_article(
            df_global=df_global,
            size_metric_col=dp_strength_col,
            color_metric_col=dp_signed_col,
            title=title,
            output_name="30_article_word_block_dp.png",
            top_k=top_k,
            use_abs_for_ranking=True,
            use_direction_colors=True,
            class_j=class_j,
            class_r=class_r,
            exclude_tokens=[
                "only",
                "open"
            ]
        )

    def plot_all(
        self,
        df_global,
        class_j,
        class_r,
        top_k=35,
        include_signed_mean=True,
        include_cdm=True,
        include_dp=True
    ):
        self.plot_mean_abs(
            df_global=df_global,
            class_j=class_j,
            class_r=class_r,
            top_k=top_k
        )

        self.plot_median_abs(
            df_global=df_global,
            class_j=class_j,
            class_r=class_r,
            top_k=top_k
        )

        if include_signed_mean:
            self.plot_signed_mean(
                df_global=df_global,
                class_j=class_j,
                class_r=class_r,
                top_k=top_k
            )

        if include_cdm:
            self.plot_cdm(
                df_global=df_global,
                class_j=class_j,
                class_r=class_r,
                top_k=top_k
            )

        if include_dp:
            self.plot_dp(
                df_global=df_global,
                class_j=class_j,
                class_r=class_r,
                top_k=top_k
            )