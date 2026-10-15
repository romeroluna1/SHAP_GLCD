# -*- coding: utf-8 -*-

# ============================================================
# Clase_Global_Text_TwoColumn_Ranked_Article.py
# ============================================================
# Visualizador global para texto en formato de dos columnas
# ordenadas por ranking de importancia.
#
# Esta versión conserva el estilo visual de la clase tipo bloque:
#   - tamaño de figura 12 x 7.2;
#   - título de 18;
#   - leyenda de 12;
#   - rango de fuente 12-38;
#   - fondo blanco;
#   - encabezado compacto;
#   - línea horizontal separando encabezado y palabras.
#
# La lógica de selección de tokens se hereda de:
#   SHAPGlobalTextTokenSizeArticleVisualizer
#
# Por tanto se conserva:
#   - selección de tokens;
#   - ranking;
#   - desempate por frecuencia;
#   - normalización de tamaño con raíz cuadrada;
#   - columnas de métricas;
#   - colores direccionales.
#
# Solo cambia la forma de presentar:
#   - antes: bloque de palabras;
#   - ahora: dos columnas por ranking global.
#
# Regla visual:
#   - columna izquierda: primera mitad del ranking;
#   - columna derecha: segunda mitad del ranking;
#   - en métricas direccionales, el color indica clase favorecida.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from ._text_token_visualizer import (
    SHAPGlobalTextTokenSizeArticleVisualizer
)


class SHAPGlobalTextTwoColumnRankedArticleVisualizer(
    SHAPGlobalTextTokenSizeArticleVisualizer
):
    """
    Visualizador global de tokens en dos columnas ordenadas por importancia.

    Hereda la lógica de preparación de datos de la clase de bloque para que
    los tokens seleccionados sean consistentes con las figuras anteriores.

    La diferencia está solo en la distribución visual:
        - las columnas no separan por clase;
        - las columnas siguen el ranking de mayor a menor importancia;
        - el color indica dirección solo en métricas direccionales.
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
        show_rank_numbers=False,
        title_fontsize=None,
        legend_fontsize=None,
        header_line_color="lightgray",
        header_line_width=1.2,
        column_line_color="lightgray",
        column_line_width=1.0
    ):
        """
        Mantiene compatibilidad con title_font_size/title_fontsize
        y legend_font_size/legend_fontsize.
        """

        if title_fontsize is not None:
            title_font_size = title_fontsize

        if legend_fontsize is not None:
            legend_font_size = legend_fontsize

        super().__init__(
            output_dir=output_dir,
            token_col=token_col,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            title_font_size=title_font_size,
            legend_font_size=legend_font_size,
            dpi=dpi,
            neutral_color=neutral_color,
            cj_color=cj_color,
            cr_color=cr_color,
            fig_width=fig_width,
            fig_height=fig_height
        )

        self.show_rank_numbers = show_rank_numbers

        self.header_line_color = header_line_color
        self.header_line_width = header_line_width
        self.column_line_color = column_line_color
        self.column_line_width = column_line_width

    # ========================================================
    # Utilidades propias
    # ========================================================

    def _safe_name(self, text):
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

    def _pair_suffix(self, class_j, class_r):
        return f"{self._safe_name(class_j)}_vs_{self._safe_name(class_r)}"

    def _save_ranked_png_and_pdf(self, output_path):
        """
        Guarda PNG y PDF en alta resolución.
        Se usa dpi mínimo de 600 para conservar nitidez en artículo.
        """

        save_dpi = max(int(self.dpi), 600)

        plt.savefig(
            output_path,
            dpi=save_dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white"
        )

        if output_path.lower().endswith(".png"):
            pdf_path = output_path[:-4] + ".pdf"
        else:
            pdf_path = output_path + ".pdf"

        plt.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white"
        )

    def _build_subtitle(self, class_j, class_r):
        if class_j is None or class_r is None:
            return "Aggregated over evaluation instances"

        return (
            f"Aggregated over evaluation instances | "
            f"ΔS = SHAP({class_j}) - SHAP({class_r})"
        )

    # ========================================================
    # Render principal ranked two-column
    # ========================================================

    def _plot_ranked_two_column(
        self,
        df_global,
        size_metric_col,
        title,
        output_name,
        top_k=20,
        legend_text=None,
        color_metric_col=None,
        use_abs_for_ranking=True,
        use_direction_colors=False,
        class_j=None,
        class_r=None,
        exclude_tokens=None
    ):
        """
        Dibuja los tokens en dos columnas siguiendo el ranking global.

        La preparación de datos se hace con self._prepare_plot_df(...),
        heredado de la clase de bloque. Esto asegura consistencia con
        las figuras tipo bloque.
        """

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

        # Misma lógica de tamaños de la clase bloque.
        font_sizes = self._compute_font_sizes(
            plot_df["ranking_value"].values
        )

        plot_df = plot_df.copy()
        plot_df["_font_size"] = font_sizes
        plot_df["_rank"] = np.arange(1, len(plot_df) + 1)

        n_tokens = len(plot_df)
        split_idx = int(np.ceil(n_tokens / 2.0))

        left_df = plot_df.iloc[:split_idx].copy()
        right_df = plot_df.iloc[split_idx:].copy()

        fig = plt.figure(
            figsize=(self.fig_width, self.fig_height),
            dpi=self.dpi,
            facecolor="white"
        )

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("white")

        # ----------------------------------------------------
        # Encabezado: título, subtítulo, leyenda y línea
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Encabezado con mayor separación vertical
        # ----------------------------------------------------

        title_y = 0.975
        subtitle_y = 0.910
        legend_y = 0.850
        header_line_y = 0.795

        ax.text(
            0.5,
            title_y,
            title,
            fontsize=self.title_font_size,
            fontweight="normal",
            ha="center",
            va="center",
            color="black",
            transform=ax.transAxes
        )

        ax.text(
            0.5,
            subtitle_y,
            self._build_subtitle(class_j, class_r),
            fontsize=self.legend_font_size,
            ha="center",
            va="center",
            color="black",
            transform=ax.transAxes
        )

        if legend_text is not None:
            ax.text(
                0.5,
                legend_y,
                legend_text,
                fontsize=self.legend_font_size,
                ha="center",
                va="center",
                color="black",
                transform=ax.transAxes
            )

        # Línea horizontal separando encabezado y palabras.
        ax.hlines(
            y=header_line_y,
            xmin=0.06,
            xmax=0.94,
            color=self.header_line_color,
            linewidth=self.header_line_width,
            alpha=0.85,
            transform=ax.transAxes
        )

        # ----------------------------------------------------
        # Área de palabras
        # ----------------------------------------------------

        top_y = 0.705
        bottom_y = 0.095

        # Línea vertical suave entre columnas, solo en área de palabras.
        ax.axvline(
            0.5,
            ymin=bottom_y - 0.01,
            ymax=header_line_y - 0.035,
            color=self.column_line_color,
            linewidth=self.column_line_width,
            alpha=0.65
        )

        def y_positions(n_items):
            if n_items <= 0:
                return []
            if n_items == 1:
                return [(top_y + bottom_y) / 2.0]
            return np.linspace(top_y, bottom_y, n_items)

        y_left = y_positions(len(left_df))
        y_right = y_positions(len(right_df))

        x_left = 0.25
        x_right = 0.75

        def draw_column(df_col, x_pos, y_values):
            for (_, row), y_pos in zip(df_col.iterrows(), y_values):

                token = str(row[token_col])
                font_size = float(row["_font_size"])

                if color_metric_col is not None:
                    color_value = float(row[color_metric_col])
                else:
                    color_value = None

                color = self._get_token_color(
                    color_value=color_value,
                    use_direction_colors=use_direction_colors
                )

                if self.show_rank_numbers:
                    text_to_draw = f"{int(row['_rank'])}. {token}"
                else:
                    text_to_draw = token

                ax.text(
                    x_pos,
                    y_pos,
                    text_to_draw,
                    fontsize=font_size,
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    transform=ax.transAxes
                )

        draw_column(left_df, x_left, y_left)
        draw_column(right_df, x_right, y_right)

        output_path = os.path.join(self.output_dir, output_name)

        self._save_ranked_png_and_pdf(output_path)
        plt.close()

        print(f"Figura ranked two-column guardada en: {output_path}")

    # ========================================================
    # Métodos públicos
    # ========================================================

    def plot_mean_abs(
        self,
        df_global,
        class_j,
        class_r,
        top_k=20
    ):
        """
        Mean |ΔS|:
        Dos columnas ranked.
        Tamaño = mean_abs_deltaS.
        Color = neutral.
        """

        metric_col = self._get_mean_abs_col(df_global)

        title = f"Global Mean |ΔS| ({class_j}/{class_r})"

        legend = "Size: mean |ΔS|"

        output_name = (
            f"26_article_two_column_ranked_mean_abs_"
            f"{self._pair_suffix(class_j, class_r)}.png"
        )

        self._plot_ranked_two_column(
            df_global=df_global,
            size_metric_col=metric_col,
            color_metric_col=None,
            title=title,
            legend_text=legend,
            output_name=output_name,
            top_k=top_k,
            use_abs_for_ranking=True,
            use_direction_colors=False,
            class_j=class_j,
            class_r=class_r,
            exclude_tokens=None
        )

    def plot_median_abs(
        self,
        df_global,
        class_j,
        class_r,
        top_k=20
    ):
        """
        Median |ΔS|:
        Dos columnas ranked.
        Tamaño = median_abs_deltaS.
        Color = neutral.
        """

        metric_col = self._get_median_abs_col(df_global)

        title = f"Global Median |ΔS| ({class_j}/{class_r})"

        legend = "Size: median |ΔS|"

        output_name = (
            f"27_article_two_column_ranked_median_abs_"
            f"{self._pair_suffix(class_j, class_r)}.png"
        )

        self._plot_ranked_two_column(
            df_global=df_global,
            size_metric_col=metric_col,
            color_metric_col=None,
            title=title,
            legend_text=legend,
            output_name=output_name,
            top_k=top_k,
            use_abs_for_ranking=True,
            use_direction_colors=False,
            class_j=class_j,
            class_r=class_r,
            exclude_tokens=None
        )

    def plot_signed_mean(
        self,
        df_global,
        class_j,
        class_r,
        top_k=20
    ):
        """
        Signed Mean ΔS:
        Dos columnas ranked.
        Tamaño = |signed_mean_deltaS|.
        Color = dirección según signed_mean_deltaS.
        """

        metric_col = self._get_signed_mean_col(df_global)

        title = f"Global Signed Mean ΔS ({class_j}/{class_r})"

        legend = (
            f"Green: {class_r}   |   Magenta: {class_j}   |   "
            "Size: |signed mean ΔS|"
        )

        output_name = (
            f"28_article_two_column_ranked_signed_mean_"
            f"{self._pair_suffix(class_j, class_r)}.png"
        )

        self._plot_ranked_two_column(
            df_global=df_global,
            size_metric_col=metric_col,
            color_metric_col=metric_col,
            title=title,
            legend_text=legend,
            output_name=output_name,
            top_k=top_k,
            use_abs_for_ranking=True,
            use_direction_colors=True,
            class_j=class_j,
            class_r=class_r,
            exclude_tokens=None
        )

    def plot_cdm(
        self,
        df_global,
        class_j,
        class_r,
        top_k=20
    ):
        """
        Conditional Difference Mean:
        Dos columnas ranked.
        Tamaño = cdm_strength.
        Color = dirección según cdm_value.
        """

        cdm_strength_col = self._get_cdm_strength_col(df_global)
        cdm_value_col = self._get_cdm_value_col(df_global)

        title = f"Conditional  Mean ΔS ({class_j}/{class_r})"

        legend = (
            f"Green: {class_r}   |   Magenta: {class_j}   |   "
            "Size: CdM strength"
        )

        output_name = (
            f"29_article_two_column_ranked_cdm_"
            f"{self._pair_suffix(class_j, class_r)}.png"
        )

        self._plot_ranked_two_column(
            df_global=df_global,
            size_metric_col=cdm_strength_col,
            color_metric_col=cdm_value_col,
            title=title,
            legend_text=legend,
            output_name=output_name,
            top_k=top_k,
            use_abs_for_ranking=True,
            use_direction_colors=True,
            class_j=class_j,
            class_r=class_r,
            exclude_tokens=None
        )

    def plot_dp(
        self,
        df_global,
        class_j,
        class_r,
        top_k=20
    ):
        """
        Weighted Dominance Plot:
        Dos columnas ranked.
        Tamaño = dp_strength.
        Color = dirección según dp_signed_value.

        Se conserva el mismo filtro que se usaba en la figura tipo bloque
        de DP.
        """

        dp_strength_col = self._get_dp_strength_col(df_global)
        dp_signed_col = self._get_dp_signed_col(df_global)

        title = f"Weighted Dominance  ({class_j}/{class_r})"

        legend = (
            f"Green: {class_r}   |   Magenta: {class_j}   |   "
            "Size: DP strength"
        )

        output_name = (
            f"30_article_two_column_ranked_dp_"
            f"{self._pair_suffix(class_j, class_r)}.png"
        )

        self._plot_ranked_two_column(
            df_global=df_global,
            size_metric_col=dp_strength_col,
            color_metric_col=dp_signed_col,
            title=title,
            legend_text=legend,
            output_name=output_name,
            top_k=top_k,
            use_abs_for_ranking=True,
            use_direction_colors=True,
            class_j=class_j,
            class_r=class_r,
            exclude_tokens=[
                "he",
                "had",
                "http",
                "https",
                "www",
                "only",
                "open",
                "en"
            ]
        )

    def plot_all(
        self,
        df_global,
        class_j,
        class_r,
        top_k=20,
        include_signed_mean=True,
        include_cdm=True,
        include_dp=True
    ):
        """
        Genera todas las figuras ranked two-column.
        """

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



