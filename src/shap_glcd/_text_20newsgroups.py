# ============================================================
# Clase_Global_Text_SHAP_GLCD_20Newsgroups.py
# ============================================================
# Adaptación de SHAP-GLCD global para 20 Newsgroups + RoBERTa.
#
# Esta clase hereda de SHAPGlobalTextGLCD y conserva exactamente:
#   - la definición de deltaS = SHAP(Cj) - SHAP(Cr);
#   - las métricas globales mean |deltaS|, median |deltaS|,
#     signed mean, CDM y DP;
#   - los nombres de columnas requeridos por las figuras globales.
#
# La única diferencia metodológica es la reconstrucción segura de
# palabras a partir de subpalabras de RoBERTa (byte-level BPE).
# También conserva compatibilidad con WordPiece y SentencePiece.
#
# Convención RoBERTa:
#   - Ġword indica el comienzo de una palabra nueva.
#   - un token sin Ġ continúa la palabra anterior.
#
# Seguridad:
#   - se vuelve a tokenizar el texto original;
#   - esa tokenización solo se usa cuando su longitud coincide
#     exactamente con la primera dimensión de los valores SHAP;
#   - si no coincide, se usa la información de SHAP sin realizar
#     uniones agresivas.
# ============================================================

from __future__ import annotations

import html as html_lib
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ._text_global import SHAPGlobalTextGLCD


class SHAPGlobalTextGLCD20Newsgroups(SHAPGlobalTextGLCD):
    """
    SHAP-GLCD global para 20 Newsgroups con reconstrucción de
    subpalabras compatible con RoBERTa.

    La agregación y las métricas globales se heredan de
    ``SHAPGlobalTextGLCD``. Esta clase únicamente reemplaza la
    preparación de tokens y valores SHAP por instancia.
    """

    def __init__(
        self,
        lcd_explainer,
        output_dir: str = "results_shap_glcd_20newsgroups_roberta_base",
        min_token_frequency: int = 2,
        normalize_token_case: bool = True,
        combine_subwords: bool = True,
        remove_punctuation_only: bool = True,
        verbose_alignment: bool = False,
    ):
        super().__init__(
            lcd_explainer=lcd_explainer,
            output_dir=output_dir,
            min_token_frequency=min_token_frequency,
        )

        self.normalize_token_case = bool(normalize_token_case)
        self.combine_subwords = bool(combine_subwords)
        self.remove_punctuation_only = bool(remove_punctuation_only)
        self.verbose_alignment = bool(verbose_alignment)

        self._alignment_exact_count = 0
        self._alignment_fallback_count = 0

    # ============================================================
    # Utilidades de limpieza
    # ============================================================

    @staticmethod
    def _is_special_raw_token(token: Any) -> bool:
        token_lower = str(token).strip().lower()

        return token_lower in {
            "[cls]", "[sep]", "[pad]", "[mask]",
            "<s>", "</s>", "<pad>", "<mask>",
            "<unk>", "[unk]",
        }

    @staticmethod
    def _is_punctuation_only(token: str) -> bool:
        token = str(token).strip()

        if token == "":
            return True

        punctuation = set(
            ".,;:!?()[]{}\"'`´“”‘’-/\\|_+=*&^%$#@~…"
        )

        return all(character in punctuation for character in token)

    @staticmethod
    def _strip_outer_punctuation(token: str) -> str:
        """
        Elimina puntuación únicamente en los extremos.

        Conserva signos internos útiles, por ejemplo:
            bd's, can't, state-of-the-art
        """

        punctuation = ".,;:!?()[]{}\"`´“”‘’/\\|_+=*&^%$#@~…"
        return str(token).strip().strip(punctuation).strip()

    def _normalize_reconstructed_token(self, token: Any) -> str:
        token = html_lib.unescape(str(token))

        token = token.replace("&amp;", "&")
        token = token.replace("&quot;", '"')
        token = token.replace("&#39;", "'")
        token = token.replace("# 39 ;", "'")

        token = token.replace("Ġ", " ")
        token = token.replace("▁", " ")
        token = token.replace("##", "")

        token = " ".join(token.split())
        token = self._strip_outer_punctuation(token)

        if self.normalize_token_case:
            token = token.lower()

        return token.strip()

    def _is_valid_reconstructed_token(self, token: str) -> bool:
        token = str(token).strip()
        token_lower = token.lower()

        if token == "":
            return False

        if self._is_special_raw_token(token):
            return False

        if self.remove_punctuation_only and self._is_punctuation_only(token):
            return False

        # Excluye direcciones de correo electrónico reconstruidas como token.
        # Se usa una regla general y no una dirección específica.
        if "@" in token:
            return False

        # Excluye URL explícitas, que corresponden a metadatos o enlaces
        # y no a vocabulario semántico del dominio.
        if token_lower.startswith(("http://", "https://", "www.")):
            return False

        # Mantiene el criterio del flujo global original: se eliminan
        # piezas aisladas de un único carácter.
        if len(token) <= 1:
            return False

        noisy_tokens = {"amp", "quot", "39", "nbsp"}
        if token_lower in noisy_tokens:
            return False

        return True

    # ============================================================
    # Extracción de tokens y valores SHAP
    # ============================================================

    def _extract_raw_tokens_from_shap(self, instance_idx: int) -> List[str]:
        shap_values = self.lcd_explainer.shap_values

        try:
            data_i = shap_values[instance_idx].data
        except Exception:
            data_i = shap_values.data[instance_idx]

        if isinstance(data_i, str):
            return [data_i]

        try:
            return [str(token) for token in list(data_i)]
        except Exception:
            return [str(data_i)]

    def _retokenize_for_shap_alignment(
        self,
        instance_idx: int,
        target_length: int,
    ) -> Optional[List[str]]:
        """
        Recupera tokens del tokenizador con sus marcadores originales.

        La tokenización solo se acepta cuando su longitud coincide
        exactamente con el número de filas de SHAP. Esto evita unir
        palabras completas cuando SHAP devuelve segmentos diferentes.
        """

        tokenizer = getattr(self.lcd_explainer, "tokenizer", None)
        texts_eval = getattr(self.lcd_explainer, "texts_eval", None)

        if tokenizer is None or texts_eval is None:
            return None

        try:
            text = str(texts_eval[instance_idx])
        except Exception:
            return None

        for add_special_tokens in (True, False):
            try:
                encoded = tokenizer(
                    text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=self.lcd_explainer.max_length,
                )

                input_ids = encoded["input_ids"]
                tokens = tokenizer.convert_ids_to_tokens(input_ids)

                if len(tokens) == int(target_length):
                    return [str(token) for token in tokens]

            except Exception:
                continue

        return None

    def _extract_values_from_instance(self, instance_idx: int) -> np.ndarray:
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
                f"{values.shape}. Se esperaba (n_tokens, n_clases)."
            )

        return np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    @staticmethod
    def _detect_subword_scheme(tokens_raw: Sequence[str]) -> str:
        token_strings = [str(token) for token in tokens_raw]

        if any(token.strip().startswith("##") for token in token_strings):
            return "wordpiece"

        if any(token.strip().startswith("Ġ") for token in token_strings):
            return "byte_bpe"

        if any(token.strip().startswith("▁") for token in token_strings):
            return "sentencepiece"

        return "none"

    @staticmethod
    def _starts_new_word(raw_token: str, scheme: str) -> bool:
        raw_token = str(raw_token)
        raw_strip = raw_token.strip()

        if scheme == "wordpiece":
            return not raw_strip.startswith("##")

        if scheme == "byte_bpe":
            return (
                raw_strip.startswith("Ġ")
                or (len(raw_token) > 0 and raw_token[0].isspace())
            )

        if scheme == "sentencepiece":
            return (
                raw_strip.startswith("▁")
                or (len(raw_token) > 0 and raw_token[0].isspace())
            )

        # Fallback conservador: cada segmento se trata como token completo.
        return True

    def _tokens_to_text(self, raw_tokens: Sequence[str], scheme: str) -> str:
        tokenizer = getattr(self.lcd_explainer, "tokenizer", None)

        if tokenizer is not None:
            try:
                reconstructed = tokenizer.convert_tokens_to_string(
                    list(raw_tokens)
                )
                return self._normalize_reconstructed_token(reconstructed)
            except Exception:
                pass

        pieces = []

        for raw_token in raw_tokens:
            piece = str(raw_token)
            piece = piece.replace("Ġ", "")
            piece = piece.replace("▁", "")
            piece = piece.replace("##", "")
            pieces.append(piece)

        if scheme in {"wordpiece", "byte_bpe", "sentencepiece"}:
            reconstructed = "".join(pieces)
        else:
            reconstructed = " ".join(pieces)

        return self._normalize_reconstructed_token(reconstructed)

    # ============================================================
    # Reconstrucción por instancia
    # ============================================================

    def _reconstruct_tokens_and_values(
        self,
        tokens_raw: Sequence[str],
        values: np.ndarray,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Reconstruye palabras y suma los vectores SHAP de sus subpalabras.

        Returns
        -------
        tokens:
            Palabras normalizadas.
        aggregated_values:
            Matriz (n_palabras, n_clases), donde cada fila es la suma
            de los valores SHAP de las subpalabras que forman la palabra.
        """

        n = min(len(tokens_raw), values.shape[0])
        tokens_raw = list(tokens_raw[:n])
        values = values[:n, :]

        scheme = self._detect_subword_scheme(tokens_raw)

        # Sin marcadores fiables no se realizan uniones agresivas.
        if not self.combine_subwords or scheme == "none":
            tokens = []
            value_rows = []

            for raw_token, value_row in zip(tokens_raw, values):
                if self._is_special_raw_token(raw_token):
                    continue

                token = self._normalize_reconstructed_token(raw_token)

                if not self._is_valid_reconstructed_token(token):
                    continue

                tokens.append(token)
                value_rows.append(np.asarray(value_row, dtype=float))

            if not value_rows:
                return [], np.empty((0, values.shape[1]), dtype=float)

            return tokens, np.vstack(value_rows)

        reconstructed_tokens: List[str] = []
        reconstructed_values: List[np.ndarray] = []

        current_raw_tokens: List[str] = []
        current_value: Optional[np.ndarray] = None

        def flush_current_group() -> None:
            nonlocal current_raw_tokens, current_value

            if not current_raw_tokens or current_value is None:
                current_raw_tokens = []
                current_value = None
                return

            token = self._tokens_to_text(
                raw_tokens=current_raw_tokens,
                scheme=scheme,
            )

            if self._is_valid_reconstructed_token(token):
                reconstructed_tokens.append(token)
                reconstructed_values.append(current_value.copy())

            current_raw_tokens = []
            current_value = None

        for raw_token, value_row in zip(tokens_raw, values):
            raw_token = str(raw_token)
            value_row = np.asarray(value_row, dtype=float)

            if self._is_special_raw_token(raw_token):
                flush_current_group()
                continue

            starts_new_word = self._starts_new_word(
                raw_token=raw_token,
                scheme=scheme,
            )

            if current_raw_tokens and starts_new_word:
                flush_current_group()

            current_raw_tokens.append(raw_token)

            if current_value is None:
                current_value = value_row.copy()
            else:
                current_value = current_value + value_row

        flush_current_group()

        if not reconstructed_values:
            return [], np.empty((0, values.shape[1]), dtype=float)

        return reconstructed_tokens, np.vstack(reconstructed_values)

    def _get_instance_values_and_tokens(
        self,
        instance_idx: int,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Método utilizado por la clase base para construir los registros deltaS.
        """

        values = self._extract_values_from_instance(instance_idx)

        aligned_tokens = self._retokenize_for_shap_alignment(
            instance_idx=instance_idx,
            target_length=values.shape[0],
        )

        if aligned_tokens is not None:
            tokens_raw = aligned_tokens
            self._alignment_exact_count += 1
        else:
            tokens_raw = self._extract_raw_tokens_from_shap(instance_idx)
            self._alignment_fallback_count += 1

            if self.verbose_alignment:
                print(
                    "Advertencia de alineación: se usará la segmentación "
                    f"almacenada por SHAP para la instancia {instance_idx}."
                )

        tokens, aggregated_values = self._reconstruct_tokens_and_values(
            tokens_raw=tokens_raw,
            values=values,
        )

        if len(tokens) != aggregated_values.shape[0]:
            raise RuntimeError(
                "La reconstrucción produjo cantidades incompatibles: "
                f"{len(tokens)} tokens y {aggregated_values.shape[0]} "
                "filas SHAP."
            )

        return tokens, aggregated_values

    # ============================================================
    # Diagnóstico opcional
    # ============================================================

    def get_alignment_summary(self) -> dict:
        total = self._alignment_exact_count + self._alignment_fallback_count

        return {
            "exact_alignment_instances": int(self._alignment_exact_count),
            "fallback_instances": int(self._alignment_fallback_count),
            "total_processed_instances": int(total),
            "exact_alignment_rate": (
                float(self._alignment_exact_count / total)
                if total > 0
                else 0.0
            ),
        }

    def print_alignment_summary(self) -> None:
        summary = self.get_alignment_summary()

        print("\nResumen de alineación de tokens:")
        print(
            "  Alineación exacta tokenizador-SHAP: "
            f"{summary['exact_alignment_instances']}"
        )
        print(
            "  Fallback conservador: "
            f"{summary['fallback_instances']}"
        )
        print(
            "  Tasa de alineación exacta: "
            f"{summary['exact_alignment_rate']:.3f}"
        )


# Alias corto opcional.
SHAPGlobalTextGLCD20NG = SHAPGlobalTextGLCD20Newsgroups
