# ============================================================
# Clase_Metricas_Text_SHAP_LCD_FINAL_100926.py
# ============================================================
# Métricas de estabilidad y consistencia para explicaciones
# diferenciales SHAP-LCD / SHAP-GLCD en texto.
#
# Protocolo final del artículo:
#   - selección balanceada por etiqueta verdadera en el main;
#   - perturbación textual mediante token masking;
#   - Top-K sobre |DeltaS| para cada explicación;
#   - normalización L2 después de Top-K;
#   - Stability = distancia L2 cuadrática entre explicaciones;
#   - Cimp basada en implicación lógica con similitud coseno;
#   - Coverage y PredKeep como métricas auxiliares;
#   - semilla por instancia = seed + t, con t=0,...,N-1.
#
# IMPORTANTE:
# No se truncan ni rellenan artificialmente los vectores para
# hacerlos coincidir. Las perturbaciones se generan con el mismo
# shap.maskers.Text del explicador y se aceptan solo cuando la
# retokenización conserva exactamente las posiciones no enmascaradas.
# Además, las dimensiones SHAP deben coincidir entre original y
# perturbación.
# ============================================================

import math
import os

import numpy as np
import pandas as pd


class SHAPTextLCDMetrics:
    """
    Evalúa estabilidad y consistencia de explicaciones diferenciales
    entre pares de clases en modelos de clasificación de texto.

    La clase recibe un objeto ya inicializado de SHAPTextExplainerLCD.
    """

    def __init__(self, lcd_explainer):
        self.lcd_explainer = lcd_explainer
        self.model = lcd_explainer.model
        self.tokenizer = lcd_explainer.tokenizer
        self.class_names = lcd_explainer.class_names
        self.max_length = lcd_explainer.max_length

        if getattr(self.lcd_explainer, "explainer", None) is None:
            self.lcd_explainer._init_explainer()

    # ============================================================
    # Utilidades generales
    # ============================================================

    @staticmethod
    def _safe_filename(text):
        """Limpia un texto para usarlo como nombre de archivo."""
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
        """Convierte nombre de clase o índice a índice entero."""
        if isinstance(class_name_or_idx, (int, np.integer)):
            return int(class_name_or_idx)

        if self.class_names is None:
            raise ValueError("No hay class_names definido. Usa índices.")

        inverse = {
            class_name: class_idx
            for class_idx, class_name in self.class_names.items()
        }

        if class_name_or_idx not in inverse:
            raise ValueError(
                f"No se encontró la clase '{class_name_or_idx}'. "
                f"Clases disponibles: {list(inverse.keys())}"
            )

        return int(inverse[class_name_or_idx])

    def _class_label(self, class_idx):
        """Devuelve el nombre legible de una clase."""
        class_idx = int(class_idx)
        if self.class_names is None:
            return f"Class {class_idx}"
        return self.class_names[class_idx]

    @staticmethod
    def _cosine_similarity(vector_a, vector_b):
        """Calcula la similitud coseno entre dos vectores."""
        vector_a = np.asarray(vector_a, dtype=float)
        vector_b = np.asarray(vector_b, dtype=float)

        if vector_a.shape != vector_b.shape:
            raise ValueError(
                "La similitud coseno requiere vectores con la misma forma: "
                f"{vector_a.shape} vs {vector_b.shape}."
            )

        norm_a = np.linalg.norm(vector_a, ord=2)
        norm_b = np.linalg.norm(vector_b, ord=2)

        if norm_a == 0 and norm_b == 0:
            return 1.0
        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)
        return float(np.clip(similarity, -1.0, 1.0))

    @staticmethod
    def _prepare_explanation(vector, top_k=10, normalize="l2"):
        """
        Aplica el protocolo Top-K + normalización a un vector DeltaS.

        Top-K se determina de forma independiente para cada explicación
        usando |DeltaS|. Los componentes no seleccionados se fijan en 0,
        por lo que se conserva la posición original de cada token.
        """
        vector = np.asarray(vector, dtype=float).reshape(-1)
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        if top_k is not None:
            top_k = int(top_k)
            if top_k < 1:
                raise ValueError("top_k debe ser >= 1 o None.")

            if top_k < vector.size:
                # Orden estable para hacer deterministas los empates.
                ranking = np.argsort(-np.abs(vector), kind="mergesort")
                keep = ranking[:top_k]
                sparse = np.zeros_like(vector, dtype=float)
                sparse[keep] = vector[keep]
                vector = sparse

        if normalize is None:
            return vector

        normalize_str = str(normalize).strip().lower()

        if normalize_str in {"none", "no", "false"}:
            return vector

        if normalize_str != "l2":
            raise ValueError(
                "normalize debe ser 'l2', 'none' o None. "
                f"Valor recibido: {normalize}"
            )

        norm = np.linalg.norm(vector, ord=2)
        if norm > 0:
            vector = vector / norm

        return vector

    # ============================================================
    # Perturbaciones textuales con control de alineación
    # ============================================================

    def _get_text_masker(self):
        """
        Devuelve el mismo masker de texto utilizado por el explicador SHAP.

        Usar el mismo masker evita reconstruir manualmente una secuencia de
        WordPiece/BPE mediante decode/convert_tokens_to_string, operación que
        no es necesariamente invertible.
        """
        masker = getattr(self.lcd_explainer, "masker", None)
        if masker is None:
            raise RuntimeError(
                "El explicador SHAP-LCD no dispone de un masker de texto."
            )
        if not hasattr(masker, "token_segments"):
            raise RuntimeError(
                "El masker de texto no dispone de token_segments()."
            )
        return masker

    def _masker_tokenization(self, text):
        """Obtiene segmentos e IDs exactamente con el masker usado por SHAP."""
        masker = self._get_text_masker()
        segments, token_ids = masker.token_segments(text)

        token_ids = np.asarray(token_ids).reshape(-1)
        token_ids = [int(token_id) for token_id in token_ids]
        segments = list(segments)

        if len(segments) != len(token_ids):
            raise RuntimeError(
                "El masker devolvió diferente número de segmentos e IDs: "
                f"{len(segments)} vs {len(token_ids)}."
            )

        return segments, token_ids

    def _model_visible_positions(self, token_ids):
        """
        Devuelve las posiciones de contenido que pueden entrar al modelo.

        SHAP puede segmentar el texto completo, mientras que predict_proba()
        tokeniza con truncation=True y max_length=self.max_length. Por ello,
        las perturbaciones se restringen a posiciones de contenido que quedan
        dentro de la ventana efectiva del clasificador.
        """
        masker = self._get_text_masker()
        n_tokens = len(token_ids)

        keep_prefix = int(getattr(masker, "keep_prefix", 0) or 0)
        keep_suffix = int(getattr(masker, "keep_suffix", 0) or 0)

        content_start = keep_prefix
        content_end_full = max(content_start, n_tokens - keep_suffix)

        if self.max_length is None:
            content_end_active = content_end_full
        else:
            max_length = int(self.max_length)
            active_content_capacity = max(
                0,
                max_length - keep_prefix - keep_suffix,
            )
            content_end_active = min(
                content_end_full,
                content_start + active_content_capacity,
            )

        special_ids = {
            int(token_id)
            for token_id in getattr(self.tokenizer, "all_special_ids", [])
        }

        positions = [
            position
            for position in range(content_start, content_end_active)
            if int(token_ids[position]) not in special_ids
        ]

        return np.asarray(positions, dtype=int)

    def _mask_positions_and_check(self, text, original_ids, positions):
        """
        Enmascara posiciones mediante shap.maskers.Text y comprueba que la
        retokenización sea exactamente la esperada.

        La condición aceptada es fuerte: los IDs no enmascarados deben quedar
        idénticos y cada posición seleccionada debe convertirse exactamente en
        mask_token_id. Así se conserva un espacio de componentes posicionales
        común para comparar DeltaS sin pad ni truncate.
        """
        masker = self._get_text_masker()
        n_tokens = len(original_ids)

        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(self.tokenizer, "unk_token_id", None)
        if mask_token_id is None:
            raise RuntimeError(
                "El tokenizador no dispone de mask_token_id ni unk_token_id."
            )
        mask_token_id = int(mask_token_id)

        keep_mask = np.ones(n_tokens, dtype=bool)
        positions = np.asarray(positions, dtype=int)
        keep_mask[positions] = False

        masked_output = masker(keep_mask, text)
        if not isinstance(masked_output, tuple) or len(masked_output) == 0:
            raise RuntimeError(
                "El masker de SHAP devolvió una salida inesperada."
            )

        first_output = np.asarray(masked_output[0], dtype=object).reshape(-1)
        if first_output.size != 1:
            raise RuntimeError(
                "Se esperaba un único texto perturbado producido por el masker."
            )

        perturbed_text = str(first_output[0])
        _, retokenized_ids = self._masker_tokenization(perturbed_text)

        expected_ids = [int(token_id) for token_id in original_ids]
        for position in positions:
            expected_ids[int(position)] = mask_token_id

        exact_alignment = retokenized_ids == expected_ids

        return perturbed_text, exact_alignment, retokenized_ids, expected_ids

    def generate_token_mask_perturbation(
        self,
        text,
        perturbation_rate=0.05,
        rng=None,
        return_metadata=False,
        max_alignment_attempts=200,
    ):
        """
        Genera una perturbación local mediante token masking.

        La perturbación se construye con el mismo ``shap.maskers.Text`` usado
        por el explicador. Debido a que WordPiece/BPE no siempre admite una
        reconstrucción texto->IDs perfectamente invertible, se emplea rechazo
        de perturbaciones: solo se aceptan máscaras cuya retokenización
        conserva exactamente todos los IDs no modificados y sustituye por el
        token de máscara únicamente las posiciones seleccionadas.

        De esta forma se garantiza alineación posicional exacta sin recurrir a
        truncado o relleno artificial de los vectores SHAP.
        """
        if not 0 < perturbation_rate <= 1:
            raise ValueError(
                "perturbation_rate debe estar en el intervalo (0, 1]."
            )

        if rng is None:
            rng = np.random.default_rng(42)

        _, original_ids = self._masker_tokenization(text)

        if len(original_ids) == 0:
            metadata = {
                "original_ids": [],
                "perturbed_ids": [],
                "masked_positions": [],
                "n_eligible_tokens": 0,
                "n_safe_tokens": 0,
                "n_masked_tokens": 0,
                "alignment_attempts": 0,
            }
            return (text, metadata) if return_metadata else text

        eligible_positions = self._model_visible_positions(original_ids)

        if eligible_positions.size == 0:
            metadata = {
                "original_ids": original_ids,
                "perturbed_ids": original_ids.copy(),
                "masked_positions": [],
                "n_eligible_tokens": 0,
                "n_safe_tokens": 0,
                "n_masked_tokens": 0,
                "alignment_attempts": 0,
            }
            return (text, metadata) if return_metadata else text

        n_to_mask = int(
            math.ceil(perturbation_rate * int(eligible_positions.size))
        )
        n_to_mask = max(1, n_to_mask)
        n_to_mask = min(n_to_mask, int(eligible_positions.size))

        # --------------------------------------------------------
        # Identificar posiciones individualmente seguras.
        # --------------------------------------------------------
        # Esto evita seleccionar sistemáticamente sub-tokens cuya sustitución
        # altera la tokenización de sus vecinos. La comprobación es determinista
        # y no consume números aleatorios.
        safe_positions = []
        for position in eligible_positions:
            _, ok, _, _ = self._mask_positions_and_check(
                text=text,
                original_ids=original_ids,
                positions=[int(position)],
            )
            if ok:
                safe_positions.append(int(position))

        safe_positions = np.asarray(safe_positions, dtype=int)

        if safe_positions.size < n_to_mask:
            raise RuntimeError(
                "No existen suficientes posiciones de token que puedan "
                "enmascararse preservando exactamente la alineación. "
                f"Elegibles={eligible_positions.size}, "
                f"seguras={safe_positions.size}, requeridas={n_to_mask}."
            )

        # --------------------------------------------------------
        # Seleccionar la combinación y verificarla exactamente.
        # --------------------------------------------------------
        for attempt in range(1, int(max_alignment_attempts) + 1):
            positions = rng.choice(
                safe_positions,
                size=n_to_mask,
                replace=False,
            )
            positions = np.sort(np.asarray(positions, dtype=int))

            perturbed_text, ok, retokenized_ids, expected_ids = (
                self._mask_positions_and_check(
                    text=text,
                    original_ids=original_ids,
                    positions=positions,
                )
            )

            if ok:
                metadata = {
                    "original_ids": original_ids,
                    "perturbed_ids": retokenized_ids,
                    "expected_ids": expected_ids,
                    "masked_positions": [
                        int(position) for position in positions
                    ],
                    "n_eligible_tokens": int(eligible_positions.size),
                    "n_safe_tokens": int(safe_positions.size),
                    "n_masked_tokens": int(n_to_mask),
                    "alignment_attempts": int(attempt),
                }

                if return_metadata:
                    return perturbed_text, metadata
                return perturbed_text

        raise RuntimeError(
            "No fue posible generar una perturbación con alineación exacta "
            f"después de {max_alignment_attempts} intentos. "
            f"Tokens elegibles={eligible_positions.size}, "
            f"tokens seguros={safe_positions.size}, "
            f"tokens a enmascarar={n_to_mask}."
        )

    # ============================================================
    # Extracción robusta de explicaciones diferenciales
    # ============================================================

    def _get_values_for_position(self, shap_output, position):
        """Extrae los valores SHAP de una instancia de un lote."""
        try:
            instance_exp = shap_output[position]
            return instance_exp.values
        except Exception:
            pass

        values_all = shap_output.values

        if isinstance(values_all, (list, tuple)):
            return values_all[position]

        values_all = np.asarray(values_all)

        if values_all.dtype == object and values_all.ndim == 1:
            return values_all[position]
        if values_all.ndim == 3:
            return values_all[position]
        if values_all.ndim == 2:
            return values_all
        if values_all.ndim == 1:
            return values_all

        raise ValueError(f"Forma SHAP no esperada: {values_all.shape}")

    def _extract_exp_vector_from_shap_output(
        self,
        shap_output,
        position,
        class_j_idx,
        class_r_idx,
    ):
        """
        Extrae el vector diferencial:
            DeltaS = SHAP(C_j) - SHAP(C_r)
        """
        values = self._get_values_for_position(
            shap_output=shap_output,
            position=position,
        )
        values = np.asarray(values)

        if values.dtype == object:
            try:
                values = np.asarray(values.tolist(), dtype=float)
            except Exception:
                values = np.asarray(values, dtype=float)

        if values.ndim == 2:
            if values.shape[1] <= max(class_j_idx, class_r_idx):
                raise ValueError(
                    "No hay suficientes clases en los valores SHAP. "
                    f"Forma recibida: {values.shape}; clases requeridas: "
                    f"{class_j_idx}, {class_r_idx}."
                )
            exp_vector = (
                values[:, class_j_idx]
                - values[:, class_r_idx]
            )

        elif values.ndim == 1:
            exp_vector = values

        elif values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
            if values.shape[1] <= max(class_j_idx, class_r_idx):
                raise ValueError(
                    "No hay suficientes clases en los valores SHAP. "
                    f"Forma recibida: {values.shape}."
                )
            exp_vector = (
                values[:, class_j_idx]
                - values[:, class_r_idx]
            )

        else:
            raise ValueError(
                f"Forma SHAP por instancia no esperada: {values.shape}"
            )

        exp_vector = np.asarray(exp_vector, dtype=float).reshape(-1)
        return np.nan_to_num(
            exp_vector,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    # ============================================================
    # Métricas por instancia
    # ============================================================

    def evaluate_instance_metrics(
        self,
        text,
        class_j,
        class_r,
        M=40,
        perturbation_rate=0.05,
        tau=0.95,
        top_k=10,
        normalize="l2",
        seed=42,
    ):
        """
        Evalúa estabilidad y consistencia para una instancia textual.

        Stability:
            (1/M) sum_m ||Exp_K(x_i^(m)) - Exp_K(x_i)||_2^2

        donde Exp_K representa DeltaS después de Top-K y normalización.

        Consistency basada en implicación:
            cos(Exp_K(x_i), Exp_K(x_i^(m))) >= tau
            => y_hat(x_i) = y_hat(x_i^(m))
        """
        if M < 1:
            raise ValueError("M debe ser un entero mayor o igual a 1.")
        if not 0 < tau <= 1:
            raise ValueError("tau debe estar en el intervalo (0, 1].")

        class_j_idx = self._class_name_to_index(class_j)
        class_r_idx = self._class_name_to_index(class_r)

        rng = np.random.default_rng(seed)

        perturbed_texts = []
        perturbation_metadata = []

        for _ in range(M):
            perturbed_text, metadata = self.generate_token_mask_perturbation(
                text=text,
                perturbation_rate=perturbation_rate,
                rng=rng,
                return_metadata=True,
            )
            perturbed_texts.append(perturbed_text)
            perturbation_metadata.append(metadata)

        all_texts = [text, *perturbed_texts]

        # Predicciones original + perturbaciones.
        predictions = self.lcd_explainer.predict_labels(all_texts)
        base_prediction = int(predictions[0])
        perturbed_predictions = predictions[1:]

        # SHAP original + perturbaciones en el mismo lote.
        shap_output = self.lcd_explainer.explainer(all_texts)

        base_raw = self._extract_exp_vector_from_shap_output(
            shap_output=shap_output,
            position=0,
            class_j_idx=class_j_idx,
            class_r_idx=class_r_idx,
        )

        explanation_dimension = int(len(base_raw))

        base_explanation = self._prepare_explanation(
            base_raw,
            top_k=top_k,
            normalize=normalize,
        )

        stability_values = []
        consistency_values = []
        cosine_values = []
        coverage_values = []
        prediction_keep_values = []

        for perturbation_index in range(M):
            perturbed_raw = self._extract_exp_vector_from_shap_output(
                shap_output=shap_output,
                position=perturbation_index + 1,
                class_j_idx=class_j_idx,
                class_r_idx=class_r_idx,
            )

            # No se permite pad/truncate: la dimensión debe coincidir.
            if len(perturbed_raw) != explanation_dimension:
                metadata = perturbation_metadata[perturbation_index]
                raise RuntimeError(
                    "Desalineación detectada en los valores SHAP. "
                    f"Dimensión original={explanation_dimension}, "
                    f"dimensión perturbada={len(perturbed_raw)}, "
                    f"perturbación={perturbation_index}, "
                    f"posiciones enmascaradas={metadata['masked_positions']}."
                )

            perturbed_explanation = self._prepare_explanation(
                perturbed_raw,
                top_k=top_k,
                normalize=normalize,
            )

            squared_distance = float(
                np.sum((perturbed_explanation - base_explanation) ** 2)
            )
            stability_values.append(squared_distance)

            cosine_similarity = self._cosine_similarity(
                base_explanation,
                perturbed_explanation,
            )
            cosine_values.append(cosine_similarity)

            antecedent = cosine_similarity >= tau
            consequent = (
                base_prediction
                == int(perturbed_predictions[perturbation_index])
            )

            implication_holds = (not antecedent) or consequent

            consistency_values.append(int(implication_holds))
            coverage_values.append(int(antecedent))
            prediction_keep_values.append(int(consequent))

        stability_values = np.asarray(stability_values, dtype=float)
        consistency_values = np.asarray(consistency_values, dtype=float)
        cosine_values = np.asarray(cosine_values, dtype=float)
        coverage_values = np.asarray(coverage_values, dtype=float)
        prediction_keep_values = np.asarray(
            prediction_keep_values,
            dtype=float,
        )

        n_masked_values = np.asarray(
            [metadata["n_masked_tokens"] for metadata in perturbation_metadata],
            dtype=float,
        )
        n_safe_values = np.asarray(
            [metadata["n_safe_tokens"] for metadata in perturbation_metadata],
            dtype=float,
        )
        alignment_attempt_values = np.asarray(
            [metadata["alignment_attempts"] for metadata in perturbation_metadata],
            dtype=float,
        )

        return {
            "stability": float(np.mean(stability_values)),
            "consistency": float(np.mean(consistency_values)),
            "coverage_imp": float(np.mean(coverage_values)),
            "prediction_keep_imp": float(np.mean(prediction_keep_values)),
            "cosine_mean": float(np.mean(cosine_values)),
            "cosine_std": (
                float(np.std(cosine_values, ddof=1)) if M > 1 else 0.0
            ),
            "base_prediction": base_prediction,
            "base_prediction_name": self._class_label(base_prediction),
            "explanation_dimension": explanation_dimension,
            "M": int(M),
            "perturbation_rate": float(perturbation_rate),
            "tau": float(tau),
            "top_k": None if top_k is None else int(top_k),
            "normalize": normalize,
            "mean_masked_tokens": float(np.mean(n_masked_values)),
            "min_masked_tokens": int(np.min(n_masked_values)),
            "max_masked_tokens": int(np.max(n_masked_values)),
            "mean_safe_tokens": float(np.mean(n_safe_values)),
            "mean_alignment_attempts": float(np.mean(alignment_attempt_values)),
            "max_alignment_attempts_used": int(np.max(alignment_attempt_values)),
            "token_alignment_ok": True,
        }

    # ============================================================
    # Métricas por par de clases
    # ============================================================

    def evaluate_pair_metrics(
        self,
        texts,
        y_true,
        class_j,
        class_r,
        filter_mode="true",
        M=40,
        perturbation_rate=0.05,
        tau=0.95,
        top_k=10,
        normalize="l2",
        max_instances_per_pair=50,
        seed=42,
        verbose=True,
    ):
        """
        Evalúa estabilidad y consistencia para un par de clases.

        Para el protocolo final se debe usar filter_mode='true'. La muestra
        se construye balanceada y sin reemplazo con N/2 instancias por clase.
        """
        texts = list(texts)
        y_true = np.asarray(y_true)

        class_j_idx = self._class_name_to_index(class_j)
        class_r_idx = self._class_name_to_index(class_r)

        if class_j_idx == class_r_idx:
            raise ValueError("class_j y class_r deben ser clases diferentes.")

        label_j = self._class_label(class_j_idx)
        label_r = self._class_label(class_r_idx)

        rng = np.random.default_rng(seed)

        if filter_mode == "true":
            labels_for_filter = y_true
        elif filter_mode == "pred":
            if verbose:
                print("Calculando predicciones para filtrar por predicción...")
            labels_for_filter = self.lcd_explainer.predict_labels(texts)
        else:
            raise ValueError("filter_mode debe ser 'true' o 'pred'.")

        labels_for_filter = np.asarray(labels_for_filter)

        indices_j = np.where(labels_for_filter == class_j_idx)[0]
        indices_r = np.where(labels_for_filter == class_r_idx)[0]

        if len(indices_j) == 0 or len(indices_r) == 0:
            raise ValueError(
                f"No hay instancias suficientes para {label_j} vs {label_r} "
                f"con filter_mode={filter_mode}."
            )

        if max_instances_per_pair is not None:
            max_instances_per_pair = int(max_instances_per_pair)
            if max_instances_per_pair < 2:
                raise ValueError("max_instances_per_pair debe ser al menos 2.")
            if max_instances_per_pair % 2 != 0:
                raise ValueError(
                    "max_instances_per_pair debe ser par para una muestra balanceada."
                )

            n_each = min(
                len(indices_j),
                len(indices_r),
                max_instances_per_pair // 2,
            )
        else:
            n_each = min(len(indices_j), len(indices_r))

        if n_each < 1:
            raise ValueError("No fue posible construir una muestra balanceada.")

        selected_j = rng.choice(
            indices_j,
            size=n_each,
            replace=False,
        )
        selected_r = rng.choice(
            indices_r,
            size=n_each,
            replace=False,
        )

        candidate_indices = np.concatenate([selected_j, selected_r])
        candidate_indices = np.asarray(candidate_indices, dtype=int)

        # Se conserva el orden determinista utilizado en la clase anterior.
        candidate_indices = np.sort(candidate_indices)

        if verbose:
            print("\n==================================================")
            print(f"Evaluando par: {label_j} vs {label_r}")
            print(f"Selección: {'etiqueta verdadera' if filter_mode == 'true' else 'predicción'}")
            print(f"Filtro por corrección de predicción: NO")
            print(f"Semilla base: {seed}")
            print("Instancias disponibles:")
            print(f"  {label_j}: {len(indices_j)}")
            print(f"  {label_r}: {len(indices_r)}")
            print("Muestra balanceada:")
            print(f"  {label_j}: {n_each}")
            print(f"  {label_r}: {n_each}")
            print(f"N instancias evaluadas: {len(candidate_indices)}")
            print(f"M perturbaciones por instancia: {M}")
            print(f"perturbation_rate: {perturbation_rate}")
            print(f"tau: {tau}")
            print(f"top_k: {top_k}")
            print(f"normalize: {normalize}")
            print("Semilla por instancia: seed + t, t=0,...,N-1")
            print("Alineación tokenizada: ESTRICTA (masker SHAP + verificación IDs)")
            print("==================================================")

        instance_rows = []

        # t es la posición 0,...,N-1 dentro de la muestra evaluada.
        for t, test_index in enumerate(candidate_indices):
            test_index = int(test_index)
            instance_seed = int(seed + t)

            if verbose:
                print(
                    f"\nInstancia {t + 1}/{len(candidate_indices)} "
                    f"| índice test={test_index} | instance_seed={instance_seed}"
                )

            instance_metrics = self.evaluate_instance_metrics(
                text=texts[test_index],
                class_j=class_j_idx,
                class_r=class_r_idx,
                M=M,
                perturbation_rate=perturbation_rate,
                tau=tau,
                top_k=top_k,
                normalize=normalize,
                seed=instance_seed,
            )

            true_label = int(y_true[test_index])
            base_prediction = int(instance_metrics["base_prediction"])

            row = {
                "sample_position": int(t),
                "test_index": test_index,
                "true_label": true_label,
                "true_label_name": self._class_label(true_label),
                "class_j": label_j,
                "class_r": label_r,
                "pair": f"{label_j} vs {label_r}",
                "filter_mode": filter_mode,
                "sampling_seed": int(seed),
                "instance_seed": instance_seed,
                "prediction_correct": int(base_prediction == true_label),
                **instance_metrics,
            }

            instance_rows.append(row)

        instance_df = pd.DataFrame(instance_rows)

        if instance_df.empty:
            raise RuntimeError("No se generaron resultados por instancia.")

        stability_values = instance_df["stability"].to_numpy(dtype=float)
        consistency_values = instance_df["consistency"].to_numpy(dtype=float)
        coverage_values = instance_df["coverage_imp"].to_numpy(dtype=float)
        prediction_keep_values = instance_df[
            "prediction_keep_imp"
        ].to_numpy(dtype=float)

        number_of_instances = len(instance_df)

        stability_std = (
            float(np.std(stability_values, ddof=1))
            if number_of_instances > 1
            else 0.0
        )
        consistency_std = (
            float(np.std(consistency_values, ddof=1))
            if number_of_instances > 1
            else 0.0
        )

        stability_se = (
            stability_std / np.sqrt(number_of_instances)
            if number_of_instances > 0
            else np.nan
        )
        consistency_se = (
            consistency_std / np.sqrt(number_of_instances)
            if number_of_instances > 0
            else np.nan
        )

        if filter_mode == "pred":
            balance_labels = instance_df["base_prediction"].to_numpy(dtype=int)
        else:
            balance_labels = instance_df["true_label"].to_numpy(dtype=int)

        number_j = int(np.sum(balance_labels == class_j_idx))
        number_r = int(np.sum(balance_labels == class_r_idx))
        balance_string = f"{number_j}/{number_r}"

        n_correct = int(instance_df["prediction_correct"].sum())
        n_incorrect = int(number_of_instances - n_correct)

        summary = {
            "class_j": label_j,
            "class_r": label_r,
            "pair": f"{label_j} vs {label_r}",
            "filter_mode": filter_mode,
            "sampling_seed": int(seed),
            "N_instances": int(number_of_instances),
            "Bal.": balance_string,
            "N_correct": n_correct,
            "N_incorrect": n_incorrect,
            "M_perturbations": int(M),
            "perturbation_rate": float(perturbation_rate),
            "tau": float(tau),
            "top_k": None if top_k is None else int(top_k),
            "normalize": normalize,
            "token_alignment": "strict_shap_masker_exact_ids",
            "instance_seed_rule": "seed+t",
            "stability_mean": float(np.mean(stability_values)),
            "stability_std": stability_std,
            "stability_se": float(stability_se),
            "consistency_mean": float(np.mean(consistency_values)),
            "consistency_std": consistency_std,
            "consistency_se": float(consistency_se),
            "coverage_imp_mean": float(np.mean(coverage_values)),
            "prediction_keep_imp_mean": float(np.mean(prediction_keep_values)),
            "cosine_mean_avg": float(instance_df["cosine_mean"].mean()),
            "cosine_std_avg": float(instance_df["cosine_std"].mean()),
            "mean_masked_tokens_avg": float(
                instance_df["mean_masked_tokens"].mean()
            ),
        }

        summary_df = pd.DataFrame([summary])

        if verbose:
            print("\nControl de selección:")
            print(f"  Predicciones correctas   : {n_correct}")
            print(f"  Predicciones incorrectas : {n_incorrect}")
            print("  Filtro por predicción    : NO")
            print("  Alineación tokens        : OK (IDs exactos)")

        return summary_df, instance_df

    # ============================================================
    # Métricas para varios pares
    # ============================================================

    def evaluate_multiple_pairs(
        self,
        texts,
        y_true,
        pairs,
        filter_mode="true",
        M=40,
        perturbation_rate=0.05,
        tau=0.95,
        top_k=10,
        normalize="l2",
        max_instances_per_pair=50,
        seed=42,
        output_dir=None,
    ):
        """Evalúa estabilidad y consistencia para uno o varios pares."""
        if not pairs:
            raise ValueError("La lista de pares está vacía.")

        all_summary = []
        all_instances = []

        for class_j, class_r in pairs:
            summary_df, instance_df = self.evaluate_pair_metrics(
                texts=texts,
                y_true=y_true,
                class_j=class_j,
                class_r=class_r,
                filter_mode=filter_mode,
                M=M,
                perturbation_rate=perturbation_rate,
                tau=tau,
                top_k=top_k,
                normalize=normalize,
                max_instances_per_pair=max_instances_per_pair,
                seed=seed,
                verbose=True,
            )

            all_summary.append(summary_df)
            all_instances.append(instance_df)

            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)

                safe_j = self._safe_filename(summary_df.loc[0, "class_j"])
                safe_r = self._safe_filename(summary_df.loc[0, "class_r"])

                summary_path = os.path.join(
                    output_dir,
                    f"metrics_summary_{safe_j}_vs_{safe_r}.csv",
                )
                instance_path = os.path.join(
                    output_dir,
                    f"metrics_instances_{safe_j}_vs_{safe_r}.csv",
                )

                summary_df.to_csv(
                    summary_path,
                    index=False,
                    encoding="utf-8",
                )
                instance_df.to_csv(
                    instance_path,
                    index=False,
                    encoding="utf-8",
                )

                print("\nResumen del par guardado en:")
                print(os.path.abspath(summary_path))
                print("\nResultados por instancia guardados en:")
                print(os.path.abspath(instance_path))

        summary_all_df = pd.concat(all_summary, ignore_index=True)
        instances_all_df = pd.concat(all_instances, ignore_index=True)

        if output_dir is not None:
            summary_all_path = os.path.join(
                output_dir,
                "metrics_summary_all_pairs.csv",
            )
            instances_all_path = os.path.join(
                output_dir,
                "metrics_instances_all_pairs.csv",
            )
            table_path = os.path.join(
                output_dir,
                "metrics_table_format_all_pairs.csv",
            )

            summary_all_df.to_csv(
                summary_all_path,
                index=False,
                encoding="utf-8",
            )
            instances_all_df.to_csv(
                instances_all_path,
                index=False,
                encoding="utf-8",
            )

            table_df = self.make_latex_like_summary_table(summary_all_df)
            table_df.to_csv(
                table_path,
                index=False,
                encoding="utf-8",
            )

        return summary_all_df, instances_all_df

    # ============================================================
    # Tabla con formato de artículo
    # ============================================================

    @staticmethod
    def make_latex_like_summary_table(summary_df):
        """
        Genera:
        Pair | N | Bal. | S_bar ± sS | Cimp_bar ± sC |
        Cov_imp / PredKeep_imp
        """
        rows = []

        for _, row in summary_df.iterrows():
            rows.append(
                {
                    "Pair (Cj, Cr)": row["pair"],
                    "N": int(row["N_instances"]),
                    "Bal.": row["Bal."],
                    "S_bar ± sS": (
                        f"{row['stability_mean']:.4f} ± "
                        f"{row['stability_std']:.4f}"
                    ),
                    "Cimp_bar ± sC": (
                        f"{row['consistency_mean']:.4f} ± "
                        f"{row['consistency_std']:.4f}"
                    ),
                    "Cov_imp / PredKeep_imp": (
                        f"{row['coverage_imp_mean']:.4f} / "
                        f"{row['prediction_keep_imp_mean']:.4f}"
                    ),
                }
            )

        return pd.DataFrame(rows)
