# Clase_Explain_Text.py

import os
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import shap
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class SHAPTextExplainerLCD:
    """
    Clase para aplicar SHAP-LCD a modelos de clasificación de texto.

    Esta clase está pensada para trabajar con modelos tipo Transformer
    de Hugging Face, por ejemplo BERT, RoBERTa o DistilBERT.

    La idea metodológica es equivalente a la versión usada en imágenes:

    1. Algorithm 1:
       Calcular diferencias absolutas entre probabilidades de clase |ΔP|.

    2. Algorithm 2:
       Calcular diferencias SHAP entre dos clases:
           ΔS = SHAP(C_j) - SHAP(C_r)

    3. Algorithm 3:
       Visualizar el texto resaltando los tokens que favorecen una clase
       frente a la otra.

    En imágenes, la unidad explicativa era el píxel.
    En texto, la unidad explicativa será el token o palabra.
    """

    def __init__(
        self,
        model,
        tokenizer,
        texts_eval: List[str],
        y_eval: Optional[List[int]] = None,
        class_names: Optional[Dict[int, str]] = None,
        max_length: int = 128,
        batch_size: int = 16,
        device: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Inicializa el explicador SHAP-LCD para texto.

        Parameters
        ----------
        model:
            Modelo de clasificación de texto de Hugging Face.
            Debe devolver logits para cada clase.

        tokenizer:
            Tokenizador asociado al modelo.

        texts_eval:
            Textos que se van a explicar. Normalmente corresponden
            a un subconjunto del conjunto de prueba.

        y_eval:
            Etiquetas verdaderas de los textos de evaluación.

        class_names:
            Diccionario con nombres de clases, por ejemplo:
            {
                0: "World",
                1: "Sports",
                2: "Business",
                3: "Sci/Tech"
            }

        max_length:
            Longitud máxima de tokenización.

        batch_size:
            Tamaño de lote para predicción.

        device:
            "cuda" si hay GPU disponible o "cpu".
            Si no se especifica, se detecta automáticamente.

        random_state:
            Semilla para reproducibilidad.
        """

        self.model = model
        self.tokenizer = tokenizer
        self.texts_eval = list(texts_eval)
        self.y_eval = None if y_eval is None else np.array(y_eval)

        self.class_names = class_names
        self.max_length = max_length
        self.batch_size = batch_size
        self.random_state = random_state

        # Selección automática de dispositivo.
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Se mueve el modelo al dispositivo elegido.
        self.model.to(self.device)
        self.model.eval()

        # Objetos internos de SHAP.
        self.masker = None
        self.explainer = None
        self.shap_values = None
        self.max_eval_used = None

        # Metadatos para verificar compatibilidad de caché.
        self.model_hash = self._fingerprint_model()
        self.tokenizer_hash = self._fingerprint_tokenizer()

        self._loaded_meta = None

        # Se inicializa el explicador SHAP para texto.
        self._init_explainer()

    # ============================================================
    # Funciones auxiliares de metadatos y caché
    # ============================================================

    def _fingerprint_model(self) -> str:
        """
        Calcula una huella simple del modelo.

        En modelos Transformer, calcular una huella de todos los pesos puede ser
        costoso. Aquí se usa la configuración del modelo y el número de parámetros.
        Esto no reemplaza una trazabilidad experimental completa, pero ayuda a
        evitar mezclar cachés de modelos diferentes.
        """
        h = hashlib.sha256()

        try:
            config_str = self.model.config.to_json_string()
            h.update(config_str.encode("utf-8"))
        except Exception:
            h.update(str(self.model).encode("utf-8"))

        try:
            n_params = sum(p.numel() for p in self.model.parameters())
            h.update(str(n_params).encode("utf-8"))
        except Exception:
            pass

        return h.hexdigest()

    def _fingerprint_tokenizer(self) -> str:
        """
        Calcula una huella simple del tokenizador.
        """
        h = hashlib.sha256()

        try:
            h.update(str(self.tokenizer.__class__.__name__).encode("utf-8"))
            h.update(str(self.tokenizer.vocab_size).encode("utf-8"))
            h.update(str(self.tokenizer.model_max_length).encode("utf-8"))
        except Exception:
            h.update(str(self.tokenizer).encode("utf-8"))

        return h.hexdigest()

    def is_cache_compatible(self, desired_max_eval: Optional[int]) -> Tuple[bool, str]:
        """
        Verifica si un archivo SHAP cargado corresponde al mismo modelo,
        tokenizador y tamaño de evaluación.
        """
        if self._loaded_meta is None:
            return False, "no hay metadata cargada"

        meta = self._loaded_meta

        if meta.get("model_hash") != self.model_hash:
            return False, "modelo distinto"

        if meta.get("tokenizer_hash") != self.tokenizer_hash:
            return False, "tokenizador distinto"

        if meta.get("max_eval_used") != desired_max_eval:
            return False, "MAX_EVAL distinto"

        return True, "ok"

    # ============================================================
    # Predicción del modelo
    # ============================================================

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Devuelve las probabilidades de clase para una lista de textos.

        Esta función es la que SHAP usa internamente para consultar el modelo.
        Por eso recibe textos crudos y devuelve una matriz:

            shape = (n_textos, n_clases)
        """

        # SHAP puede enviar un solo texto o un arreglo de textos.
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)

        all_probs = []

        # Se predice por lotes para evitar consumir demasiada memoria.
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]

            # Tokenización: convierte texto en input_ids y attention_mask.
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Se mueven los tensores al dispositivo.
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # No necesitamos gradientes para predecir.
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits

                # Softmax convierte logits en probabilidades.
                probs = torch.softmax(logits, dim=-1)

            all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def predict_labels(self, texts: List[str]) -> np.ndarray:
        """
        Devuelve la clase predicha para cada texto.
        """
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

    # ============================================================
    # Inicialización y cálculo de SHAP
    # ============================================================

    def _init_explainer(self):
        """
        Inicializa el explicador SHAP para texto.

        shap.maskers.Text se encarga de ocultar tokens/palabras para estimar
        la contribución de cada unidad textual.
        """
        self.masker = shap.maskers.Text(self.tokenizer)
        self.explainer = shap.Explainer(
            self.predict_proba,
            self.masker,
            output_names=self._output_names()
        )

    def _output_names(self) -> Optional[List[str]]:
        """
        Devuelve los nombres de las clases en orden.
        """
        if self.class_names is None:
            return None

        return [self.class_names[i] for i in sorted(self.class_names.keys())]

    def _get_eval_slice(self, max_eval: Optional[int]) -> List[str]:
        """
        Devuelve todos los textos de evaluación o solo los primeros max_eval.
        """
        if max_eval is None:
            return self.texts_eval
        return self.texts_eval[:int(max_eval)]

    def calculate_shap_values(self, max_eval: Optional[int] = None):
        """
        Calcula los valores SHAP para los textos de evaluación.

        En texto, SHAP devuelve un objeto Explanation. Para cada instancia,
        este objeto contiene:
            - tokens o fragmentos textuales
            - valores SHAP por token
            - valores SHAP por clase
        """
        if self.explainer is None:
            raise ValueError("Explainer no inicializado.")

        texts = self._get_eval_slice(max_eval)
        self.max_eval_used = None if max_eval is None else int(max_eval)

        print(f"Calculando SHAP para {len(texts)} textos...")
        self.shap_values = self.explainer(texts)

        try:
            print("Valores SHAP calculados. Forma:", self.shap_values.values.shape)
        except Exception:
            print("Valores SHAP calculados.")

    def save_shap_values(self, filename: str = "shap_values_text.pickle"):
        """
        Guarda los valores SHAP y metadatos para reutilizarlos.
        """
        if self.shap_values is None:
            raise ValueError("No hay valores SHAP para guardar.")

        payload = {
            "shap_values": self.shap_values,
            "max_eval_used": self.max_eval_used,
            "model_hash": self.model_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "class_names": self.class_names,
            "max_length": self.max_length
        }

        with open(filename, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Valores SHAP guardados en: {filename}")

    def load_shap_values(self, filename: str = "shap_values_text.pickle"):
        """
        Carga valores SHAP previamente guardados.
        """
        with open(filename, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and "shap_values" in payload:
            self.shap_values = payload["shap_values"]
            self.max_eval_used = payload.get("max_eval_used", None)
            self._loaded_meta = payload
        else:
            self.shap_values = payload
            self.max_eval_used = None
            self._loaded_meta = None

        print(f"Valores SHAP cargados desde: {filename}")

    # ============================================================
    # Utilidades de clases
    # ============================================================

    def _class_name_to_index(self, class_name_or_idx: Any) -> int:
        """
        Convierte un nombre de clase a índice.

        Permite usar tanto:
            - índice: 0
            - nombre: "World"
        """

        if isinstance(class_name_or_idx, int):
            return class_name_or_idx

        if self.class_names is None:
            raise ValueError("No hay class_names definido. Usa índices de clase.")

        inverse = {v: k for k, v in self.class_names.items()}

        if class_name_or_idx not in inverse:
            raise ValueError(
                f"No se encontró la clase '{class_name_or_idx}'. "
                f"Clases disponibles: {list(inverse.keys())}"
            )

        return inverse[class_name_or_idx]

    def _class_label(self, class_idx: int) -> str:
        """
        Devuelve el nombre legible de una clase.
        """
        if self.class_names is None:
            return f"Class {class_idx}"
        return self.class_names[class_idx]

    # ============================================================
    # Algorithm 1: diferencias de probabilidad |ΔP|
    # ============================================================

    def plot_probability_differences(
        self,
        instance_idx: int,
        top_k: int = 6,
        instance_label: str = "Instance",
        show: bool = True,
        save_path: Optional[str] = None
    ) -> List[Tuple[float, str, str]]:
        """
        Calcula y grafica las diferencias absolutas entre probabilidades
        para todos los pares de clases.

        Esto permite seleccionar un par de clases relevante para el análisis LCD.
        """

        text = self.texts_eval[instance_idx]
        probs = self.predict_proba([text])[0]

        num_classes = len(probs)
        class_labels = [self._class_label(i) for i in range(num_classes)]

        diffs = []

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                diffs.append(
                    (
                        abs(probs[i] - probs[j]),
                        class_labels[i],
                        class_labels[j]
                    )
                )

        diffs = sorted(diffs, key=lambda x: x[0], reverse=True)
        diffs_plot = diffs[:top_k]

        labels = [f"{d[1]} vs {d[2]}" for d in diffs_plot]
        values = [d[0] for d in diffs_plot]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=values, y=labels, hue=labels, palette="viridis", legend=False)
        plt.title(f"Absolute differences between class probabilities\nfor {instance_label}", fontsize=18)
        plt.xlabel(r"Absolute difference $|\Delta P|$", fontsize=16)
        plt.ylabel("Class pairs", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return diffs

    # ============================================================
    # Extracción de ΔS para una instancia
    # ============================================================

    def get_shap_diff_for_instance(
        self,
        instance_idx: int,
        class_i,
        class_j
    ) -> pd.DataFrame:
        """
        Calcula ΔS token a token para una instancia y un par de clases.

        ΔS = SHAP(C_i) - SHAP(C_j)

        Si ΔS > 0, el token favorece a C_i.
        Si ΔS < 0, el token favorece a C_j.
        """

        if self.shap_values is None:
            raise ValueError("Primero calcula o carga los valores SHAP.")

        idx_i = self._class_name_to_index(class_i)
        idx_j = self._class_name_to_index(class_j)

        # Tokens de la instancia.
        tokens = list(self.shap_values.data[instance_idx])

        # Valores SHAP de la instancia.
        values = self.shap_values.values[instance_idx]

        # En clasificación multiclase, values suele tener forma:
        # (n_tokens, n_classes)
        if values.ndim != 2:
            raise ValueError(f"Forma SHAP no esperada para texto: {values.shape}")

        shap_i = values[:, idx_i]
        shap_j = values[:, idx_j]
        shap_diff = shap_i - shap_j

        df = pd.DataFrame({
            "token": tokens,
            "SHAP_class_i": shap_i,
            "SHAP_class_j": shap_j,
            "SHAP_Difference": shap_diff,
            "abs_SHAP_Difference": np.abs(shap_diff)
        })

        df = df.sort_values(
            by="abs_SHAP_Difference",
            ascending=False
        ).reset_index(drop=True)

        return df

    # ============================================================
    # Algorithm 2: gráfico de barras de ΔS
    # ============================================================

    def explain_selected_pairs(
        self,
        instance_idx: int,
        selected_pairs: List[Tuple[Any, Any]],
        top_n: int = 15,
        instance_label: str = "Instance",
        show: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Genera un gráfico de barras con los tokens más relevantes
        según |ΔS| para el par de clases seleccionado.
        """

        for class_i, class_j in selected_pairs:
            df = self.get_shap_diff_for_instance(instance_idx, class_i, class_j)
            df_top = df.head(top_n).copy()

            label_i = self._class_label(self._class_name_to_index(class_i))
            label_j = self._class_label(self._class_name_to_index(class_j))

            plt.figure(figsize=(12, 7))
            sns.barplot(
                data=df_top,
                x="SHAP_Difference",
                y="token",
                hue="token",
                palette="viridis",
                legend=False
            )

            plt.title(
                f"Top |ΔS| tokens between {label_i} and {label_j}\n{instance_label}",
                fontsize=18
            )
            plt.xlabel(r"SHAP difference $\Delta S$", fontsize=15)
            plt.ylabel("Tokens", fontsize=15)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()

            if save_path is not None:
                root, ext = os.path.splitext(save_path)
                out_path = f"{root}_{label_i}_vs_{label_j}{ext}"
                plt.savefig(out_path, dpi=600, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close()

    # ============================================================
    # Algorithm 3: texto resaltado con colores SHAP-LCD
    # ============================================================

    @staticmethod
    def _clean_token(token: Any) -> str:
        """
        Limpia algunos marcadores comunes de tokenización.

        RoBERTa usa 'Ġ' para indicar espacio.
        SentencePiece usa '▁'.
        BERT usa '##' para subpalabras.
        """
        token = str(token)

        token = token.replace("Ġ", " ")
        token = token.replace("▁", " ")
        token = token.replace("##", "")

        return token

    @staticmethod
    def _color_span(token: str, value: float, max_abs: float) -> str:
        """
        Devuelve un token en formato HTML con color de fondo.

        Magenta: favorece la clase C_i.
        Verde: favorece la clase C_j.

        La intensidad del color depende de |ΔS|.
        """
        if max_abs <= 0:
            alpha = 0.0
        else:
            alpha = min(abs(value) / max_abs, 1.0)

        # Se evita que los colores sean demasiado claros.
        alpha = 0.15 + 0.70 * alpha

        # Magenta para contribución positiva.
        if value > 0:
            color = f"rgba(214, 39, 180, {alpha:.3f})"
        # Verde para contribución negativa.
        elif value < 0:
            color = f"rgba(44, 160, 44, {alpha:.3f})"
        # Sin color si ΔS es exactamente cero.
        else:
            color = "rgba(255, 255, 255, 0.0)"

        return (
            f'<span style="background-color:{color}; '
            f'padding:2px 3px; margin:1px; border-radius:3px;">'
            f'{token}</span>'
        )

    def build_text_highlight_html(
        self,
        instance_idx: int,
        class_i,
        class_j,
        instance_label: str = "Instance",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Construye una visualización HTML del texto resaltado.

        ΔS > 0: token favorece a class_i.
        ΔS < 0: token favorece a class_j.
        """

        if self.shap_values is None:
            raise ValueError("Primero calcula o carga los valores SHAP.")

        idx_i = self._class_name_to_index(class_i)
        idx_j = self._class_name_to_index(class_j)

        label_i = self._class_label(idx_i)
        label_j = self._class_label(idx_j)

        tokens = list(self.shap_values.data[instance_idx])
        values = self.shap_values.values[instance_idx]

        if values.ndim != 2:
            raise ValueError(f"Forma SHAP no esperada para texto: {values.shape}")

        shap_diff = values[:, idx_i] - values[:, idx_j]

        if max_tokens is not None:
            tokens = tokens[:max_tokens]
            shap_diff = shap_diff[:max_tokens]

        max_abs = np.max(np.abs(shap_diff)) + 1e-12

        colored_tokens = []

        for token, value in zip(tokens, shap_diff):
            clean_tok = self._clean_token(token)

            # Se ignoran tokens vacíos.
            if clean_tok.strip() == "":
                continue

            colored_tokens.append(
                self._color_span(clean_tok, value, max_abs)
            )

        probs = self.predict_proba([self.texts_eval[instance_idx]])[0]

        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>SHAP-LCD Text Explanation</title>
        </head>
        <body style="font-family:Arial, sans-serif; line-height:1.8; max-width:1100px;">
            <h2>SHAP-LCD text explanation: {label_i} vs {label_j}</h2>
            <h3>{instance_label}</h3>

            <p>
                <b>Predicted probabilities:</b>
                {label_i}: {probs[idx_i]:.4f} |
                {label_j}: {probs[idx_j]:.4f}
            </p>

            <p>
                <span style="background-color:rgba(214, 39, 180, 0.75); padding:3px 6px; border-radius:3px;">
                    Magenta
                </span>
                favors <b>{label_i}</b>
                &nbsp;&nbsp;
                <span style="background-color:rgba(44, 160, 44, 0.75); padding:3px 6px; border-radius:3px;">
                    Green
                </span>
                favors <b>{label_j}</b>
            </p>

            <div style="font-size:18px; border:1px solid #ddd; padding:14px; border-radius:6px;">
                {" ".join(colored_tokens)}
            </div>
        </body>
        </html>
        """

        return html

    def plot_text_highlight(
        self,
        instance_idx: int,
        class_i,
        class_j,
        instance_label: str = "Instance",
        max_tokens: Optional[int] = None,
        save_path: Optional[str] = "shap_lcd_text_highlight.html"
    ) -> str:
        """
        Guarda una visualización HTML del texto resaltado.

        Retorna el HTML como cadena, por si se quiere mostrar en un notebook.
        """
        html = self.build_text_highlight_html(
            instance_idx=instance_idx,
            class_i=class_i,
            class_j=class_j,
            instance_label=instance_label,
            max_tokens=max_tokens
        )

        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html)

            print(f"Visualización SHAP-LCD guardada en: {save_path}")

        return html