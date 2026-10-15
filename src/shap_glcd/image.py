import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt


class SHAP_GLCD_ImageExplainer_MapsOnly:
    """
    SHAP-GLCD global PAIRWISE (solo mapas) para imágenes.

    - SHAP se calcula/carga sobre un subconjunto del test -> x_eval/y_eval
    - Se guarda idx_eval en pickle -> alineación exacta (SHAP-x-y)
    - Todo lo global se calcula SOLO con imágenes del par (j_sel vs r_sel) dentro de x_eval

    Mapas que genera (todos PAIRWISE):
      1) mean |ΔS|
      2) signed mean ΔS
      3) Top-p% mask sobre mean|ΔS|
      4) Dominance map (GDP por pixel): W+(p) in [0,1]
      5) Conditional mean maps: mean(ΔS | ΔS>0) y mean(ΔS | ΔS<0)
      6) Separability map (Cohen's d) por pixel usando intensidades de imagen del par

    Nuevo:
      - eval_pair_only=True: x_eval se toma SOLO de las clases j_sel y r_sel (balanceado).
      - background se toma de x_train (global) para mantener referencia estable.
    """

    def __init__(
        self,
        model,
        x_train,
        x_test,
        y_test,
        j_sel,
        r_sel,
        class_names=None,                 # dict o lista/tupla
        shap_pickle="shap_values_fashion.pickle",
        out_dir="Figuras_GLCD_Maps",
        dpi=600,
        random_state=42,
        show_plots=True,
        display_class_names=None,
        strict_pair_balance=True
    ):
        self.model = model
        self.x_train = np.asarray(x_train)
        self.x_test_full = np.asarray(x_test)
        self.y_test_full = np.asarray(y_test)

        self.j_sel = int(j_sel)
        self.r_sel = int(r_sel)
        self.class_names = class_names
        self.display_class_names = display_class_names

        self.shap_pickle = shap_pickle
        self.out_dir = out_dir
        self.dpi = int(dpi)
        self.random_state = int(random_state)
        self.show_plots = bool(show_plots)
        self.strict_pair_balance = bool(strict_pair_balance)

        os.makedirs(self.out_dir, exist_ok=True)

        # Asegura (N,H,W,C)
        if self.x_train.ndim == 3:
            self.x_train = self.x_train[..., np.newaxis]
        if self.x_test_full.ndim == 3:
            self.x_test_full = self.x_test_full[..., np.newaxis]

        if self.x_train.ndim != 4 or self.x_test_full.ndim != 4:
            raise ValueError("x_train/x_test deben ser (N,H,W) o (N,H,W,C).")

        self.H = self.x_test_full.shape[1]
        self.W = self.x_test_full.shape[2]
        self.C = self.x_test_full.shape[3]
        self.P = self.H * self.W * self.C

        # Eval subset (alineado a SHAP)
        self.idx_eval = None
        self.x_eval = None
        self.y_eval = None

        # SHAP (N_eval,P,K)
        self.shap_arr = None

        # Pairwise
        self.pair_mask = None
        self.delta = None  # (N_pair,P)

    # -----------------------
    # utilidades
    # -----------------------
    def _set_seed(self):
        os.environ["PYTHONHASHSEED"] = str(self.random_state)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def _cls_name(self, c: int) -> str:
        if self.class_names is None:
            return str(c)
        if isinstance(self.class_names, dict):
            return str(self.class_names.get(c, c))
        if isinstance(self.class_names, (list, tuple)):
            return str(self.class_names[c])
        return str(c)

    def _display_cls_name(self, c: int) -> str:
        """
        Nombre corto usado SOLO en títulos, leyendas y barras de color.

        Si display_class_names=None, se usa exactamente el nombre normal
        devuelto por _cls_name(), por lo que los mains antiguos siguen
        funcionando sin cambios.
        """
        if self.display_class_names is None:
            return self._cls_name(c)

        if isinstance(self.display_class_names, dict):
            return str(
                self.display_class_names.get(
                    c,
                    self._cls_name(c)
                )
            )

        if isinstance(self.display_class_names, (list, tuple)):
            return str(self.display_class_names[c])

        return self._cls_name(c)

    def _pair_display_label(self) -> str:
        """
        Etiqueta corta del par usada SOLO en las figuras.
        """
        return (
            f"{self._display_cls_name(self.j_sel)} and "
            f"{self._display_cls_name(self.r_sel)}"
        )

    def _safe_name(self, s: str) -> str:
        s = str(s)
        for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
            s = s.replace(ch, "_")
        return s.replace(" ", "_")

    def _pair_label(self) -> str:
        return f"{self._cls_name(self.j_sel)} and {self._cls_name(self.r_sel)}"

    def _pair_tag(self) -> str:
        a = self._safe_name(self._cls_name(self.j_sel))
        b = self._safe_name(self._cls_name(self.r_sel))
        return f"{a}_and_{b}"

    def _to_shap_array_NPK(self, shap_values):
        """
        Convierte SHAP a (N,P,K).
        Acepta:
          - list de K arrays: (N,H,W,C) o (N,P)
          - array: (N,H,W,C,K) o (N,P,K)
        """
        if isinstance(shap_values, list):
            K = len(shap_values)
            sv0 = np.asarray(shap_values[0])
            N = sv0.shape[0]
            mats = []
            for k in range(K):
                svk = np.asarray(shap_values[k])
                if svk.ndim == 4:      # (N,H,W,C)
                    mats.append(svk.reshape(N, -1))
                elif svk.ndim == 2:    # (N,P)
                    mats.append(svk)
                else:
                    raise ValueError(f"Formato inesperado shap_values[{k}]: {svk.shape}")
            return np.stack(mats, axis=2)  # (N,P,K)

        arr = np.asarray(shap_values)
        if arr.ndim == 5:  # (N,H,W,C,K)
            return arr.reshape(arr.shape[0], -1, arr.shape[-1])  # (N,P,K)
        if arr.ndim == 3:  # (N,P,K)
            return arr
        raise ValueError(f"Formato inesperado de shap_values: {arr.shape}")

    def _align_eval_to_shap(self):
        if self.shap_arr is None or self.x_eval is None or self.y_eval is None:
            return
        N_shap = int(self.shap_arr.shape[0])
        if len(self.x_eval) != N_shap or len(self.y_eval) != N_shap:
            self.x_eval = self.x_eval[:N_shap]
            self.y_eval = self.y_eval[:N_shap]

    def _apply_strict_pair_balance(self):
        """Balance the cached pair without recomputing SHAP values."""
        if not self.strict_pair_balance:
            return
        if self.shap_arr is None or self.x_eval is None or self.y_eval is None:
            return

        pos_j = np.flatnonzero(self.y_eval == self.j_sel)
        pos_r = np.flatnonzero(self.y_eval == self.r_sel)

        if pos_j.size == 0 or pos_r.size == 0:
            raise ValueError(
                f"No hay suficientes imágenes del par "
                f"({self._pair_label()}) para balancear."
            )

        n_each = min(pos_j.size, pos_r.size)

        # A cached sample that is already balanced is left untouched. This
        # preserves the exact Fashion-MNIST and HAM10000 ordering/results.
        if pos_j.size == pos_r.size:
            return

        rng = np.random.default_rng(self.random_state)
        selected_j = rng.choice(pos_j, size=n_each, replace=False)
        selected_r = rng.choice(pos_r, size=n_each, replace=False)
        selected = np.concatenate([selected_j, selected_r])
        rng.shuffle(selected)

        self.shap_arr = self.shap_arr[selected]
        self.x_eval = self.x_eval[selected]
        self.y_eval = self.y_eval[selected]
        if self.idx_eval is not None:
            self.idx_eval = self.idx_eval[selected]

        print(
            "Balanceo estricto aplicado al pickle: "
            f"{self._cls_name(self.j_sel)}={n_each}, "
            f"{self._cls_name(self.r_sel)}={n_each}."
        )

    # -----------------------
    # SHAP compute / load
    # -----------------------
    def compute_shap(self, background_size=100, max_test=300, explainer_type="gradient", eval_pair_only=True):
        self._set_seed()
        import shap

        # Background (global desde train)
        X_bg = self.x_train
        if len(X_bg) > background_size:
            idx_bg = np.random.choice(len(X_bg), size=background_size, replace=False)
            X_bg = X_bg[idx_bg]

        # Eval subset
        Ntot = len(self.x_test_full)

        if eval_pair_only:
            idx_j = np.where(self.y_test_full == self.j_sel)[0]
            idx_r = np.where(self.y_test_full == self.r_sel)[0]

            if idx_j.size == 0 or idx_r.size == 0:
                raise ValueError(f"No hay suficientes imágenes del par ({self._pair_label()}) en el test.")

            half = max_test // 2

            if self.strict_pair_balance:
                n_each = min(len(idx_j), len(idx_r), half)
                n_j = n_each
                n_r = n_each
            else:
                n_j = min(len(idx_j), half)
                n_r = min(len(idx_r), max_test - n_j)

                if n_j + n_r < max_test:
                    faltan = max_test - (n_j + n_r)
                    rem_j = len(idx_j) - n_j
                    rem_r = len(idx_r) - n_r

                    extra_j = min(rem_j, faltan) if rem_j >= rem_r else min(rem_j, max(0, faltan - min(rem_r, faltan)))
                    extra_r = min(rem_r, faltan - extra_j)

                    n_j += extra_j
                    n_r += extra_r

            sel_j = np.random.choice(idx_j, size=n_j, replace=False) if n_j > 0 else np.array([], dtype=int)
            sel_r = np.random.choice(idx_r, size=n_r, replace=False) if n_r > 0 else np.array([], dtype=int)

            idx_eval = np.concatenate([sel_j, sel_r])
            np.random.shuffle(idx_eval)

        else:
            if Ntot > max_test:
                idx_eval = np.random.choice(Ntot, size=max_test, replace=False)
            else:
                idx_eval = np.arange(Ntot)

        self.idx_eval = np.asarray(idx_eval, dtype=int)
        self.x_eval = self.x_test_full[self.idx_eval]
        self.y_eval = self.y_test_full[self.idx_eval]

        # Explainer
        if str(explainer_type).lower() == "deep":
            explainer = shap.DeepExplainer(self.model, X_bg)
        else:
            explainer = shap.GradientExplainer(self.model, X_bg)

        shap_values = explainer.shap_values(self.x_eval)
        self.shap_arr = self._to_shap_array_NPK(shap_values)

        if self.shap_arr.shape[1] != self.P:
            raise ValueError(f"P SHAP={self.shap_arr.shape[1]} no coincide con H*W*C={self.P}.")

        obj = {"shap_values": shap_values, "idx_eval": self.idx_eval}
        with open(self.shap_pickle, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._align_eval_to_shap()
        self._apply_strict_pair_balance()
        return self.shap_arr

    def load_shap(self):
        if not os.path.exists(self.shap_pickle):
            raise FileNotFoundError(f"No existe: {self.shap_pickle}")

        with open(self.shap_pickle, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict) and "shap_values" in obj:
            shap_values = obj["shap_values"]
            idx_eval = obj.get("idx_eval", None)
        else:
            # pickle antiguo
            shap_values = obj
            idx_eval = None

        self.shap_arr = self._to_shap_array_NPK(shap_values)

        if self.shap_arr.shape[1] != self.P:
            raise ValueError(f"P SHAP={self.shap_arr.shape[1]} no coincide con H*W*C={self.P}.")

        if idx_eval is not None:
            self.idx_eval = np.asarray(idx_eval, dtype=int)
            self.x_eval = self.x_test_full[self.idx_eval]
            self.y_eval = self.y_test_full[self.idx_eval]
        else:
            N_shap = int(self.shap_arr.shape[0])
            self.idx_eval = np.arange(N_shap)
            self.x_eval = self.x_test_full[:N_shap]
            self.y_eval = self.y_test_full[:N_shap]

        self._align_eval_to_shap()
        self._apply_strict_pair_balance()
        return self.shap_arr

    def ensure_shap(self, mode="auto", **compute_kwargs):
        if mode == "auto":
            if os.path.exists(self.shap_pickle):
                return self.load_shap()
            return self.compute_shap(**compute_kwargs)
        if mode == "always":
            return self.compute_shap(**compute_kwargs)
        if mode == "only_load":
            return self.load_shap()
        raise ValueError("mode debe ser: auto, always, only_load")

    # -----------------------
    # Pairwise core
    # -----------------------
    def compute_delta_pairwise(self):
        if self.shap_arr is None or self.x_eval is None or self.y_eval is None:
            raise RuntimeError("Primero compute_shap() o load_shap().")

        self._align_eval_to_shap()

        self.pair_mask = (self.y_eval == self.j_sel) | (self.y_eval == self.r_sel)
        n_pair = int(self.pair_mask.sum())
        if n_pair == 0:
            raise ValueError(
                f"No hay imágenes del par ({self._pair_label()}) dentro del eval subset. "
                f"Sube max_test o cambia el par."
            )

        shap_pair = self.shap_arr[self.pair_mask]  # (N_pair,P,K)
        self.delta = shap_pair[:, :, self.j_sel] - shap_pair[:, :, self.r_sel]  # (N_pair,P)
        return self.delta

    # -----------------------
    # Mapas
    # -----------------------
    def _reshape_to_HW(self, vP: np.ndarray) -> np.ndarray:
        if self.C == 1:
            return vP.reshape(self.H, self.W)
        return vP.reshape(self.H, self.W, self.C).mean(axis=2)

    def map_mean_abs_delta(self):
        if self.delta is None:
            self.compute_delta_pairwise()
        mean_abs = np.mean(np.abs(self.delta), axis=0)
        return self._reshape_to_HW(mean_abs)

    def map_signed_mean_delta(self):
        if self.delta is None:
            self.compute_delta_pairwise()
        signed_m = np.mean(self.delta, axis=0)
        return self._reshape_to_HW(signed_m)

    def map_top_percent_mask(self, percent=5.0):
        A = self.map_mean_abs_delta().astype(np.float64)
        flat = A.ravel()
        thr = np.percentile(flat, 100.0 - float(percent))
        return (A >= thr).astype(float)

    def map_conditional_pos_neg(self):
        if self.delta is None:
            self.compute_delta_pairwise()

        D = self.delta

        pos_mask = D > 0
        neg_mask = D < 0

        pos_count = pos_mask.sum(axis=0)
        neg_count = neg_mask.sum(axis=0)

        pos_sum = np.where(pos_mask, D, 0.0).sum(axis=0)
        neg_sum = np.where(neg_mask, D, 0.0).sum(axis=0)

        pos_m = np.divide(
            pos_sum,
            pos_count,
            out=np.zeros_like(pos_sum, dtype=float),
            where=pos_count > 0
        )

        neg_m = np.divide(
            neg_sum,
            neg_count,
            out=np.zeros_like(neg_sum, dtype=float),
            where=neg_count > 0
        )

        return (
            self._reshape_to_HW(pos_m),
            self._reshape_to_HW(neg_m)
        )

    def map_dominance_gdp(self):
        if self.delta is None:
            self.compute_delta_pairwise()

        D = self.delta
        N = D.shape[0]

        pos_mask = D > 0
        neg_mask = D < 0

        p_pos = pos_mask.sum(axis=0) / max(N, 1)
        p_neg = neg_mask.sum(axis=0) / max(N, 1)

        pos_count = pos_mask.sum(axis=0)
        neg_count = neg_mask.sum(axis=0)

        pos_sum = np.where(pos_mask, D, 0.0).sum(axis=0)
        neg_sum = np.where(neg_mask, D, 0.0).sum(axis=0)

        m_pos = np.divide(
            pos_sum,
            pos_count,
            out=np.zeros_like(pos_sum, dtype=float),
            where=pos_count > 0
        )

        m_neg = np.divide(
            neg_sum,
            neg_count,
            out=np.zeros_like(neg_sum, dtype=float),
            where=neg_count > 0
        )

        w_pos = p_pos * np.abs(m_pos)
        w_neg = p_neg * np.abs(m_neg)

        denom = w_pos + w_neg

        Wplus = np.divide(
            w_pos,
            denom,
            out=np.full_like(w_pos, 0.5, dtype=float),
            where=denom > 0
        )

        return self._reshape_to_HW(Wplus)

    def map_separability_cohens_d(self):
        if self.x_eval is None or self.y_eval is None:
            raise RuntimeError("x_eval/y_eval no existen; primero ensure_shap().")

        mask = (self.y_eval == self.j_sel) | (self.y_eval == self.r_sel)
        Xp = self.x_eval[mask]
        yp = self.y_eval[mask]

        Xflat = Xp.reshape(len(Xp), -1)
        A = Xflat[yp == self.j_sel]
        B = Xflat[yp == self.r_sel]

        if A.shape[0] < 2 or B.shape[0] < 2:
            d = np.zeros((self.P,), dtype=float)
            return self._reshape_to_HW(d)

        meanA = A.mean(axis=0)
        meanB = B.mean(axis=0)
        varA = A.var(axis=0, ddof=1)
        varB = B.var(axis=0, ddof=1)

        n1 = A.shape[0]
        n2 = B.shape[0]
        pooled = ((n1 - 1) * varA + (n2 - 1) * varB) / max((n1 + n2 - 2), 1)
        pooled_std = np.sqrt(np.maximum(pooled, 1e-12))

        d = (meanA - meanB) / pooled_std
        return self._reshape_to_HW(d)

    # -----------------------
    # Guardado/visualización
    # -----------------------
    def _save_map(
            self,
            M,
            title,
            filename,
            cmap="viridis",
            vmin=None,
            vmax=None,
            cbar_label=None,
            cbar_ticks=None,
            cbar_ticklabels=None
    ):
        out = os.path.join(self.out_dir, filename)

        # Se reserva espacio suficiente a la derecha para que las etiquetas
        # largas de la barra de color no queden recortadas (por ejemplo,
        # "Favours Eosinophil" en BloodMNIST).
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=18, labelpad=10)

        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

        if cbar_ticklabels is not None:
            cbar.set_ticklabels(cbar_ticklabels)

        cbar.ax.tick_params(labelsize=16)

        ax.set_title(title, fontsize=22)
        ax.axis("off")

        # Márgenes fijos para que todos los mapas queden con el mismo layout
        fig.subplots_adjust(left=0.06, right=0.78, top=0.90, bottom=0.06)

        # Guardar sin bbox_inches="tight" para evitar recortes distintos
        fig.savefig(out, dpi=self.dpi)

        if self.show_plots:
            plt.show()

        plt.close(fig)
        return out
    def plot_parallel_summary(self, top_percent=5.0, filename=None):
        if self.x_eval is None or self.y_eval is None:
            raise RuntimeError("Primero corre ensure_shap() para tener x_eval/y_eval.")
        if self.delta is None:
            self.compute_delta_pairwise()

        x = self.x_eval
        y = self.y_eval
        j = self.j_sel
        r = self.r_sel

        mean_j = x[y == j].mean(axis=0).squeeze()
        mean_r = x[y == r].mean(axis=0).squeeze()
        diff_jr = mean_j - mean_r

        # Para imágenes RGB, la diferencia se resume espacialmente
        # mediante el promedio de los tres canales.
        if diff_jr.ndim == 3:
            diff_jr_map = diff_jr.mean(axis=2)
        else:
            diff_jr_map = diff_jr

        diff_lim = np.nanmax(np.abs(diff_jr_map))

        if not np.isfinite(diff_lim) or diff_lim == 0:
            diff_lim = 1.0

        map1 = self.map_mean_abs_delta()
        map2 = self.map_signed_mean_delta()
        map3 = self.map_top_percent_mask(percent=top_percent)
        map4 = self.map_dominance_gdp()
        map7 = self.map_separability_cohens_d()

        pair = self._pair_display_label()
        tag = self._pair_tag()

        fig = plt.figure(figsize=(16, 8))

        ax = plt.subplot(2, 4, 1)
        ax.imshow(mean_j, cmap="gray")
        ax.set_title(f"Mean image: {self._display_cls_name(j)}")
        ax.axis("off")

        ax = plt.subplot(2, 4, 2)
        ax.imshow(mean_r, cmap="gray")
        ax.set_title(f"Mean image: {self._display_cls_name(r)}")
        ax.axis("off")

        ax = plt.subplot(2, 4, 3)

        im = ax.imshow(
            diff_jr_map,
            cmap="PiYG",
            vmin=-diff_lim,
            vmax=diff_lim
        )

        ax.set_title(
            f"Mean diff: "
            f"{self._display_cls_name(j)} - "
            f"{self._display_cls_name(r)}"
        )

        ax.axis("off")
        plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )
        ax = plt.subplot(2, 4, 4)
        im = ax.imshow(map7, cmap="PiYG")
        ax.set_title(f"Cohen's d (data)  ({pair})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = plt.subplot(2, 4, 5)
        im = ax.imshow(map1, cmap="viridis")
        ax.set_title(f"Mean |ΔS|  ({pair})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = plt.subplot(2, 4, 6)
        signed_lim_parallel = np.nanmax(np.abs(map2))
        if not np.isfinite(signed_lim_parallel) or signed_lim_parallel == 0:
            signed_lim_parallel = 1.0
        im = ax.imshow(
            map2,
            cmap="PiYG",
            vmin=-signed_lim_parallel,
            vmax=signed_lim_parallel
        )
        ax.set_title(f"Signed mean ΔS  ({pair})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = plt.subplot(2, 4, 7)
        im = ax.imshow(map4, cmap="coolwarm", vmin=0, vmax=1)
        ax.set_title(f"Dominance W+(p)  ({pair})")
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pixel-wise dominance")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels([
            f"Favours {self._display_cls_name(r)}",
            "Balanced dominance",
            f"Favours {self._display_cls_name(j)}"
        ])

        ax = plt.subplot(2, 4, 8)
        im = ax.imshow(map3, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Top-{top_percent:.1f}% mask on mean|ΔS|")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if filename is None:
            filename = f"Parallel_Summary_{tag}.png"
        out = os.path.join(self.out_dir, filename)
        plt.savefig(out, dpi=self.dpi, bbox_inches="tight")

        if self.show_plots:
            plt.show()
        plt.close()

        print("Parallel summary:", out)

    def run_all_maps(self, shap_mode="auto", compute_kwargs=None, top_percent=5.0, make_parallel=True):
        compute_kwargs = compute_kwargs or {}
        self.ensure_shap(mode=shap_mode, **compute_kwargs)
        self.compute_delta_pairwise()

        tag = self._pair_tag()
        pair_txt = self._pair_display_label()

        A = self.map_mean_abs_delta()
        self._save_map(
            A,
            f"Global Mean |ΔS| \n ({pair_txt})",
            f"Map1_MeanAbsDeltaS_{tag}.png",
            cmap="viridis"
        )

        S = self.map_signed_mean_delta()

        signed_lim = np.nanmax(np.abs(S))
        if not np.isfinite(signed_lim) or signed_lim == 0:
            signed_lim = 1.0

        self._save_map(
            S,
            f"Global Signed Mean ΔS \n ({pair_txt})",
            f"Map2_SignedMeanDeltaS_{tag}.png",
            cmap="PiYG",
            vmin=-signed_lim,
            vmax=signed_lim
        )

        M = self.map_top_percent_mask(percent=top_percent)
        self._save_map(
            M,
            f"Top-{top_percent:.1f}%  mask on mean |ΔS| \n ({pair_txt})",
            f"Map3_Top{top_percent:.1f}pctMask_{tag}.png",
            cmap="gray",
            vmin=0.0,
            vmax=1.0
        )

        Wp = self.map_dominance_gdp()
        self._save_map(
            Wp,
            f"Weighted Dominance map (DP)\n({pair_txt})",
            f"Map4_DominanceWplus_{tag}.png",
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Weighted dominance",
            cbar_ticks=[0.0, 0.5, 1.0],
            cbar_ticklabels=[
                f"Favours\n{self._display_cls_name(self.r_sel)}",
                "Balanced\ndominance",
                f"Favours\n{self._display_cls_name(self.j_sel)}"
            ]
        )

        PosMap, NegMap = self.map_conditional_pos_neg()
        self._save_map(
            PosMap,
            f"Conditional Mean  ΔS>0 \n ({pair_txt})",
            f"Map5_CondMean_Pos_{tag}.png",
            cmap="YlOrRd"
        )
        self._save_map(
            NegMap,
            f"Conditional Mean  ΔS<0 \n ({pair_txt})",
            f"Map6_CondMean_Neg_{tag}.png",
            cmap="PuBu"
        )

        Dmap = self.map_separability_cohens_d()
        self._save_map(
            Dmap,
            f"Separability Map  (Cohen’s d)  \n  ({pair_txt})",
            f"Map7_CohensD_{tag}.png",
            cmap="PiYG"
        )

        # guardados de depuración/verificación
        np.save(os.path.join(self.out_dir, f"x_eval_{tag}.npy"), self.x_eval)
        np.save(os.path.join(self.out_dir, f"y_eval_{tag}.npy"), self.y_eval)
        np.save(os.path.join(self.out_dir, f"delta_{tag}.npy"), self.delta)

        print("\nOK")
        print("Par:", pair_txt)
        print("x_eval:", self.x_eval.shape)
        print("y_eval:", self.y_eval.shape)
        print("shap_arr (N,P,K):", self.shap_arr.shape)
        print("delta (N_pair,P):", self.delta.shape)
        print(f"Mapas guardados en: {self.out_dir}")

        if make_parallel:
            self.plot_parallel_summary(top_percent=top_percent)
