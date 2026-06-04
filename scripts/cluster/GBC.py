# -*- coding: utf-8 -*-
"""
Granular Ball Clustering (GBC) for COARSE grouping (no preset final K)
Pure NumPy, self-contained. Optional matplotlib for visualization.

Designed for small-N, high-dim embeddings (e.g., CLIP 512-d):
- L2 normalize embeddings (recommended for CLIP)
- Density uses n / r^2 to avoid high-dim instability
- Clusters are formed via ball-graph connected components
- Post-process merges singleton/tiny clusters to nearest big cluster (coarse-friendly)

Run:
  python GBC.py --npy_path D:/Code/Hsg_gen/scripts/output/text_rel.npy --save gbc.png

Tips:
- If clusters too碎: increase neighbor_ratio OR increase min_child_size
- If clusters too粘: decrease neighbor_ratio
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np


# -----------------------------
# Optional matplotlib
# -----------------------------
def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


# -----------------------------
# Granular Ball
# -----------------------------
@dataclass
class GranularBall:
    idx: np.ndarray
    center: np.ndarray
    radius: float
    density: float
    cc: float
    cluster_id: int = -1
    is_noise: bool = False

    @property
    def n(self) -> int:
        return int(self.idx.size)


def _l2norm_to_center(pts: np.ndarray, center: np.ndarray) -> np.ndarray:
    diff = pts - center[None, :]
    return np.sqrt(np.sum(diff * diff, axis=1))


def compute_ball(X: np.ndarray, idx: np.ndarray, eps: float = 1e-12) -> GranularBall:
    """
    Compute ball properties:
      - center: mean
      - radius: max distance to center
      - density: n / r^2  (stable for high-dim embeddings)
      - cc: compare density inside (r/2) vs overall density
    """
    if idx.size == 0:
        raise ValueError("Empty ball idx")

    pts = X[idx]
    center = pts.mean(axis=0)
    dist = _l2norm_to_center(pts, center)
    r = float(max(dist.max(), eps))

    # density robust for high-dim
    density = float(len(idx) / (r * r + eps))

    inner = dist <= (r * 0.5)
    inner_cnt = int(inner.sum())
    if inner_cnt > 0:
        inner_density = float(inner_cnt / ((r * 0.5) ** 2 + eps))
        cc = float(min(density, inner_density) / (max(density, inner_density) + eps))
    else:
        cc = 0.0

    return GranularBall(idx=idx.copy(), center=center, radius=r, density=density, cc=cc)


# -----------------------------
# KMeans++ (pure NumPy)
# -----------------------------
def kmeans_pp(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 0) -> np.ndarray:
    """
    Simple kmeans with kmeans++ init. Returns labels in [0..k-1].
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if k <= 1:
        return np.zeros(n, dtype=np.int64)
    if k > n:
        k = n

    # init centers
    centers = np.empty((k, d), dtype=X.dtype)
    centers[0] = X[rng.integers(0, n)]
    d2 = np.sum((X - centers[0]) ** 2, axis=1)

    for i in range(1, k):
        probs = d2 / (d2.sum() + 1e-12)
        centers[i] = X[rng.choice(n, p=probs)]
        new_d2 = np.sum((X - centers[i]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = dist2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            idx = np.where(labels == i)[0]
            if len(idx) > 0:
                centers[i] = X[idx].mean(axis=0)
            else:
                centers[i] = X[rng.integers(0, n)]
    return labels


# -----------------------------
# Post-process: merge small clusters
# -----------------------------
def merge_small_clusters(X: np.ndarray, labels: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    """
    Merge tiny clusters (size < min_cluster_size) into nearest BIG cluster (by centroid distance).
    This stabilizes coarse clustering for small N and avoids many singletons.
    """
    labels = labels.astype(np.int64).copy()
    X = X.astype(np.float64, copy=False)

    uniq = [int(u) for u in np.unique(labels) if u >= 0]
    if len(uniq) <= 1:
        return labels

    # sizes + centers
    sizes: Dict[int, int] = {u: int(np.sum(labels == u)) for u in uniq}
    centers: Dict[int, np.ndarray] = {}
    for u in uniq:
        idx = np.where(labels == u)[0]
        centers[u] = X[idx].mean(axis=0)

    big = [u for u in uniq if sizes[u] >= min_cluster_size]
    if len(big) == 0:
        return labels  # all tiny, do nothing

    # merge each tiny cluster into nearest big cluster
    for u in uniq:
        if sizes[u] >= min_cluster_size:
            continue
        cu = centers[u]
        best_v = None
        best_d = 1e18
        for v in big:
            d = float(np.linalg.norm(cu - centers[v]))
            if d < best_d:
                best_d = d
                best_v = v
        if best_v is not None:
            labels[labels == u] = best_v

    # relabel contiguous 0..K-1
    uniq2 = sorted([int(u) for u in np.unique(labels) if u >= 0])
    remap = {old: i for i, old in enumerate(uniq2)}
    labels = np.vectorize(lambda x: remap.get(int(x), -1))(labels).astype(np.int64)
    return labels


# -----------------------------
# GBC Coarse
# -----------------------------
class GranularBallClusteringCoarse:
    """
    No preset final K.
    Steps:
      1) initial division into k0 balls
      2) adaptive splitting by CC + density gain
      3) build ball graph by boundary distance and connected components -> clusters
      4) optional: merge tiny clusters -> nearest big cluster (recommended for small N)
    """

    def __init__(
        self,
        # splitting
        cc_threshold: float = 0.82,
        min_ball_size: int = 4,
        min_child_size: int = 3,
        density_gain: float = 1.02,
        # ball-graph neighbor threshold
        neighbor_ratio: float = 0.12,   # eps = neighbor_ratio * median_radius
        # merge tiny clusters
        merge_min_cluster_size: int = 2,
        # init k0
        init_k0_min: int = 6,
        init_k0_max: int = 12,
        random_state: int = 0,
        verbose: bool = True,
    ):
        self.cc_threshold = float(cc_threshold)
        self.min_ball_size = int(min_ball_size)
        self.min_child_size = int(min_child_size)
        self.density_gain = float(density_gain)
        self.neighbor_ratio = float(neighbor_ratio)
        self.merge_min_cluster_size = int(merge_min_cluster_size)
        self.init_k0_min = int(init_k0_min)
        self.init_k0_max = int(init_k0_max)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self.balls_: List[GranularBall] = []
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: int = 0
        self.n_splits_: int = 0

    def fit(self, X: np.ndarray) -> "GranularBallClusteringCoarse":
        X = self._validate(X)

        balls = self._initial_division(X)
        if self.verbose:
            print(f"[Init] balls={len(balls)}")

        balls = self._adaptive_splitting(X, balls)
        if self.verbose:
            print(f"[Split] balls={len(balls)}, splits={self.n_splits_}")

        self._cluster_balls_by_graph(balls)
        self._assign_point_labels(balls, X)

        # merge singletons/tiny clusters (recommended for coarse)
        if self.merge_min_cluster_size > 1 and self.labels_ is not None:
            self.labels_ = merge_small_clusters(X, self.labels_, min_cluster_size=self.merge_min_cluster_size)
            self.n_clusters_ = len(np.unique(self.labels_))

        self.balls_ = balls
        if self.verbose:
            print(f"[Done] clusters={self.n_clusters_}, cluster_sizes={self.cluster_sizes()}")
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    # ----- Step1: initial division (k0, not final K) -----
    def _initial_division(self, X: np.ndarray) -> List[GranularBall]:
        n = X.shape[0]
        k0 = max(self.init_k0_min, min(self.init_k0_max, int(np.sqrt(n)) * 2))  # N=29 -> 10
        labels0 = kmeans_pp(X, k0, max_iter=50, seed=self.random_state)

        balls = []
        for i in range(k0):
            idx = np.where(labels0 == i)[0]
            if len(idx) > 0:
                balls.append(compute_ball(X, idx))
        return balls

    # ----- Step2: split -----
    def _adaptive_splitting(self, X: np.ndarray, balls: List[GranularBall]) -> List[GranularBall]:
        max_iter = 200
        rng = np.random.default_rng(self.random_state)
        it = 0

        while it < max_iter:
            it += 1
            changed = False
            new_balls: List[GranularBall] = []

            for b in balls:
                # split condition
                if b.n < self.min_ball_size or b.cc >= self.cc_threshold:
                    new_balls.append(b)
                    continue

                pts = X[b.idx]
                if pts.shape[0] < (self.min_child_size * 2):
                    new_balls.append(b)
                    continue

                labels2 = kmeans_pp(pts, 2, max_iter=30, seed=int(rng.integers(0, 10**9)))
                idx1 = b.idx[np.where(labels2 == 0)[0]]
                idx2 = b.idx[np.where(labels2 == 1)[0]]

                if len(idx1) < self.min_child_size or len(idx2) < self.min_child_size:
                    new_balls.append(b)
                    continue

                c1 = compute_ball(X, idx1)
                c2 = compute_ball(X, idx2)

                # accept if avg density improves enough
                parent = b.density
                avg_child = 0.5 * (c1.density + c2.density)

                if avg_child > parent * self.density_gain:
                    new_balls.extend([c1, c2])
                    self.n_splits_ += 1
                    changed = True
                else:
                    new_balls.append(b)

            balls = new_balls
            if not changed:
                if self.verbose:
                    print(f"[Split] converged at iter={it}")
                break

        return balls

    # ----- Step3: ball graph clustering -----
    def _cluster_balls_by_graph(self, balls: List[GranularBall]) -> None:
        # build adjacency
        if len(balls) == 0:
            self.n_clusters_ = 0
            return

        radii = np.array([b.radius for b in balls], dtype=np.float64)
        r_med = float(np.median(radii))
        eps = self.neighbor_ratio * r_med

        adj = [[] for _ in range(len(balls))]
        for i in range(len(balls)):
            bi = balls[i]
            for j in range(i + 1, len(balls)):
                bj = balls[j]
                center_dist = float(np.linalg.norm(bi.center - bj.center))
                boundary_dist = center_dist - (bi.radius + bj.radius)
                # neighbor if overlap or within eps
                if boundary_dist <= eps:
                    adj[i].append(j)
                    adj[j].append(i)

        # connected components
        comp = -np.ones(len(balls), dtype=np.int64)
        cid = 0
        for i in range(len(balls)):
            if comp[i] != -1:
                continue
            stack = [i]
            comp[i] = cid
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if comp[v] == -1:
                        comp[v] = cid
                        stack.append(v)
            cid += 1

        for i, b in enumerate(balls):
            b.cluster_id = int(comp[i])
        self.n_clusters_ = cid

    # ----- Step4: assign point labels from balls -----
    def _assign_point_labels(self, balls: List[GranularBall], X: np.ndarray) -> None:
        n = X.shape[0]
        labels = np.full(n, -1, dtype=np.int64)
        for b in balls:
            labels[b.idx] = b.cluster_id

        # make contiguous
        uniq = sorted([int(u) for u in np.unique(labels) if u >= 0])
        remap = {old: i for i, old in enumerate(uniq)}
        labels = np.vectorize(lambda x: remap.get(int(x), -1))(labels).astype(np.int64)

        self.labels_ = labels
        self.n_clusters_ = len(np.unique(labels))

    # ----- stats -----
    def cluster_sizes(self) -> List[int]:
        if self.labels_ is None:
            return []
        sizes = []
        for k in range(self.n_clusters_):
            sizes.append(int(np.sum(self.labels_ == k)))
        return sizes

    # ----- PCA plot -----
    @staticmethod
    def _pca2(X: np.ndarray) -> np.ndarray:
        Xc = X - X.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:2].T

    def plot(
        self,
        X: np.ndarray,
        title: str = "GBC Coarse",
        show_balls: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        plt = _try_import_matplotlib()
        if plt is None:
            raise RuntimeError("matplotlib not available. Install via: pip install matplotlib")

        if self.labels_ is None:
            raise RuntimeError("Call fit() before plot().")

        X = self._validate(X)
        X2 = self._pca2(X) if X.shape[1] > 2 else X

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=self.labels_, s=35, alpha=0.85)
        ax.set_title(f"{title} (clusters={self.n_clusters_}, balls={len(self.balls_)})")
        ax.set_xlabel("PCA-1" if X.shape[1] > 2 else "x")
        ax.set_ylabel("PCA-2" if X.shape[1] > 2 else "y")
        ax.grid(True, alpha=0.25)
        plt.colorbar(sc, ax=ax)

        if show_balls and self.balls_:
            for b in self.balls_:
                pts2 = X2[b.idx]
                c2 = pts2.mean(axis=0)
                r2 = float(np.max(np.linalg.norm(pts2 - c2[None, :], axis=1)))
                circ = plt.Circle((c2[0], c2[1]), r2, fill=False, linewidth=1.2, alpha=0.55)
                ax.add_patch(circ)
                ax.scatter([c2[0]], [c2[1]], marker="x", s=80, alpha=0.9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()

    # ----- validate -----
    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be (N,D) with N>=2")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN/Inf")
        return X


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--npy_path", type=str, required=True)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_norm", action="store_true", help="disable L2 normalization (not recommended for CLIP)")
    args = parser.parse_args()

    X = np.load(r"D:\Code\Hsg_gen\scripts\output\text_rel.npy")
    print(f"Loaded X: shape={X.shape}, dtype={X.dtype}")

    X = X.astype(np.float64)
    if not args.no_norm:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    gbc = GranularBallClusteringCoarse(
        cc_threshold=0.90,      # 0.82 -> 0.90  (更容易触发分裂)
        min_ball_size=1,        # 4 -> 3
        min_child_size=8,       # 3 -> 2  (小样本必须降)
        density_gain=1.01,      # 1.02 -> 1.00 (只要不变差就允许分裂)
        neighbor_ratio=0.12,    # 0.12 -> 0.06 (更严格的连边)
        merge_min_cluster_size=2,
        init_k0_min=6,          # 6 -> 8 (初始更多球)
        init_k0_max=12,         # 12 -> 14
        random_state=0,
        verbose=True,
    )

    labels = gbc.fit_predict(X)
    print("labels:", labels.tolist())
    print("cluster_sizes:", gbc.cluster_sizes())

    if not args.no_plot:
        gbc.plot(X, title="My CLIP Embedding GBC (coarse)", show_balls=True, save_path=args.save)


if __name__ == "__main__":
    main()
