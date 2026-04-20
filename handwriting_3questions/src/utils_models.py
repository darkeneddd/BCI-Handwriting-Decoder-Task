"""
utils_models.py
────────────────
Model factory, grid search, and evaluation utilities shared across all
experiment scripts.

Supports:
  Linear:          LogisticRegression, LinearSVC, RidgeClassifier
  Non-linear:      SVC (RBF kernel), MLPClassifier

Rules enforced here:
  - StandardScaler is always fitted on training data only (inside the pipeline)
  - Hyperparameter selection uses validation accuracy exclusively
  - Test set is evaluated only once, after hyperparameter selection
"""

from __future__ import annotations

import pickle
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model   import LogisticRegression, RidgeClassifier
from sklearn.metrics        import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.svm            import LinearSVC, SVC


# ─── Label encoding ───────────────────────────────────────────────────────────

def make_label_encoder(labels: np.ndarray) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(labels)
    return le


# ─── Pipeline factories ───────────────────────────────────────────────────────

def make_linear_pipeline(model_name: str, hparams: dict) -> Pipeline:
    """
    Build a StandardScaler → classifier pipeline for linear models.

    Parameters
    ----------
    model_name : 'LogisticRegression' | 'LinearSVC' | 'RidgeClassifier'
    hparams    : dict of hyperparameters for the chosen classifier
    """
    if model_name == "LogisticRegression":
        clf = LogisticRegression(
            C=hparams["C"],
            max_iter=hparams.get("max_iter", 2000),
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )
    elif model_name == "LinearSVC":
        clf = LinearSVC(
            C=hparams["C"],
            max_iter=hparams.get("max_iter", 3000),
            random_state=42,
        )
    elif model_name == "RidgeClassifier":
        clf = RidgeClassifier(alpha=hparams["alpha"])
    else:
        raise ValueError(f"Unknown linear model: {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def make_svc_rbf_pipeline(hparams: dict) -> Pipeline:
    """Build a StandardScaler → SVC(RBF kernel) pipeline."""
    clf = SVC(
        C=hparams["C"],
        kernel="rbf",
        gamma=hparams.get("gamma", "scale"),
        random_state=42,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def make_mlp_pipeline(hparams: dict) -> Pipeline:
    """Build a StandardScaler → MLPClassifier pipeline."""
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(hparams["hidden_layer_sizes"]),
        alpha=hparams["alpha"],
        learning_rate_init=hparams["learning_rate_init"],
        max_iter=hparams.get("max_iter", 500),
        early_stopping=hparams.get("early_stopping", True),
        validation_fraction=hparams.get("validation_fraction", 0.1),
        n_iter_no_change=hparams.get("n_iter_no_change", 15),
        random_state=42,
        verbose=False,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", mlp)])


# ─── Hyperparameter grids from config ────────────────────────────────────────

def hparam_grid_from_config(model_name: str, model_cfg: dict) -> dict[str, list]:
    """
    Extract list-valued hyperparameters from the config sub-dict for a model.

    Returns a dict whose values are lists (suitable for itertools.product).
    """
    return {k: v for k, v in model_cfg.items() if isinstance(v, list)}


# ─── Generic grid search ──────────────────────────────────────────────────────

def grid_search_linear(model_name: str,
                       hparam_grid: dict,
                       X_train, y_train,
                       X_val, y_val,
                       extra_hparams: dict | None = None
                       ) -> tuple[Pipeline, dict, float]:
    """
    Grid-search linear model hyperparameters; select by val accuracy.

    Parameters
    ----------
    model_name   : classifier name string
    hparam_grid  : dict of {param_name: [values]}
    X_train / y_train / X_val / y_val : numpy arrays
    extra_hparams : fixed hyperparams (e.g. max_iter) not in the grid

    Returns
    -------
    (best_pipeline, best_hparams_dict, best_val_accuracy)
    """
    base = extra_hparams or {}
    best_acc, best_pipe, best_hp = -1.0, None, {}

    param_names = list(hparam_grid.keys())
    for combo in product(*[hparam_grid[k] for k in param_names]):
        hp = {**base, **dict(zip(param_names, combo))}
        pipe = make_linear_pipeline(model_name, hp)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, pred)
        if acc > best_acc:
            best_acc, best_pipe, best_hp = acc, pipe, hp

    return best_pipe, best_hp, best_acc


def grid_search_svc_rbf(svc_cfg: dict,
                        X_train, y_train,
                        X_val, y_val) -> tuple[Pipeline, dict, float]:
    """Grid-search SVC(RBF); select by val accuracy."""
    best_acc, best_pipe, best_hp = -1.0, None, {}

    C_options     = svc_cfg.get("C",     [0.1, 1.0, 10.0, 100.0])
    gamma_options = svc_cfg.get("gamma", ["scale", "auto"])

    n_combos = len(C_options) * len(gamma_options)
    idx = 0
    for C, gamma in product(C_options, gamma_options):
        idx += 1
        hp = {"C": C, "gamma": gamma}
        pipe = make_svc_rbf_pipeline(hp)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, pred)
        print(f"    [{idx}/{n_combos}] C={C} gamma={gamma} → val acc={acc:.3f}")
        if acc > best_acc:
            best_acc, best_pipe, best_hp = acc, pipe, hp

    return best_pipe, best_hp, best_acc


def grid_search_mlp(mlp_cfg: dict,
                    X_train, y_train,
                    X_val, y_val) -> tuple[Pipeline, dict, float]:
    """Grid-search MLP; select by val accuracy."""
    best_acc, best_pipe, best_hp = -1.0, None, {}

    hidden_opts = [list(h) for h in mlp_cfg.get("hidden_layer_sizes", [[128]])]
    alpha_opts  = mlp_cfg.get("alpha", [1e-4])
    lr_opts     = mlp_cfg.get("learning_rate_init", [1e-3])

    n_combos = len(hidden_opts) * len(alpha_opts) * len(lr_opts)
    idx = 0
    for hidden, alpha, lr in product(hidden_opts, alpha_opts, lr_opts):
        idx += 1
        hp = {
            "hidden_layer_sizes":   hidden,
            "alpha":                alpha,
            "learning_rate_init":   lr,
            "max_iter":             mlp_cfg.get("max_iter", 500),
            "early_stopping":       mlp_cfg.get("early_stopping", True),
            "validation_fraction":  mlp_cfg.get("validation_fraction", 0.1),
            "n_iter_no_change":     mlp_cfg.get("n_iter_no_change", 15),
        }
        pipe = make_mlp_pipeline(hp)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        acc = accuracy_score(y_val, pred)
        print(f"    [{idx}/{n_combos}] hidden={hidden} alpha={alpha} lr={lr} "
              f"→ val f1={f1:.3f} acc={acc:.3f}")
        if acc > best_acc:
            best_acc, best_pipe, best_hp = acc, pipe, hp

    return best_pipe, best_hp, best_acc


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(pipeline: Pipeline,
                   X: np.ndarray,
                   y: np.ndarray,
                   label_encoder: LabelEncoder) -> dict[str, Any]:
    """
    Evaluate a fitted pipeline on a given split.

    Returns
    -------
    dict with keys: accuracy, macro_f1, confusion_matrix (np array), y_pred
    """
    y_pred = pipeline.predict(X)
    acc    = accuracy_score(y, y_pred)
    f1     = f1_score(y, y_pred, average="macro", zero_division=0)
    cm     = confusion_matrix(y, y_pred,
                              labels=np.arange(len(label_encoder.classes_)))
    return {
        "accuracy":         float(acc),
        "macro_f1":         float(f1),
        "confusion_matrix": cm,
        "y_pred":           y_pred,
    }


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline,
               hparams: dict,
               label_encoder: LabelEncoder,
               path: Path) -> None:
    """Pickle a fitted pipeline + metadata to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "pipeline":      pipeline,
            "hparams":       hparams,
            "label_encoder": label_encoder,
            "classes":       list(label_encoder.classes_),
        }, f)


def load_model(path: Path) -> dict:
    """Load a pickled model bundle from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
