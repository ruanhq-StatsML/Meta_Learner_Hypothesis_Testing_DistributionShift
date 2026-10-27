"""Validate a site upload against the stage-0 contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_SCHEMA_PATH = Path(__file__).with_name("site_upload.schema.json")
CORE_FIELDS = (
    "site_id",
    "vimp_cov",
    "vimp_con",
    "z_mmd",
    "z_r",
    "calibration_ok",
    "D_k",
    "D_k_ci",
    "n_k",
    "feature_mask",
)


def load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_upload(payload: Mapping[str, Any], *, simplex_tol: float = 1e-6) -> None:
    """Raise ValueError if `payload` breaks the upload contract.

    `diagnostics` is optional and unchecked beyond being an object. Every
    other top-level key is rejected so a causal estimand cannot be smuggled
    in beside the association vectors.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("upload must be an object")
    extra = set(payload) - set(CORE_FIELDS) - {"diagnostics"}
    if extra:
        raise ValueError("unexpected fields: {0}".format(sorted(extra)))
    missing = [key for key in CORE_FIELDS if key not in payload]
    if missing:
        raise ValueError("missing fields: {0}".format(missing))
    if not isinstance(payload["site_id"], str) or not payload["site_id"]:
        raise ValueError("site_id must be a non-empty string")

    cov = _vector("vimp_cov", payload["vimp_cov"])
    con = _vector("vimp_con", payload["vimp_con"])
    mask = payload["feature_mask"]
    if not isinstance(mask, (list, tuple)) or not mask:
        raise ValueError("feature_mask must be a non-empty bool array")
    if not all(isinstance(item, bool) for item in mask):
        raise ValueError("feature_mask must be boolean")
    if not (len(cov) == len(con) == len(mask)):
        raise ValueError("vimp_cov, vimp_con, and feature_mask must share length p")
    _simplex("vimp_cov", cov, simplex_tol)
    _simplex("vimp_con", con, simplex_tol)
    for key in ("z_mmd", "z_r", "D_k"):
        if not _is_number(payload[key]):
            raise ValueError("{0} must be a float".format(key))
    ci = payload["D_k_ci"]
    if not isinstance(ci, (list, tuple)) or len(ci) != 2 or not all(_is_number(v) for v in ci):
        raise ValueError("D_k_ci must be a pair of floats")
    if float(ci[0]) > float(ci[1]):
        raise ValueError("D_k_ci lower bound exceeds upper bound")
    if not isinstance(payload["calibration_ok"], bool):
        raise ValueError("calibration_ok must be a bool")
    n_k = payload["n_k"]
    if isinstance(n_k, bool) or not isinstance(n_k, int) or n_k < 0:
        raise ValueError("n_k must be a non-negative int")
    if "diagnostics" in payload and not isinstance(payload["diagnostics"], Mapping):
        raise ValueError("diagnostics must be an object")


def _vector(name: str, value: Any) -> list:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("{0} must be a non-empty array".format(name))
    if not all(_is_number(item) for item in value):
        raise ValueError("{0} must contain only numbers".format(name))
    return [float(item) for item in value]


def _simplex(name: str, values: list, tol: float) -> None:
    if any(item < -tol for item in values):
        raise ValueError("{0} has a negative coordinate".format(name))
    total = sum(max(item, 0.0) for item in values)
    if abs(total - 1.0) > tol:
        raise ValueError("{0} simplex error {1:.3e} exceeds {2:.3e}".format(name, abs(total - 1.0), tol))
