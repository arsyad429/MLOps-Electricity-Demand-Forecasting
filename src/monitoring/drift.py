"""Versioned reference histograms and Population Stability Index for model inputs."""

import json
import math
import re
from bisect import bisect_right
from dataclasses import asdict, dataclass
from enum import IntEnum


class DriftStatus(IntEnum):
    INSUFFICIENT_DATA = -1
    STABLE = 0
    MODERATE_DRIFT = 1
    SIGNIFICANT_DRIFT = 2


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    version: str

    def __post_init__(self):
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name)
                or not re.fullmatch(r"[1-9][0-9]*", self.version)):
            raise ValueError("Expected a registered model name and an immutable version.")

    @property
    def reference_filename(self):
        return f"{self.name}__{self.version}.json"


@dataclass(frozen=True)
class FeatureReference:
    boundaries: tuple[float, ...]
    proportions: tuple[float, ...]
    constant: float | None = None


@dataclass(frozen=True)
class DriftReference:
    model: ModelIdentity
    sample_count: int
    requested_bins: int
    features: dict[str, FeatureReference]

    def to_dict(self):
        return {"schema_version": 1, "feature_names": list(self.features), **asdict(self)}


@dataclass(frozen=True)
class FeatureDrift:
    production_samples: int
    psi: float | None = None
    status: DriftStatus = DriftStatus.INSUFFICIENT_DATA


@dataclass(frozen=True)
class DriftResult:
    status: DriftStatus
    production_samples: int
    reference_samples: int
    invalid_payloads: int
    features: dict[str, FeatureDrift]


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _counts(values, boundaries, constant):
    if constant is not None:
        counts = [0, 0, 0]
        for value in values:
            counts[int(value >= constant) + int(value > constant)] += 1
    else:
        counts = [0] * (len(boundaries) + 1)
        for value in values:
            counts[bisect_right(boundaries, value)] += 1
    return counts


def _bin_boundaries(values, bins):
    distinct = sorted(set(values))
    if len(distinct) == 1:
        return (), distinct[0]
    if len(distinct) <= bins:
        cuts = [left / 2 + right / 2 for left, right in zip(distinct, distinct[1:])]
        return tuple(cut if cut > left else right for cut, left, right in zip(cuts, distinct, distinct[1:])), None
    # Nearest-rank quantiles. Remove repeated cuts and a cut at the training minimum.
    cuts = {values[(index * len(values) - 1) // bins] for index in range(1, bins)}
    return tuple(sorted(cuts - {values[0]})), None


def build_reference(model, columns, *, bins=10):
    """Build fixed bins from ordered, already filtered training feature columns."""
    if type(bins) is not int or bins < 3 or not columns:
        raise ValueError("A reference requires features and at least three requested bins.")
    lengths = {len(column) for column in columns.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("Reference columns must contain an equal, positive number of rows.")
    sample_count, = lengths
    features = {}
    for name, column in columns.items():
        if not name or not all(map(_finite, column)):
            raise ValueError("Reference features must have finite numeric values.")
        values = sorted(column)
        boundaries, constant = _bin_boundaries(values, bins)
        counts = _counts(values, boundaries, constant)
        features[name] = FeatureReference(boundaries, tuple(count / sample_count for count in counts), constant)
    return DriftReference(model, sample_count, bins, features)


def load_reference(path, model):
    """Read and validate a versioned reference, including its model, names and bins."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if type(data) is not dict or data.get("schema_version") != 1 or data.get("model") != asdict(model):
        raise ValueError("Reference format or registered model version does not match.")
    count = data.get("sample_count")
    bins = data.get("requested_bins")
    entries = data.get("features")
    if (type(count) is not int or count < 1 or type(bins) is not int or bins < 3
            or type(entries) is not dict or not entries or data.get("feature_names") != list(entries)):
        raise ValueError("Reference feature schema is invalid.")
    features = {}
    for name, entry in entries.items():
        if not name or type(entry) is not dict:
            raise ValueError("Reference feature schema is invalid.")
        boundaries = entry.get("boundaries")
        proportions = entry.get("proportions")
        constant = entry.get("constant")
        if (type(boundaries) is not list or not all(map(_finite, boundaries))
                or sorted(set(boundaries)) != boundaries or type(proportions) is not list
                or len(proportions) != (3 if constant is not None else len(boundaries) + 1)
                or len(proportions) > bins or not all(_finite(p) and 0 <= p <= 1 for p in proportions)
                or not math.isclose(sum(proportions), 1.0)
                or (constant is not None and (not _finite(constant) or boundaries))):
            raise ValueError("Reference histogram is invalid.")
        features[name] = FeatureReference(tuple(boundaries), tuple(proportions), constant)
    return DriftReference(model, count, bins, features)


def _psi(expected, counts):
    # Smooth proportions symmetrically, so sample sizes alone do not move identical distributions.
    epsilon = 0.0001
    total = sum(counts)
    scale = 1 + len(expected) * epsilon
    expected = [(p + epsilon) / scale for p in expected]
    actual = [(count / total + epsilon) / scale for count in counts]
    return math.fsum((a - e) * math.log(a / e) for a, e in zip(actual, expected))


def evaluate_drift(model, reference, records, *, minimum=100, moderate=0.10, significant=0.25):
    """Score unique origin/payload pairs using only the specified model's training reference."""
    if (type(minimum) is not int or minimum < 1 or not _finite(moderate) or not _finite(significant)
            or not 0 < moderate < significant):
        raise ValueError("Invalid PSI sample gate or thresholds.")
    if reference is None or reference.model != model:
        return DriftResult(DriftStatus.INSUFFICIENT_DATA, 0, 0, 0, {})

    names = tuple(reference.features)
    columns = {name: [] for name in names}
    seen = set()
    samples = invalid = 0
    for observation, payload in records:
        if observation in seen:
            continue
        seen.add(observation)
        if not isinstance(payload, dict) or tuple(payload) != names:
            invalid += 1
            continue
        samples += 1
        for name in names:
            if _finite(payload[name]):
                columns[name].append(payload[name])

    features = {}
    for name, spec in reference.features.items():
        column = columns[name]
        if len(column) < minimum:
            features[name] = FeatureDrift(len(column))
            continue
        psi = _psi(spec.proportions, _counts(column, spec.boundaries, spec.constant))
        if psi >= significant:
            status = DriftStatus.SIGNIFICANT_DRIFT
        elif psi >= moderate:
            status = DriftStatus.MODERATE_DRIFT
        else:
            status = DriftStatus.STABLE
        features[name] = FeatureDrift(len(column), psi, status)

    statuses = [feature.status for feature in features.values()]
    overall = max(statuses)
    if overall == DriftStatus.STABLE and DriftStatus.INSUFFICIENT_DATA in statuses:
        overall = DriftStatus.INSUFFICIENT_DATA
    return DriftResult(overall, samples, reference.sample_count, invalid, features)
