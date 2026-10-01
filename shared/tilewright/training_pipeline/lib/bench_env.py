# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Common env knobs every hipblaslt-bench invocation in the pipeline
must set, in one place.

`TENSILE_PREDICTION_LIB=1` makes Tensile route MasterSolutionLibrary's
findTopSolutions through `ProblemPredictionLibrary` -- the path
stage06_deploy patches the ML recommender into. Without it,
hipblaslt-bench takes the legacy Equality/Range-only path and we never
exercise the deployed model.

Stages that invoke hipblaslt-bench should funnel through
`bench_env_with(...)` rather than building their own env dict, so this
flag is impossible to forget.
"""
from __future__ import annotations

import os
from typing import Dict, Optional


# Set on EVERY hipblaslt-bench invocation made by this pipeline.
# Centralised here so any future addition (TENSILE_DB, fault-injection
# flags, etc.) is a one-line change.
_REQUIRED_BENCH_ENV: Dict[str, str] = {
    "TENSILE_PREDICTION_LIB": "1",
    # Disable HSA hot-swap for every bench: without it the runtime can migrate
    # queues mid-measurement, which shows up as unexplained latency outliers and
    # (occasionally) a wedged queue.
    "HSA_HOTSWAP_DISABLE": "1",
}


def bench_env_with(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a full env dict (a copy of os.environ) with the required
    bench knobs merged in. `extra` -- if provided -- overrides the
    defaults on a per-key basis (e.g. stage07_sel_time passes
    ``ML_RECOMMENDER_DISABLE=1`` to turn the ML recommender OFF while
    keeping ``TENSILE_PREDICTION_LIB=1``)."""
    env = os.environ.copy()
    env.update(_REQUIRED_BENCH_ENV)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env
