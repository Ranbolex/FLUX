# FLUX Anonymous Algorithm Core

This package contains only FLUX algorithm kernels and their supporting
structures. It excludes benchmark adapters, experiment drivers, comparison
methods, paper tables, historical data, run records, and provider credentials.

## Modules

- `hybrid_adaptive_controller.py`: frozen budget-, scale-, difficulty-, and
  collapse-aware gates for candidate expansion, node addition/removal,
  rewiring, and minimum-team preservation. Original profile constants remain
  unchanged.
- `runtime_metrics.py`: runtime feedback from graph events, message delivery
  and consumption, reachable participation, and expert reports.
- `graph_metrics.py` and `collapse.py`: structural and semantic collapse
  diagnostics.
- `role_forge.py` and `roles.py`: task-architect instructions, role-plan
  parsing/validation, and role selection.
- `budget.py`: shared resource limits, reservations, and accounting.

The implementations retain their original logic. Only package import paths
were adjusted to form the standalone `flux_core` namespace.

## Use

Python 3.11 or newer is recommended. From this directory:

```text
python -m pip install -r requirements.txt
```

```python
from flux_core import get_hybrid_control_profile, role_architect_instruction
from flux_core.runtime_metrics import build_runtime_trace

profile = get_hybrid_control_profile(
    "webshop-hybrid-budget-collapse-v2", "webshop"
)
instruction = role_architect_instruction("reasoning")
```

The caller supplies visible task signals and runtime records, invokes a model
if required, and applies eligible graph edits in its own execution system.
This is an algorithm library, not an end-to-end experiment runner. It does
not reproduce the paper tables by itself.

Semantic diagnostics optionally use `torch` and `sentence-transformers`
with a locally available encoder. These dependencies are not required for
structural metrics; the original lexical fallback is retained. Encoder
downloads remain disabled unless explicitly enabled by the caller.

## Anonymity and rights

No author identity, affiliation, contact address, private repository link,
personal path, or historical run metadata is included. ZIP timestamps and
attributes are normalized. `MANIFEST_SHA256.csv` indexes release files.

No new blanket license is assigned by this packaging operation. See
`RELEASE_NOTICE.txt` for the distribution scope.
