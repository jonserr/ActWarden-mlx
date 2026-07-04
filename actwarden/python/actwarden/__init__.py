"""ActWarden training plane.

Python owns: trace synthesis and replay, the reference feature packer, model
training, export to .mlxfn, and offline evaluation. The C++ runtime under
actwarden/runtime owns live scoring and policy enforcement.

Feature semantics are defined by actwarden/feature_spec/v1.json, not by code.
"""

__version__ = "0.1.0"

from actwarden.traces import Span, group_by_session, read_jsonl  # noqa: F401
from actwarden.features import FeaturePacker, load_spec  # noqa: F401
