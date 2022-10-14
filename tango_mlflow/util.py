from collections.abc import Iterable
from enum import Enum
from typing import Any, Dict, Optional


class RunKind(Enum):
    STEP = "step"
    TANGO_RUN = "tango_run"


def flatten_dict(d: Dict[str, Any], parent_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single level dictionary.
    """
    result = {}
    for key, value in d.items():
        subkey = f"{parent_key}.{key}" if parent_key else key
        value = {str(i): v for i, v in enumerate(value)} if isinstance(value, Iterable) else value
        if isinstance(value, dict):
            result.update(flatten_dict(value, subkey))
        else:
            result[key] = value
    return result
