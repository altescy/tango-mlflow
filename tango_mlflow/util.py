from enum import Enum
from typing import Any, Dict


class RunKind(Enum):
    STEP = "step"
    TANGO_RUN = "tango_run"


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single level dictionary.
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, (list, tuple)):
            value = {str(i): v for i, v in enumerate(value)}
        if isinstance(value, dict):
            result.update({f"{key}.{k}": v for k, v in flatten_dict(value).items()})
        else:
            result[key] = value
    return result
