"""Regression checks for the method-native LADA 16-shot protocol."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "src/lada/native_protocol.py"
_SPEC = importlib.util.spec_from_file_location("native_protocol", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
LADA_16SHOT_EPOCHS = _MODULE.LADA_16SHOT_EPOCHS
LADA_NATIVE_DEFAULTS = _MODULE.LADA_NATIVE_DEFAULTS


def main():
    assert LADA_16SHOT_EPOCHS == {
        "aircraft": 40, "caltech101": 10, "dtd": 30, "eurosat": 100,
        "flowers": 30, "food101": 5, "mnist": 200, "oxford_pets": 10,
        "stanford_cars": 30, "sun397": 10,
    }
    assert LADA_NATIVE_DEFAULTS["batch_size"] == 64
    assert LADA_NATIVE_DEFAULTS["lr"] == 1e-3
    assert LADA_NATIVE_DEFAULTS["weight_decay"] == 5e-4
    assert LADA_NATIVE_DEFAULTS["prototype_k"] == 4
    print("native LADA protocol: passed")


if __name__ == "__main__":
    main()
