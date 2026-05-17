from pathlib import Path

import yaml
from pydantic import BaseModel

_DEFAULT_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


class Settings(BaseModel):
    contamination: dict[str, float]
    detectors: dict[str, dict[str, float | int | str]]
    data: dict[str, str | int | float]


def load_config(path: Path | None = None) -> Settings:
    p = path or _DEFAULT_PATH
    with p.open() as f:
        raw = yaml.safe_load(f)
    return Settings(**raw)
