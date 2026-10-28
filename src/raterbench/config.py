"""Run configuration (plain pydantic + YAML)."""

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field

from raterbench.llm.cost import Pricing


class CalibrationConfig(BaseModel):
    # test_remainder: test items not drawn for evaluation (same distribution as eval).
    # train_holdout: held out from the training split, then removed from the training pool.
    source: Literal["test_remainder", "train_holdout"]
    size: int | None = Field(default=None, description="None = all of test_remainder")


class DatasetConfig(BaseModel):
    name: str
    seed: int = 13
    eval_size: int
    calibration: CalibrationConfig


class ModelConfig(BaseModel):
    name: str  # identifier used in results
    backend: str  # "anthropic" | "ollama"
    model: str  # provider model ID / tag
    params: dict[str, Any] = Field(default_factory=dict)
    pricing: Pricing = Field(default_factory=Pricing)
    concurrency: int = 4


class SelfConsistencyConfig(BaseModel):
    n: int = 0  # extra samples per item; 0 disables
    temperature: float = 0.7  # used only by backends that accept temperature
    conditions: list[str] = Field(default_factory=list)


class RunConfig(BaseModel):
    name: str
    main: Path = Path("configs/main.yaml")
    models: list[Path]
    conditions: list[str]
    roles: list[str] = Field(default_factory=lambda: ["eval"])
    max_items_per_role: int | None = None  # stratified subset (e.g. 20 for a dry run)
    fewshot_k: int = 5
    seed: int = 13
    self_consistency: SelfConsistencyConfig = Field(default_factory=SelfConsistencyConfig)
    budget_usd: float
    cache_path: Path = Path(".cache/llm_cache.sqlite")
    output_dir: Path = Path("results/runs")
    prompts_dir: Path = Path("prompts")


class MainConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    samples_dir: Path = Path("data/samples")
    datasets: list[Path]

    def dataset_configs(self) -> list[DatasetConfig]:
        return [load_yaml(p, DatasetConfig) for p in self.datasets]


T = TypeVar("T", bound=BaseModel)


def load_yaml(path: Path, model: type[T]) -> T:
    with Path(path).open(encoding="utf-8") as f:
        return model.model_validate(yaml.safe_load(f))
