"""Prompt rendering from versioned Jinja2 templates, task specs, and codebooks."""

import hashlib
import random
from collections import defaultdict
from pathlib import Path

import jinja2
import yaml
from pydantic import BaseModel

from raterbench.data.schema import Item

CONDITIONS = {
    "zero_shot": "zero_shot.j2",
    "few_shot": "few_shot.j2",
    "codebook": "codebook.j2",
    "codebook_reasoning": "codebook_reasoning.j2",
}
REASONING_CONDITIONS = {"codebook_reasoning"}


class LabelDef(BaseModel):
    name: str
    definition: str


class TaskSpec(BaseModel):
    dataset: str
    description: str
    labels: list[LabelDef]
    targets: dict[str, str] = {}
    fewshot_by_subgroup: bool = False

    @property
    def label_names(self) -> tuple[str, ...]:
        return tuple(lab.name for lab in self.labels)


def select_fewshot(
    pool: list[Item], labels: tuple[str, ...], k: int, seed: int, by_subgroup: bool
) -> dict[str | None, list[Item]]:
    """Label-balanced few-shot examples (round-robin over labels), fixed by seed.

    Returns one example list per subgroup when `by_subgroup`, else a single list under None.
    """
    buckets: dict[str | None, list[Item]] = defaultdict(list)
    for it in pool:
        buckets[it.subgroup if by_subgroup else None].append(it)
    out = {}
    for key in sorted(buckets, key=str):
        rng = random.Random(f"{seed}:{key}")
        per_label = {}
        for lab in labels:
            cands = sorted((it for it in buckets[key] if it.label == lab), key=lambda x: x.id)
            rng.shuffle(cands)
            per_label[lab] = cands
        chosen: list[Item] = []
        i = 0
        while len(chosen) < k and any(i < len(v) for v in per_label.values()):
            for lab in labels:
                if len(chosen) < k and i < len(per_label[lab]):
                    chosen.append(per_label[lab][i])
            i += 1
        rng.shuffle(chosen)  # avoid a fixed label order in the prompt
        out[key] = chosen
    return out


class PromptBuilder:
    def __init__(
        self, dataset: str, prompts_dir: Path, fewshot_pool: list[Item], k: int, seed: int
    ):
        self.prompts_dir = prompts_dir
        task_path = prompts_dir / "tasks" / f"{dataset}.yaml"
        codebook_path = prompts_dir / "codebooks" / f"{dataset}.md"
        self.task = TaskSpec.model_validate(yaml.safe_load(task_path.read_text(encoding="utf-8")))
        self.codebook = codebook_path.read_text(encoding="utf-8").strip()
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(prompts_dir / "templates"),
            undefined=jinja2.StrictUndefined,
            keep_trailing_newline=False,
            autoescape=False,
        )
        self.system = self.env.get_template("system.j2").render().strip()
        self.fewshot = select_fewshot(
            fewshot_pool, self.task.label_names, k, seed, self.task.fewshot_by_subgroup
        )
        self._source_hash = self._hash_sources([task_path, codebook_path])

    def _hash_sources(self, extra: list[Path]) -> str:
        h = hashlib.sha256()
        files = sorted((self.prompts_dir / "templates").glob("*.j2")) + extra
        for p in files:
            h.update(p.name.encode())
            h.update(p.read_bytes())
        return h.hexdigest()

    def examples_for(self, item: Item) -> list[Item]:
        return self.fewshot[item.subgroup if self.task.fewshot_by_subgroup else None]

    def prompt_hash(self, condition: str) -> str:
        """Hash of templates + task spec + codebook (+ few-shot example IDs)."""
        h = hashlib.sha256(self._source_hash.encode())
        h.update(condition.encode())
        if condition == "few_shot":
            for key in sorted(self.fewshot, key=str):
                h.update(",".join(it.id for it in self.fewshot[key]).encode())
        return h.hexdigest()[:16]

    def render(self, condition: str, item: Item) -> str:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {condition!r}; known: {sorted(CONDITIONS)}")
        target = self.task.targets.get(item.subgroup, item.subgroup)
        description = self.env.from_string(self.task.description).render(target=target).strip()
        examples = (
            [{"text": ex.text, "label": ex.label} for ex in self.examples_for(item)]
            if condition == "few_shot"
            else []
        )
        return (
            self.env.get_template(CONDITIONS[condition])
            .render(
                description=description,
                labels=self.task.labels,
                text=item.text,
                examples=examples,
                codebook=self.codebook,
            )
            .strip()
        )

    def retry_message(self, error: str) -> str:
        return self.env.get_template("retry.j2").render(error=error).strip()
