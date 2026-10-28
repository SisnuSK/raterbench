"""Common item representation shared by all dataset loaders."""

import hashlib
from dataclasses import dataclass, field


@dataclass
class Item:
    id: str
    text: str
    label: str
    subgroup: str
    split: str  # source split as published: "train", "val", or "test"
    group: str | None = None  # items sharing a group (e.g. one Reddit post) are never split apart
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def unit(self) -> str:
        """Sampling unit: the group if there is one, otherwise the item itself."""
        return self.group if self.group is not None else self.id

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()
