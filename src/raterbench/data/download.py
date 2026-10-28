"""Download raw datasets from pinned sources and verify their checksums.

Raw files live under `data/raw/` (git-ignored) and are never redistributed.
"""

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RemoteFile:
    url: str
    path: str  # relative to data/raw/<dataset>/
    sha256: str


DREADDIT_FILES = [
    RemoteFile(
        url="http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip",
        path="dreaddit.zip",
        sha256="6c7d8859764231cca47410d6995d8941ac45b08a446a561923fd73b34f3f61cb",
    ),
]

# Official TweetEval repo, pinned to the commit the files were verified against.
TWEETEVAL_COMMIT = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
_TWEETEVAL_BASE = (
    f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/{TWEETEVAL_COMMIT}/datasets/stance"
)
_TWEETEVAL_SHA256 = {
    "mapping.txt": "0f8a803dc1773c84279506f25d54b5a9eec7edfc277b41eb75076744dc89b26f",
    "abortion/test_labels.txt": "c90e6d36d863f876d6661620d37b613b4b07858a5277c8d6623713ee59ca451c",
    "abortion/test_text.txt": "bf0e16a0b8ca4cf0ab90efbc560db3151c288fc842f5e3c6554e8589d521556a",
    "abortion/train_labels.txt": "e6786a594bd9a083c524a0f420c690351140b52af288f487cb4772d29675b014",
    "abortion/train_text.txt": "a421d5b8fd9f972970b9275b83f65745bf81986d2a412b4caa2ba071f3efa916",
    "abortion/val_labels.txt": "8df57a50823d5f3683ecf75d824a42e3b08eb52e25e3e2d6928f523097a0c050",
    "abortion/val_text.txt": "0428ab3f2894936f2445a9020763c2bd19ed42986872168bb65886dede5843fd",
    "atheism/test_labels.txt": "4ef7c9398d265cfac625092c834e43cef9da9cb318e563493abb64f65dfe1b52",
    "atheism/test_text.txt": "16c5336b2cba606ca63a6afcc50241be63a8fccf021628c6505449439b9d54b3",
    "atheism/train_labels.txt": "a764aac1a75ccb32c4ffc4c03c66dc365cb50f013d3e94549bf775636cbc8373",
    "atheism/train_text.txt": "0e82f1d4a16d79a38a68aee761762cf8a846bc8f7f9395670ca44e2ecf2f58f7",
    "atheism/val_labels.txt": "638095b3582f927fd1481cdb8d1f9f670f8d27880baf32c0b26c5946fd8f8292",
    "atheism/val_text.txt": "5fe14c4c01f87a45dba640dddbb1d1909a893f9565f159c48fa1ba35bb46c209",
    "climate/test_labels.txt": "193c9f2358f61d9efe558324ec89ecaf08e600a44b68128f47838c01d9f98dfd",
    "climate/test_text.txt": "41ee8ee2ad3c36e0629654fdb271f37775197c79be8b299adbeadd2003b63c53",
    "climate/train_labels.txt": "d6274f55bc95f5a7f2ae591b886c1414a7664aaf4e0c609f4ba6cf377929af18",
    "climate/train_text.txt": "4803211832d318026323a8e5014cff1b95e1c8c3854378101e5d1a8c82582eb7",
    "climate/val_labels.txt": "0cb133ab9b137292f075210db45f7e293dc52798a4e21e59037bfcfe66c97aa6",
    "climate/val_text.txt": "fc5714703add266801ee2fd98296ea20ec0879e89cdb9f906d9812d9f640f2ba",
    "feminist/test_labels.txt": "ddbde6d253ee47c5d5ef8bc5386270fde45cf088d3be70bba9c382b8a024897a",
    "feminist/test_text.txt": "1bfdbdc2af64fd62dcc775d1288e192ac8ff805ef27ccf3aaac54a98616eefda",
    "feminist/train_labels.txt": "abd4f196d801423bb0daba8c0ecf5b3efba1f10e8f410c3dfa360b50c8b9c685",
    "feminist/train_text.txt": "c176e6663973c8e78bfa92ba1e8874a70cc5358567d71584a90943bc6525eaab",
    "feminist/val_labels.txt": "399e0d468d0e4ead7a445f69efdf35876c835acf4cefc00a16f451a5d42e5c13",
    "feminist/val_text.txt": "3518b2ddcf696626a7243d7cea720a975718c7a52a5a086931be87897c1de58b",
    "hillary/test_labels.txt": "068468f6a72b85dfb65bf10e45f2453fa082d1ea9d7a40e7f560d5b6d75027f3",
    "hillary/test_text.txt": "5c4e020285a62cfd88f264849e1db242ded356c171b1a68dd0050b76635053aa",
    "hillary/train_labels.txt": "0ea5753d13a717a9e91581d1d89c0b5206c8f905f0a717b2b27d02dbf419250d",
    "hillary/train_text.txt": "0bd735de895cb74d63c224e64e3d955cac99be97aa225f803fe4d2f5978a2c99",
    "hillary/val_labels.txt": "e5d44c771b7349a4a74309f56ca072fdf8f1c015068d519ca2ed3a931c833606",
    "hillary/val_text.txt": "9714b7dcc8617e095433d7b63df8aa155eb84216b9ac9195105ab83d85cd248d",
}
TWEETEVAL_FILES = [
    RemoteFile(url=f"{_TWEETEVAL_BASE}/{p}", path=p, sha256=h) for p, h in _TWEETEVAL_SHA256.items()
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(files: list[RemoteFile], dest_dir: Path, force: bool = False) -> list[Path]:
    """Download each file into dest_dir (skipping verified existing copies)."""
    out = []
    for rf in files:
        target = dest_dir / rf.path
        if not force and target.exists() and sha256_file(target) == rf.sha256:
            out.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(rf.url, timeout=60) as resp, tmp.open("wb") as f:
            f.write(resp.read())
        actual = sha256_file(tmp)
        if actual != rf.sha256:
            tmp.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {rf.url}: expected {rf.sha256}, got {actual}. "
                "The upstream file changed; re-verify the dataset before updating the pin."
            )
        tmp.replace(target)
        out.append(target)
    return out


def verify(files: list[RemoteFile], dest_dir: Path) -> None:
    for rf in files:
        target = dest_dir / rf.path
        if not target.exists():
            raise FileNotFoundError(f"{target} missing; run `raterbench data download` first.")
        if sha256_file(target) != rf.sha256:
            raise RuntimeError(f"{target} does not match its pinned checksum.")
