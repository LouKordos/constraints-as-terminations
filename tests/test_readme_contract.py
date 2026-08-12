from __future__ import annotations

import re
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_presents_locomposition_and_attributes_cat() -> None:
    text = README.read_text(encoding="utf-8")

    assert text.startswith("# LoComposition\n")
    assert "https://arxiv.org/abs/2606.15896" in text
    assert "https://sites.google.com/view/locomposition" in text
    assert "Constraints as Terminations" in text
    assert "https://arxiv.org/abs/2403.18765" in text


def test_readme_relative_links_resolve() -> None:
    text = README.read_text(encoding="utf-8")
    destinations = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)

    missing: list[str] = []
    for destination in destinations:
        destination = destination.strip().split(maxsplit=1)[0].strip("<>")
        if destination.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = destination.split("#", maxsplit=1)[0]
        if path and not (ROOT / path).exists():
            missing.append(destination)

    assert not missing, f"README links to missing repository paths: {missing}"


def test_documentation_media_are_real_images_with_useful_resolution() -> None:
    expected = {
        "assets/locomposition-overview.png": (1200, 500),
        "assets/cot-and-contact-patterns.png": (1200, 500),
        "assets/sim2real-overview.png": (1200, 500),
        "assets/terrain-contact-adaptation.png": (1200, 500),
        "assets/swing-height-adaptation.png": (1200, 500),
        "assets/demos/go2-sim2real.gif": (960, 540),
        "assets/demos/anymal-c.gif": (540, 540),
        "assets/demos/spot.gif": (540, 540),
    }

    for relative_path, minimum_size in expected.items():
        with Image.open(ROOT / relative_path) as image:
            assert image.width >= minimum_size[0]
            assert image.height >= minimum_size[1]


def test_citation_metadata_names_the_project_and_all_authors() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == "LoComposition"
    assert citation["preferred-citation"]["title"] == (
        "LoComposition: Terrain-Adaptive Energy-Efficient Quadruped "
        "Locomotion without Gait Priors"
    )
    assert citation["preferred-citation"]["identifiers"] == [
        {"type": "other", "value": "arXiv:2606.15896"}
    ]
    family_names = {
        author["family-names"] for author in citation["preferred-citation"]["authors"]
    }
    assert family_names == {
        "Kordos",
        "Franz",
        "Rappenecker",
        "Hausdörfer",
        "Schoellig",
        "Kolev",
        "Martius",
    }
