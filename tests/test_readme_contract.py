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


def test_readme_qualifies_cross_embodiment_transfer_without_retuning() -> None:
    text = README.read_text(encoding="utf-8")

    assert "without an embodiment-specific hyperparameter search" in text
    assert "Action scaling and operational-limit bounds" in text
    assert "mass randomization, disturbance magnitudes, and the energy coefficient" in text
    assert "mass ratio" in text


def test_readme_documents_locked_local_and_cluster_setup_without_secrets() -> None:
    text = README.read_text(encoding="utf-8")

    assert "requirements.txt" not in text
    assert "uv.lock" in text
    assert "ddb044eb5b2300792de41e82d53b032f3632b489" in text
    assert 'UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"' in text
    assert "uv sync --frozen --inexact" in text
    assert "train-locomposition.sbatch" in text
    assert "train-locomposition-2080ti.sbatch" in text
    assert "one-time" in text.lower()
    assert "shared home" in text.lower()
    assert re.search(r"(?i)(?:export\s+)?WANDB_API_KEY\s*=", text) is None
    text_without_isaaclab_revision = text.replace(
        "ddb044eb5b2300792de41e82d53b032f3632b489", ""
    )
    assert re.search(r"(?i)\b[0-9a-f]{40}\b", text_without_isaaclab_revision) is None


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
