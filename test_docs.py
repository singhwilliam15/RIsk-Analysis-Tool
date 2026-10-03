"""
The README and docs (review P2): no stale status lines, and every local link and image they use exists.
"""

import os
import re

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = ["README.md", "docs/features.md", "docs/case_studies.md", "docs/methodology.md", "docs/deploy.md",
        "docs/interview_story.md"]
STALE = ["is being extended", "Market risk** is complete", "Market risk is complete", "Coming next (Phase 6)",
         "434 tests", "434 offline tests", "measures those links", "until the Phase 7 case studies",
         "banks are not modelled beyond the manual panel"]


def _read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("path", DOCS)
def test_no_stale_lines(path):
    text = _read(path)
    found = [phrase for phrase in STALE if phrase in text]
    assert not found, f"{path}: {found}"


@pytest.mark.parametrize("path", DOCS)
def test_local_links_and_images_exist(path):
    base = os.path.dirname(os.path.join(ROOT, path))
    for target in re.findall(r"\]\(([^)\s]+)\)", _read(path)):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        local = target.split("#")[0]
        assert os.path.exists(os.path.normpath(os.path.join(base, local))), f"{path}: broken link {target}"


def test_readme_is_one_screen_plus_a_little():
    assert len(_read("README.md").splitlines()) < 150
