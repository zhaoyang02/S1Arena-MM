"""Every checkpoint the adapters download is pinned to a commit (docs/models.md lists them); no weights are loaded."""

import re

from jevarena_mm.models import clef, jev_omni, onejev, raya, xor

SHA = re.compile(r"[0-9a-f]{40}")


def test_downloads_are_pinned_to_commits():
    pins = [onejev.REVISION, jev_omni.REVISION, raya.REVISION, raya.PROCESSOR_REVISION, xor.COMMIT,
            *clef.REVISIONS.values(), *jev_omni.HEAD_REVISIONS.values()]
    assert all(SHA.fullmatch(p) for p in pins)
    assert set(clef.REVISIONS) == {"Cloudflare/clef", "Cloudflare/clef-flash", "Cloudflare/clef-omni"}
