import pytest

from jevarena_mm.models.base import split_options
from jevarena_mm.registry import make_model, matches, register_task


def test_names_need_a_modality_prefix_and_a_matching_leaf(clean_registry):
    class Good:
        name, leaf = "vid.basic/toy/source", "vid.basic/toy"

    class NoModality:
        name, leaf = "basic/temporal/ssv2", "basic/temporal"

    class WrongLeaf:
        name, leaf = "aud.basic/intent/slurp", "aud.basic/other"

    register_task(Good)
    with pytest.raises(ValueError):
        register_task(Good)
    for bad in (NoModality, WrongLeaf):
        with pytest.raises(ValueError):
            register_task(bad)


def test_prefixes_select_whole_components():
    name = "vid.basic/temporal/ssv2"
    assert all(matches(name, p) for p in ("", "vid", "vid.basic", "vid.basic/temporal", name))
    assert not matches(name, "lvid") and not matches(name, "vid.bas") and not matches(name, "vid.basic/temp")
    assert not matches("lvid.know/needle/generated", "vid")


def test_model_specs_and_options():
    assert split_options("org/model@fps=1,max_frames=64") == ("org/model", {"fps": "1", "max_frames": "64"})
    assert split_options("@fps=2") == ("", {"fps": "2"})
    with pytest.raises(ValueError):
        split_options("@fps")
    assert make_model("random").model_id == "test/random"
    limited = make_model("random@modalities=image")
    assert limited.modalities == {"image"} and limited.model_id == "test/random@modalities=image"
    assert make_model("random").slug != limited.slug
    with pytest.raises(KeyError):
        make_model("nonexistent@fps=1")
