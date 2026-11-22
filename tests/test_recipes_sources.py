"""Recipes of sources whose terms forbid redistributing their annotations carry tokens, never the annotations."""

import json
import re

from jevarena_mm import recipes
from jevarena_mm.registry import TASKS, discover


def rows(name):
    discover()
    return recipes.load(TASKS[name]())


def test_charades_sta_recipe_keeps_queries_windows_and_quarters_out():
    loaded = rows("lvid.ret/moment/charades_sta")
    assert len(loaded) == 114
    for row in loaded:
        item = row["item"]
        assert re.search(r"\n⟦charades_sta:query:\d+⟧$", item["request"]["questions"]["answer"]["instructions"])
        assert re.fullmatch(r"⟦charades_sta:quarter:[\d+]+@[\d.]+⟧", item["gold"]["answer"]["label"])
        assert re.fullmatch(r"⟦charades_sta:windows:[\d+]+⟧", item["meta"]["evidence_s"])


def test_escalation_quotes_charades_sta_through_a_token():
    for row in rows("lvid.route/escalation/jev_pair"):
        state = row["item"]["request"]["state"]
        if state.get("video", {}).get("uri", "").startswith("charades/"):
            assert "⟦charades_sta" in json.dumps(state, ensure_ascii=False)


def test_ssv2_recipe_keeps_labels_out():
    """Scene, question, options and answer all depend on the video's label, so all of them are tokens."""
    loaded = rows("vid.basic/temporal/ssv2")
    assert len(loaded) == 500
    for row in loaded:
        item = row["item"]
        question = item["request"]["questions"]["answer"]
        assert re.fullmatch(r"⟦ssv2:scene:\d+⟧", item["request"]["state"]["text"])
        if question["type"] == "choice":
            assert re.fullmatch(r"⟦ssv2:choice_question:\d+⟧", question["instructions"])
            assert re.fullmatch(r"⟦ssv2:options:\d+⟧", question["criteria"])
            assert re.fullmatch(r"⟦ssv2:direction:\d+⟧", item["gold"]["answer"]["label"])
        else:
            assert re.fullmatch(r"⟦ssv2:yes_no_question:\d+/[ab]⟧", question["instructions"])
            assert re.fullmatch(r"⟦ssv2:yes_no_answer:\d+/[ab]⟧", item["gold"]["answer"]["label"])


def test_ssv2_templates():
    assert recipes.ssv2_template("Pushing [something] from left to right") == ("push", "left_to_right")
    assert recipes.ssv2_template("Taking [something] out of [something]") == ("container", "out_of")
    try:
        recipes.ssv2_template("Spinning [something] that quickly stops spinning")
    except KeyError:
        pass
    else:
        raise AssertionError("a template the task does not ask about must not be filled in")


def test_hateful_memes_recipe_keeps_texts_and_labels_out():
    loaded = rows("img.safety/hateful_meme/hateful_memes")
    assert len(loaded) == 500
    for row in loaded:
        item = row["item"]
        key = item["request"]["state"]["meme"]["uri"].rsplit("/", 1)[1].removesuffix(".png")
        assert item["request"]["state"]["meme_text"] == f"⟦hateful_memes:text:{key}⟧"
        assert item["gold"]["hateful"]["label"] == f"⟦hateful_memes:label:{key}⟧"


def test_artemis_recipe_keeps_emotions_out():
    loaded = rows("img.social/art_emotion/artemis")
    assert len(loaded) == 500
    for row in loaded:
        gold = row["item"]["gold"]["emotion"]
        key = row["item"]["meta"]["group"].removeprefix("artemis:")
        assert gold == {"label": f"⟦artemis:label:{key}⟧", "dist": f"⟦artemis:dist:{key}⟧"}


def test_winoground_recipe_keeps_captions_out():
    loaded = rows("img.ret/compositional/winoground")
    assert len(loaded) == 500
    for row in loaded:
        question, gold = row["item"]["request"]["questions"]["caption"], row["item"]["gold"]["caption"]
        assert re.fullmatch(r"⟦winoground:options:\d+/[01]/(01|10)/[ti]⟧", question["criteria"])
        assert gold["label"] == question["criteria"].replace(":options:", ":label:")


def test_soccernet_mvfoul_recipe_keeps_answers_out():
    """The questions are the task's own wording; only the answers come from SoccerNet's annotations."""
    loaded = rows("vid.pro/foul/soccernet_mvfoul")
    assert len(loaded) == 500
    for row in loaded:
        item = row["item"]
        action = item["meta"]["group"].removeprefix("soccernet-mvfoul:test:event:")
        assert re.fullmatch(rf"⟦soccernet_mvfoul:(action|offence|no_offence|severity):{action}⟧",
                            item["gold"]["answer"]["label"])


def test_hateful_memes_text_drops_surrounding_spaces(workspace):
    """As the items were built: one test_seen text ends in a space that the item does not have."""
    path = recipes.source_file("hateful_memes")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"id": 51924, "img": "img/51924.png", "label": 0, "text": "fridays got me like "}\n', encoding="utf-8")
    assert recipes.value("hateful_memes", "text", "51924") == "fridays got me like"
    assert recipes.value("hateful_memes", "label", "51924") == "false"
