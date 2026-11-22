import pytest
from pydantic import ValidationError

from jevarena_mm.schema import DecisionRequest, Gold, Item, MediaRef, ModelResponse, Question

SHA_A = "a" * 64
SHA_B = "b" * 64


def video(sha=SHA_A, window=(0.0, 4.0)):
    return {"kind": "video", "uri": "ssv2/videos/1.webm", "sha256": sha, "window": window}


def item(state, task="vid.basic/temporal/ssv2"):
    return Item(
        leaf=task.rsplit("/", 1)[0],
        task=task,
        request=DecisionRequest(state=state, questions={"moved": Question(type="boolean", instructions="Did it move?")}),
        gold={"moved": Gold(label="true")},
    )


def test_state_values_become_media_refs_and_round_trip():
    request = DecisionRequest(
        state={"text": "a clip", "video": video(), "views": [video(SHA_A), video(SHA_B)]},
        questions={"q": Question(type="boolean", instructions="?")},
    )
    assert isinstance(request.state["video"], MediaRef)
    assert [ref.sha256 for _, ref in request.media()] == [SHA_A, SHA_A, SHA_B]
    assert request.modalities() == {"video"}
    again = DecisionRequest.model_validate_json(request.model_dump_json())
    assert again == request
    assert again.state["text"] == "a clip"


def test_item_id_follows_media_hash_and_window():
    base = item({"video": video()})
    assert base.id.startswith("vid.basic.temporal.ssv2:")
    assert item({"video": video()}).id == base.id
    assert item({"video": video(SHA_B)}).id != base.id
    assert item({"video": video(window=(0.0, 3.0))}).id != base.id


def test_items_round_trip_with_the_same_id():
    original = item({"video": video(), "audio": {"kind": "audio", "uri": "ami/a.wav", "sha256": SHA_B}})
    again = Item.model_validate_json(original.model_dump_json())
    assert again.id == original.id and again.request.modalities() == {"video", "audio"}


@pytest.mark.parametrize("bad", [
    {"kind": "image", "uri": "x/a.png", "sha256": SHA_A, "window": [0, 1]},  # images take no window
    {"kind": "video", "uri": "x/a.mp4", "sha256": "ABC"},                    # not a sha256
    {"kind": "video", "uri": "/abs/a.mp4", "sha256": SHA_A},                 # not relative
    {"kind": "video", "uri": "x/../a.mp4", "sha256": SHA_A},                 # escapes the media root
    {"kind": "video", "uri": "x/a.mp4", "sha256": SHA_A, "window": [3, 1]},  # empty window
    {"kind": "video", "uri": "x/a.mp4", "sha256": SHA_A, "fps": 2},          # sampling is not data
])
def test_invalid_media_refs_are_rejected(bad):
    with pytest.raises(ValidationError):
        DecisionRequest(state={"m": bad}, questions={"q": Question(type="boolean", instructions="?")})


def test_items_cannot_reference_in_memory_media():
    with pytest.raises(ValidationError):
        item({"frame": {"kind": "image", "uri": "mem:" + SHA_A, "sha256": SHA_A}})


def test_question_limits_are_jevarenas():
    with pytest.raises(ValidationError):
        Question(type="score", instructions="?", criteria=[str(i) for i in range(11)])
    with pytest.raises(ValidationError):
        Question(type="choice", instructions="?", criteria={str(i): str(i) for i in range(256)})


def test_response_status():
    assert ModelResponse().status == "ok"
    assert ModelResponse(error="HTTP 500").status == "failed"
    assert ModelResponse(status="unsupported").status == "unsupported"
