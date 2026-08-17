"""The AI4Bharat layer: language routing, degradation, and the endpoints.

Every model is stubbed. What these tests actually assert is the *contract* around
the models — that the right checkpoint is chosen for each direction, that a
missing model degrades instead of raising, and that composition is never silently
replaced by translation.
"""

from __future__ import annotations

import pytest

from app.tutor import lesson_generator, mock_rag_service, tamil_quality
from app.tutor.indic import languages, translate, transliterate
from app.tutor.schemas import ExplanationOrigin, LessonSource
from tests.conftest import GOOD_LESSON, GOOD_TAMIL, FakeClient

CHUNK = mock_rag_service.PHOTOSYNTHESIS


@pytest.fixture(autouse=True)
def no_backtranslation(monkeypatch):
    """Default: IndicTrans2 validation unavailable, so tests don't try to load 1GB."""
    monkeypatch.setattr(tamil_quality.translate, "to_english", lambda text, lang: None)


# ------------------------------------------------------------------ language registry


def test_five_languages_are_registered():
    assert set(languages.LANGUAGES) == {"ta", "hi", "te", "kn", "ml"}


@pytest.mark.parametrize(
    ("code", "flores", "asr_code"),
    [
        ("ta", "tam_Taml", "ta"),
        ("hi", "hin_Deva", "hi"),
        ("te", "tel_Telu", "te"),
        ("kn", "kan_Knda", "kn"),
        ("ml", "mal_Mlym", "ml"),
    ],
)
def test_every_language_carries_the_codes_each_model_wants(code, flores, asr_code):
    lang = languages.get(code)
    assert lang.flores == flores
    assert lang.asr == asr_code
    assert lang.tts_voice and lang.local_analogy_hint


def test_unknown_language_falls_back_to_tamil_rather_than_raising():
    assert languages.get("klingon").code == "ta"
    assert languages.get(None).code == "ta"


# ------------------------------------------------------------------ direction routing


@pytest.mark.parametrize(
    ("src", "tgt", "expected"),
    [
        ("eng_Latn", "tam_Taml", "EN_INDIC"),
        ("tam_Taml", "eng_Latn", "INDIC_EN"),
        ("tam_Taml", "tel_Telu", "INDIC_INDIC"),
        ("hin_Deva", "mal_Mlym", "INDIC_INDIC"),
    ],
)
def test_the_right_checkpoint_is_chosen_for_each_direction(src, tgt, expected):
    """Indic→Indic must not be routed through English — that is the 320M model's job."""
    holder = translate._engine_for(src, tgt)
    assert holder is getattr(translate, expected)


def test_translate_returns_none_when_the_model_cannot_load(monkeypatch):
    monkeypatch.setattr(translate.EN_INDIC, "get", lambda: None)

    assert translate.from_english("hello", "ta") is None


def test_translate_short_circuits_when_source_and_target_match():
    assert translate.translate("same", src_flores="tam_Taml", tgt_flores="tam_Taml") == "same"


# ------------------------------------------------------------------ IndicXlit routing


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("thavaram epdi saapdum", True),
        ("prakasha samsleshanam", True),
        ("how do plants make food", False),  # plain English — must be left alone
        ("what is photosynthesis", False),
        ("தாவரம் எப்படி சாப்பிடும்", False),  # already native script
    ],
)
def test_only_romanised_regional_queries_are_transliterated(text, expected, monkeypatch):
    monkeypatch.setattr(transliterate, "to_native", lambda t, lang: "தாவரம் எப்படி சாப்பிடும்")

    _, was_transliterated = transliterate.normalise_query(text, "ta")

    assert was_transliterated is expected


def test_query_is_unchanged_when_indicxlit_is_unavailable(monkeypatch):
    monkeypatch.setattr(transliterate, "to_native", lambda t, lang: None)

    query, changed = transliterate.normalise_query("thavaram epdi saapdum", "ta")

    assert query == "thavaram epdi saapdum"
    assert changed is False


# ------------------------------------------------------------------ composition vs translation


def test_regional_explanation_is_composed_not_translated(monkeypatch):
    """The whole point of the branch: en→indic must not run on the happy path."""
    calls: list[str] = []
    monkeypatch.setattr(
        translate, "from_english", lambda text, lang: calls.append(lang) or "translated"
    )

    client = FakeClient(json_responses=[GOOD_LESSON], tamil_responses=[GOOD_TAMIL])
    lesson = lesson_generator.generate(CHUNK, None, client=client)

    assert lesson.regional_origin is ExplanationOrigin.COMPOSED
    assert lesson.tamil_explanation == GOOD_TAMIL
    assert calls == [], "en→indic ran on the happy path — composition was bypassed"


def test_en_indic_is_the_fallback_when_composition_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        translate, "from_english", lambda text, lang: "पौधे सूरज की रोशनी से भोजन बनाते हैं।"
    )

    # No tamil_responses -> the composition call raises LLMUnavailable.
    client = FakeClient(json_responses=[GOOD_LESSON], tamil_responses=[])
    lesson = lesson_generator.generate(CHUNK, None, language="hi", client=client)

    assert lesson.regional_origin is ExplanationOrigin.TRANSLATED_FROM_ENGLISH
    assert lesson.tamil_explanation.startswith("पौधे")
    assert "IndicTrans2 en→indic (fallback)" in lesson.trace.ai4bharat_models_used
    # Still a real lesson, not a refusal.
    assert lesson.grounded is True


def test_extra_languages_translate_from_the_composed_text_not_the_english(monkeypatch):
    """Indic→Indic keeps the teacher voice; going via English would flatten it."""
    seen: dict = {}

    def fake_between_indic(text, *, source_code, target_code):
        seen[target_code] = (text, source_code)
        return f"[{target_code}] {text[:12]}"

    monkeypatch.setattr(translate, "between_indic", fake_between_indic)
    monkeypatch.setattr(
        translate, "from_english", lambda text, lang: pytest.fail("en→indic should not run")
    )

    client = FakeClient(json_responses=[GOOD_LESSON], tamil_responses=[GOOD_TAMIL])
    lesson = lesson_generator.generate(
        CHUNK, None, translate_to=["te", "ml"], client=client
    )

    assert [t.language for t in lesson.translations] == ["te", "ml"]
    assert all(t.origin is ExplanationOrigin.TRANSLATED_FROM_REGIONAL for t in lesson.translations)
    # It translated the composed Tamil, not the English explanation.
    assert seen["te"] == (GOOD_TAMIL, "ta")


def test_extra_languages_fall_back_to_english_when_indic_indic_is_down(monkeypatch):
    monkeypatch.setattr(translate, "between_indic", lambda text, **kw: None)
    monkeypatch.setattr(translate, "from_english", lambda text, lang: f"[{lang}] from english")

    client = FakeClient(json_responses=[GOOD_LESSON], tamil_responses=[GOOD_TAMIL])
    lesson = lesson_generator.generate(CHUNK, None, translate_to=["hi"], client=client)

    assert lesson.translations[0].origin is ExplanationOrigin.TRANSLATED_FROM_ENGLISH


def test_extra_languages_are_skipped_not_faked_when_no_translator_exists(monkeypatch):
    monkeypatch.setattr(translate, "between_indic", lambda text, **kw: None)
    monkeypatch.setattr(translate, "from_english", lambda text, lang: None)

    client = FakeClient(json_responses=[GOOD_LESSON], tamil_responses=[GOOD_TAMIL])
    lesson = lesson_generator.generate(CHUNK, None, translate_to=["hi"], client=client)

    assert lesson.translations == []
    assert any("IndicTrans2 unavailable" in n for n in lesson.trace.notes)


def test_requesting_the_composed_language_again_is_a_no_op():
    client = FakeClient(json_responses=[GOOD_LESSON], tamil_responses=[GOOD_TAMIL])
    lesson = lesson_generator.generate(CHUNK, None, translate_to=["ta"], client=client)

    assert lesson.translations == []


# ------------------------------------------------------------------ non-Tamil composition


HINDI_LESSON = (
    "देखिए, पत्ती को एक रसोई समझिए। उसमें जो chlorophyll है वही रसोइया है। "
    "sunlight चूल्हे की आग है। हवा से आने वाली carbon dioxide और जड़ों से चढ़ने वाला "
    "water हमारा सामान है। इनसे पौधा glucose बनाता है और बचा हुआ oxygen बाहर छोड़ देता है।"
)


def test_hindi_uses_the_generic_teacher_prompt_with_its_own_analogies():
    client = FakeClient(json_responses=[GOOD_LESSON], tamil_responses=[HINDI_LESSON])
    lesson = lesson_generator.generate(CHUNK, None, language="hi", client=client)

    assert lesson.language == "hi"
    assert lesson.language_name == "Hindi"
    assert lesson.regional_origin is ExplanationOrigin.COMPOSED
    # The generic persona is filled with this language's script name and local scenes.
    system = client.tamil_calls[0]["system"]
    assert "Hindi" in system and "हिन्दी" in system
    assert "चूल्हा" in system  # Hindi kitchen vocabulary, not the Tamil hint
    # One call only: the Hinglish technical terms mean no concept went missing.
    assert len(client.tamil_calls) == 1


def test_refusal_never_loads_a_translation_model(monkeypatch):
    """The refusal must stay instant — no model load, even for a non-Tamil language."""
    monkeypatch.setattr(
        translate.EN_INDIC, "get", lambda: pytest.fail("refusal triggered a model load")
    )

    from app.shared.schemas import GroundingResult

    lesson = lesson_generator.generate(
        GroundingResult(query="unrelated", chunks=[], is_in_scope=False),
        language="te",
    )

    assert lesson.source is LessonSource.REFUSED
    assert lesson.language == "te"
    assert lesson.refusal_reason


# ------------------------------------------------------------------ endpoints


@pytest.fixture
def api():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_capabilities_reports_what_actually_loaded(api):
    body = api.get("/api/tutor/capabilities").json()

    assert set(body["ai4bharat"]) == {"translate", "transliterate", "asr", "tts"}
    # Nothing is loaded in a test run, and the endpoint says so rather than lying.
    assert body["ai4bharat"]["asr"]["asr"] == "not loaded yet"
    assert len(body["languages"]) == 5


def test_languages_endpoint_lists_all_five(api):
    body = api.get("/api/tutor/languages").json()

    assert body["default"] == "ta"
    assert {lang["code"] for lang in body["languages"]} == {"ta", "hi", "te", "kn", "ml"}


def test_speak_returns_503_rather_than_500_when_tts_is_missing(api):
    response = api.post("/api/tutor/speak", json={"text": "வணக்கம்", "language": "ta"})

    assert response.status_code == 503
    assert "parler" in response.json()["detail"].lower()


def test_listen_returns_503_rather_than_500_when_asr_is_missing(api):
    response = api.post(
        "/api/tutor/listen",
        files={"audio": ("q.wav", b"RIFF....fake", "audio/wav")},
        data={"language": "ta"},
    )

    assert response.status_code == 503


def test_transliterate_returns_503_rather_than_500_when_indicxlit_is_missing(api):
    response = api.post(
        "/api/tutor/transliterate", json={"text": "vanakkam", "language": "ta"}
    )

    assert response.status_code == 503


# ------------------------------------------------------------------ query preparation


def test_tanglish_question_reaches_english_for_retrieval(monkeypatch):
    """The textbook is English, so a Tanglish question must arrive in English.

    Transliterating to Tamil script and then searching an English page matches
    nothing — a refusal on an in-scope question, which looks exactly like the
    grounding guardrail working correctly. This test pins the second hop.
    """
    from app.tutor import query_prep

    monkeypatch.setattr(transliterate, "to_native", lambda t, l: "தாவரம் எப்படி சாப்பிடும்")
    monkeypatch.setattr(translate, "to_english", lambda t, l: "how does a plant eat")

    prepared = query_prep.prepare("thavaram epdi saapdum", "ta")

    assert prepared.for_teaching == "தாவரம் எப்படி சாப்பிடும்"   # what the teacher answers
    assert prepared.for_retrieval == "how does a plant eat"      # what the page is searched with
    assert prepared.transliterated and prepared.translated
    assert prepared.models_used == [
        "IndicXlit roman→native",
        "IndicTrans2 indic→en (query)",
    ]


def test_english_question_is_never_transliterated_or_translated(monkeypatch):
    from app.tutor import query_prep

    monkeypatch.setattr(
        transliterate, "to_native", lambda t, l: pytest.fail("IndicXlit ran on English")
    )
    monkeypatch.setattr(
        translate, "to_english", lambda t, l: pytest.fail("indic→en ran on English")
    )

    prepared = query_prep.prepare("how do plants make food", "ta")

    assert prepared.for_retrieval == "how do plants make food"
    assert not prepared.transliterated and not prepared.translated


def test_native_script_question_is_translated_without_transliteration(monkeypatch):
    """Typed directly in Tamil script: no IndicXlit needed, but still needs English."""
    from app.tutor import query_prep

    monkeypatch.setattr(translate, "to_english", lambda t, l: "how does a plant eat")

    prepared = query_prep.prepare("தாவரம் எப்படி சாப்பிடும்", "ta")

    assert prepared.transliterated is False
    assert prepared.translated is True
    assert prepared.for_retrieval == "how does a plant eat"


def test_query_prep_degrades_when_neither_model_is_installed(monkeypatch):
    """Today's real state: nothing installed, so the question is used as typed."""
    from app.tutor import query_prep

    monkeypatch.setattr(transliterate, "to_native", lambda t, l: None)
    monkeypatch.setattr(translate, "to_english", lambda t, l: None)

    prepared = query_prep.prepare("thavaram epdi saapdum", "ta")

    assert prepared.for_retrieval == "thavaram epdi saapdum"
    assert prepared.models_used == []


def test_missing_translator_falls_back_to_the_native_script_query(monkeypatch):
    from app.tutor import query_prep

    monkeypatch.setattr(transliterate, "to_native", lambda t, l: "தாவரம் எப்படி சாப்பிடும்")
    monkeypatch.setattr(translate, "to_english", lambda t, l: None)

    prepared = query_prep.prepare("thavaram epdi saapdum", "ta")

    assert prepared.for_retrieval == "தாவரம் எப்படி சாப்பிடும்"
    assert prepared.translated is False


# ------------------------------------------------------------------ LLM fallback


def test_llm_translates_the_query_when_indictrans2_is_unavailable(monkeypatch):
    """Windows can't build IndicTransToolkit, so the LLM covers the query hop."""
    from app.tutor import llm_translate, query_prep

    monkeypatch.setattr(transliterate, "to_native", lambda t, l: "தாவரம் எப்படி சாப்பிடும்")
    monkeypatch.setattr(translate, "to_english", lambda t, l: None)          # IndicTrans2 down
    monkeypatch.setattr(llm_translate, "to_english", lambda t, l: "how does a plant eat")

    prepared = query_prep.prepare("thavaram epdi saapdum", "ta")

    assert prepared.for_retrieval == "how does a plant eat"
    assert prepared.translated is True
    assert "LLM query translation (IndicTrans2 unavailable)" in prepared.models_used


def test_indictrans2_is_preferred_over_the_llm_when_both_are_available(monkeypatch):
    from app.tutor import llm_translate, query_prep

    monkeypatch.setattr(transliterate, "to_native", lambda t, l: "தாவரம்")
    monkeypatch.setattr(translate, "to_english", lambda t, l: "from indictrans2")
    monkeypatch.setattr(
        llm_translate, "to_english", lambda t, l: pytest.fail("LLM ran despite IndicTrans2")
    )

    assert query_prep.prepare("thavaram", "ta").for_retrieval == "from indictrans2"


def test_llm_backtranslation_is_labelled_as_the_weaker_check(monkeypatch):
    """Self-validation must never be reported as an independent check."""
    from app.tutor import llm_translate

    monkeypatch.setattr(tamil_quality.translate, "to_english", lambda t, l: None)
    monkeypatch.setattr(
        llm_translate,
        "to_english",
        lambda t, l: "The chlorophyll uses sunlight, carbon dioxide, water to make glucose "
        "and releases oxygen, giving energy.",
    )

    report = tamil_quality.evaluate(GOOD_TAMIL, CHUNK.concepts, "ta")

    assert report.backtranslation_available is True
    assert report.backtranslation_engine == "llm"     # not "indictrans2"
    assert "weaker" in report.note
    assert report.passed


def test_backtranslation_engine_is_none_when_nothing_can_translate(monkeypatch):
    monkeypatch.setattr(tamil_quality.translate, "to_english", lambda t, l: None)

    report = tamil_quality.evaluate(GOOD_TAMIL, CHUNK.concepts, "ta")

    assert report.backtranslation_engine == "none"
    assert report.backtranslation_available is False


def test_tanglish_is_translated_even_when_indicxlit_is_missing(monkeypatch):
    """Regression: this exact query was refused as out of scope on a real run.

    With IndicXlit unavailable the text stays romanised, so gating translation on
    "was transliterated or is native script" skipped it entirely — and a refusal
    on a valid question is indistinguishable from the guardrail working.
    """
    from app.tutor import llm_translate, query_prep

    monkeypatch.setattr(transliterate, "to_native", lambda t, l: None)   # IndicXlit missing
    monkeypatch.setattr(translate, "to_english", lambda t, l: None)      # IndicTrans2 missing
    monkeypatch.setattr(llm_translate, "to_english", lambda t, l: "how does a plant eat")

    prepared = query_prep.prepare("thavaram epdi saapdum", "ta")

    assert prepared.transliterated is False
    assert prepared.translated is True
    assert prepared.for_retrieval == "how does a plant eat"


def test_romanised_regional_detection_excludes_english():
    assert transliterate.is_romanised_regional("thavaram epdi saapdum") is True
    assert transliterate.is_romanised_regional("prakasha samsleshanam") is True
    assert transliterate.is_romanised_regional("how do plants make food") is False
    assert transliterate.is_romanised_regional("தாவரம் எப்படி சாப்பிடும்") is False


def test_rate_limited_translation_is_never_reported_as_out_of_scope(monkeypatch):
    """A quota failure must not be dressed up as the grounding guardrail.

    Observed live: Gemini returned 429, query translation failed, retrieval
    missed, and the student was told "that isn't in this chapter" — a false
    claim about their textbook caused entirely by our quota.
    """
    from fastapi.testclient import TestClient

    from app.main import app
    from app.tutor import llm_translate
    from app.tutor.llm_client import LLMRateLimited

    monkeypatch.setattr(transliterate, "to_native", lambda t, l: None)
    monkeypatch.setattr(translate, "to_english", lambda t, l: None)

    def rate_limited(*a, **k):
        raise LLMRateLimited("gemini call failed: 429 quota exceeded")

    monkeypatch.setattr(llm_translate, "to_english", rate_limited)

    response = TestClient(app).post(
        "/api/tutor/explain", json={"page_id": "pg42", "query": "thavaram epdi saapdum"}
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "try again" in detail.lower()
    assert "chapter" not in detail.lower()


def test_a_genuine_out_of_scope_question_still_refuses_normally(monkeypatch):
    """The guard must not swallow real refusals — only untrustworthy ones."""
    from fastapi.testclient import TestClient

    from app.main import app

    response = TestClient(app).post(
        "/api/tutor/explain", json={"page_id": "pg42", "query": "what is the capital of france"}
    )

    assert response.status_code == 200
    assert response.json()["grounded"] is False


def test_transient_errors_are_classified_apart_from_missing_models():
    from app.tutor.llm_client import LLMRateLimited, LLMUnavailable, _classify

    assert isinstance(_classify(Exception("Error code: 429 quota"), "x"), LLMRateLimited)
    assert isinstance(_classify(Exception("503 overloaded"), "x"), LLMRateLimited)
    missing = _classify(Exception("no GEMINI_API_KEY"), "x")
    assert isinstance(missing, LLMUnavailable) and not isinstance(missing, LLMRateLimited)


def test_every_route_that_prepares_a_query_can_actually_run():
    """Guards a NameError class of bug: /explain/speak once called a deleted helper.

    Endpoints with heavyweight dependencies are easy to leave untested and easy
    to break during a refactor — reaching the 503 proves the handler body ran.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    response = TestClient(app).post(
        "/api/tutor/explain/speak", json={"page_id": "pg42", "query": "how do plants make food"}
    )

    assert response.status_code == 503          # TTS not installed, not a crash
    assert "text-to-speech" in response.json()["detail"].lower()
