"""Concept extraction.

The tutor's analogy-coverage check requires every concept to be mapped to a
concrete component, and its regional-language check requires every concept to
survive into the Tamil lesson. So a bad concept is not cosmetic — it costs a
regeneration, and an empty list disables both checks silently.
"""

import pytest
from tests.rag.conftest import fake_response

from app.rag import concepts
from app.rag.concepts import _fallback, _singular, _parse, extract


@pytest.fixture(autouse=True)
def clear_cache():
    extract.cache_clear()
    yield
    extract.cache_clear()


PAGE = (
    "Photosynthesis is the process by which green plants make their own food. "
    "It takes place mainly in the leaves, inside tiny structures called chloroplasts. "
    "Chloroplasts contain a green pigment called chlorophyll, which absorbs sunlight. "
    "The plant takes in carbon dioxide from the air through small pores on the leaf "
    "called stomata, and absorbs water from the soil through its roots."
)


class TestSingular:
    @pytest.mark.parametrize(
        "word,expected",
        [
            ("plants", "plant"),
            ("chloroplasts", "chloroplast"),
            ("leaves", "leave"),
        ],
    )
    def test_strips_real_plurals(self, word, expected):
        assert _singular(word) == expected

    @pytest.mark.parametrize(
        "word", ["photosynthesis", "analysis", "process", "gas", "nucleus", "stomata"]
    )
    def test_leaves_science_nouns_alone(self, word):
        """Stripping the 's' off 'photosynthesis' invents 'photosynthesi' — a
        concept the tutor then hunts for in the lesson and never finds."""
        assert _singular(word) == word


class TestFallback:
    def test_finds_the_terms_a_teacher_would_explain(self):
        found = _fallback(PAGE, 6)
        assert "carbon dioxide" in found, "multi-word terms must survive tokenisation"
        assert "chlorophyll" in found or "chloroplast" in found

    def test_respects_the_cap(self):
        assert len(_fallback(PAGE, 3)) == 3

    def test_drops_filler_words(self):
        found = _fallback(PAGE, 8)
        assert not ({"the", "which", "from", "takes", "called"} & set(found))

    def test_never_emits_a_multiword_term_and_its_parts(self):
        """'carbon' and 'dioxide' competing separately would ask the tutor to
        map three concepts where the page has one."""
        found = _fallback(PAGE, 8)
        if "carbon dioxide" in found:
            assert "carbon" not in found and "dioxide" not in found

    def test_empty_text_yields_nothing(self):
        assert _fallback("", 6) == []

    def test_output_is_lowercase_and_unique(self):
        found = _fallback(PAGE, 8)
        assert found == [f.lower() for f in found]
        assert len(set(found)) == len(found)


class TestParse:
    def test_comma_separated(self):
        assert _parse("chlorophyll, sunlight, glucose", 6) == (
            "chlorophyll",
            "sunlight",
            "glucose",
        )

    def test_tolerates_bullets_and_newlines(self):
        """The model occasionally ignores 'comma-separated' on short pages."""
        assert _parse("- chlorophyll\n- sunlight\n- glucose", 6) == (
            "chlorophyll",
            "sunlight",
            "glucose",
        )

    def test_tolerates_numbering(self):
        assert _parse("1. chlorophyll\n2. sunlight", 6) == ("chlorophyll", "sunlight")

    def test_drops_duplicates_and_overlong_junk(self):
        long = "x" * 60
        assert _parse(f"glucose, glucose, {long}", 6) == ("glucose",)

    def test_respects_the_cap(self):
        assert len(_parse("a, b, c, d, e, f, g, h", 3)) == 3


class TestExtract:
    def test_uses_the_model_when_available(self, stub_client):
        stub_client(fake_response("chlorophyll, sunlight, carbon dioxide"))
        assert extract(PAGE) == ("chlorophyll", "sunlight", "carbon dioxide")

    def test_falls_back_when_the_model_is_unavailable(self, stub_client):
        stub_client(None, raises=RuntimeError("no api key"))
        found = extract(PAGE)
        assert found, "ingestion must still produce concepts without a key"
        assert "carbon dioxide" in found

    def test_falls_back_when_the_model_returns_nothing(self, stub_client):
        stub_client(fake_response(None))
        assert extract(PAGE)

    def test_empty_text_yields_nothing(self, stub_client):
        stub_client(fake_response("should not be called"))
        assert extract("") == ()
        assert extract("   ") == ()

    def test_result_is_hashable_for_the_cache(self, stub_client):
        stub_client(fake_response("glucose"))
        assert isinstance(extract(PAGE), tuple)

    def test_repeated_text_is_extracted_once(self, stub_client):
        calls = stub_client(fake_response("glucose"))
        for _ in range(3):
            extract(PAGE)
        assert len(calls) == 1, "ingest cost is per upload, not per re-chunk"
