"""Regional-language validation via AI4Bharat IndicTrans2 backtranslation.

(Module name kept from the branch spec; it now handles all five languages —
Tamil is just the one we tuned.)

This is **validation, not generation**. The explanation is composed by the lesson
model in its own voice; IndicTrans2 only tells us whether the concepts survived:

    composed Tamil --IndicTrans2 (ta -> en)--> backtranslation
    backtranslation + original Tamil --fuzzy match--> concepts still present?

Concepts are checked against the regional text *and* the backtranslation, because
Tanglish deliberately keeps terms like "chlorophyll" in English — those match in
the source directly, and round-tripping them through a translator is pointless.

If IndicTrans2 is missing, this degrades to a concept check against the raw
regional text and says so in the report. A missing optional dependency must never
fail a lesson on demo day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.tutor.grounding_check import missing_concepts
from app.tutor.indic import translate

log = logging.getLogger(__name__)


@dataclass
class TamilQualityReport:
    """What the check found. `missing` non-empty means: regenerate once."""

    missing: list[str] = field(default_factory=list)
    backtranslation: str | None = None
    backtranslation_available: bool = False
    note: str = ""

    @property
    def passed(self) -> bool:
        return not self.missing


def evaluate(
    regional_text: str, concepts: list[str], language_code: str = "ta"
) -> TamilQualityReport:
    """Check that `concepts` survived into the composed regional explanation."""
    if not (regional_text or "").strip():
        return TamilQualityReport(missing=list(concepts), note="empty regional explanation")

    backtranslation = translate.to_english(regional_text, language_code)

    if backtranslation is None:
        return TamilQualityReport(
            missing=missing_concepts(concepts, regional_text),
            backtranslation_available=False,
            note=(
                "IndicTrans2 backtranslation unavailable — concepts checked against "
                "the regional text only"
            ),
        )

    return TamilQualityReport(
        missing=missing_concepts(concepts, regional_text, backtranslation),
        backtranslation=backtranslation,
        backtranslation_available=True,
        note="checked against IndicTrans2 backtranslation",
    )
