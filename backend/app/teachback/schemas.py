"""
Data contracts for the teach-back evaluation module.

Keep this file dependency-light: it should only ever import pydantic.
Other modules (API routes, orchestration layer) depend on these shapes,
so changes here should be rare and communicated to the team.
"""

from typing import Literal

from pydantic import BaseModel, Field

UnderstandingLevel = Literal["strong", "partial", "weak"]
NextAction = Literal["practice", "retry_teachback", "reteach"]


class TeachbackRequest(BaseModel):
    """Input to evaluate_teachback."""

    concept: str = Field(..., description="The concept/topic being taught, e.g. 'Newton's First Law'.")
    tutor_explanation: str = Field(..., description="The explanation Bodhi gave the student.")
    student_answer: str = Field(..., description="The student's free-text teach-back attempt.")


class TeachbackResult(BaseModel):
    """Output of evaluate_teachback. Machine-readable, consumed by other modules/frontend."""

    understanding_level: UnderstandingLevel
    understood: list[str] = Field(default_factory=list, description="What the student got right.")
    misconceptions: list[str] = Field(default_factory=list, description="Specific incorrect beliefs, not generic 'wrong'.")
    missing_concepts: list[str] = Field(default_factory=list, description="Important parts the student left out.")
    feedback: str = Field(..., description="Natural, tutor-like corrective feedback shown to the student.")
    needs_reteach: bool
    next_action: NextAction
