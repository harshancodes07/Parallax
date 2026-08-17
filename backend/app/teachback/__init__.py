"""
Teach-back evaluation module.

Public interface:
    evaluate_teachback(concept, tutor_explanation, student_answer) -> TeachbackResult

This module is intentionally self-contained. It does not import from
rag/, tutor/, practice/, or the frontend. Callers elsewhere in the app
should only ever import from here:

    from app.teachback import evaluate_teachback, TeachbackResult
"""

from app.teachback.evaluator import evaluate_teachback
from app.teachback.schemas import TeachbackRequest, TeachbackResult

__all__ = ["evaluate_teachback", "TeachbackRequest", "TeachbackResult"]
