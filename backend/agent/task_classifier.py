"""
backend/agent/task_classifier.py
Authoritative Agent Task Classifier for SatQuery AI.
Determines optimal specialist model routing based on sensor modality combinations,
natural language query intent, and UI task hints.

Owner: Member 1
"""

from typing import Optional

class TaskDecision:
    """Encapsulates the agent's authoritative routing decision."""
    def __init__(
        self,
        task_name: str,
        specialist_key: str,
        task_hint_provided: str,
        is_override: bool,
        rationale: str
    ):
        self.task_name = task_name
        self.specialist_key = specialist_key
        self.task_hint_provided = task_hint_provided
        self.is_override = is_override
        self.rationale = rationale

class TaskClassifier:
    """
    Evaluates multi-modal sensor inputs and text queries to make an authoritative
    task selection decision.
    """

    def classify_task(
        self,
        mode: str,
        task_hint: Optional[str] = None,
        question: Optional[str] = None
    ) -> TaskDecision:
        """
        Classifies input configuration into one of the 4 supported tasks:
        - vqa
        - caption
        - change
        - multimodal_fusion
        """
        hint_clean = (task_hint or "").strip().lower()
        hint_display = task_hint if task_hint else "none"
        q_lower = (question or "").lower()

        # 1. Bi-temporal Modality Priority
        if mode == "bitemporal":
            is_override = bool(hint_clean and hint_clean not in ("change", "bitemporal", "difference"))
            rationale = (
                "Dual-temporal rasters (T1 + T2) detected. Authoritatively routed to Bi-Temporal Change Specialist."
                if not is_override
                else f"Dual-temporal rasters detected. Overrode UI hint '{hint_display}' to authoritative task 'change'."
            )
            return TaskDecision(
                task_name="change",
                specialist_key="change",
                task_hint_provided=hint_display,
                is_override=is_override,
                rationale=rationale
            )

        # 2. Optical + SAR Modality Priority
        if mode in ("optical_sar", "tri_modal"):
            is_override = bool(hint_clean and hint_clean not in ("multimodal", "optical_sar", "fusion", "sar"))
            rationale = (
                "Optical + SAR radar sensor channels detected. Authoritatively routed to Optical-SAR Fusion Specialist."
                if not is_override
                else f"Optical + SAR channels detected. Overrode UI hint '{hint_display}' to authoritative task 'multimodal_fusion'."
            )
            return TaskDecision(
                task_name="multimodal_fusion",
                specialist_key="fusion",
                task_hint_provided=hint_display,
                is_override=is_override,
                rationale=rationale
            )

        # 3. Single Scene Modality (VQA vs. Captioning)
        is_caption_hint = hint_clean in ("caption", "describe", "summary", "overview")
        is_caption_query = any(k in q_lower for k in ["caption", "describe all", "scene overview", "summary of the scene", "generate caption"])

        if is_caption_hint or is_caption_query:
            return TaskDecision(
                task_name="caption",
                specialist_key="caption",
                task_hint_provided=hint_display,
                is_override=False,
                rationale="Single satellite scene classified as automated scene captioning based on intent and task hint."
            )

        # Default Single Scene Task is Visual Question Answering (VQA)
        is_override = bool(hint_clean and hint_clean not in ("vqa", "none", ""))
        return TaskDecision(
            task_name="vqa",
            specialist_key="vqa",
            task_hint_provided=hint_display,
            is_override=is_override,
            rationale=(
                "Single satellite scene classified as Visual Question Answering (VQA)."
                if not is_override
                else f"Single image provided without change/SAR pairs. Overrode '{hint_display}' to 'vqa'."
            )
        )
