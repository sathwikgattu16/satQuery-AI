"""
backend/agent/controller.py
SatQuery Agent Controller orchestrating the full end-to-end inference lifecycle.
Owner: Member 1
"""

import time
from typing import Optional, List
from fastapi import UploadFile
from backend.schemas.api_models import QueryResponse, ExecutionSummary, VisualEvidence
from backend.agent.compatibility import CompatibilityChecker, CompatibilityResult
from backend.agent.data_adapter import RasterDataAdapter, AdaptedDataPayload
from backend.agent.task_classifier import TaskClassifier, TaskDecision
from backend.agent.registry import ModelRegistry

class SatQueryController:
    """Main agentic orchestrator for SatQuery AI."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        compatibility_checker: Optional[CompatibilityChecker] = None,
        data_adapter: Optional[RasterDataAdapter] = None,
        task_classifier: Optional[TaskClassifier] = None
    ):
        self.registry = registry or ModelRegistry()
        self.compatibility_checker = compatibility_checker or CompatibilityChecker()
        self.data_adapter = data_adapter or RasterDataAdapter()
        self.task_classifier = task_classifier or TaskClassifier()

    async def process_analysis_request(
        self,
        image: UploadFile,
        image_t2: Optional[UploadFile] = None,
        sar: Optional[UploadFile] = None,
        question: Optional[str] = None,
        task_hint: Optional[str] = None
    ) -> QueryResponse:
        """
        Executes the 6-stage auditable agent lifecycle:
        1. Input Validation & Modality Check
        2. Data Ingestion & Tensor Adaptation
        3. Task Classification & Routing
        4. Specialist Selection & Dispatch
        5. Specialist Execution
        6. Response & Evidence Synthesis
        """
        start_time = time.time()
        trace_steps: List[str] = []

        # Stage 1: Input Validation & Modality Compatibility
        compat_result: CompatibilityResult = await self.compatibility_checker.check_compatibility(
            image=image,
            image_t2=image_t2,
            sar=sar,
            task_hint=task_hint
        )
        trace_steps.append(
            f"Stage 1 [Validation]: Ingested {compat_result.num_images} file(s) with query: '{question or 'N/A'}'"
        )
        trace_steps.append(
            f"Stage 2 [Sensor Compatibility]: {compat_result.compatibility_notes}"
        )

        # Stage 2: Data Adaptation & Tensor Preprocessing
        adapted_payload: AdaptedDataPayload = await self.data_adapter.adapt_inputs(
            image=image,
            image_t2=image_t2,
            sar=sar,
            task_hint=task_hint
        )
        if adapted_payload.is_multispectral and adapted_payload.primary_tensor is not None:
            adapter_trace = f"Stage 3 [Data Adapter]: Formatted multi-spectral tensor to {list(adapted_payload.primary_tensor.shape)} (B02-B07) for Prithvi backbone."
        else:
            adapter_trace = "Stage 3 [Data Adapter]: Processed on safe Demo/RGB path without multispectral band fabrication."
        trace_steps.append(adapter_trace)

        # Stage 3: Task Classification & Authoritative Routing
        task_decision: TaskDecision = self.task_classifier.classify_task(
            mode=compat_result.detected_mode,
            task_hint=task_hint,
            question=question
        )
        trace_steps.append(
            f"Stage 4 [Task Classification]: Hint '{task_decision.task_hint_provided}' -> Agent Authoritative Task: '{task_decision.task_name}'"
        )
        if task_decision.is_override:
            trace_steps.append(f"Stage 4 [Routing Override]: {task_decision.rationale}")

        # Stage 4: Specialist Lookup & Model Selection
        specialist = self.registry.get(task_decision.specialist_key)
        specialist_name = specialist.name if specialist else f"DefaultMock-{task_decision.task_name.upper()}"
        is_mock_specialist = getattr(specialist, "is_mock", True) if specialist else True
        mode_label = "MOCK / DEMO" if is_mock_specialist else "REAL NEURAL WEIGHTS"
        
        trace_steps.append(
            f"Stage 5 [Specialist Selection]: Dispatched [{specialist_name}] (Execution Mode: {mode_label})"
        )

        # Stage 5: Specialist Execution
        primary_input = adapted_payload.primary_tensor if adapted_payload.primary_tensor is not None else adapted_payload.primary_raw
        secondary_input = adapted_payload.secondary_tensor if adapted_payload.secondary_tensor is not None else adapted_payload.secondary_raw

        if specialist:
            result = specialist.predict(
                query=question or "",
                image_primary=primary_input,
                image_secondary=secondary_input,
                context=adapted_payload.metadata
            )
            answer = result.get("answer", "Analysis complete.")
            confidence = float(result.get("confidence", 0.90))
            raw_evidence = result.get("evidence")
            exec_detail = result.get("execution_detail", f"Executed {specialist_name}")
            trace_steps.append(f"Stage 6 [Execution]: {exec_detail}")
        else:
            answer = f"[MOCK DEMO] Analysis completed for task '{task_decision.task_name}' on query: '{question or 'N/A'}'"
            confidence = 0.88
            raw_evidence = {
                "type": "overlay",
                "title": f"Default {task_decision.task_name.upper()} Evidence",
                "description": "Default mock visual evidence.",
                "metrics": {"Status": "Mock Default"}
            }
            trace_steps.append(f"Stage 6 [Execution]: Executed fallback mock for {task_decision.task_name}")

        # Stage 6: Response & Evidence Synthesis
        processing_time = round(time.time() - start_time, 3)
        if processing_time <= 0:
            processing_time = 0.015

        trace_steps.append(
            f"Stage 7 [Synthesis]: Synthesized response with confidence score: {round(confidence * 100, 1)}% (Latency: {processing_time}s)"
        )

        visualization: Optional[VisualEvidence] = None
        if raw_evidence and isinstance(raw_evidence, dict):
            visualization = VisualEvidence(
                type=raw_evidence.get("type", "overlay"),
                title=raw_evidence.get("title", f"{task_decision.task_name.upper()} Visual Evidence"),
                url=raw_evidence.get("url") or raw_evidence.get("data_url"),
                base64=raw_evidence.get("base64"),
                description=raw_evidence.get("description"),
                metrics=raw_evidence.get("metrics")
            )

        execution_summary = ExecutionSummary(
            selected_task=task_decision.task_name,
            task_hint_provided=task_decision.task_hint_provided,
            models_used=[specialist_name],
            num_images_provided=compat_result.num_images,
            compatibility_notes=compat_result.compatibility_notes,
            trace_steps=trace_steps
        )

        return QueryResponse(
            success=True,
            task=task_decision.task_name,
            answer=answer,
            confidence=confidence,
            processing_time=processing_time,
            execution_summary=execution_summary,
            visualization=visualization
        )
