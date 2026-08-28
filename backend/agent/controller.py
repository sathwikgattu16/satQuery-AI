"""
backend/agent/controller.py
SatQuery Agent Controller orchestrating the full end-to-end inference lifecycle.
Owner: Member 1
"""

import time
from typing import Dict, Any
from backend.schemas.api_models import QueryRequest, QueryResponse, ExecutionSummary, TraceStep, EvidencePayload
from backend.agent.compatibility import CompatibilityChecker
from backend.agent.task_classifier import TaskClassifier
from backend.agent.registry import ModelRegistry

class SatQueryController:
    """Main agentic orchestrator for SatQuery AI."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.compatibility_checker = CompatibilityChecker()
        self.task_classifier = TaskClassifier()

    def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Orchestrate request validation, task classification, specialist dispatch,
        and auditable execution trace generation.
        """
        start_time = time.time()
        steps = []

        # 1. Compatibility Check
        is_valid, err_msg = self.compatibility_checker.validate(request)
        if not is_valid:
            steps.append(TraceStep(step_name="Input Validation", status="failed", detail=err_msg))
            return QueryResponse(
                answer=f"Input Error: {err_msg}",
                confidence=0.0,
                task="invalid_input",
                specialists=[],
                evidence=None,
                execution_summary=ExecutionSummary(
                    steps=steps,
                    total_duration_ms=int((time.time() - start_time) * 1000)
                )
            )
        steps.append(TraceStep(step_name="Input Validation", status="success", detail=f"Valid {request.input_type} inputs."))

        # 2. Task Classification
        task_info = self.task_classifier.classify(request)
        task_name = task_info["task"]
        specialist_key = task_info["specialist_key"]
        steps.append(TraceStep(step_name="Task Classification", status="success", detail=f"Target: {task_name}"))

        # 3. Specialist Lookup & Execution
        specialist = self.registry.get(specialist_key)
        specialist_names = [specialist.name] if specialist else ["PlaceholderMockSpecialist"]
        
        if specialist:
            result = specialist.predict(
                query=request.query,
                image_primary=request.image_primary,
                image_secondary=request.image_secondary
            )
            answer = result.get("answer", "Analysis completed.")
            confidence = result.get("confidence", 0.85)
            evidence_data = result.get("evidence")
            steps.append(TraceStep(
                step_name="Specialist Execution",
                status="success",
                detail=result.get("execution_detail", f"Executed {specialist.name}")
            ))
        else:
            answer = f"Placeholder response for {task_name} on query: '{request.query}'"
            confidence = 0.80
            evidence_data = {"type": "overlay", "description": "Mock placeholder visual evidence."}
            steps.append(TraceStep(
                step_name="Specialist Execution",
                status="success",
                detail=f"Executed placeholder mock for {task_name}"
            ))

        evidence = EvidencePayload(**evidence_data) if evidence_data else None
        duration_ms = int((time.time() - start_time) * 1000)

        return QueryResponse(
            answer=answer,
            confidence=confidence,
            task=task_name,
            specialists=specialist_names,
            evidence=evidence,
            execution_summary=ExecutionSummary(steps=steps, total_duration_ms=duration_ms)
        )
