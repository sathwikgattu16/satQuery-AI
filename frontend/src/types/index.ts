/**
 * frontend/src/types/index.ts
 * Shared TypeScript interfaces matching the backend API contract.
 * Owner: Member 2
 */

export type InputType = 'single' | 'optical_sar' | 'bitemporal';

export interface QueryRequest {
  query: string;
  input_type: InputType;
  image_primary: string;
  image_secondary?: string;
}

export interface EvidencePayload {
  type: string;
  data_url?: string;
  description?: string;
}

export interface TraceStep {
  step_name: string;
  status: 'success' | 'failed' | 'running';
  detail?: string;
}

export interface ExecutionSummary {
  steps: TraceStep[];
  total_duration_ms?: number;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  task: string;
  specialists: string[];
  evidence?: EvidencePayload;
  execution_summary: ExecutionSummary;
}
