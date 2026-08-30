/**
 * SATQUERY AI - Core Type Definitions
 * Locked Backend Contract
 */

export type InputMode = 'single' | 'bitemporal' | 'optical_sar';

export type SingleImageTaskHint = 'vqa' | 'caption';
export type TaskHint = SingleImageTaskHint | 'change' | 'multimodal' | string;

export interface FileInputs {
  image?: File | null;
  image_t2?: File | null;
  sar?: File | null;
}

export interface ImagePreviewUrls {
  image?: string | null;
  image_t2?: string | null;
  sar?: string | null;
}

/**
 * Locked Request Payload Structure for POST /api/analyze (multipart/form-data)
 */
export interface QueryRequest {
  task_hint?: string;
  question?: string;
  image?: File | null;
  image_t2?: File | null;
  sar?: File | null;
}

/**
 * Locked Execution Summary Structure (REQUIRED in response)
 */
export interface ExecutionSummary {
  selected_task: string;
  task_hint_provided?: string;
  models_used: string[];
  num_images_provided: number;
  compatibility_notes: string;
  trace_steps?: string[];
  implementation_status?: string;
}

/**
 * Visual Evidence / Visualization object for future spatial overlays
 */
export interface VisualEvidence {
  type?: 'overlay' | 'heatmap' | 'change_mask' | 'split' | 'image' | 'diff';
  title?: string;
  url?: string;
  base64?: string;
  description?: string;
  metrics?: Record<string, string | number>;
}

/**
 * Locked Response Structure from POST /api/analyze
 */
export interface QueryResponse {
  success: boolean;
  task: string;
  answer: string;
  confidence: number | null;
  processing_time: number;
  execution_summary: ExecutionSummary;
  visualization?: VisualEvidence | string | null;
  error?: string;
}

/**
 * UI State for active execution step visualization
 */
export interface TraceStepItem {
  id: string;
  title: string;
  detail: string;
  status: 'completed' | 'active' | 'pending';
}
