import React from 'react';
import {
  CheckCircle2,
  Clock,
  Zap,
  Cpu,
  Layers,
  ShieldCheck,
  AlertTriangle,
  Bot
} from 'lucide-react';
import { QueryResponse } from '../types';

interface ResultsDisplayProps {
  response: QueryResponse | null;
  loading: boolean;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  response,
  loading,
}) => {
  if (loading) {
    return (
      <div className="card results-card loading-state">
        <div className="loading-animation-container">
          <div className="radar-spinner">
            <div className="radar-sweep"></div>
          </div>
          <h3 className="loading-title">Executing Geospatial Agent Pipeline</h3>
          <p className="loading-subtitle">
            Ingesting multispectral / SAR rasters, dispatching neural specialists, and synthesizing remote sensing answer...
          </p>
        </div>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="card results-card empty-state">
        <div className="empty-content">
          <Bot size={44} className="empty-icon" />
          <h3 className="empty-title">Ready for Remote Sensing Analysis</h3>
          <p className="empty-subtitle">
            Configure image inputs and query above, then press <strong>Analyze Satellite Data</strong> to view AI answers, confidence metrics, and agent execution details.
          </p>
        </div>
      </div>
    );
  }

  const {
    task,
    answer,
    confidence,
    processing_time,
    execution_summary,
  } = response;

  const confidencePct = Math.round(confidence * 100);
  const isHighConfidence = confidence >= 0.85;
  const isOverride =
    execution_summary.task_hint_provided &&
    execution_summary.task_hint_provided !== 'none' &&
    execution_summary.task_hint_provided !== execution_summary.selected_task;

  return (
    <div className="card results-card">
      <div className="card-header">
        <div className="card-title-group">
          <div className="status-indicator-badge">
            <CheckCircle2 size={16} className="text-emerald" />
            <h2 className="card-title">Analysis Results</h2>
          </div>
          <span className="card-hint">Authoritative Earth Observation Synthesis</span>
        </div>
        <div className="task-badge-container">
          <span className="selected-task-badge">
            Task: <strong>{task.toUpperCase()}</strong>
          </span>
          {isOverride && (
            <span className="override-pill" title="Backend agent autonomously selected a different task than UI hint based on sensor inputs">
              <AlertTriangle size={12} /> Agent Overrode UI Hint (`{execution_summary.task_hint_provided}`)
            </span>
          )}
        </div>
      </div>

      {/* Primary AI Answer Box */}
      <div className="answer-box">
        <div className="answer-header">
          <span className="answer-label">
            <Bot size={16} /> Geospatial Intelligence Answer
          </span>
        </div>
        <div className="answer-content">
          <p className="answer-text">{answer}</p>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="metrics-grid">
        {/* Confidence Metric */}
        <div className="metric-card">
          <div className="metric-top">
            <span className="metric-name">
              <Zap size={14} className="metric-icon text-cyan" /> Confidence
            </span>
            <span className={`metric-tag ${isHighConfidence ? 'tag-high' : 'tag-med'}`}>
              {isHighConfidence ? 'High' : 'Moderate'}
            </span>
          </div>
          <div className="metric-value-row">
            <span className="metric-main-value">{confidencePct}%</span>
            <span className="metric-sub-value">({confidence.toFixed(2)})</span>
          </div>
          <div className="confidence-meter-track">
            <div
              className={`confidence-meter-fill ${isHighConfidence ? 'fill-emerald' : 'fill-cyan'}`}
              style={{ width: `${Math.min(100, confidencePct)}%` }}
            ></div>
          </div>
        </div>

        {/* Processing Time Metric */}
        <div className="metric-card">
          <div className="metric-top">
            <span className="metric-name">
              <Clock size={14} className="metric-icon text-purple" /> Latency
            </span>
            <span className="metric-tag tag-neutral">Inference</span>
          </div>
          <div className="metric-value-row">
            <span className="metric-main-value">{processing_time.toFixed(2)}s</span>
          </div>
          <span className="metric-detail-text">End-to-end model execution</span>
        </div>

        {/* Images Ingested Metric */}
        <div className="metric-card">
          <div className="metric-top">
            <span className="metric-name">
              <Layers size={14} className="metric-icon text-blue" /> Rasters
            </span>
            <span className="metric-tag tag-neutral">Input Sensor(s)</span>
          </div>
          <div className="metric-value-row">
            <span className="metric-main-value">{execution_summary.num_images_provided}</span>
            <span className="metric-sub-value">channel(s)</span>
          </div>
          <span className="metric-detail-text">Processed through agent pipeline</span>
        </div>
      </div>

      {/* Models Used */}
      <div className="models-section">
        <div className="section-label">
          <Cpu size={14} /> Models / Specialists Deployed:
        </div>
        <div className="models-pills">
          {execution_summary.models_used && execution_summary.models_used.length > 0 ? (
            execution_summary.models_used.map((model, idx) => (
              <span key={idx} className="model-pill">
                <span className="pill-dot"></span> {model}
              </span>
            ))
          ) : (
            <span className="model-pill">Standard EO Neural Specialist</span>
          )}
        </div>
      </div>

      {/* Compatibility Notes */}
      {execution_summary.compatibility_notes && (
        <div className="compatibility-box">
          <div className="comp-header">
            <ShieldCheck size={14} className="text-cyan" />
            <span className="comp-title">Sensor & Alignment Compatibility Notes</span>
          </div>
          <p className="comp-text">{execution_summary.compatibility_notes}</p>
        </div>
      )}
    </div>
  );
};
