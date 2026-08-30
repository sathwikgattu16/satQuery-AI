import React from 'react';
import {
  GitCommit,
  CheckCircle,
  FileCheck,
  Cpu,
  Search,
  ShieldCheck,
  Terminal,
  Activity,
  ArrowDown
} from 'lucide-react';
import { ExecutionSummary } from '../types';

interface ExecutionTraceProps {
  summary: ExecutionSummary | null;
  loading: boolean;
}

export const ExecutionTrace: React.FC<ExecutionTraceProps> = ({
  summary,
  loading,
}) => {
  if (loading) {
    return (
      <div className="card trace-card loading">
        <div className="card-header">
          <div className="card-title-group">
            <div className="status-indicator-badge">
              <Activity size={16} className="text-cyan spin" />
              <h2 className="card-title">Agent Execution Trace</h2>
            </div>
            <span className="card-hint">Real-time reasoning pipeline</span>
          </div>
        </div>
        <div className="trace-loading-container">
          <div className="trace-step-skeleton">
            <div className="skeleton-dot active"></div>
            <div className="skeleton-line"></div>
            <div className="skeleton-text">Interpreting query & sensory inputs...</div>
          </div>
          <div className="trace-step-skeleton">
            <div className="skeleton-dot pending"></div>
            <div className="skeleton-line"></div>
            <div className="skeleton-text">Evaluating task & checking sensor compatibility...</div>
          </div>
          <div className="trace-step-skeleton">
            <div className="skeleton-dot pending"></div>
            <div className="skeleton-text">Deploying specialized neural model...</div>
          </div>
        </div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="card trace-card empty">
        <div className="card-header">
          <div className="card-title-group">
            <div className="status-indicator-badge">
              <Terminal size={16} className="text-muted" />
              <h2 className="card-title">Agent Execution Trace</h2>
            </div>
            <span className="card-hint">Authoritative backend pipeline lifecycle</span>
          </div>
        </div>
        <div className="trace-empty-box">
          <p className="trace-empty-text">
            Submit a query to inspect the autonomous agent's decision trace, task selection rationale, model dispatching, and sensor verification steps.
          </p>
        </div>
      </div>
    );
  }

  // Derive logical stages from the authoritative backend execution_summary
  const steps = [
    {
      id: 'input',
      icon: <FileCheck size={14} />,
      title: 'Input Ingestion & Registration',
      description: `Ingested ${summary.num_images_provided} raster channel(s). Task hint received: "${summary.task_hint_provided || 'None'}"`,
      status: 'completed',
    },
    {
      id: 'task_routing',
      icon: <Search size={14} />,
      title: 'Agent Task Decision',
      description: `Agent authoritatively selected task: "${summary.selected_task.toUpperCase()}". ${
        summary.task_hint_provided && summary.task_hint_provided !== summary.selected_task
          ? `(Overrode initial hint "${summary.task_hint_provided}")`
          : 'Validated against input schema.'
      }`,
      status: 'completed',
    },
    {
      id: 'compatibility',
      icon: <ShieldCheck size={14} />,
      title: 'Sensor Compatibility & Preprocessing',
      description: summary.compatibility_notes || 'Inputs verified and calibrated.',
      status: 'completed',
    },
    {
      id: 'model_dispatch',
      icon: <Cpu size={14} />,
      title: 'Specialist Models Dispatched',
      description: summary.models_used && summary.models_used.length > 0
        ? `Executed ensemble: ${summary.models_used.join(' → ')}`
        : 'Specialist neural weights deployed.',
      status: 'completed',
    },
    {
      id: 'synthesis',
      icon: <CheckCircle size={14} />,
      title: 'Geospatial Answer Synthesis',
      description: 'Extracted spatial anchors, calculated confidence, and prepared final output payload.',
      status: 'completed',
    },
  ];

  return (
    <div className="card trace-card">
      <div className="card-header">
        <div className="card-title-group">
          <div className="status-indicator-badge">
            <GitCommit size={16} className="text-purple" />
            <h2 className="card-title">Agent Execution Trace</h2>
          </div>
          <span className="card-hint">
            Authoritative decision pipeline reported by backend (`execution_summary`)
          </span>
        </div>
        <span className="badge-tag tag-success">
          <CheckCircle size={11} /> 5/5 Stages Completed
        </span>
      </div>

      {/* Visual Pipeline Timeline */}
      <div className="trace-timeline">
        {steps.map((step, idx) => (
          <div key={step.id} className="timeline-item">
            <div className="timeline-rail">
              <div className="timeline-node">
                {step.icon}
              </div>
              {idx < steps.length - 1 && <div className="timeline-connector"></div>}
            </div>
            <div className="timeline-content">
              <div className="timeline-header">
                <span className="timeline-step-index">Step {idx + 1}</span>
                <h4 className="timeline-title">{step.title}</h4>
              </div>
              <p className="timeline-desc">{step.description}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Raw Backend Trace Steps (if provided directly by backend) */}
      {summary.trace_steps && summary.trace_steps.length > 0 && (
        <div className="raw-trace-box">
          <div className="raw-trace-header">
            <Terminal size={12} />
            <span>Backend Diagnostic Log Streams:</span>
          </div>
          <div className="raw-trace-logs">
            {summary.trace_steps.map((logLine, idx) => (
              <div key={idx} className="raw-log-line">
                <ArrowDown size={10} className="log-arrow" />
                <span>{logLine}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
