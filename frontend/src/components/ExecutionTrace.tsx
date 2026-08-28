import React from 'react';
import { ExecutionSummary } from '../types';

interface ExecutionTraceProps {
  summary?: ExecutionSummary;
}

export const ExecutionTrace: React.FC<ExecutionTraceProps> = ({ summary }) => {
  if (!summary || !summary.steps || summary.steps.length === 0) {
    return null;
  }

  return (
    <div className="p-4 bg-slate-800/80 rounded border border-slate-700">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-sm font-semibold text-slate-300">Auditable Execution Trace</h3>
        {summary.total_duration_ms && (
          <span className="text-xs text-slate-400">{summary.total_duration_ms} ms</span>
        )}
      </div>
      <div className="space-y-2">
        {summary.steps.map((step, idx) => (
          <div key={idx} className="flex items-start gap-2 text-xs">
            <span
              className={`w-2 h-2 mt-1 rounded-full ${
                step.status === 'success' ? 'bg-emerald-400' : 'bg-rose-400'
              }`}
            />
            <div className="flex-1">
              <strong className="text-slate-200">{step.step_name}:</strong>{' '}
              <span className="text-slate-400">{step.detail}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
