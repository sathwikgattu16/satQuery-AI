import React from 'react';
import { QueryResponse } from '../types';

interface ResultsDisplayProps {
  response: QueryResponse | null;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ response }) => {
  if (!response) {
    return (
      <div className="p-4 bg-slate-800/50 rounded border border-slate-700/50 text-slate-400 text-center">
        Submit a query to view multimodal remote sensing analysis.
      </div>
    );
  }

  return (
    <div className="p-4 bg-slate-800 rounded border border-slate-700 flex flex-col gap-3">
      <div className="flex justify-between items-center">
        <span className="text-xs font-semibold px-2 py-0.5 bg-sky-950 text-sky-400 rounded border border-sky-800">
          Task: {response.task}
        </span>
        <span className="text-xs text-slate-300">
          Confidence: <strong className="text-emerald-400">{(response.confidence * 100).toFixed(1)}%</strong>
        </span>
      </div>
      <div>
        <h3 className="text-sm font-semibold text-slate-400 mb-1">Answer</h3>
        <p className="text-slate-100 text-base leading-relaxed">{response.answer}</p>
      </div>
      <div className="text-xs text-slate-400">
        Specialists: {response.specialists.join(', ')}
      </div>
    </div>
  );
};
