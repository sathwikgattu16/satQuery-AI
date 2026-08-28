import React from 'react';
import { EvidencePayload } from '../types';

interface EvidenceViewerProps {
  evidence?: EvidencePayload;
}

export const EvidenceViewer: React.FC<EvidenceViewerProps> = ({ evidence }) => {
  if (!evidence) {
    return null;
  }

  return (
    <div className="p-4 bg-slate-800 rounded border border-slate-700">
      <h3 className="text-sm font-semibold text-slate-400 mb-2">Visual Evidence ({evidence.type})</h3>
      {evidence.data_url ? (
        <img src={evidence.data_url} alt="Visual Evidence" className="w-full h-auto rounded border border-slate-600" />
      ) : (
        <div className="p-6 bg-slate-900 rounded border border-dashed border-slate-700 text-center text-xs text-slate-500">
          Visual evidence preview placeholder ({evidence.type})
        </div>
      )}
      {evidence.description && (
        <p className="text-xs text-slate-400 mt-2">{evidence.description}</p>
      )}
    </div>
  );
};
