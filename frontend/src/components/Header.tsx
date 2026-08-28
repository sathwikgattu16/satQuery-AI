import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="p-4 border-b border-slate-700 bg-slate-800 flex justify-between items-center">
      <div>
        <h1 className="text-xl font-bold text-sky-400">SATQUERY AI</h1>
        <p className="text-xs text-slate-400">Multimodal Remote Sensing Vision-Language Assistant</p>
      </div>
      <span className="text-xs px-2 py-1 bg-sky-900/60 text-sky-300 rounded border border-sky-700">
        ISRO Hackathon Prototype
      </span>
    </header>
  );
};
