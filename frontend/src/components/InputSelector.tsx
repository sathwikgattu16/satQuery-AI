import React from 'react';
import { InputType } from '../types';

interface InputSelectorProps {
  inputType: InputType;
  onSelectType: (type: InputType) => void;
}

export const InputSelector: React.FC<InputSelectorProps> = ({ inputType, onSelectType }) => {
  return (
    <div className="flex gap-2 p-3 bg-slate-800 rounded border border-slate-700">
      {(['single', 'optical_sar', 'bitemporal'] as InputType[]).map((type) => (
        <button
          key={type}
          onClick={() => onSelectType(type)}
          className={`px-3 py-1.5 text-sm rounded font-medium transition ${
            inputType === type
              ? 'bg-sky-600 text-white'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          {type === 'single' && 'Single Image'}
          {type === 'optical_sar' && 'Optical + SAR Pair'}
          {type === 'bitemporal' && 'Bi-temporal Pair (T1 + T2)'}
        </button>
      ))}
    </div>
  );
};
