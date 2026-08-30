import React from 'react';
import {
  Send,
  Loader2,
  HelpCircle,
  FileText,
  Clock,
  Sparkles,
  Sliders
} from 'lucide-react';
import { InputMode, SingleImageTaskHint } from '../types';

interface QueryBarProps {
  mode: InputMode;
  question: string;
  onQuestionChange: (q: string) => void;
  singleImageHint: SingleImageTaskHint;
  onSingleImageHintChange: (h: SingleImageTaskHint) => void;
  onSubmit: () => void;
  loading: boolean;
  disabled: boolean;
}

export const QueryBar: React.FC<QueryBarProps> = ({
  mode,
  question,
  onQuestionChange,
  singleImageHint,
  onSingleImageHintChange,
  onSubmit,
  loading,
  disabled,
}) => {
  // Preset queries based on current mode for quick evaluation
  const presets: Record<InputMode, string[]> = {
    single: [
      'What are the primary land cover classifications present in this scene?',
      'Count the number of agricultural storage units or structures.',
      'Describe the coastal geographical formations and port activity.',
    ],
    bitemporal: [
      'What urban expansion or deforestation changes occurred between T1 and T2?',
      'Assess the water body area shrinkage or flood inundation between T1 and T2.',
      'Quantify infrastructure development and vegetation loss.',
    ],
    optical_sar: [
      'Identify maritime vessels hidden beneath optical cloud cover using SAR.',
      'Classify agricultural crop types and estimate soil moisture with optical-SAR fusion.',
      'Detect water channels obscured by dense canopy using radar penetration.',
    ],
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !disabled && !loading) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className="card query-bar-card">
      <div className="card-header">
        <div className="card-title-group">
          <h2 className="card-title">2. Query & Task Hint Configuration</h2>
          <span className="card-hint">
            Specify analysis intent. Backend agent will independently select optimal models.
          </span>
        </div>
      </div>

      {/* Task Hint Selector */}
      <div className="task-hint-container">
        <div className="hint-header">
          <span className="hint-title">
            <Sliders size={14} /> UI Task Hint (`task_hint`):
          </span>
          <span className="hint-disclaimer">
            * Backend agent autonomously validates and determines the final execution task.
          </span>
        </div>

        {mode === 'single' ? (
          <div className="hint-pill-selector">
            <button
              type="button"
              className={`hint-pill ${singleImageHint === 'vqa' ? 'active' : ''}`}
              onClick={() => onSingleImageHintChange('vqa')}
            >
              <HelpCircle size={14} />
              <span>VQA (Visual Question Answering)</span>
            </button>
            <button
              type="button"
              className={`hint-pill ${singleImageHint === 'caption' ? 'active' : ''}`}
              onClick={() => onSingleImageHintChange('caption')}
            >
              <FileText size={14} />
              <span>Captioning (Scene Description)</span>
            </button>
          </div>
        ) : (
          <div className="hint-auto-badge">
            <span className="auto-pill">
              {mode === 'bitemporal' && <><Clock size={13} /> Auto-suggested hint: <strong>`change`</strong> (Bi-temporal Difference)</>}
              {mode === 'optical_sar' && <><Sparkles size={13} /> Auto-suggested hint: <strong>`multimodal`</strong> (Optical-SAR Fusion)</>}
            </span>
          </div>
        )}
      </div>

      {/* Question Input Area */}
      <div className="query-input-wrapper">
        <textarea
          className="query-textarea"
          rows={3}
          placeholder={
            mode === 'single'
              ? singleImageHint === 'caption'
                ? 'Optional: Guide captioning focus (e.g., focus on transport infrastructure, agricultural density)...'
                : 'Enter your remote sensing question (e.g., "Identify the runway orientation and apron layout")...'
              : mode === 'bitemporal'
              ? 'Ask about changes between T1 & T2 (e.g., "Quantify flood inundation extent and affected farmland")...'
              : 'Ask a multimodal query (e.g., "Detect ship coordinates through cloud cover using SAR backscatter")...'
          }
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />

        <div className="query-footer">
          {/* Preset Chips */}
          <div className="preset-container">
            <span className="preset-label">Suggestions:</span>
            <div className="preset-chips">
              {presets[mode].map((presetText, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="preset-chip"
                  onClick={() => onQuestionChange(presetText)}
                  disabled={loading}
                >
                  {presetText.length > 50 ? `${presetText.substring(0, 48)}...` : presetText}
                </button>
              ))}
            </div>
          </div>

          {/* Submit Action */}
          <div className="action-row">
            <span className="shortcut-hint">Press Ctrl + Enter to analyze</span>
            <button
              type="button"
              className={`btn-analyze ${loading ? 'loading' : ''}`}
              onClick={onSubmit}
              disabled={disabled || loading}
            >
              {loading ? (
                <>
                  <Loader2 className="spin" size={18} />
                  <span>Processing Satellite Data...</span>
                </>
              ) : (
                <>
                  <Send size={18} />
                  <span>Analyze Satellite Data</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
