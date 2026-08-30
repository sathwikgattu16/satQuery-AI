import React, { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import { Header } from './components/Header';
import { InputSelector } from './components/InputSelector';
import { QueryBar } from './components/QueryBar';
import { ResultsDisplay } from './components/ResultsDisplay';
import { EvidenceViewer } from './components/EvidenceViewer';
import { ExecutionTrace } from './components/ExecutionTrace';
import { submitAnalysis, USE_MOCK_DEFAULT } from './services/api';
import {
  InputMode,
  SingleImageTaskHint,
  FileInputs,
  ImagePreviewUrls,
  QueryResponse,
  QueryRequest,
} from './types';

// Helper to create synthetic satellite SVG files & preview URLs for instant demo testing
function createSampleSatelliteFile(label: string, color: string, pattern: 'optical' | 'sar' | 't2', filename: string): { file: File; url: string } {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="400" height="250" viewBox="0 0 400 250">
      <defs>
        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="${pattern === 'sar' ? '#180a33' : '#071726'}" />
          <stop offset="100%" stop-color="${pattern === 'sar' ? '#2e125c' : '#0e2d42'}" />
        </linearGradient>
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="${color}" stroke-width="0.5" opacity="0.3" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#grad)" />
      <rect width="100%" height="100%" fill="url(#grid)" />
      
      <!-- Simulated terrain / satellite contours -->
      ${pattern === 'optical' ? `
        <path d="M 0,140 Q 90,80 200,160 T 400,100 L 400,250 L 0,250 Z" fill="#0d4a45" opacity="0.6" />
        <path d="M 120,40 Q 240,110 380,50 L 400,0 L 0,0 L 0,60 Z" fill="#1b4332" opacity="0.5" />
        <circle cx="280" cy="120" r="24" fill="#0284c7" opacity="0.7" />
      ` : pattern === 't2' ? `
        <path d="M 0,140 Q 90,80 200,160 T 400,100 L 400,250 L 0,250 Z" fill="#0d4a45" opacity="0.6" />
        <!-- Expanded urban & flooded zone -->
        <circle cx="280" cy="120" r="42" fill="#0284c7" opacity="0.85" />
        <rect x="80" y="110" width="70" height="50" fill="#dc2626" opacity="0.6" />
      ` : `
        <!-- SAR speckle / polarimetric backscatter simulation -->
        <circle cx="150" cy="90" r="30" fill="none" stroke="#a855f7" stroke-width="3" opacity="0.8" />
        <circle cx="150" cy="90" r="15" fill="none" stroke="#c084fc" stroke-width="2" />
        <line x1="50" y1="180" x2="350" y2="180" stroke="#a855f7" stroke-width="2" stroke-dasharray="4" />
        <rect x="240" y="70" width="60" height="30" fill="#a855f7" opacity="0.4" />
      `}

      <!-- Overlay metadata -->
      <rect x="12" y="12" width="170" height="24" rx="4" fill="#000000" opacity="0.6" />
      <text x="20" y="28" fill="${color}" font-family="monospace" font-size="11" font-weight="bold">${label}</text>
      <text x="320" y="235" fill="#ffffff" font-family="sans-serif" font-size="10" opacity="0.7">ISRO SENSING</text>
    </svg>
  `;
  const blob = new Blob([svg], { type: 'image/svg+xml' });
  const file = new File([blob], filename, { type: 'image/svg+xml' });
  const url = URL.createObjectURL(file);
  return { file, url };
}

export const App: React.FC = () => {
  const [mode, setMode] = useState<InputMode>('single');
  const [files, setFiles] = useState<FileInputs>({});
  const [previews, setPreviews] = useState<ImagePreviewUrls>({});
  const [question, setQuestion] = useState<string>('What are the primary land cover classifications present in this scene?');
  const [singleImageHint, setSingleImageHint] = useState<SingleImageTaskHint>('vqa');
  const [useMock, setUseMock] = useState<boolean>(USE_MOCK_DEFAULT);
  const [loading, setLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Handle Mode Change
  const handleModeChange = (newMode: InputMode) => {
    setMode(newMode);
    setError(null);

    // Adjust default sample question based on mode
    if (newMode === 'single') {
      setQuestion(
        singleImageHint === 'caption'
          ? 'Generate a detailed caption describing all geographical features and structures.'
          : 'What are the primary land cover classifications present in this scene?'
      );
    } else if (newMode === 'bitemporal') {
      setQuestion('What urban expansion or deforestation changes occurred between T1 and T2?');
    } else if (newMode === 'optical_sar') {
      setQuestion('Identify maritime vessels hidden beneath optical cloud cover using SAR.');
    }
  };

  // Handle File Change & Preview Generation
  const handleFileChange = (slot: keyof FileInputs, file: File | null) => {
    setFiles((prev) => ({ ...prev, [slot]: file }));
    setError(null);

    if (file) {
      const url = URL.createObjectURL(file);
      setPreviews((prev) => ({ ...prev, [slot]: url }));
    } else {
      setPreviews((prev) => ({ ...prev, [slot]: null }));
    }
  };

  // Load Demonstration Samples for Fast Judging
  const handleLoadSample = () => {
    setError(null);
    if (mode === 'single') {
      const sample = createSampleSatelliteFile('OPTICAL L2A [RGB]', '#00d4ff', 'optical', 'isro_cartosat_scene.svg');
      setFiles({ image: sample.file });
      setPreviews({ image: sample.url });
    } else if (mode === 'bitemporal') {
      const sample1 = createSampleSatelliteFile('T1 BASELINE [2022]', '#00d4ff', 'optical', 't1_baseline_2022.svg');
      const sample2 = createSampleSatelliteFile('T2 TARGET [2024]', '#ef4444', 't2', 't2_observation_2024.svg');
      setFiles({ image: sample1.file, image_t2: sample2.file });
      setPreviews({ image: sample1.url, image_t2: sample2.url });
    } else if (mode === 'optical_sar') {
      const sample1 = createSampleSatelliteFile('OPTICAL SENTINEL-2', '#00d4ff', 'optical', 'optical_cloudy_rgb.svg');
      const sample2 = createSampleSatelliteFile('RISAT-1A SAR (VV/VH)', '#a855f7', 'sar', 'sar_polarimetric_radar.svg');
      setFiles({ image: sample1.file, sar: sample2.file });
      setPreviews({ image: sample1.url, sar: sample2.url });
    }
  };

  // Handle Analysis Submission
  const handleSubmit = async () => {
    setError(null);

    // Validation
    if (!files.image) {
      setError('Please provide the primary satellite image before running analysis.');
      return;
    }
    if (mode === 'bitemporal' && !files.image_t2) {
      setError('Bi-temporal mode requires both T1 Baseline and T2 Target observation images.');
      return;
    }
    if (mode === 'optical_sar' && !files.sar) {
      setError('Optical + SAR mode requires both Optical imagery and a SAR radar file.');
      return;
    }

    // Determine task hint for the request
    let taskHint: string = singleImageHint;
    if (mode === 'bitemporal') {
      taskHint = 'change';
    } else if (mode === 'optical_sar') {
      taskHint = 'multimodal';
    }

    // Construct locked QueryRequest structure
    const requestPayload: QueryRequest = {
      task_hint: taskHint,
      question: question.trim() || undefined,
      image: files.image,
      image_t2: mode === 'bitemporal' ? files.image_t2 : undefined,
      sar: mode === 'optical_sar' ? files.sar : undefined,
    };

    setLoading(true);
    setResponse(null);

    try {
      const result = await submitAnalysis(requestPayload, { useMock });
      setResponse(result);
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : 'An unexpected error occurred';
      setError(errMsg);
    } finally {
      setLoading(false);
    }
  };

  const isSubmitDisabled = !files.image || (mode === 'bitemporal' && !files.image_t2) || (mode === 'optical_sar' && !files.sar);

  return (
    <div className="app-container">
      {/* App Header with branding and mock/live toggle */}
      <Header useMock={useMock} onToggleMock={setUseMock} />

      <main className="main-content">
        {/* Error notification banner */}
        {error && (
          <div className="error-banner">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* 2-Column Responsive Workspace Grid */}
        <div className="workspace-grid">
          {/* Left Column: Input Configurations & Query Submission */}
          <div className="column-left">
            <InputSelector
              mode={mode}
              onModeChange={handleModeChange}
              files={files}
              previews={previews}
              onFileChange={handleFileChange}
              onLoadSample={handleLoadSample}
            />

            <QueryBar
              mode={mode}
              question={question}
              onQuestionChange={setQuestion}
              singleImageHint={singleImageHint}
              onSingleImageHintChange={(hint) => {
                setSingleImageHint(hint);
                if (hint === 'caption') {
                  setQuestion('Generate a detailed caption describing all geographical features and structures.');
                } else {
                  setQuestion('What are the primary land cover classifications present in this scene?');
                }
              }}
              onSubmit={handleSubmit}
              loading={loading}
              disabled={isSubmitDisabled}
            />
          </div>

          {/* Right Column: Authoritative AI Results, Visual Evidence, and Execution Trace */}
          <div className="column-right">
            <ResultsDisplay
              response={response}
              loading={loading}
            />

            <EvidenceViewer
              visualization={response?.visualization}
              loading={loading}
            />

            <ExecutionTrace
              summary={response?.execution_summary || null}
              loading={loading}
            />
          </div>
        </div>
      </main>
    </div>
  );
};
