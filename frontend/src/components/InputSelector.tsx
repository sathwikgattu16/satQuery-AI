import React, { useRef } from 'react';
import {
  Layers,
  Clock,
  Radio,
  Upload,
  X,
  FileImage,
  Sparkles,
  Info
} from 'lucide-react';
import { InputMode, FileInputs, ImagePreviewUrls } from '../types';

interface InputSelectorProps {
  mode: InputMode;
  onModeChange: (mode: InputMode) => void;
  files: FileInputs;
  previews: ImagePreviewUrls;
  onFileChange: (slot: keyof FileInputs, file: File | null) => void;
  onLoadSample: () => void;
}

export const InputSelector: React.FC<InputSelectorProps> = ({
  mode,
  onModeChange,
  files,
  previews,
  onFileChange,
  onLoadSample,
}) => {
  const fileInputRef1 = useRef<HTMLInputElement>(null);
  const fileInputRef2 = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent, slot: keyof FileInputs) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileChange(slot, e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="card input-selector-card">
      <div className="card-header">
        <div className="card-title-group">
          <h2 className="card-title">1. Satellite Observation Mode</h2>
          <span className="card-hint">Select sensing workflow and load target imagery</span>
        </div>
        <button
          type="button"
          onClick={onLoadSample}
          className="btn-sample"
          title="Load sample satellite images for instant demonstration"
        >
          <Sparkles size={13} /> Load Demo Images
        </button>
      </div>

      {/* Mode Selector Tabs */}
      <div className="mode-tabs">
        <button
          type="button"
          className={`mode-tab ${mode === 'single' ? 'active' : ''}`}
          onClick={() => onModeChange('single')}
        >
          <Layers size={16} className="tab-icon" />
          <div className="tab-text">
            <span className="tab-title">Single Image</span>
            <span className="tab-sub">VQA & Captioning</span>
          </div>
        </button>

        <button
          type="button"
          className={`mode-tab ${mode === 'bitemporal' ? 'active' : ''}`}
          onClick={() => onModeChange('bitemporal')}
        >
          <Clock size={16} className="tab-icon" />
          <div className="tab-text">
            <span className="tab-title">Bi-temporal Change</span>
            <span className="tab-sub">T1 vs T2 Comparison</span>
          </div>
        </button>

        <button
          type="button"
          className={`mode-tab ${mode === 'optical_sar' ? 'active' : ''}`}
          onClick={() => onModeChange('optical_sar')}
        >
          <Radio size={16} className="tab-icon" />
          <div className="tab-text">
            <span className="tab-title">Optical + SAR</span>
            <span className="tab-sub">Radar & Multispectral</span>
          </div>
        </button>
      </div>

      {/* Mode Explanation Banner */}
      <div className="mode-guidance-banner">
        <Info size={14} className="info-icon" />
        <span className="guidance-text">
          {mode === 'single' && (
            <>
              <strong>Single-Scene Mode:</strong> Provide 1 primary optical/multispectral tile (6-band TIFF for Real Mode, or PNG/SVG for Demo Mode).
            </>
          )}
          {mode === 'bitemporal' && (
            <>
              <strong>Bi-temporal Change:</strong> Provide <strong>T1 (Baseline)</strong> and <strong>T2 (Observation)</strong>. Real Mode requires stacked 6-band Sentinel-2 TIFFs (B02-B07). For quick UI testing, click <em>Load Demo Images</em>.
            </>
          )}
          {mode === 'optical_sar' && (
            <>
              <strong>Multimodal Fusion:</strong> Provide <strong>Optical (6-band TIFF)</strong> and <strong>SAR (2-band VV/VH TIFF)</strong> for all-weather penetrating analytics.
            </>
          )}
        </span>
      </div>

      {/* Dynamic Image Upload Slots */}
      <div className={`upload-grid grid-${mode === 'single' ? '1' : '2'}`}>
        {/* Slot 1: Primary Image (image) */}
        <div className="upload-slot">
          <div className="slot-header">
            <span className="slot-label">
              {mode === 'single' && 'Primary Satellite Image (`image`)'}
              {mode === 'bitemporal' && 'Time 1 Baseline (`image`)'}
              {mode === 'optical_sar' && 'Optical / Multispectral (`image`)'}
            </span>
            {files.image && (
              <button
                type="button"
                className="btn-clear"
                onClick={() => onFileChange('image', null)}
                title="Remove image"
              >
                <X size={13} /> Remove
              </button>
            )}
          </div>

          {previews.image ? (
            <div className="preview-container">
              <img src={previews.image} alt="Primary preview" className="preview-image" />
              <div className="preview-overlay">
                <span className="file-name">{files.image?.name || 'satellite_primary.png'}</span>
                <span className="file-meta">
                  {files.image ? `${(files.image.size / 1024).toFixed(1)} KB` : 'Demo Asset'}
                </span>
              </div>
            </div>
          ) : (
            <div
              className="dropzone"
              onDragOver={handleDragOver}
              onDrop={(e) => handleDrop(e, 'image')}
              onClick={() => fileInputRef1.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef1}
                accept="image/*"
                className="hidden-file-input"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    onFileChange('image', e.target.files[0]);
                  }
                }}
              />
              <div className="dropzone-content">
                <div className="upload-icon-circle">
                  <Upload size={20} />
                </div>
                <p className="dropzone-main-text">Click or drag & drop primary image</p>
                <p className="dropzone-sub-text">GeoTIFF, PNG, JPEG (Sent as `image` field)</p>
              </div>
            </div>
          )}
        </div>

        {/* Slot 2: Secondary Image (image_t2 for bi-temporal, sar for optical_sar) */}
        {mode === 'bitemporal' && (
          <div className="upload-slot">
            <div className="slot-header">
              <span className="slot-label">Time 2 Observation (`image_t2`)</span>
              {files.image_t2 && (
                <button
                  type="button"
                  className="btn-clear"
                  onClick={() => onFileChange('image_t2', null)}
                  title="Remove image"
                >
                  <X size={13} /> Remove
                </button>
              )}
            </div>

            {previews.image_t2 ? (
              <div className="preview-container">
                <img src={previews.image_t2} alt="T2 preview" className="preview-image" />
                <div className="preview-overlay">
                  <span className="file-name">{files.image_t2?.name || 'satellite_t2.png'}</span>
                  <span className="file-meta">
                    {files.image_t2 ? `${(files.image_t2.size / 1024).toFixed(1)} KB` : 'Demo Asset'}
                  </span>
                </div>
              </div>
            ) : (
              <div
                className="dropzone"
                onDragOver={handleDragOver}
                onDrop={(e) => handleDrop(e, 'image_t2')}
                onClick={() => fileInputRef2.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef2}
                  accept="image/*"
                  className="hidden-file-input"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      onFileChange('image_t2', e.target.files[0]);
                    }
                  }}
                />
                <div className="dropzone-content">
                  <div className="upload-icon-circle">
                    <FileImage size={20} />
                  </div>
                  <p className="dropzone-main-text">Click or drag & drop T2 target image</p>
                  <p className="dropzone-sub-text">Post-event image (Sent as `image_t2` field)</p>
                </div>
              </div>
            )}
          </div>
        )}

        {mode === 'optical_sar' && (
          <div className="upload-slot">
            <div className="slot-header">
              <span className="slot-label">SAR Polarimetric Image (`sar`)</span>
              {files.sar && (
                <button
                  type="button"
                  className="btn-clear"
                  onClick={() => onFileChange('sar', null)}
                  title="Remove image"
                >
                  <X size={13} /> Remove
                </button>
              )}
            </div>

            {previews.sar ? (
              <div className="preview-container">
                <img src={previews.sar} alt="SAR preview" className="preview-image" />
                <div className="preview-overlay">
                  <span className="file-name">{files.sar?.name || 'satellite_sar.png'}</span>
                  <span className="file-meta">
                    {files.sar ? `${(files.sar.size / 1024).toFixed(1)} KB` : 'Demo Asset'}
                  </span>
                </div>
              </div>
            ) : (
              <div
                className="dropzone"
                onDragOver={handleDragOver}
                onDrop={(e) => handleDrop(e, 'sar')}
                onClick={() => fileInputRef2.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef2}
                  accept="image/*"
                  className="hidden-file-input"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      onFileChange('sar', e.target.files[0]);
                    }
                  }}
                />
                <div className="dropzone-content">
                  <div className="upload-icon-circle sar-circle">
                    <Radio size={20} />
                  </div>
                  <p className="dropzone-main-text">Click or drag & drop SAR radar file</p>
                  <p className="dropzone-sub-text">C/L-Band VV/VH backscatter (Sent as `sar` field)</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
