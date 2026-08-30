import React from 'react';
import { Satellite, Radio, ShieldCheck, Database } from 'lucide-react';

interface HeaderProps {
  useMock: boolean;
  onToggleMock: (val: boolean) => void;
}

export const Header: React.FC<HeaderProps> = ({ useMock, onToggleMock }) => {
  return (
    <header className="app-header">
      <div className="header-container">
        <div className="brand-section">
          <div className="logo-badge">
            <Satellite className="logo-icon text-cyan" size={28} />
            <div className="pulse-ring"></div>
          </div>
          <div className="brand-text">
            <div className="title-row">
              <h1 className="brand-title">
                SATQUERY <span className="brand-highlight">AI</span>
              </h1>
              <span className="badge isro-badge">
                <ShieldCheck size={12} className="badge-icon" /> ISRO HACKATHON
              </span>
            </div>
            <p className="brand-subtitle">
              Intelligent Remote Sensing & Geospatial Earth Observation Assistant
            </p>
          </div>
        </div>

        <div className="header-controls">
          <div className="mode-toggle-card">
            <div className="toggle-label-group">
              <span className="toggle-label">DataSource:</span>
              <span className={`status-pill ${useMock ? 'pill-mock' : 'pill-live'}`}>
                {useMock ? (
                  <>
                    <Database size={11} /> Standalone Mock Mode
                  </>
                ) : (
                  <>
                    <Radio size={11} className="spin-slow" /> Live Backend (Port 8000)
                  </>
                )}
              </span>
            </div>
            <label className="switch-toggle" title="Toggle between standalone mock data and live Member 1 backend">
              <input
                type="checkbox"
                checked={useMock}
                onChange={(e) => onToggleMock(e.target.checked)}
              />
              <span className="slider round"></span>
            </label>
          </div>
        </div>
      </div>

      <div className="header-subbar">
        <div className="subbar-tags">
          <span className="tech-tag">🛰️ Single-Scene VQA & Captioning</span>
          <span className="tech-tag">⏱️ Bi-Temporal Change Detection</span>
          <span className="tech-tag">📡 Optical + SAR Polarimetric Fusion</span>
          <span className="tech-tag highlight">⚡ Agent-Authoritative Task Routing</span>
        </div>
      </div>
    </header>
  );
};
