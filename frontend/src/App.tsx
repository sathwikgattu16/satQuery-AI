import React, { useState } from 'react';
import { Header } from './components/Header';
import { InputSelector } from './components/InputSelector';
import { QueryBar } from './components/QueryBar';
import { ResultsDisplay } from './components/ResultsDisplay';
import { EvidenceViewer } from './components/EvidenceViewer';
import { ExecutionTrace } from './components/ExecutionTrace';
import { InputType, QueryResponse } from './types';
import { submitQuery } from './services/api';

export const App: React.FC = () => {
  const [inputType, setInputType] = useState<InputType>('single');
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleQuery = async (query: string) => {
    setIsLoading(true);
    try {
      // Minimal placeholder mock query payload
      const res = await submitQuery({
        query,
        input_type: inputType,
        image_primary: 'sample_primary.tif',
        image_secondary: inputType !== 'single' ? 'sample_secondary.tif' : undefined,
      });
      setResponse(res);
    } catch (err) {
      console.error('Query error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      <Header />
      <main className="flex-1 max-w-5xl w-full mx-auto p-4 flex flex-col gap-4">
        <InputSelector inputType={inputType} onSelectType={setInputType} />
        <QueryBar onSubmit={handleQuery} isLoading={isLoading} />
        <ResultsDisplay response={response} />
        {response?.evidence && <EvidenceViewer evidence={response.evidence} />}
        {response?.execution_summary && <ExecutionTrace summary={response.execution_summary} />}
      </main>
    </div>
  );
};

export default App;
