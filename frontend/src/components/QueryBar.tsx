import React, { useState } from 'react';

interface QueryBarProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
}

export const QueryBar: React.FC<QueryBarProps> = ({ onSubmit, isLoading }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 w-full">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a question about the satellite imagery..."
        disabled={isLoading}
        className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500"
      />
      <button
        type="submit"
        disabled={isLoading || !query.trim()}
        className="px-5 py-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-medium rounded transition"
      >
        {isLoading ? 'Analyzing...' : 'Query'}
      </button>
    </form>
  );
};
