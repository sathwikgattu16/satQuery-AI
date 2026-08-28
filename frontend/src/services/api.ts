/**
 * frontend/src/services/api.ts
 * API service for communicating with the SatQuery FastAPI backend.
 * Owner: Member 2
 */

import { QueryRequest, QueryResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export async function submitQuery(request: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`API query failed with status: ${response.status}`);
  }

  return response.json();
}
