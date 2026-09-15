import type { AgentResponse, HealthResponse } from '../types';

const API_BASE = '/api';

export async function checkBackendHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!res.ok) {
      throw new Error(`Health check failed with HTTP ${res.status}: ${res.statusText}`);
    }

    return await res.json();
  } catch (err: any) {
    // If proxy failed, try direct connection to localhost:8000 as secondary check
    try {
      const direct = await fetch('http://127.0.0.1:8000/health');
      if (direct.ok) return await direct.json();
    } catch {}
    throw new Error(err.message || 'Unable to connect to FastAPI backend at http://127.0.0.1:8000');
  }
}

export async function submitCustomerMessage(message: string): Promise<AgentResponse> {
  const clean = message.trim();
  if (!clean) {
    throw new Error('Message cannot be empty.');
  }
  if (clean.length > 1000) {
    throw new Error('Message exceeds maximum length of 1,000 characters.');
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/handle_message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ message: clean }),
    });
  } catch (netErr: any) {
    // Attempt direct host if dev proxy fails
    try {
      res = await fetch('http://127.0.0.1:8000/handle_message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ message: clean }),
      });
    } catch {
      throw new Error(`Connection refused: Could not connect to FastAPI server at http://127.0.0.1:8000. Ensure 'uvicorn src.agent.api:app --port 8000' is running.`);
    }
  }

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status} ${res.statusText}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' 
          ? errJson.detail 
          : JSON.stringify(errJson.detail);
      }
    } catch {}
    throw new Error(`Agent pipeline error: ${errorDetail}`);
  }

  return await res.json();
}
