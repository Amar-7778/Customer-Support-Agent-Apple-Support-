import React from 'react';
import { AlertOctagon, RefreshCw } from 'lucide-react';

interface ErrorAlertProps {
  message: string;
  onRetry: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({ message, onRetry }) => {
  return (
    <div className="rounded-2xl p-5 bg-rose-950/40 border border-rose-500/30 text-rose-200 space-y-3 animate-fade-in shadow-lg">
      <div className="flex items-start space-x-3">
        <AlertOctagon className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-semibold text-rose-100 tracking-tight">
            Backend Pipeline Execution Error
          </h4>
          <p className="text-xs text-rose-300 mt-1 leading-relaxed font-mono break-words">
            {message}
          </p>
          <div className="mt-2 text-[11px] text-rose-300/80">
            Ensure the FastAPI server is running with <code className="bg-black/30 px-1 py-0.5 rounded text-rose-100">uvicorn src.agent.api:app --port 8000</code> and Groq API keys are active in <code className="bg-black/30 px-1 py-0.5 rounded text-rose-100">.env</code>.
          </div>
        </div>
      </div>

      <div className="flex justify-end pt-2 border-t border-rose-500/20">
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/30 text-xs font-medium transition"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Execution</span>
        </button>
      </div>
    </div>
  );
};
