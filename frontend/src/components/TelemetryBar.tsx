import React from 'react';
import { Cpu, Clock, Zap } from 'lucide-react';
import type { AgentMetrics } from '../types';

interface TelemetryBarProps {
  metrics: AgentMetrics;
}

export const TelemetryBar: React.FC<TelemetryBarProps> = ({ metrics }) => {
  return (
    <div className="glass-panel rounded-2xl p-4 border border-white/[0.08] shadow-sm flex flex-wrap items-center justify-between gap-4 text-xs">
      {/* Model & Latency */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center space-x-2 text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-apple-blue" />
          <span className="text-slate-400">Model:</span>
          <span className="font-mono text-white font-medium">{metrics.model || 'qwen/qwen3.8-27b'}</span>
        </div>

        <div className="flex items-center space-x-2 text-slate-300">
          <Clock className="w-3.5 h-3.5 text-indigo-400" />
          <span className="text-slate-400">Total Latency:</span>
          <span className="font-mono text-white font-semibold">
            {metrics.total_latency_seconds.toFixed(2)}s
          </span>
          <span className="text-slate-500 font-mono text-[11px] hidden sm:inline">
            (cls: {metrics.classification_latency_seconds.toFixed(2)}s | dft: {metrics.drafting_latency_seconds.toFixed(2)}s | ver: {metrics.verification_latency_seconds.toFixed(2)}s)
          </span>
        </div>
      </div>

      {/* Token Metrics */}
      <div className="flex items-center space-x-3 text-slate-400 font-mono text-[11px]">
        <div className="flex items-center space-x-1.5 bg-white/5 px-2.5 py-1 rounded-md border border-white/5">
          <Zap className="w-3 h-3 text-amber-400" />
          <span>Tokens:</span>
          <span className="text-white font-medium">{metrics.total_tokens.toLocaleString()}</span>
          <span className="text-slate-500">
            ({metrics.total_prompt_tokens.toLocaleString()} prompt / {metrics.total_completion_tokens.toLocaleString()} comp)
          </span>
        </div>
      </div>
    </div>
  );
};
