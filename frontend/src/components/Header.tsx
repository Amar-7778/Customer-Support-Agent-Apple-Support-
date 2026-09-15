import React from 'react';
import { Server, AlertCircle, RefreshCw } from 'lucide-react';
import type { HealthResponse } from '../types';

interface HeaderProps {
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  onRefreshHealth: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  healthLoading,
  healthError: _healthError,
  onRefreshHealth,
}) => {
  const isHealthy = health?.status === 'healthy';

  return (
    <header className="w-full border-b border-white/[0.08] bg-canvas/80 backdrop-blur-xl sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand & Identity */}
        <div className="flex items-center space-x-3.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-b from-white/15 to-white/5 border border-white/10 flex items-center justify-center shadow-inner">
            {/* Apple Icon */}
            <svg
              className="w-5 h-5 text-white fill-current"
              viewBox="0 0 170 170"
            >
              <path d="M150.37 130.25c-2.45 5.66-5.35 10.87-8.71 15.66-4.58 6.53-8.33 11.05-11.22 13.56-4.48 4.12-9.28 6.23-14.42 6.35-3.69 0-8.14-1.05-13.32-3.18-5.19-2.12-9.97-3.17-14.34-3.17-4.58 0-9.49 1.05-14.75 3.17-5.26 2.13-9.5 3.24-12.74 3.35-4.35.13-9.16-1.9-14.42-6.08-3.69-3.08-7.77-7.97-12.23-14.67-6.02-8.99-10.85-19.34-14.51-31.06-3.65-11.72-5.48-23.01-5.48-33.88 0-14.77 3.73-27.17 11.19-37.21 7.46-10.04 17.06-15.17 28.8-15.4 5.37 0 11.24 1.44 17.62 4.34 6.38 2.9 10.45 4.39 12.22 4.49 1.46 0 5.68-1.55 12.65-4.66 6.98-3.11 12.82-4.55 17.53-4.32 13.48.64 24.31 5.39 32.51 14.26-11.75 7.1-17.5 16.9-17.26 29.41.24 9.94 4.09 18.27 11.54 25 7.46 6.72 16.32 10.4 26.58 11.05-2.28 6.98-5.06 14.12-8.35 21.43zM119.22 31.84c0-7.72 2.76-14.89 8.28-21.52 5.53-6.63 12.44-10.32 20.73-11.07.22 1.1.33 2.14.33 3.12 0 7.56-2.91 14.88-8.73 21.96-5.83 7.07-12.82 10.88-20.97 11.43-.11-1.28-.16-2.58-.16-3.92z" />
            </svg>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-sm tracking-tight text-white">Apple Support Agent</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/10 text-slate-300 border border-white/10 uppercase tracking-wider">
                Stage 5
              </span>
            </div>
            <p className="text-xs text-slate-400 font-normal">
              Autonomous Grounded Dispatch & Verification Console
            </p>
          </div>
        </div>

        {/* Backend & Index Status Indicator */}
        <div className="flex items-center space-x-3 text-xs">
          <div className="hidden sm:flex items-center space-x-2 px-3 py-1.5 rounded-full glass-panel text-slate-300">
            <Server className="w-3.5 h-3.5 text-slate-400" />
            <span>ChromaDB Index:</span>
            <span className="font-mono text-white font-medium">3,000 precedents</span>
          </div>

          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full border border-white/10 bg-canvas-elevated">
            {healthLoading ? (
              <>
                <RefreshCw className="w-3 h-3 text-slate-400 animate-spin" />
                <span className="text-slate-400">Pinging backend...</span>
              </>
            ) : isHealthy ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span className="text-emerald-400 font-medium font-mono text-[11px]">System Online</span>
                <span className="text-slate-500 font-mono hidden md:inline">| {health?.model || 'qwen/qwen3.8-27b'}</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
                <span className="text-rose-400 font-medium">Backend Offline</span>
                <button
                  onClick={onRefreshHealth}
                  className="hover:text-white underline text-[11px] ml-1"
                  title="Retry connection"
                >
                  Retry
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
