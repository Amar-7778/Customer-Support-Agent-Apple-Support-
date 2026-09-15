import React, { useState } from 'react';
import { ChevronDown, ChevronUp, History, CheckCircle2 } from 'lucide-react';
import type { Precedent } from '../types';

interface PrecedentsDrawerProps {
  precedents: Precedent[];
}

export const PrecedentsDrawer: React.FC<PrecedentsDrawerProps> = ({ precedents }) => {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0); // First expanded by default

  const toggleExpand = (idx: number) => {
    setExpandedIndex(expandedIndex === idx ? null : idx);
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-white/[0.08] shadow-subtle-card space-y-4">
      {/* Drawer Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <History className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-semibold tracking-tight text-white">
            Grounding Evidence: Historical Precedents
          </h3>
          <span className="text-[10px] font-mono text-slate-400 bg-white/5 px-2 py-0.5 rounded border border-white/5">
            {precedents.length} Retrieved from ChromaDB
          </span>
        </div>
        <span className="text-xs text-slate-400 font-mono hidden sm:inline">
          Stratified 3,000 Index
        </span>
      </div>

      <p className="text-xs text-slate-400 leading-relaxed">
        The agent's reply and escalation decisions are grounded directly in these nearest-neighbor verified historical resolutions from Apple Support:
      </p>

      {/* Precedent Cards List */}
      <div className="space-y-3">
        {precedents.map((prec, idx) => {
          const isExpanded = expandedIndex === idx;
          const simScore = (prec.similarity_score * 100).toFixed(1);

          return (
            <div
              key={prec.thread_id || idx}
              className={`rounded-xl border transition-all duration-200 overflow-hidden ${
                isExpanded
                  ? 'bg-canvas-elevated/90 border-white/15 shadow-sm'
                  : 'bg-canvas-elevated/40 border-white/5 hover:border-white/10'
              }`}
            >
              {/* Card Summary Bar (Clickable) */}
              <button
                type="button"
                onClick={() => toggleExpand(idx)}
                className="w-full text-left px-4 py-3.5 flex items-center justify-between gap-3 focus:outline-none"
              >
                <div className="flex items-center space-x-3 min-w-0">
                  <span className="w-6 h-6 rounded-full bg-white/10 text-white font-mono text-xs flex items-center justify-center font-semibold">
                    {idx + 1}
                  </span>

                  <div className="truncate">
                    <span className="text-xs font-semibold text-white mr-2">
                      Precedent {prec.thread_id}
                    </span>
                    <span className="text-xs text-slate-400 truncate hidden md:inline">
                      "{prec.customer_message.slice(0, 60)}..."
                    </span>
                  </div>
                </div>

                <div className="flex items-center space-x-3 flex-shrink-0">
                  {/* Action Taken Pill */}
                  <span className="text-[10px] font-mono bg-white/5 text-slate-300 border border-white/10 px-2 py-0.5 rounded">
                    {prec.action_taken || 'resolution'}
                  </span>

                  {/* Similarity Pill */}
                  <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">
                    {simScore}% match
                  </span>

                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-slate-400" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-slate-400" />
                  )}
                </div>
              </button>

              {/* Card Expanded Detail Body */}
              {isExpanded && (
                <div className="px-4 pb-4 pt-1 border-t border-white/[0.06] space-y-3.5 animate-fade-in text-xs">
                  {/* Customer Context */}
                  <div className="space-y-1">
                    <span className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
                      Historical Customer Inquiry:
                    </span>
                    <div className="p-2.5 rounded-lg bg-black/40 border border-white/5 text-slate-200 leading-relaxed font-normal">
                      "{prec.customer_message}"
                    </div>
                  </div>

                  {/* Metadata Row: Action Taken & Outcome */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                    <div className="p-2 rounded-lg bg-white/[0.03] border border-white/5">
                      <span className="text-[10px] font-mono uppercase text-slate-400 block">
                        Action Taken:
                      </span>
                      <span className="text-xs font-medium text-sky-300 font-mono">
                        {prec.action_taken}
                      </span>
                    </div>

                    <div className="p-2 rounded-lg bg-white/[0.03] border border-white/5">
                      <span className="text-[10px] font-mono uppercase text-slate-400 block">
                        Outcome:
                      </span>
                      <span className="text-xs font-medium text-emerald-300 font-mono">
                        {prec.outcome}
                      </span>
                    </div>
                  </div>

                  {/* Original Brand Reply */}
                  <div className="space-y-1">
                    <span className="font-mono text-[10px] uppercase tracking-wider text-slate-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                      Historical @AppleSupport Brand Reply (Grounding Truth):
                    </span>
                    <div className="p-3 rounded-lg bg-black/40 border border-emerald-500/20 text-slate-200 italic font-sans leading-relaxed">
                      "{prec.brand_reply_text}"
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
