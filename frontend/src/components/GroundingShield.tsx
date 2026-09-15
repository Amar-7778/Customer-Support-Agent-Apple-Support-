import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, Scale } from 'lucide-react';
import type { GroundingVerification } from '../types';

interface GroundingShieldProps {
  grounding: GroundingVerification;
  agreementScore: number;
}

export const GroundingShield: React.FC<GroundingShieldProps> = ({
  grounding,
  agreementScore,
}) => {
  const isGrounded = grounding.grounded;
  const claims = grounding.unsupported_claims || [];
  const agreementPct = Math.round(agreementScore * 100);
  const meetsThreshold = agreementScore >= 0.70;

  return (
    <div className="glass-panel rounded-2xl p-6 border border-white/[0.08] shadow-subtle-card space-y-5">
      {/* Top Title */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Scale className="w-4 h-4 text-apple-blue" />
          <h3 className="text-sm font-semibold tracking-tight text-white">
            Grounding & Factual Verification
          </h3>
        </div>
        <span className="text-[10px] font-mono uppercase bg-white/5 text-slate-400 px-2 py-0.5 rounded border border-white/5">
          LLM Self-Critique
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Verification Status Card */}
        <div
          className={`p-4 rounded-xl border flex flex-col justify-between ${
            isGrounded
              ? 'bg-emerald-500/[0.06] border-emerald-500/20'
              : 'bg-rose-500/[0.06] border-rose-500/20'
          }`}
        >
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              {isGrounded ? (
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
              ) : (
                <ShieldAlert className="w-4 h-4 text-rose-400" />
              )}
              <span
                className={`text-xs font-semibold uppercase tracking-wider font-mono ${
                  isGrounded ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {isGrounded ? 'Factual Grounding Verified' : 'Factual Hallucination Flagged'}
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              {isGrounded
                ? 'All diagnostic steps, DM invitations, and advice in this draft are strictly substantiated by the retrieved historical precedents.'
                : 'The draft contains one or more claims, unauthorized promises, or canonical URLs not found in historical precedents.'}
            </p>
          </div>

          {/* Unsupported Claims List */}
          {claims.length > 0 && (
            <div className="mt-3 pt-3 border-t border-rose-500/20 space-y-1.5">
              <span className="text-[11px] font-mono text-rose-300 font-semibold uppercase">
                Unsupported Claims Detected:
              </span>
              <ul className="space-y-1">
                {claims.map((claim, idx) => (
                  <li
                    key={idx}
                    className="text-xs text-rose-200 bg-rose-950/40 border border-rose-500/30 px-2.5 py-1 rounded flex items-center gap-1.5"
                  >
                    <AlertTriangle className="w-3 h-3 text-rose-400 flex-shrink-0" />
                    <span>"{claim}"</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Precedent Agreement Score Card */}
        <div className="p-4 rounded-xl border border-white/10 bg-canvas-elevated/50 flex flex-col justify-between space-y-3">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 font-mono">
                Precedent Agreement Score
              </span>
              <span
                className={`text-sm font-bold font-mono ${
                  meetsThreshold ? 'text-emerald-400' : 'text-amber-400'
                }`}
              >
                {agreementScore.toFixed(2)} ({agreementPct}%)
              </span>
            </div>

            {/* Agreement Progress Bar */}
            <div className="mt-2.5 relative w-full h-2 rounded-full bg-white/10 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  meetsThreshold ? 'bg-emerald-500' : 'bg-amber-500'
                }`}
                style={{ width: `${Math.min(100, Math.max(5, agreementPct))}%` }}
              />
            </div>

            {/* Threshold Marker Indicator */}
            <div className="mt-1 flex justify-between text-[10px] font-mono text-slate-500">
              <span>0% (Disagreement)</span>
              <span className="text-slate-400">Threshold: 0.70</span>
              <span>100% (Consensus)</span>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 leading-relaxed">
            {meetsThreshold
              ? 'Retrieved precedents agree on the resolution pathway, satisfying the autonomous consensus threshold.'
              : 'Retrieved precedents exhibit conflicting resolution paths, requiring specialist human escalation.'}
          </p>
        </div>
      </div>
    </div>
  );
};
