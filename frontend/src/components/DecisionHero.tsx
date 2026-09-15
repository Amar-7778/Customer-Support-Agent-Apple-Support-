import React from 'react';
import { ShieldCheck, ShieldAlert, Check, Lock } from 'lucide-react';
import type { AgentResponse } from '../types';

interface DecisionHeroProps {
  response: AgentResponse;
}

export const DecisionHero: React.FC<DecisionHeroProps> = ({ response }) => {
  const isAuto = response.decision === 'auto_handle';
  const confPercent = Math.round(response.intent_confidence * 100);
  const agreementPercent = Math.round(response.precedent_agreement_score * 100);
  const isGrounded = response.grounding_verification.grounded;

  return (
    <div
      id="decision-hero-banner"
      className={`rounded-2xl p-6 sm:p-7 border transition-all duration-300 relative overflow-hidden animate-slide-up ${
        isAuto
          ? 'bg-decision-auto-bg/80 border-decision-auto-border shadow-hero-auto'
          : 'bg-decision-escalate-bg/80 border-decision-escalate-border shadow-hero-escalate'
      }`}
    >
      {/* Background Ambient Aura */}
      <div
        className={`absolute -right-16 -top-16 w-64 h-64 rounded-full blur-3xl pointer-events-none opacity-20 ${
          isAuto ? 'bg-emerald-500' : 'bg-rose-500'
        }`}
      />

      <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
        {/* Main Verdict & Description */}
        <div className="space-y-2.5 max-w-2xl">
          {/* Top Pill / Badge */}
          <div className="flex items-center space-x-2.5">
            <span
              className={`inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold tracking-wide uppercase font-mono ${
                isAuto
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                  : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
              }`}
            >
              {isAuto ? (
                <>
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Autonomous Dispatch Authorized</span>
                </>
              ) : (
                <>
                  <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
                  <span>Specialist Escalation Required</span>
                </>
              )}
            </span>

            <span className="text-xs font-mono text-slate-400">
              Intent: <strong className="text-slate-200 font-semibold">{response.predicted_intent}</strong>
            </span>
          </div>

          {/* Large Headline */}
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            {isAuto ? (
              <span>Safe to Dispatch to Customer</span>
            ) : (
              <span>Intercepted for Human Review</span>
            )}
          </h2>

          {/* Stated Reason Callout */}
          <div className="p-3 rounded-xl bg-black/30 border border-white/5 backdrop-blur-sm">
            <p className="text-xs sm:text-sm text-slate-200 leading-relaxed font-normal">
              <span className="text-slate-400 font-medium">Policy Audit Reason: </span>
              {response.escalation_reason}
            </p>
          </div>
        </div>

        {/* Safety Gating Checklist */}
        <div className="flex-shrink-0 bg-canvas-elevated/90 border border-white/10 rounded-xl p-4 min-w-[240px] space-y-2.5 shadow-sm">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 pb-1.5 border-b border-white/10 flex justify-between">
            <span>Safety Gate Audit</span>
            <span className="text-slate-500">5-Stage Pass</span>
          </div>

          {/* Gate 1: Restricted Intent Check */}
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-300">Policy Eligibility</span>
            {response.escalation_reason.includes('always escalate') ? (
              <span className="inline-flex items-center text-rose-400 font-mono text-[11px] space-x-1">
                <Lock className="w-3 h-3" />
                <span>Restricted</span>
              </span>
            ) : (
              <span className="inline-flex items-center text-emerald-400 font-mono text-[11px] space-x-1">
                <Check className="w-3 h-3" />
                <span>Permitted</span>
              </span>
            )}
          </div>

          {/* Gate 2: Intent Confidence */}
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-300">Classifier Confidence</span>
            <span className={`font-mono text-[11px] ${confPercent >= 85 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {confPercent}% {confPercent >= 85 ? '✓' : '⚠'}
            </span>
          </div>

          {/* Gate 3: Precedent Agreement */}
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-300">Precedent Agreement</span>
            <span className={`font-mono text-[11px] ${agreementPercent >= 70 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {agreementPercent}% {agreementPercent >= 70 ? '(≥70%)' : '(<70%)'}
            </span>
          </div>

          {/* Gate 4: Grounding Self-Critique */}
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-300">Factual Grounding</span>
            {isGrounded ? (
              <span className="text-emerald-400 font-mono text-[11px]">100% Grounded</span>
            ) : (
              <span className="text-rose-400 font-mono text-[11px]">Ungrounded Claim</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
