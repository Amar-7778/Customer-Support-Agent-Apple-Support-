import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { InquiryInput } from './components/InquiryInput';
import { PipelineStepper } from './components/PipelineStepper';
import { DecisionHero } from './components/DecisionHero';
import { DraftReplyCard } from './components/DraftReplyCard';
import { GroundingShield } from './components/GroundingShield';
import { PrecedentsDrawer } from './components/PrecedentsDrawer';
import { TelemetryBar } from './components/TelemetryBar';
import { ErrorAlert } from './components/ErrorAlert';
import type { AgentResponse, HealthResponse } from './types';
import { checkBackendHealth, submitCustomerMessage } from './services/api';
import { Sparkles } from 'lucide-react';

export const App: React.FC = () => {
  const [message, setMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<AgentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  const fetchHealth = async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const data = await checkBackendHealth();
      setHealth(data);
    } catch (err: any) {
      setHealthError(err.message || 'Health check failed');
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  const handleEvaluate = async () => {
    if (!message.trim() || isLoading) return;
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await submitCustomerMessage(message);
      setResponse(res);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred during execution.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-canvas text-slate-100 flex flex-col selection:bg-apple-blue selection:text-white">
      {/* Navigation Header */}
      <Header
        health={health}
        healthLoading={healthLoading}
        healthError={healthError}
        onRefreshHealth={fetchHealth}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 space-y-8">
        {/* Top Hero Section: Context & Purpose */}
        <div className="max-w-3xl space-y-2">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-apple-blue-subtle text-sky-400 text-xs font-mono font-medium border border-apple-blue/20">
            <span>Production Multi-Agent Demo</span>
            <span>•</span>
            <span>No Mock Data</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-bold tracking-tight text-white text-balance">
            Autonomous Customer Support & Escalation Safeguard
          </h1>
          <p className="text-sm sm:text-base text-slate-400 font-normal leading-relaxed text-balance">
            Evaluates incoming customer tweets across an 8-intent taxonomy, grounds responses in 3,000 historical Apple Support precedents, validates factual fidelity with an independent self-critique pass, and enforces verifiable escalation boundaries.
          </p>
        </div>

        {/* Primary Input Command Section */}
        <section className="space-y-4">
          <InquiryInput
            message={message}
            onChangeMessage={setMessage}
            onSubmit={handleEvaluate}
            isLoading={isLoading}
          />
        </section>

        {/* Loading Stepper (Shown during inference) */}
        {isLoading && (
          <section className="animate-fade-in">
            <PipelineStepper isLoading={isLoading} />
          </section>
        )}

        {/* Error Alert */}
        {error && (
          <section className="animate-fade-in">
            <ErrorAlert message={error} onRetry={handleEvaluate} />
          </section>
        )}

        {/* Inspection Deck: Revealed after pipeline completes */}
        {response && !isLoading && (
          <section className="space-y-6 animate-fade-in">
            {/* 1. Decision Hero: High visual weight */}
            <DecisionHero response={response} />

            {/* 2. Draft Reply Twitter Simulation */}
            <DraftReplyCard
              draftReply={response.drafted_reply}
              precedentCount={response.retrieved_precedents.length}
            />

            {/* 3. Grounding Shield: LLM Self-Critique & Agreement */}
            <GroundingShield
              grounding={response.grounding_verification}
              agreementScore={response.precedent_agreement_score}
            />

            {/* 4. Precedents Drawer: Top-3 ChromaDB Grounding Receipts */}
            <PrecedentsDrawer precedents={response.retrieved_precedents} />

            {/* 5. Telemetry & Execution Performance */}
            {response.metrics && (
              <TelemetryBar metrics={response.metrics} />
            )}
          </section>
        )}

        {/* Empty State / Architectural Overview (Shown before first evaluation) */}
        {!response && !isLoading && !error && (
          <div className="glass-panel rounded-2xl p-7 sm:p-9 border border-white/[0.08] shadow-subtle-card space-y-6">
            <div className="flex items-center space-x-2 text-slate-300">
              <Sparkles className="w-5 h-5 text-apple-blue" />
              <h3 className="text-base font-semibold text-white tracking-tight">
                How This Autonomous Pipeline Operates
              </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs sm:text-sm">
              <div className="p-4 rounded-xl bg-canvas-elevated/70 border border-white/5 space-y-2">
                <div className="w-7 h-7 rounded-lg bg-apple-blue/20 flex items-center justify-center text-sky-400 font-mono font-bold text-xs">
                  1
                </div>
                <h4 className="font-semibold text-white">8-Intent Stratification</h4>
                <p className="text-slate-400 leading-relaxed text-xs">
                  Classifies tweets into data-derived intents (e.g. Battery & Power, OS Bugs, Account Access) via few-shot Groq inference (<code className="text-sky-300 font-mono text-[11px]">qwen/qwen3.8-27b</code>).
                </p>
              </div>

              <div className="p-4 rounded-xl bg-canvas-elevated/70 border border-white/5 space-y-2">
                <div className="w-7 h-7 rounded-lg bg-indigo-500/20 flex items-center justify-center text-indigo-400 font-mono font-bold text-xs">
                  2
                </div>
                <h4 className="font-semibold text-white">3,000 Precedent ChromaDB</h4>
                <p className="text-slate-400 leading-relaxed text-xs">
                  Retrieves top-3 historical Apple precedents to ground the reply. Disagreements among historical resolutions trigger automatic human escalation.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-canvas-elevated/70 border border-white/5 space-y-2">
                <div className="w-7 h-7 rounded-lg bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-mono font-bold text-xs">
                  3
                </div>
                <h4 className="font-semibold text-white">Zero-Hallucination Gating</h4>
                <p className="text-slate-400 leading-relaxed text-xs">
                  An independent self-critique pass verifies that the drafted reply contains no ungrounded claims or unauthorized policies before approving autonomous dispatch.
                </p>
              </div>
            </div>

            <div className="pt-4 border-t border-white/[0.08] flex items-center justify-between text-xs text-slate-500">
              <span>Select an inquiry from the dropdown above or type your own to evaluate live.</span>
              <span className="font-mono text-slate-400">FastAPI backend • Live ChromaDB</span>
            </div>
          </div>
        )}
      </main>

      {/* Minimal Footer */}
      <footer className="border-t border-white/[0.08] py-6 text-center text-xs text-slate-500 font-mono">
        <p>Apple Support Autonomous Agent Demo • Connected to live FastAPI service (<code className="text-slate-400">/handle_message</code>)</p>
      </footer>
    </div>
  );
};
export default App;
