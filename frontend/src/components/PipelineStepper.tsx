import React, { useEffect, useState } from 'react';
import { CheckCircle2, Loader2, Sparkles, Database, FileText, ShieldAlert, Cpu } from 'lucide-react';

interface PipelineStepperProps {
  isLoading: boolean;
}

interface StepItem {
  id: number;
  label: string;
  detail: string;
  icon: React.ReactNode;
}

export const PipelineStepper: React.FC<PipelineStepperProps> = ({ isLoading }) => {
  const [activeStep, setActiveStep] = useState<number>(1);
  const [elapsed, setElapsed] = useState<number>(0);

  const steps: StepItem[] = [
    {
      id: 1,
      label: 'Intent Classification',
      detail: 'Groq LLM prompt across 8 canonical taxonomy intents',
      icon: <Sparkles className="w-4 h-4 text-sky-400" />,
    },
    {
      id: 2,
      label: 'Precedent Retrieval',
      detail: 'ChromaDB nearest-neighbor search across 3,000 precedents',
      icon: <Database className="w-4 h-4 text-indigo-400" />,
    },
    {
      id: 3,
      label: 'Reply Synthesis',
      detail: 'Grounded Apple Support reply generation (strict 280-char limit)',
      icon: <FileText className="w-4 h-4 text-emerald-400" />,
    },
    {
      id: 4,
      label: 'Grounding Verification',
      detail: 'Independent LLM self-critique pass against unsupported claims',
      icon: <ShieldAlert className="w-4 h-4 text-amber-400" />,
    },
    {
      id: 5,
      label: 'Policy Escalation Gating',
      detail: 'Evaluating restricted intents, precedent agreement & claims',
      icon: <Cpu className="w-4 h-4 text-purple-400" />,
    },
  ];

  useEffect(() => {
    if (!isLoading) {
      setActiveStep(1);
      setElapsed(0);
      return;
    }

    const startTime = Date.now();
    const interval = setInterval(() => {
      const ms = Date.now() - startTime;
      setElapsed(ms / 1000);

      // Realistic pipeline timeline transitions
      if (ms < 1400) {
        setActiveStep(1);
      } else if (ms < 2300) {
        setActiveStep(2);
      } else if (ms < 3300) {
        setActiveStep(3);
      } else if (ms < 4200) {
        setActiveStep(4);
      } else {
        setActiveStep(5);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [isLoading]);

  if (!isLoading) return null;

  return (
    <div className="glass-panel rounded-2xl p-6 border border-apple-blue/30 shadow-2xl animate-fade-in relative overflow-hidden">
      {/* Top Header */}
      <div className="flex items-center justify-between pb-4 mb-4 border-b border-white/[0.08]">
        <div className="flex items-center space-x-2.5">
          <div className="relative">
            <span className="animate-ping absolute inline-flex h-3 w-3 rounded-full bg-apple-blue opacity-75"></span>
            <div className="w-3 h-3 rounded-full bg-apple-blue"></div>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">
              Executing Autonomous Agent Pipeline
            </h4>
            <p className="text-xs text-slate-400">
              Live multi-agent workflow powered by Groq and ChromaDB
            </p>
          </div>
        </div>

        <div className="font-mono text-xs text-slate-400 bg-white/5 px-2.5 py-1 rounded-md border border-white/5">
          Elapsed: <span className="text-white font-medium">{elapsed.toFixed(1)}s</span>
        </div>
      </div>

      {/* Stepper Vertical Flow */}
      <div className="space-y-3.5">
        {steps.map((step) => {
          const isDone = activeStep > step.id;
          const isCurrent = activeStep === step.id;

          return (
            <div
              key={step.id}
              className={`flex items-start space-x-3.5 p-2.5 rounded-xl transition-all duration-300 ${
                isCurrent
                  ? 'bg-white/[0.06] border border-white/10 shadow-sm'
                  : 'opacity-70'
              }`}
            >
              {/* Status Icon */}
              <div className="mt-0.5 flex-shrink-0">
                {isDone ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 animate-fade-in" />
                ) : isCurrent ? (
                  <Loader2 className="w-4 h-4 text-apple-blue animate-spin" />
                ) : (
                  <div className="w-4 h-4 rounded-full border border-slate-600 flex items-center justify-center">
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600" />
                  </div>
                )}
              </div>

              {/* Step Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2">
                  <span
                    className={`text-xs font-medium tracking-tight ${
                      isCurrent
                        ? 'text-white'
                        : isDone
                        ? 'text-slate-300'
                        : 'text-slate-500'
                    }`}
                  >
                    {step.label}
                  </span>
                  {isCurrent && (
                    <span className="text-[10px] font-mono uppercase bg-apple-blue/20 text-sky-300 px-1.5 py-0.2 rounded border border-apple-blue/30 animate-pulse">
                      In Progress
                    </span>
                  )}
                  {isDone && (
                    <span className="text-[10px] font-mono text-emerald-400/80">
                      Completed
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-slate-400 mt-0.5 truncate">
                  {step.detail}
                </p>
              </div>

              {/* Step Type Icon */}
              <div className="opacity-50">{step.icon}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
