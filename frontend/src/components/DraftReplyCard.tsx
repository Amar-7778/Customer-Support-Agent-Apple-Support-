import React, { useState } from 'react';
import { Copy, Check, MessageSquareQuote } from 'lucide-react';

interface DraftReplyCardProps {
  draftReply: string;
  precedentCount: number;
}

export const DraftReplyCard: React.FC<DraftReplyCardProps> = ({
  draftReply,
  precedentCount,
}) => {
  const [copied, setCopied] = useState(false);
  const charLength = draftReply.length;
  const isWithinLimit = charLength <= 280;

  const handleCopy = () => {
    navigator.clipboard.writeText(draftReply);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-white/[0.08] shadow-subtle-card space-y-4">
      {/* Header with Title & Twitter Simulation Tag */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <MessageSquareQuote className="w-4 h-4 text-apple-blue" />
          <h3 className="text-sm font-semibold tracking-tight text-white">
            Synthesized Support Reply
          </h3>
          <span className="text-[10px] font-mono text-slate-400 bg-white/5 px-2 py-0.5 rounded border border-white/5">
            Grounded on {precedentCount} precedents
          </span>
        </div>

        {/* Copy Button */}
        <button
          onClick={handleCopy}
          className="inline-flex items-center space-x-1.5 text-xs text-slate-300 hover:text-white px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 transition"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400 font-medium">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy Reply</span>
            </>
          )}
        </button>
      </div>

      {/* Simulated @AppleSupport Twitter Card */}
      <div className="rounded-xl p-4 sm:p-5 bg-black/40 border border-white/10 space-y-3">
        {/* Twitter Account Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {/* Apple Avatar */}
            <div className="w-10 h-10 rounded-full bg-slate-900 border border-white/10 flex items-center justify-center p-1.5 shadow-sm">
              <svg className="w-6 h-6 text-white fill-current" viewBox="0 0 170 170">
                <path d="M150.37 130.25c-2.45 5.66-5.35 10.87-8.71 15.66-4.58 6.53-8.33 11.05-11.22 13.56-4.48 4.12-9.28 6.23-14.42 6.35-3.69 0-8.14-1.05-13.32-3.18-5.19-2.12-9.97-3.17-14.34-3.17-4.58 0-9.49 1.05-14.75 3.17-5.26 2.13-9.5 3.24-12.74 3.35-4.35.13-9.16-1.9-14.42-6.08-3.69-3.08-7.77-7.97-12.23-14.67-6.02-8.99-10.85-19.34-14.51-31.06-3.65-11.72-5.48-23.01-5.48-33.88 0-14.77 3.73-27.17 11.19-37.21 7.46-10.04 17.06-15.17 28.8-15.4 5.37 0 11.24 1.44 17.62 4.34 6.38 2.9 10.45 4.39 12.22 4.49 1.46 0 5.68-1.55 12.65-4.66 6.98-3.11 12.82-4.55 17.53-4.32 13.48.64 24.31 5.39 32.51 14.26-11.75 7.1-17.5 16.9-17.26 29.41.24 9.94 4.09 18.27 11.54 25 7.46 6.72 16.32 10.4 26.58 11.05-2.28 6.98-5.06 14.12-8.35 21.43zM119.22 31.84c0-7.72 2.76-14.89 8.28-21.52 5.53-6.63 12.44-10.32 20.73-11.07.22 1.1.33 2.14.33 3.12 0 7.56-2.91 14.88-8.73 21.96-5.83 7.07-12.82 10.88-20.97 11.43-.11-1.28-.16-2.58-.16-3.92z" />
              </svg>
            </div>

            <div>
              <div className="flex items-center space-x-1.5">
                <span className="text-sm font-semibold text-white">Apple Support</span>
                {/* Verified Badge */}
                <svg className="w-4 h-4 text-sky-400 fill-current" viewBox="0 0 24 24">
                  <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                </svg>
              </div>
              <span className="text-xs text-slate-400 font-mono">@AppleSupport</span>
            </div>
          </div>

          {/* Twitter / X Stamp */}
          <div className="text-slate-500">
            <span className="text-[11px] font-mono">Twitter reply</span>
          </div>
        </div>

        {/* Reply Body Text */}
        <p className="text-slate-100 text-sm sm:text-base leading-relaxed font-normal pt-1">
          {draftReply}
        </p>

        {/* Constraint Indicator Footer */}
        <div className="pt-3 border-t border-white/[0.08] flex items-center justify-between text-xs font-mono">
          <span className="text-slate-400">
            Length: <strong className={isWithinLimit ? 'text-emerald-400' : 'text-rose-400'}>{charLength}</strong> / 280 characters
          </span>
          <span className={isWithinLimit ? 'text-emerald-400 flex items-center gap-1' : 'text-rose-400'}>
            {isWithinLimit ? '✓ Compliant with Twitter 280 limit' : '⚠ Exceeds 280 limit'}
          </span>
        </div>
      </div>
    </div>
  );
};
