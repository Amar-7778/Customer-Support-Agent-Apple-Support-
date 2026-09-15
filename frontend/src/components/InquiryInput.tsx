import React, { useState } from 'react';
import { Send, X, ChevronDown } from 'lucide-react';
import type { ExampleInquiry } from '../types';
import examplesData from '../data/examples.json';

interface InquiryInputProps {
  message: string;
  onChangeMessage: (val: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

export const InquiryInput: React.FC<InquiryInputProps> = ({
  message,
  onChangeMessage,
  onSubmit,
  isLoading,
}) => {
  const examples: ExampleInquiry[] = examplesData as ExampleInquiry[];
  const [selectedExampleId, setSelectedExampleId] = useState<string>('');

  const charCount = message.length;
  const isOverLimit = charCount > 1000;
  const isValid = charCount > 0 && !isOverLimit && !isLoading;

  const handleSelectExample = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    setSelectedExampleId(id);
    if (!id) return;
    const found = examples.find((ex) => ex.id === id);
    if (found) {
      onChangeMessage(found.text);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      if (isValid) {
        onSubmit();
      }
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-5 shadow-subtle-card border border-white/[0.08] transition-all">
      {/* Top Bar: Label & Example Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3.5">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Customer Support Inquiry
          </span>
          <span className="text-[10px] font-mono text-slate-500 bg-white/5 px-2 py-0.5 rounded-full border border-white/5">
            Twitter / @AppleSupport
          </span>
        </div>

        {/* Real Examples Dropdown */}
        <div className="relative inline-block w-full sm:w-auto">
          <select
            id="example-picker"
            value={selectedExampleId}
            onChange={handleSelectExample}
            disabled={isLoading}
            className="w-full sm:w-72 text-xs bg-canvas-elevated hover:bg-canvas-hover text-slate-200 border border-white/10 rounded-lg px-3 py-1.5 pr-8 appearance-none focus:outline-none focus:ring-1 focus:ring-apple-blue cursor-pointer transition truncate font-sans"
          >
            <option value="">-- Load Real Candidate Inquiry --</option>
            {examples.map((ex) => (
              <option key={ex.id} value={ex.id}>
                [{ex.intent_label}] {ex.text.slice(0, 50)}...
              </option>
            ))}
          </select>
          <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-2.5 pointer-events-none" />
        </div>
      </div>

      {/* Textarea Container */}
      <div className="relative glass-input rounded-xl focus-within:ring-2 focus-within:ring-apple-blue/50">
        <textarea
          id="customer-inquiry-input"
          value={message}
          onChange={(e) => {
            onChangeMessage(e.target.value);
            setSelectedExampleId(''); // reset dropdown if user edits
          }}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          rows={4}
          placeholder="Enter incoming customer inquiry (e.g., '@AppleSupport my battery dies every 2 hours after the iOS 11 update...')"
          className="w-full bg-transparent px-4 py-3.5 text-sm sm:text-base text-slate-100 placeholder-slate-500 resize-none focus:outline-none font-normal leading-relaxed"
        />

        {/* Floating Clear Button */}
        {message && !isLoading && (
          <button
            type="button"
            onClick={() => {
              onChangeMessage('');
              setSelectedExampleId('');
            }}
            className="absolute right-3 top-3 p-1 rounded-md text-slate-400 hover:text-white hover:bg-white/10 transition"
            title="Clear text"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Footer Controls: Character Count & Dispatch Button */}
      <div className="flex items-center justify-between mt-3.5 pt-2 border-t border-white/[0.06]">
        <div className="flex items-center space-x-3 text-xs">
          <span
            className={`font-mono text-[11px] ${
              isOverLimit
                ? 'text-rose-400 font-semibold'
                : charCount > 800
                ? 'text-amber-400'
                : 'text-slate-500'
            }`}
          >
            {charCount} / 1,000 characters
          </span>
          <span className="hidden md:inline text-slate-500 text-[11px]">
            Press <kbd className="font-mono bg-white/10 px-1 py-0.5 rounded text-slate-300">⌘</kbd> + <kbd className="font-mono bg-white/10 px-1 py-0.5 rounded text-slate-300">Enter</kbd> to evaluate
          </span>
        </div>

        <button
          id="evaluate-inquiry-btn"
          type="button"
          onClick={onSubmit}
          disabled={!isValid}
          className={`relative inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl font-medium text-xs tracking-tight transition-all duration-200 shadow-tactile-btn ${
            isValid
              ? 'bg-apple-blue hover:bg-apple-blue-hover text-white active:scale-[0.98]'
              : 'bg-white/5 text-slate-500 cursor-not-allowed border border-white/5'
          }`}
        >
          {isLoading ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Evaluating Pipeline...</span>
            </>
          ) : (
            <>
              <Send className="w-3.5 h-3.5" />
              <span>Evaluate & Dispatch</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
