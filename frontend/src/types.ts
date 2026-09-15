export interface Precedent {
  thread_id: string;
  tweet_id: string;
  intent: string;
  customer_message: string;
  action_taken: string;
  outcome: string;
  brand_reply_text: string;
  similarity_score: number;
}

export interface GroundingVerification {
  grounded: boolean;
  unsupported_claims: string[];
}

export interface AgentMetrics {
  model: string;
  total_latency_seconds: number;
  classification_latency_seconds: number;
  drafting_latency_seconds: number;
  verification_latency_seconds: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
}

export interface AgentResponse {
  customer_message: string;
  predicted_intent: string;
  intent_confidence: number;
  retrieved_precedents: Precedent[];
  precedent_agreement_score: number;
  drafted_reply: string;
  grounding_verification: GroundingVerification;
  decision: 'auto_handle' | 'escalate';
  escalation_reason: string;
  metrics: AgentMetrics;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | string;
  service: string;
  provider: string;
  model: string;
  groq_configured: boolean;
  collection: string;
  index_size: number;
}

export interface ExampleInquiry {
  id: string;
  intent: string;
  intent_label: string;
  category_tag: string;
  expected_decision: string;
  text: string;
  length: number;
}

export type PipelineStage = 
  | 'idle'
  | 'classifying'
  | 'retrieving'
  | 'drafting'
  | 'verifying'
  | 'complete'
  | 'error';
