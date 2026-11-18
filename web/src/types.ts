// API contract: see ../ui/README.md. Ground truth never reaches the UI.
export interface Trip { trip_id: string; n_gps: number; duration_s: number; length_km?: number }
export interface Metrics {
  n_obs: number;
  pointwise_acc_classifier: number; pointwise_acc_hmm: number;
  route_continuity_classifier: number; route_continuity_hmm: number;
  candidate_miss_rate: number;
  latency_ms: { candidates: number; classifier: number; hmm: number };
}
export interface PredictResult {
  trip_id: string;
  gps: [number, number][];
  network: [number, number, number, number][];
  classifier_only: { arcs: [number, number, number, number][]; breaks: [number, number][] };
  hmm: { route: [number, number][] };
  metrics: Metrics;
}
export interface Health { status: string; model?: string }
export type LayerKey = 'net' | 'gps' | 'clf' | 'hmm' | 'brk';
