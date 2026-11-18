import { Component, type ReactNode } from 'react';

// Shows the error text instead of a blank area. Changing `resetKey` clears the error (e.g. on a new result).
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: unknown; title?: string }, { err: Error | null }> {
  state = { err: null as Error | null };
  static getDerivedStateFromError(err: Error) { return { err }; }
  componentDidUpdate(prev: { resetKey?: unknown }) {
    if (this.state.err && prev.resetKey !== this.props.resetKey) this.setState({ err: null });
  }
  render() {
    if (!this.state.err) return this.props.children;
    return (
      <div className="empty"><div>
        <h2>{this.props.title ?? 'MAP ERROR'}</h2>
        <span className="label" style={{ color: 'var(--warn)', wordBreak: 'break-all' }}>{this.state.err.message}</span>
      </div></div>
    );
  }
}
