import { createRoot } from 'react-dom/client';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';
import '@fontsource/space-grotesk/500.css';
import '@fontsource/space-grotesk/700.css';
import './styles.css';
import { App } from './App';
import { ErrorBoundary } from './ErrorBoundary';

// Top-level boundary: an unexpected response shape shows a message instead of a blank page. Reload to recover.
createRoot(document.getElementById('root')!).render(<ErrorBoundary title="SOMETHING WENT WRONG, RELOAD THE PAGE"><App /></ErrorBoundary>);
