import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

const mountEl = document.getElementById('workflow-root') || document.getElementById('root');
if (mountEl) {
  createRoot(mountEl).render(<StrictMode><App /></StrictMode>);
}
