/// <reference types="vite/client" />

// Declare Iconify web component — offline via /static/iconify-preload.js
import type React from 'react';

declare global {
  namespace JSX {
    interface IntrinsicElements {
      'iconify-icon': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement> & {
        icon?: string;
      }, HTMLElement>;
    }
  }
}
