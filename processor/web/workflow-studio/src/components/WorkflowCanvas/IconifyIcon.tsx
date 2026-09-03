/** Thin wrapper around the iconify-icon web component (offline, no CDN).
 *  Icons are preloaded via /static/iconify-preload.js. */

import { createElement } from 'react';

interface Props {
  icon: string;
  className?: string;
  style?: React.CSSProperties;
}

export function IconifyIcon({ icon, className, style }: Props) {
  return createElement('iconify-icon', { icon, class: className, style });
}
