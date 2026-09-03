/** Auth API — bootstrap from the shared layout, with API fallback. */
import { req } from './workflows';

export interface CurrentUser {
  id: string;
  username: string;
  role: string;
}

declare global {
  interface Window {
    __EGO_USER__?: CurrentUser;
  }
}

export function getCurrentUser(): Promise<CurrentUser> {
  const bootstrap = window.__EGO_USER__;
  if (bootstrap?.role) return Promise.resolve(bootstrap);
  return req<CurrentUser>('/api/v1/auth/me');
}
