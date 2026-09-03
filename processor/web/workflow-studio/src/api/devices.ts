/** Device APIs — online collector capabilities drive the Studio input palette. */
import { req } from './workflows';
import type { InputSourcesResponse } from '../types/workflow';

export function getDeviceInputSources() {
  return req<InputSourcesResponse>('/api/v1/devices/input-sources');
}
