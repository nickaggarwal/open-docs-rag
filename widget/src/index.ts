// Main entry point for npm package
import { OpenDocsRAGWidget } from './widget';
import type { WidgetConfig, WidgetInstance } from './types';

export { OpenDocsRAGWidget } from './widget';
export type { WidgetConfig, WidgetInstance, ChatMessage, ApiResponse } from './types';
export { parseConfigFromScript, defaultConfig } from './config';

// Export CSS for manual import when needed
export * from './styles';

// Factory function for creating widgets programmatically
export function createWidget(config: WidgetConfig): WidgetInstance {
  return new OpenDocsRAGWidget(config);
}

// Global type declarations for window object
declare global {
  interface Window {
    OpenDocsRAGWidget?: {
      create: typeof createWidget;
      Widget: typeof OpenDocsRAGWidget;
    };
  }
}
