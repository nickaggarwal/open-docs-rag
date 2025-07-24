// Bundle entry point - auto-initializes widget from script tag attributes
import { OpenDocsRAGWidget } from './widget';
import { parseConfigFromScript } from './config';
import { createWidget } from './index';

// Auto-initialize widget when script loads
function initWidget(): void {
  const config = parseConfigFromScript();
  if (config) {
    new OpenDocsRAGWidget(config);
  }
}

// Expose global API for manual control
window.OpenDocsRAGWidget = {
  create: createWidget,
  Widget: OpenDocsRAGWidget
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initWidget);
} else {
  // DOM is already loaded
  initWidget();
}
