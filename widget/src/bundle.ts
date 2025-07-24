// Bundle entry point - auto-initializes widget from script tag attributes
import { OpenDocsRAGWidget } from './widget';
import { parseConfigFromScript } from './config';
import { createWidget } from './index';

// Expose global API for manual control IMMEDIATELY
(window as any).OpenDocsRAGWidget = {
  create: createWidget,
  Widget: OpenDocsRAGWidget,
  // Add a test function to debug
  test: () => {
    console.log('Test function called!');
    const testWidget = createWidget({
      websiteId: 'test-123',
      projectName: 'Test Project',
      projectColor: '#ff6b6b',
      projectLogo:
        'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSIjZmY2YjZiIi8+Cjwvc3ZnPgo=',
      modalExampleQuestions: 'Test question'
    });
    console.log('Test widget created:', testWidget);
    console.log('Test widget has open method:', typeof testWidget.open);
    return testWidget;
  }
};

console.log('OpenDocsRAGWidget global API exposed:', (window as any).OpenDocsRAGWidget);
console.log('createWidget function:', createWidget);
console.log('OpenDocsRAGWidget class:', OpenDocsRAGWidget);
console.log('create function type:', typeof (window as any).OpenDocsRAGWidget.create);

// Auto-initialize widget when script loads
function initWidget(): void {
  console.log('Initializing Open Docs RAG Widget...');
  console.log('Document ready state:', document.readyState);

  // Give a small delay for DOM to be fully parsed
  setTimeout(() => {
    const config = parseConfigFromScript();
    if (config) {
      console.log('Config found, creating widget:', config);
      new OpenDocsRAGWidget(config);
    } else {
      console.log('No valid config found - trying again in 100ms...');
      // Retry once more after a short delay
      setTimeout(() => {
        const retryConfig = parseConfigFromScript();
        if (retryConfig) {
          console.log('Config found on retry, creating widget:', retryConfig);
          new OpenDocsRAGWidget(retryConfig);
        } else {
          console.warn('Still no valid config found after retry');
        }
      }, 100);
    }
  }, 10);
}

// Initialize when DOM is ready with multiple fallbacks
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initWidget);
  // Backup timer in case DOMContentLoaded doesn't fire
  setTimeout(initWidget, 100);
} else if (document.readyState === 'interactive') {
  // DOM is parsed but maybe resources aren't loaded yet
  setTimeout(initWidget, 10);
} else {
  // DOM is already loaded
  initWidget();
}
