// Test React component to verify the widget works properly
import React, { useEffect } from 'react';
import { createWidget } from './dist/index.js';

function TestWidget() {
  useEffect(() => {
    console.log('Creating widget...');
    const widget = createWidget({
      websiteId: 'test-123',
      projectName: 'Test Project',
      projectColor: '#007bff',
      projectLogo:
        'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSIjMDA3YmZmIi8+Cjwvc3ZnPgo=',
      modalExampleQuestions: 'How to get started?, API docs, Troubleshooting',
      apiEndpoint: '/api/chat'
    });

    // Check if widget is positioned correctly
    setTimeout(() => {
      const container = document.querySelector('.open-docs-rag-widget-container');
      if (container) {
        const styles = window.getComputedStyle(container);
        console.log('Widget position:', styles.position);
        console.log('Widget bottom:', styles.bottom);
        console.log('Widget right:', styles.right);
        console.log('Widget z-index:', styles.zIndex);
      }
    }, 100);

    return () => widget.destroy();
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h1>Test React App</h1>
      <p>Check the bottom-right corner for the widget!</p>
      <p>Open browser console to see positioning info.</p>
    </div>
  );
}

export default TestWidget;
