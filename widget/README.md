# Open Docs RAG Widget

A customizable AI-powered documentation widget that can be easily integrated into any website or React application. Easy to integrate and highly configurable.

## Features

- 🎨 **Highly Customizable** - Configure colors, styling, behavior, and content through simple data attributes
- 📱 **Mobile Responsive** - Works seamlessly on desktop, tablet, and mobile devices
- 🚀 **Easy Integration** - Use as npm package or include via script tag
- ⚡ **Lightweight** - Minimal bundle size with no external dependencies
- 🔧 **TypeScript Support** - Full TypeScript definitions included
- 🎯 **Accessibility** - Built with accessibility best practices
- ⚛️ **React Compatible** - Works seamlessly with React applications
- 🌐 **Framework Agnostic** - Can be used with any JavaScript framework or vanilla HTML
- ⚡ **SSE Streaming Support** - Real-time streaming responses with Server-Sent Events for better UX

## Installation

```bash
npm install open-docs-rag-widget
```

## Quick Start

### 🎉 Zero Configuration

✅ **No CSS Import Required**: The widget automatically injects its styles when imported - no manual CSS imports needed!

**Optional CSS Import** (for custom builds or CDN usage):

```html
<!-- Only needed for CDN usage without module imports -->
<link rel="stylesheet" href="https://unpkg.com/open-docs-rag-widget@latest/dist/styles.css" />
```

### React Integration

#### Option 1: React Hook (Recommended)

```tsx
import React, { useEffect, useRef } from 'react';
import { createWidget, WidgetInstance, WidgetConfig } from 'open-docs-rag-widget';
// No CSS import needed - styles are auto-injected!

interface OpenDocsWidgetProps {
  config: WidgetConfig;
}

export const OpenDocsWidget: React.FC<OpenDocsWidgetProps> = ({ config }) => {
  const widgetRef = useRef<WidgetInstance | null>(null);

  useEffect(() => {
    // Create widget instance
    widgetRef.current = createWidget(config);

    // Cleanup on unmount
    return () => {
      if (widgetRef.current) {
        widgetRef.current.destroy();
      }
    };
  }, []);

  // Update config when props change
  useEffect(() => {
    if (widgetRef.current) {
      widgetRef.current.updateConfig(config);
    }
  }, [config]);

  return null; // Widget renders itself to document.body
};

// Usage in your React app
function App() {
  const widgetConfig = {
    websiteId: 'your-website-id',
    projectName: 'Your Project',
    projectColor: '#1D4716',
    projectLogo: '/logo.svg',
    modalExampleQuestions: 'How to get started?, How to deploy?, API documentation',
    modalDisclaimer: 'This is an AI assistant for your documentation.',
    apiEndpoint: '/api/chat'
  };

  return (
    <div>
      <h1>Your App</h1>
      <OpenDocsWidget config={widgetConfig} />
    </div>
  );
}
```

#### Option 2: Direct Usage in React

```tsx
import React, { useEffect } from 'react';
import { createWidget } from 'open-docs-rag-widget';
// No CSS import needed - styles are auto-injected!

function App() {
  useEffect(() => {
    const widget = createWidget({
      websiteId: 'your-website-id',
      projectName: 'Your Project',
      projectColor: '#1D4716',
      projectLogo: '/logo.svg',
      modalExampleQuestions: 'How to get started?, How to deploy?',
      apiEndpoint: '/api/chat'
    });

    // Open widget programmatically if needed
    // widget.open();

    return () => {
      widget.destroy();
    };
  }, []);

  return <div>Your App Content</div>;
}
```

#### Option 3: Custom React Component with Controls

```tsx
import React, { useEffect, useRef, useState } from 'react';
import { createWidget, WidgetInstance } from 'open-docs-rag-widget';
// No CSS import needed - styles are auto-injected!

export const AdvancedOpenDocsWidget: React.FC = () => {
  const widgetRef = useRef<WidgetInstance | null>(null);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    widgetRef.current = createWidget({
      websiteId: 'advanced-widget',
      projectName: 'Advanced Docs',
      projectColor: '#007bff',
      projectLogo: '/logo.svg',
      buttonHide: true, // Hide default button
      modalExampleQuestions: 'Advanced features, Integration guide, API reference',
      apiEndpoint: '/api/chat'
    });

    return () => {
      if (widgetRef.current) {
        widgetRef.current.destroy();
      }
    };
  }, []);

  const openWidget = () => {
    if (widgetRef.current) {
      widgetRef.current.open();
      setIsOpen(true);
    }
  };

  const closeWidget = () => {
    if (widgetRef.current) {
      widgetRef.current.close();
      setIsOpen(false);
    }
  };

  return (
    <div>
      <button onClick={openWidget} className='bg-blue-500 text-white px-4 py-2 rounded'>
        Ask AI Assistant
      </button>
      {isOpen && (
        <button onClick={closeWidget} className='ml-2 bg-gray-500 text-white px-4 py-2 rounded'>
          Close
        </button>
      )}
    </div>
  );
};
```

### Vanilla JavaScript / HTML

#### Option 1: Script Tag (Quick Setup)

```html
<script
  async
  src="https://unpkg.com/open-docs-rag-widget@latest/dist/bundle.js"
  data-website-id="your-website-id"
  data-project-name="Your Project"
  data-project-color="#1D4716"
  data-project-logo="/logo.svg"
  data-modal-example-questions="How to get started?, How to deploy?, API documentation"
  data-modal-disclaimer="This is an AI assistant for your documentation."
  data-api-endpoint="/api/chat"
></script>
```

#### Option 2: Import from CDN

```html
<script type="module">
  // No CSS link needed - styles are auto-injected when imported!
  import { createWidget } from 'https://unpkg.com/open-docs-rag-widget@latest/dist/index.esm.js';

  const widget = createWidget({
    websiteId: 'your-website-id',
    projectName: 'Your Project',
    projectColor: '#1D4716',
    projectLogo: '/logo.svg',
    modalExampleQuestions: 'How to get started?, How to deploy?',
    apiEndpoint: '/api/chat'
  });
</script>
```

### Next.js Integration

```tsx
// components/OpenDocsWidget.tsx
'use client'; // For Next.js 13+ app directory

import dynamic from 'next/dynamic';
import { WidgetConfig } from 'open-docs-rag-widget';

// Dynamically import to avoid SSR issues
const OpenDocsWidgetComponent = dynamic(() => import('./OpenDocsWidgetClient'), { ssr: false });

interface Props {
  config: WidgetConfig;
}

export default function OpenDocsWidget({ config }: Props) {
  return <OpenDocsWidgetComponent config={config} />;
}

// components/OpenDocsWidgetClient.tsx
('use client');

import { useEffect, useRef } from 'react';
import { createWidget, WidgetInstance, WidgetConfig } from 'open-docs-rag-widget';
// No CSS import needed - styles are auto-injected!

interface Props {
  config: WidgetConfig;
}

export default function OpenDocsWidgetClient({ config }: Props) {
  const widgetRef = useRef<WidgetInstance | null>(null);

  useEffect(() => {
    widgetRef.current = createWidget(config);

    return () => {
      if (widgetRef.current) {
        widgetRef.current.destroy();
      }
    };
  }, []);

  return null;
}

// Usage in app/page.tsx or pages/index.tsx
import OpenDocsWidget from '@/components/OpenDocsWidget';

export default function Home() {
  return (
    <div>
      <h1>Welcome to Next.js</h1>
      <OpenDocsWidget
        config={{
          websiteId: 'nextjs-app',
          projectName: 'Next.js Docs',
          projectColor: '#000000',
          projectLogo: '/vercel.svg',
          apiEndpoint: '/api/chat'
        }}
      />
    </div>
  );
}
```

## Configuration

### Required Parameters

| Parameter      | Type   | Description                         | Example              |
| -------------- | ------ | ----------------------------------- | -------------------- |
| `websiteId`    | string | Unique identifier for your website  | `"my-website-123"`   |
| `projectName`  | string | Name displayed in the widget header | `"My Documentation"` |
| `projectColor` | string | Primary color (HEX)                 | `"#1D4716"`          |
| `projectLogo`  | string | Logo URL or data URI                | `"/logo.svg"`        |

### Optional Parameters

#### Modal Configuration

- `modalTitle?: string` - Custom title for the modal
- `modalDisclaimer?: string` - Disclaimer text (supports Markdown)
- `modalExampleQuestions?: string` - Comma-separated example questions
- `modalExampleQuestionsTitle?: string` - Title for example questions section
- `modalAskAiInputPlaceholder?: string` - Input placeholder text

#### Button Configuration

- `buttonHide?: boolean` - Hide the default button
- `buttonHeight?: string` - Button height (default: `"54px"`)
- `buttonWidth?: string` - Button width (default: `"48px"`)
- `buttonImage?: string` - Custom button image URL

#### Behavior

- `modalOverrideOpenId?: string` - ID of custom trigger element
- `modalOverrideOpenClass?: string` - Class of custom trigger elements
- `modalOpenByDefault?: boolean` - Open modal automatically
- `modalOpenOnCommandK?: boolean` - Enable Cmd+K shortcut

#### API

- `apiEndpoint?: string` - Your API endpoint URL
- `apiKey?: string` - Optional API key for authentication

## API Integration

The widget expects your API endpoint to:

1. Accept POST requests with JSON body:

```json
{
  "question": "User's question",
  "num_results": 3
}
```

2. Return JSON response:

```json
{
  "answer": "AI response",
  "sources": ["optional", "array", "of", "source", "urls"]
}
```

Note: The widget will also accept responses with `message` or `response` fields as alternatives to `answer`.

### Server-Sent Events (SSE) Streaming Support 🆕

The widget now supports real-time streaming responses via Server-Sent Events! For a better user experience with streaming responses:

1. Set `Content-Type: text/event-stream` in your response headers
2. Stream sources immediately:

```
data: {"type": "sources", "content": ["url1", "url2", "url3"]}
```

3. Stream answer chunks:

```
data: {"type": "answer_chunk", "content": "text chunk"}
```

4. End the stream:

```
data: [DONE]
```

**Benefits of SSE Streaming:**

- ✅ Sources appear immediately
- ✅ Text streams word-by-word for better UX
- ✅ Typing indicator during streaming
- ✅ Send button disabled during response
- ✅ Backward compatible with JSON responses

### Example API Implementation (Next.js)

```typescript
// app/api/chat/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { question, num_results } = await request.json();

    // Your AI/RAG logic here
    const response = await yourAIService.query(question, num_results);

    return NextResponse.json({
      answer: response.answer,
      sources: response.sources
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to process question' }, { status: 500 });
  }
}
```

## Publishing to NPM

### Prerequisites

1. Create an npm account at [npmjs.com](https://npmjs.com)
2. Install npm CLI and login: `npm login`

### Publishing Steps

```bash
# 1. Clone and setup
git clone <your-repo>
cd open-docs-rag-widget/widget
npm install

# 2. Update package.json
# - Change "name" to your package name
# - Update "repository" URLs
# - Set correct "version"

# 3. Build the package
npm run build

# 4. Test the package locally
npm pack
# This creates a .tgz file you can test with: npm install ./package-name.tgz

# 5. Publish to npm
npm publish

# For scoped packages (recommended):
npm publish --access public
```

### Updating Your Package

```bash
# Update version
npm version patch  # 1.0.0 -> 1.0.1
npm version minor  # 1.0.0 -> 1.1.0
npm version major  # 1.0.0 -> 2.0.0

# Publish update
npm publish
```

## Advanced Usage

### TypeScript Types

```typescript
import { WidgetConfig, WidgetInstance, ChatMessage, ApiResponse } from 'open-docs-rag-widget';

const config: WidgetConfig = {
  websiteId: 'typed-widget',
  projectName: 'TypeScript App',
  projectColor: '#3178c6',
  projectLogo: '/ts-logo.svg',
  apiEndpoint: '/api/chat'
};

const widget: WidgetInstance = createWidget(config);
```

### Custom CSS Styling

The widget inherits fonts from your application, but you can override styles:

```css
/* Override widget styles */
.open-docs-rag-widget-container {
  /* Your custom styles */
}

.open-docs-rag-widget-button {
  /* Custom button styles */
}
```

### Event Handling

```typescript
// Custom trigger setup
const widget = createWidget({
  websiteId: 'custom-triggers',
  projectName: 'Custom App',
  projectColor: '#ff6b6b',
  projectLogo: '/logo.svg',
  buttonHide: true,
  modalOverrideOpenClass: 'open-ai-chat'
});

// All elements with class 'open-ai-chat' will trigger the widget
```

## Browser Support

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
