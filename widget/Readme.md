# Open Docs RAG Widget

A customizable AI-powered documentation widget, designed to bring intelligent assistance to your website. Easy to integrate and highly configurable.

## Features

- 🎨 **Highly Customizable** - Configure colors, styling, behavior, and content through simple data attributes
- 📱 **Mobile Responsive** - Works seamlessly on desktop, tablet, and mobile devices
- 🚀 **Easy Integration** - Use as npm package or include via script tag
- ⚡ **Lightweight** - Minimal bundle size with no external dependencies
- 🔧 **TypeScript Support** - Full TypeScript definitions included
- 🎯 **Accessibility** - Built with accessibility best practices

## Quick Start

### Option 1: Script Tag (Recommended)

Simply include the widget script with your configuration:

```html
<script
  async
  src="https://your-domain.com/open-docs-rag-widget.bundle.js"
  data-website-id="your-website-id"
  data-project-name="Your Project"
  data-project-color="#1D4716"
  data-project-logo="/logo.svg"
  data-modal-example-questions="How to get started?, How to deploy?, API documentation"
  data-modal-disclaimer="This is an AI assistant for your documentation."
  data-api-endpoint="/api/chat"
></script>
```

### Option 2: NPM Package

Install the package:

```bash
npm install open-docs-rag-widget
```

Use in your project:

```typescript
import { createWidget } from 'open-docs-rag-widget';

const widget = createWidget({
  websiteId: 'your-website-id',
  projectName: 'Your Project',
  projectColor: '#1D4716',
  projectLogo: '/logo.svg',
  modalExampleQuestions: 'How to get started?, How to deploy?',
  apiEndpoint: '/api/chat'
});

// Open the widget programmatically
widget.open();
```

## Configuration

### Required Parameters

| Parameter            | Description                         | Example              |
| -------------------- | ----------------------------------- | -------------------- |
| `data-website-id`    | Unique identifier for your website  | `"my-website-123"`   |
| `data-project-name`  | Name displayed in the widget header | `"My Documentation"` |
| `data-project-color` | Primary color (HEX)                 | `"#1D4716"`          |
| `data-project-logo`  | Logo URL or data URI                | `"/logo.svg"`        |

### Optional Parameters

#### Modal Configuration

- `data-modal-title` - Custom title for the modal
- `data-modal-disclaimer` - Disclaimer text (supports Markdown)
- `data-modal-example-questions` - Comma-separated example questions
- `data-modal-example-questions-title` - Title for example questions section
- `data-modal-ask-ai-input-placeholder` - Input placeholder text

#### Button Configuration

- `data-button-hide` - Set to `"true"` to hide the default button
- `data-button-height` - Button height (default: `"54px"`)
- `data-button-width` - Button width (default: `"48px"`)
- `data-button-image` - Custom button image URL

#### Styling

- `data-font-family` - Custom font family
- `data-modal-disclaimer-bg-color` - Disclaimer background color
- `data-modal-disclaimer-font-size` - Disclaimer font size
- `data-modal-title-font-size` - Title font size
- `data-example-question-button-*` - Various button styling options

#### Behavior

- `data-modal-override-open-id` - ID of custom trigger element
- `data-modal-override-open-class` - Class of custom trigger elements
- `data-modal-open-by-default` - Open modal automatically
- `data-modal-open-on-command-k` - Enable Cmd+K shortcut

#### API

- `data-api-endpoint` - Your API endpoint URL
- `data-api-key` - Optional API key for authentication

## API Integration

The widget expects your API endpoint to:

1. Accept POST requests with JSON body:

```json
{
  "message": "User's question",
  "websiteId": "your-website-id"
}
```

2. Return JSON response:

```json
{
  "message": "AI response",
  "sources": ["optional", "array", "of", "source", "urls"]
}
```

## Advanced Usage

### Custom Triggers

Hide the default button and use your own triggers:

```html
<script
  data-button-hide="true"
  data-modal-override-open-id="my-button"
  /* other config */
></script>

<button id="my-button">Ask AI</button>
```

### Programmatic Control

```typescript
// Create widget instance
const widget = createWidget(config);

// Control the widget
widget.open(); // Open modal
widget.close(); // Close modal
widget.destroy(); // Remove widget from DOM
widget.updateConfig({ projectColor: '#ff0000' }); // Update configuration
```

### Global API

When using the script tag, the widget exposes a global API:

```javascript
// Create additional widget instances
const customWidget = window.OpenDocsRAGWidget.create({
  websiteId: 'custom-123',
  projectName: 'Custom Widget',
  projectColor: '#ff6b6b',
  projectLogo: '/custom-logo.svg'
});

customWidget.open();
```

## Examples

### Basic Implementation

```html
<script
  async
  src="/open-docs-rag-widget.bundle.js"
  data-website-id="docs-site"
  data-project-name="My Docs"
  data-project-color="#007bff"
  data-project-logo="/logo.svg"
  data-modal-example-questions="Getting started, API reference, Troubleshooting"
  data-api-endpoint="/api/chat"
></script>
```

### Advanced Customization

```html
<script
  async
  src="/open-docs-rag-widget.bundle.js"
  data-website-id="advanced-site"
  data-project-name="Advanced Docs"
  data-project-color="#1D4716"
  data-project-logo="/logo.svg"
  data-modal-disclaimer="AI-powered assistant. Responses may not be accurate."
  data-modal-disclaimer-bg-color="#fff3cd"
  data-modal-disclaimer-text-color="#856404"
  data-example-question-button-border="2px solid #dee2e6"
  data-example-question-button-text-color="#495057"
  data-modal-open-on-command-k="true"
  data-api-endpoint="/api/chat"
  data-api-key="your-api-key"
></script>
```

## Development

### Prerequisites

- Node.js 16+
- npm or yarn

### Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Build just the bundle
npm run build:bundle

# Build just the npm package
npm run build:npm
```

### Project Structure

```
src/
├── bundle.ts       # Script tag entry point
├── config.ts       # Configuration parsing
├── index.ts        # NPM package entry point
├── styles.css      # Widget styles
├── types.ts        # TypeScript definitions
├── widget.ts       # Main widget class
└── example.html    # Example implementation
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
