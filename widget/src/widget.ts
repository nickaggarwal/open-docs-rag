import { WidgetConfig, WidgetInstance, ChatMessage, ApiResponse } from './types';
import { defaultConfig } from './config';
import './styles.css';

export class OpenDocsRAGWidget implements WidgetInstance {
  private config: WidgetConfig;
  private container: HTMLElement;
  private modal: HTMLElement | null = null;
  private isOpen: boolean = false;
  private messages: ChatMessage[] = [];

  constructor(config: WidgetConfig) {
    this.config = { ...defaultConfig, ...config };
    console.log('OpenDocsRAGWidget: Initializing with config:', {
      buttonHide: this.config.buttonHide,
      modalOverrideOpenId: this.config.modalOverrideOpenId,
      modalOpenByDefault: this.config.modalOpenByDefault
    });
    this.container = this.createContainer();
    this.init();
  }

  private init(): void {
    // Ensure styles are loaded
    this.ensureStylesInjected();

    document.body.appendChild(this.container);
    this.setupEventListeners();

    if (this.config.modalOpenByDefault) {
      console.log('OpenDocsRAGWidget: Opening modal by default');
      this.open();
    } else {
      console.log('OpenDocsRAGWidget: Modal will not open by default');
    }
  }

  private ensureStylesInjected(): void {
    // Check if our styles are already injected
    const existingStyle = document.querySelector('style[data-widget="open-docs-rag-widget"]');
    if (existingStyle) return;

    // Inject complete widget styles
    const style = document.createElement('style');
    style.setAttribute('data-widget', 'open-docs-rag-widget');
    style.textContent = `.open-docs-rag-widget-container{bottom:20px;font-family:inherit;position:fixed;right:20px;z-index:9999}.open-docs-rag-widget-button{align-items:center;background-color:#fff;border:2px solid #e9ecef;border-radius:50px;box-shadow:0 4px 12px rgba(0,0,0,.15);color:#495057;cursor:pointer;display:flex;font-family:inherit;font-weight:500;gap:8px;height:48px;justify-content:center;padding:0;transition:all .2s ease;width:120px}.open-docs-rag-widget-button:hover{background-color:#f8f9fa;border-color:#007bff;box-shadow:0 6px 20px rgba(0,123,255,.2);color:#007bff;transform:translateY(-2px)}.open-docs-rag-widget-button img{border-radius:4px}.open-docs-rag-widget-modal{align-items:center;display:flex;height:100%;justify-content:center;left:0;position:fixed;top:0;width:100%;z-index:10000}.open-docs-rag-widget-modal-overlay{background-color:rgba(0,0,0,.5);height:100%;left:0;position:absolute;top:0;width:100%}.open-docs-rag-widget-modal-content{background:#fff;border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,.3);display:flex;flex-direction:column;font-family:inherit;max-height:80vh;max-width:600px;overflow:hidden;position:relative;width:90%}.open-docs-rag-widget-modal-header{align-items:center;background-color:#f8f9fa;border-bottom:1px solid #e9ecef;display:flex;justify-content:space-between;padding:20px}.open-docs-rag-widget-modal-header-content{align-items:center;display:flex;gap:12px}.open-docs-rag-widget-modal-logo{border-radius:6px;height:32px;width:32px}.open-docs-rag-widget-modal-title{color:#212529;font-family:inherit;font-size:18px;font-weight:600;margin:0}.open-docs-rag-widget-modal-close{background:none;border:none;border-radius:4px;color:#6c757d;cursor:pointer;padding:4px;transition:all .2s ease}.open-docs-rag-widget-modal-close:hover{background-color:#e9ecef;color:#495057}.open-docs-rag-widget-modal-disclaimer{background-color:#f1fdd2;border-bottom:1px solid #e9ecef;color:#495057;font-family:inherit;font-size:14px;line-height:1.4;padding:12px 20px}.open-docs-rag-widget-chat-container,.open-docs-rag-widget-modal-body{display:flex;flex:1;flex-direction:column;overflow:hidden}.open-docs-rag-widget-chat-container{gap:16px;padding:20px}.open-docs-rag-widget-chat-messages{display:flex;flex:1;flex-direction:column;gap:16px;max-height:400px;overflow-y:auto;padding-right:4px}.open-docs-rag-widget-chat-messages::-webkit-scrollbar{width:6px}.open-docs-rag-widget-chat-messages::-webkit-scrollbar-track{background:#f1f3f4;border-radius:3px}.open-docs-rag-widget-chat-messages::-webkit-scrollbar-thumb{background:#c1c1c1;border-radius:3px}.open-docs-rag-widget-chat-messages::-webkit-scrollbar-thumb:hover{background:#a8a8a8}.open-docs-rag-widget-message{display:flex;flex-direction:column;gap:8px}.open-docs-rag-widget-message--user{align-items:flex-end}.open-docs-rag-widget-message--assistant{align-items:flex-start}.open-docs-rag-widget-message-content{word-wrap:break-word;border-radius:16px;font-family:inherit;line-height:1.4;max-width:80%;padding:12px 16px;font-size:12px!important}.open-docs-rag-widget-message--user .open-docs-rag-widget-message-content{background-color:#007bff;border-bottom-right-radius:4px;color:#fff}.open-docs-rag-widget-message--assistant .open-docs-rag-widget-message-content{background-color:#f8f9fa;border:1px solid #e9ecef;border-bottom-left-radius:4px;color:#212529}.open-docs-rag-widget-message-sources{color:#6c757d;font-family:inherit;font-size:12px;margin-top:4px}.open-docs-rag-widget-message-sources .sources-title{font-size:12px;font-weight:600;color:#495057;margin-bottom:6px;text-transform:uppercase;letter-spacing:.4px}.open-docs-rag-widget-message-sources .sources-list{display:flex;flex-wrap:wrap;gap:6px}.open-docs-rag-widget-message-sources .source-pill{display:inline-flex;align-items:center;gap:6px;background:#fff;border:1px solid #dee2e6;border-radius:999px;padding:6px 10px;color:#495057;text-decoration:none;transition:all .15s ease;box-shadow:0 1px 2px rgba(0,0,0,.04)}.open-docs-rag-widget-message-sources .source-pill:hover{background:#f8f9fa;border-color:#adb5bd;transform:translateY(-1px);box-shadow:0 2px 6px rgba(0,0,0,.06)}.open-docs-rag-widget-message-sources .source-index{background:#1d4716;color:#fff;border-radius:999px;display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;font-size:11px;font-weight:700}.open-docs-rag-widget-message-sources .source-host{font-weight:600}.open-docs-rag-widget-message-sources .source-sep{opacity:.5}.open-docs-rag-widget-message-sources .source-open-icon{opacity:.6}.typing-indicator{display:inline-flex;gap:4px;align-items:center;margin-left:6px}.typing-indicator .dot{width:6px;height:6px;background-color:#007bff;border-radius:50%;display:inline-block;animation:typing-bounce 1.2s infinite ease-in-out}.typing-indicator .dot:nth-child(2){animation-delay:0.15s}.typing-indicator .dot:nth-child(3){animation-delay:0.3s}@keyframes typing-bounce{0%,80%,100%{transform:scale(0.6);opacity:.4}40%{transform:scale(1);opacity:1}}.open-docs-rag-widget-example-questions{margin-bottom:16px}.open-docs-rag-widget-example-questions-title{color:#495057;font-family:inherit;font-size:14px;font-weight:600;letter-spacing:.5px;margin:0 0 12px;text-transform:uppercase}.open-docs-rag-widget-example-questions-grid{display:flex;flex-direction:column;gap:8px}.open-docs-rag-widget-example-question-btn{background:#fff;border:1px solid #dee2e6;border-radius:8px;color:#495057;cursor:pointer;font-family:inherit;font-size:14px;line-height:1.4;padding:12px 16px;text-align:left;transition:all .2s ease}.open-docs-rag-widget-example-question-btn:hover{background-color:#f8f9fa;border-color:#adb5bd;box-shadow:0 2px 8px rgba(0,0,0,.1);transform:translateY(-1px)}.open-docs-rag-widget-input-container{align-items:center;background-color:#f8f9fa;border:1px solid #e9ecef;border-radius:8px;display:flex;gap:8px;padding:0px 6px}.open-docs-rag-widget-input{background:none;border:none;color:#212529;flex:1;font-family:inherit;font-size:14px;outline:none;padding:8px 0}.open-docs-rag-widget-input::placeholder{color:#6c757d}.open-docs-rag-widget-send-button{align-items:center;background:none;border:none;border-radius:4px;color:#6c757d;cursor:pointer;display:flex;justify-content:center;padding:8px;transition:all .2s ease}.open-docs-rag-widget-send-button:hover{background-color:#e9ecef;color:#495057}.open-docs-rag-widget-send-button:disabled{cursor:not-allowed;opacity:.5}@media (max-width:768px){.open-docs-rag-widget-modal-content{margin:20px;max-height:90vh;width:95%}.open-docs-rag-widget-chat-messages{max-height:300px}.open-docs-rag-widget-chat-container,.open-docs-rag-widget-modal-header{padding:16px}}@keyframes fadeIn{0%{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}.open-docs-rag-widget-modal{animation:fadeIn .2s ease-out}.open-docs-rag-widget-button:focus,.open-docs-rag-widget-example-question-btn:focus,.open-docs-rag-widget-modal-close:focus,.open-docs-rag-widget-send-button:focus{outline:2px solid #007bff;outline-offset:2px}.open-docs-rag-widget-input:focus{box-shadow: none;outline:none}.open-docs-rag-widget-message-content .code-block-container{margin:16px 0;border-radius:8px;overflow:hidden;background-color:#1a202c;border:2px solid #2d3748;box-shadow:0 4px 12px rgba(0,0,0,.15);position:relative}.open-docs-rag-widget-message-content .code-block{background-color:#1a202c;color:#e2e8f0;padding:12px;margin:0;font-family:Monaco,Menlo,Ubuntu Mono,monospace;font-size:12px;line-height:1.6;overflow-x:auto;white-space:pre;border-top:1px solid #2d3748}.open-docs-rag-widget-message-content .inline-code{background-color:#f1f3f4;color:#d73a49;padding:2px 6px;border-radius:4px;border:1px solid #e1e4e8;font-family:Monaco,Menlo,Ubuntu Mono,monospace;font-size:12px;font-weight:500}`;
    document.head.appendChild(style);

    // Add explicit styling for code block language labels
    const languageStyle = document.createElement('style');
    languageStyle.setAttribute('data-widget', 'open-docs-rag-widget-language');
    languageStyle.textContent = `
      .open-docs-rag-widget-message-content .code-block-language{background-color:#2d3748;color:#a0aec0;padding:6px 10px;font-size:11px;font-weight:600;font-family:Monaco,Menlo,Ubuntu Mono,monospace;border-bottom:1px solid #4a5568;text-transform:uppercase;letter-spacing:.4px}
    `;
    document.head.appendChild(languageStyle);

    // Make sources smaller: 10px font-size and tighter padding
    const sourcesStyle = document.createElement('style');
    sourcesStyle.setAttribute('data-widget', 'open-docs-rag-widget-sources');
    sourcesStyle.textContent = `
      .open-docs-rag-widget-message-sources .source-pill{font-size:10px!important;padding:2px 6px!important;border-radius:6px!important;gap:4px!important}
      .open-docs-rag-widget-message-sources .sources-list{gap:4px!important}
    `;
    document.head.appendChild(sourcesStyle);
  }

  private createContainer(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'open-docs-rag-widget-container';
    container.innerHTML = `
      ${!this.config.buttonHide ? this.createButton() : ''}
      ${this.createModal()}
    `;
    return container;
  }

  private createButton(): string {
    const buttonStyle = this.getButtonStyles();
    const buttonImage = this.config.buttonImage || this.config.projectLogo;

    return `
      <button class="open-docs-rag-widget-button" style="${buttonStyle}">
        <img src="${buttonImage}" alt="Open AI Assistant" 
             style="height: ${this.config.buttonImageHeight}; width: ${this.config.buttonImageWidth};" />
        <span>Ask AI</span>
      </button>
    `;
  }

  private createModal(): string {
    const modalStyle = this.getModalStyles();
    const headerStyle = this.getHeaderStyles();
    const disclaimerStyle = this.getDisclaimerStyles();

    return `
      <div class="open-docs-rag-widget-modal" style="display: none;">
        <div class="open-docs-rag-widget-modal-overlay"></div>
        <div class="open-docs-rag-widget-modal-content" style="${modalStyle}">
          <div class="open-docs-rag-widget-modal-header" style="${headerStyle}">
            <div class="open-docs-rag-widget-modal-header-content">
              <img src="${this.config.projectLogo}" alt="${this.config.projectName}" 
                   class="open-docs-rag-widget-modal-logo" />
              <h2 class="open-docs-rag-widget-modal-title">${this.config.modalTitle || this.config.projectName + ' AI'}</h2>
            </div>
            <button class="open-docs-rag-widget-modal-close" aria-label="Close">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              </svg>
            </button>
          </div>
          
          ${
            this.config.modalDisclaimer
              ? `
            <div class="open-docs-rag-widget-modal-disclaimer" style="${disclaimerStyle}">
              ${this.config.modalDisclaimer}
            </div>
          `
              : ''
          }
          
          <div class="open-docs-rag-widget-modal-body">
            <div class="open-docs-rag-widget-chat-container">
              <div class="open-docs-rag-widget-chat-messages"></div>
              
              ${this.config.modalExampleQuestions ? this.createExampleQuestions() : ''}
              
              <div class="open-docs-rag-widget-input-container">
                <input type="text" 
                       class="open-docs-rag-widget-input" 
                       placeholder="${this.config.modalAskAiInputPlaceholder}"
                       aria-label="Ask a question" />
                <button class="open-docs-rag-widget-send-button" aria-label="Send">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" stroke="currentColor" stroke-width="2" 
                          stroke-linecap="round" stroke-linejoin="round"/>
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  private createExampleQuestions(): string {
    const questions = this.config.modalExampleQuestions!.split(',').map((q) => q.trim());
    const questionButtonStyle = this.getExampleQuestionButtonStyles();

    return `
      <div class="open-docs-rag-widget-example-questions">
        <h3 class="open-docs-rag-widget-example-questions-title">
          ${this.config.modalExampleQuestionsTitle}
        </h3>
        <div class="open-docs-rag-widget-example-questions-grid">
          ${questions
            .map(
              (question) => `
            <button class="open-docs-rag-widget-example-question-btn" 
                    style="${questionButtonStyle}"
                    data-question="${question}">
              ${question}
            </button>
          `
            )
            .join('')}
        </div>
      </div>
    `;
  }

  private setupEventListeners(): void {
    // Button click
    const button = this.container.querySelector('.open-docs-rag-widget-button');
    button?.addEventListener('click', () => this.open());

    // Modal close
    const closeBtn = this.container.querySelector('.open-docs-rag-widget-modal-close');
    const overlay = this.container.querySelector('.open-docs-rag-widget-modal-overlay');
    closeBtn?.addEventListener('click', () => this.close());
    overlay?.addEventListener('click', () => this.close());

    // Input handling
    const input = this.container.querySelector('.open-docs-rag-widget-input') as HTMLInputElement;
    const sendBtn = this.container.querySelector('.open-docs-rag-widget-send-button');

    input?.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.sendMessage(input.value);
      }
    });

    sendBtn?.addEventListener('click', () => {
      this.sendMessage(input.value);
    });

    // Example questions
    const questionBtns = this.container.querySelectorAll('.open-docs-rag-widget-example-question-btn');
    questionBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const question = btn.getAttribute('data-question');
        if (question) {
          this.sendMessage(question);
        }
      });
    });

    // Custom trigger elements - with delay to ensure DOM is ready
    if (this.config.modalOverrideOpenId) {
      // Add a small delay to ensure the custom trigger element exists
      setTimeout(() => {
        const customTrigger = document.getElementById(this.config.modalOverrideOpenId!);
        if (customTrigger) {
          console.log('OpenDocsRAGWidget: Custom trigger found:', this.config.modalOverrideOpenId);
          customTrigger.addEventListener('click', () => this.open());
        } else {
          console.warn('OpenDocsRAGWidget: Custom trigger element not found:', this.config.modalOverrideOpenId);
        }
      }, 100);
    }

    if (this.config.modalOverrideOpenClass) {
      setTimeout(() => {
        const customTriggers = document.querySelectorAll(`.${this.config.modalOverrideOpenClass}`);
        if (customTriggers.length > 0) {
          console.log(
            'OpenDocsRAGWidget: Custom trigger class found:',
            this.config.modalOverrideOpenClass,
            customTriggers.length
          );
          customTriggers.forEach((trigger) => {
            trigger.addEventListener('click', () => this.open());
          });
        } else {
          console.warn('OpenDocsRAGWidget: Custom trigger class not found:', this.config.modalOverrideOpenClass);
        }
      }, 100);
    }

    // Keyboard shortcuts
    if (this.config.modalOpenOnCommandK) {
      document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
          e.preventDefault();
          this.open();
        }
      });
    }

    // ESC to close
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen) {
        this.close();
      }
    });
  }

  private async sendMessage(content: string): Promise<void> {
    if (!content.trim()) return;

    const input = this.container.querySelector('.open-docs-rag-widget-input') as HTMLInputElement;
    const sendBtn = this.container.querySelector('.open-docs-rag-widget-send-button') as HTMLButtonElement;

    input.value = '';
    sendBtn.disabled = true;

    // Add user message
    const userMessage: ChatMessage = {
      id: this.generateId(),
      content,
      role: 'user',
      timestamp: new Date()
    };
    this.messages.push(userMessage);
    this.renderMessages();

    // Hide example questions after first message
    const exampleQuestions = this.container.querySelector('.open-docs-rag-widget-example-questions') as HTMLElement;
    if (exampleQuestions) {
      exampleQuestions.style.display = 'none';
    }

    // Add streaming assistant message with typing indicator
    const assistantMessageId = this.generateId();
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      content: '',
      role: 'assistant',
      timestamp: new Date(),
      sources: undefined
    };
    this.messages.push(assistantMessage);
    this.renderStreamingMessage(assistantMessageId, '', true); // Show typing indicator

    try {
      await this.callAPIStream(content, assistantMessageId);
    } catch (error) {
      console.error('OpenDocsRAGWidget: API call failed', error);
      // Replace the streaming message with error message
      const messageIndex = this.messages.findIndex((msg) => msg.id === assistantMessageId);
      if (messageIndex !== -1) {
        this.messages[messageIndex] = {
          id: assistantMessageId,
          content: 'Sorry, I encountered an error while processing your request. Please try again.',
          role: 'assistant',
          timestamp: new Date()
        };
        this.renderMessages();
      }
    } finally {
      sendBtn.disabled = false;
    }
  }

  private async callAPIStream(message: string, messageId: string): Promise<void> {
    const response = await fetch(this.config.apiEndpoint!, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` })
      },
      body: JSON.stringify({
        question: message,
        num_results: 3
      })
    });

    if (!response.ok) {
      throw new Error(`API call failed: ${response.status}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let accumulatedContent = '';
    let sources: string[] = [];

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6); // Remove 'data: ' prefix

            if (data === '[DONE]') {
              // End of stream
              this.updateMessageInArray(messageId, accumulatedContent, sources);
              this.renderStreamingMessage(messageId, accumulatedContent, false, sources);
              return;
            }

            try {
              const parsed = JSON.parse(data);

              if (parsed.type === 'sources') {
                // capture sources but don't render them until complete
                sources = parsed.content || [];
                // Update internal state only; keep rendering content without sources
                this.updateMessageInArray(messageId, accumulatedContent, sources);
                this.renderStreamingMessage(messageId, accumulatedContent, true);
              } else if (parsed.type === 'answer_chunk') {
                accumulatedContent += parsed.content || '';
                // Update message content with accumulated text
                this.updateMessageInArray(messageId, accumulatedContent, sources);
                this.renderStreamingMessage(messageId, accumulatedContent, true);
              }
            } catch (parseError) {
              console.warn('Failed to parse SSE data:', data, parseError);
            }
          }
        }
      }

      // If we exit the loop without [DONE], finalize the message
      this.updateMessageInArray(messageId, accumulatedContent, sources);
      this.renderStreamingMessage(messageId, accumulatedContent, false, sources);
    } catch (error) {
      console.error('Stream reading error:', error);
      throw error;
    } finally {
      reader.releaseLock();
    }
  }

  private updateMessageInArray(messageId: string, content: string, sources?: string[]): void {
    const messageIndex = this.messages.findIndex((msg) => msg.id === messageId);
    if (messageIndex !== -1) {
      this.messages[messageIndex].content = content;
      if (sources !== undefined) {
        this.messages[messageIndex].sources = sources;
      }
    }
  }

  private renderStreamingMessage(messageId: string, content: string, isStreaming: boolean, sources?: string[]): void {
    const messagesContainer = this.container.querySelector('.open-docs-rag-widget-chat-messages');
    if (!messagesContainer) return;

    // Find existing message element or create if needed
    let messageElement = messagesContainer.querySelector(`[data-message-id="${messageId}"]`);

    if (!messageElement) {
      // Create new message element
      messageElement = document.createElement('div');
      messageElement.className = 'open-docs-rag-widget-message open-docs-rag-widget-message--assistant';
      messageElement.setAttribute('data-message-id', messageId);
      messagesContainer.appendChild(messageElement);
    }

    // Update message content with real-time formatting
    const displayContent = content || '';
    const formattedContent = this.formatMessageContent(displayContent, 'assistant');
    const typingIndicator = isStreaming
      ? '<span class="typing-indicator" aria-label="Assistant is typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>'
      : '';

    const sourcesHtml = !isStreaming ? this.getSourcesHtml(sources) : '';
    messageElement.innerHTML = `
      <div class="open-docs-rag-widget-message-content">
        ${formattedContent}${typingIndicator}
      </div>
      ${sourcesHtml}
    `;

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  private renderMessages(): void {
    const messagesContainer = this.container.querySelector('.open-docs-rag-widget-chat-messages');
    if (!messagesContainer) return;

    messagesContainer.innerHTML = this.messages
      .map(
        (message) => `
      <div class="open-docs-rag-widget-message open-docs-rag-widget-message--${message.role}">
        <div class="open-docs-rag-widget-message-content">
          ${this.formatMessageContent(message.content, message.role)}
        </div>
        ${this.getSourcesHtml(message.sources)}
      </div>
    `
      )
      .join('');

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  private getSourcesHtml(sources?: string[]): string {
    if (!sources || sources.length === 0) return '';

    const items = sources
      .map((url, index) => {
        let host = '';
        let path = '';
        try {
          const u = new URL(url);
          host = u.hostname.replace(/^www\./, '');
          path = u.pathname + u.search + u.hash || '/';
        } catch {
          // Fallback for invalid URLs
          host = 'link';
          path = url;
        }

        // Trim very long paths for display
        if (path.length > 60) {
          path = path.slice(0, 57) + '…';
        }

        return `
        <a class="source-pill" href="${url}" target="_blank" rel="noopener noreferrer">
          <span class="source-host">${url}</span>
        </a>
      `;
      })
      .join('');

    return `
      <div class="open-docs-rag-widget-message-sources">
        <div class="sources-title">Sources</div>
        <div class="sources-list">${items}</div>
      </div>
    `;
  }

  private formatMessageContent(content: string, role: 'user' | 'assistant'): string {
    // For user messages, just return the content as-is (no special formatting needed)
    if (role === 'user') {
      return content;
    }

    // For assistant messages, apply rich formatting
    let formatted = content;

    // Handle complete code blocks (```) - only format if complete
    const codeBlockPlaceholders: string[] = [];
    formatted = formatted.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, language, code) => {
      const lang = language || '';
      const placeholder = `__CODEBLOCK_${codeBlockPlaceholders.length}__`;
      codeBlockPlaceholders.push(`<div class="code-block-container">
        ${lang ? `<div class="code-block-language">${lang}</div>` : ''}
        <pre class="code-block"><code>${this.escapeHtml(code.trim())}</code></pre>
      </div>`);
      return placeholder;
    });

    // Handle incomplete code blocks during streaming (show as pre)
    formatted = formatted.replace(/```(\w+)?\n?([\s\S]*)$/g, (match, language, code) => {
      // Only format if this is truly incomplete (no closing ```)
      if (!content.includes('```' + (language || '') + '\n' + code + '```')) {
        const lang = language || '';
        return `<div class="code-block-container incomplete">
          ${lang ? `<div class="code-block-language">${lang}</div>` : ''}
          <pre class="code-block"><code>${this.escapeHtml(code)}</code></pre>
        </div>`;
      }
      return match;
    });

    // Handle complete inline code (`) - only format if complete
    const inlineCodePlaceholders: string[] = [];
    formatted = formatted.replace(/`([^`]+)`/g, (match, code) => {
      const placeholder = `__INLINECODE_${inlineCodePlaceholders.length}__`;
      inlineCodePlaceholders.push(`<code class="inline-code">${this.escapeHtml(code)}</code>`);
      return placeholder;
    });

    // Split into lines for better processing
    const lines = formatted.split('\n');
    const processedLines: string[] = [];

    for (let i = 0; i < lines.length; i++) {
      let line = lines[i];

      // Handle bullet points (•)
      if (line.trim().match(/^•\s+/)) {
        line = line.replace(/^(\s*)•\s+(.+)$/, '$1<li class="bullet-item">$2</li>');
      }
      // Handle numbered lists (1. 2. 3. etc.)
      else if (line.trim().match(/^\d+\.\s+/)) {
        line = line.replace(/^(\s*)(\d+)\.\s+(.+)$/, '$1<li class="numbered-item">$3</li>');
      }

      processedLines.push(line);
    }

    formatted = processedLines.join('\n');

    // Wrap consecutive bullet points in ul tags
    formatted = formatted.replace(/(<li class="bullet-item">.*?<\/li>\n?)+/g, (match) => {
      return `<ul class="bullet-list">${match}</ul>`;
    });

    // Wrap consecutive numbered items in ol tags
    formatted = formatted.replace(/(<li class="numbered-item">.*?<\/li>\n?)+/g, (match) => {
      return `<ol class="numbered-list">${match}</ol>`;
    });

    // Handle bold text (**text**)
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Handle italic text (*text*) - but not if it's already in bold
    formatted = formatted.replace(/(?<![\*])\*([^*]+)\*(?![\*])/g, '<em>$1</em>');

    // Convert line breaks to <br> tags
    formatted = formatted.replace(/\n/g, '<br>');

    // Restore code blocks
    codeBlockPlaceholders.forEach((code, index) => {
      formatted = formatted.replace(`__CODEBLOCK_${index}__`, code);
    });

    // Restore inline code
    inlineCodePlaceholders.forEach((code, index) => {
      formatted = formatted.replace(`__INLINECODE_${index}__`, code);
    });

    return formatted;
  }

  private escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  private generateId(): string {
    return Math.random().toString(36).substr(2, 9);
  }

  // Style generators
  private getButtonStyles(): string {
    return `
      height: ${this.config.buttonHeight};
      width: ${this.config.buttonWidth};
      font-size: ${this.config.buttonTextFontSize};
      font-family: ${this.config.fontFamily};
      background-color: ${this.config.projectColor};
    `;
  }

  private getModalStyles(): string {
    return `
      font-family: ${this.config.fontFamily};
    `;
  }

  private getHeaderStyles(): string {
    return `
      color: ${this.config.projectColor};
      font-size: ${this.config.modalTitleFontSize};
    `;
  }

  private getDisclaimerStyles(): string {
    return `
      background-color: ${this.config.modalDisclaimerBgColor};
      font-size: ${this.config.modalDisclaimerFontSize};
      color: ${this.config.modalDisclaimerTextColor};
    `;
  }

  private getExampleQuestionButtonStyles(): string {
    return `
      padding: ${this.config.exampleQuestionButtonPaddingY} ${this.config.exampleQuestionButtonPaddingX};
      border: ${this.config.exampleQuestionButtonBorder};
      color: ${this.config.exampleQuestionButtonTextColor};
      font-size: ${this.config.exampleQuestionButtonFontSize};
      height: ${this.config.exampleQuestionButtonHeight};
      width: ${this.config.exampleQuestionButtonWidth};
    `;
  }

  // Public API methods
  public open(): void {
    console.log('OpenDocsRAGWidget: Opening modal');
    console.trace('OpenDocsRAGWidget: Modal open call stack');
    this.modal = this.container.querySelector('.open-docs-rag-widget-modal');
    if (this.modal) {
      this.modal.style.display = 'flex';
      this.isOpen = true;

      // Focus the input
      const input = this.container.querySelector('.open-docs-rag-widget-input') as HTMLInputElement;
      setTimeout(() => input?.focus(), 100);
    }
  }

  public close(): void {
    if (this.modal) {
      this.modal.style.display = 'none';
      this.isOpen = false;
    }
  }

  public destroy(): void {
    this.container.remove();
  }

  public updateConfig(newConfig: Partial<WidgetConfig>): void {
    this.config = { ...this.config, ...newConfig };
    // Re-render the widget with new config
    this.container.innerHTML = `
      ${!this.config.buttonHide ? this.createButton() : ''}
      ${this.createModal()}
    `;
    this.setupEventListeners();
  }
}
