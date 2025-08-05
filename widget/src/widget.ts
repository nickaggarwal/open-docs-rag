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
    style.textContent = `.open-docs-rag-widget-container{bottom:20px;font-family:inherit;position:fixed;right:20px;z-index:9999}.open-docs-rag-widget-button{align-items:center;background-color:#fff;border:2px solid #e9ecef;border-radius:50px;box-shadow:0 4px 12px rgba(0,0,0,.15);color:#495057;cursor:pointer;display:flex;font-family:inherit;font-weight:500;gap:8px;height:48px;justify-content:center;padding:0;transition:all .2s ease;width:120px}.open-docs-rag-widget-button:hover{background-color:#f8f9fa;border-color:#007bff;box-shadow:0 6px 20px rgba(0,123,255,.2);color:#007bff;transform:translateY(-2px)}.open-docs-rag-widget-button img{border-radius:4px}.open-docs-rag-widget-modal{align-items:center;display:flex;height:100%;justify-content:center;left:0;position:fixed;top:0;width:100%;z-index:10000}.open-docs-rag-widget-modal-overlay{background-color:rgba(0,0,0,.5);height:100%;left:0;position:absolute;top:0;width:100%}.open-docs-rag-widget-modal-content{background:#fff;border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,.3);display:flex;flex-direction:column;font-family:inherit;max-height:80vh;max-width:600px;overflow:hidden;position:relative;width:90%}.open-docs-rag-widget-modal-header{align-items:center;background-color:#f8f9fa;border-bottom:1px solid #e9ecef;display:flex;justify-content:space-between;padding:20px}.open-docs-rag-widget-modal-header-content{align-items:center;display:flex;gap:12px}.open-docs-rag-widget-modal-logo{border-radius:6px;height:32px;width:32px}.open-docs-rag-widget-modal-title{color:#212529;font-family:inherit;font-size:18px;font-weight:600;margin:0}.open-docs-rag-widget-modal-close{background:none;border:none;border-radius:4px;color:#6c757d;cursor:pointer;padding:4px;transition:all .2s ease}.open-docs-rag-widget-modal-close:hover{background-color:#e9ecef;color:#495057}.open-docs-rag-widget-modal-disclaimer{background-color:#f1fdd2;border-bottom:1px solid #e9ecef;color:#495057;font-family:inherit;font-size:14px;line-height:1.4;padding:12px 20px}.open-docs-rag-widget-chat-container,.open-docs-rag-widget-modal-body{display:flex;flex:1;flex-direction:column;overflow:hidden}.open-docs-rag-widget-chat-container{gap:16px;padding:20px}.open-docs-rag-widget-chat-messages{display:flex;flex:1;flex-direction:column;gap:16px;max-height:400px;overflow-y:auto;padding-right:4px}.open-docs-rag-widget-chat-messages::-webkit-scrollbar{width:6px}.open-docs-rag-widget-chat-messages::-webkit-scrollbar-track{background:#f1f3f4;border-radius:3px}.open-docs-rag-widget-chat-messages::-webkit-scrollbar-thumb{background:#c1c1c1;border-radius:3px}.open-docs-rag-widget-chat-messages::-webkit-scrollbar-thumb:hover{background:#a8a8a8}.open-docs-rag-widget-message{display:flex;flex-direction:column;gap:8px}.open-docs-rag-widget-message--user{align-items:flex-end}.open-docs-rag-widget-message--assistant{align-items:flex-start}.open-docs-rag-widget-message-content{word-wrap:break-word;border-radius:16px;font-family:inherit;line-height:1.4;max-width:80%;padding:12px 16px}.open-docs-rag-widget-message--user .open-docs-rag-widget-message-content{background-color:#007bff;border-bottom-right-radius:4px;color:#fff}.open-docs-rag-widget-message--assistant .open-docs-rag-widget-message-content{background-color:#f8f9fa;border:1px solid #e9ecef;border-bottom-left-radius:4px;color:#212529}.open-docs-rag-widget-message-sources{color:#6c757d;font-family:inherit;font-size:12px;margin-top:4px}.open-docs-rag-widget-message-sources a{color:#007bff;text-decoration:none}.open-docs-rag-widget-message-sources a:hover{text-decoration:underline}.typing-indicator{color:#007bff;animation:pulse 1.5s infinite;margin-left:4px}.open-docs-rag-widget-example-questions{margin-bottom:16px}.open-docs-rag-widget-example-questions-title{color:#495057;font-family:inherit;font-size:14px;font-weight:600;letter-spacing:.5px;margin:0 0 12px;text-transform:uppercase}.open-docs-rag-widget-example-questions-grid{display:flex;flex-direction:column;gap:8px}.open-docs-rag-widget-example-question-btn{background:#fff;border:1px solid #dee2e6;border-radius:8px;color:#495057;cursor:pointer;font-family:inherit;font-size:14px;line-height:1.4;padding:12px 16px;text-align:left;transition:all .2s ease}.open-docs-rag-widget-example-question-btn:hover{background-color:#f8f9fa;border-color:#adb5bd;box-shadow:0 2px 8px rgba(0,0,0,.1);transform:translateY(-1px)}.open-docs-rag-widget-input-container{align-items:center;background-color:#f8f9fa;border:1px solid #e9ecef;border-radius:8px;display:flex;gap:8px;padding:12px}.open-docs-rag-widget-input{background:none;border:none;color:#212529;flex:1;font-family:inherit;font-size:14px;outline:none;padding:8px 0}.open-docs-rag-widget-input::placeholder{color:#6c757d}.open-docs-rag-widget-send-button{align-items:center;background:none;border:none;border-radius:4px;color:#6c757d;cursor:pointer;display:flex;justify-content:center;padding:8px;transition:all .2s ease}.open-docs-rag-widget-send-button:hover{background-color:#e9ecef;color:#495057}.open-docs-rag-widget-send-button:disabled{cursor:not-allowed;opacity:.5}@media (max-width:768px){.open-docs-rag-widget-modal-content{margin:20px;max-height:90vh;width:95%}.open-docs-rag-widget-chat-messages{max-height:300px}.open-docs-rag-widget-chat-container,.open-docs-rag-widget-modal-header{padding:16px}}@keyframes fadeIn{0%{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}.open-docs-rag-widget-modal{animation:fadeIn .2s ease-out}.open-docs-rag-widget-button:focus,.open-docs-rag-widget-example-question-btn:focus,.open-docs-rag-widget-modal-close:focus,.open-docs-rag-widget-send-button:focus{outline:2px solid #007bff;outline-offset:2px}.open-docs-rag-widget-input:focus{box-shadow:0 0 0 2px rgba(0,123,255,.25);outline:none}`;
    document.head.appendChild(style);
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
                sources = parsed.content || [];
                // Update message with sources immediately
                this.updateMessageInArray(messageId, accumulatedContent, sources);
                this.renderStreamingMessage(messageId, accumulatedContent, true, sources);
              } else if (parsed.type === 'answer_chunk') {
                accumulatedContent += parsed.content || '';
                // Update message content with accumulated text
                this.updateMessageInArray(messageId, accumulatedContent, sources);
                this.renderStreamingMessage(messageId, accumulatedContent, true, sources);
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

    // Update message content
    const displayContent = content || '';
    const typingIndicator = isStreaming ? '<span class="typing-indicator">●</span>' : '';

    messageElement.innerHTML = `
      <div class="open-docs-rag-widget-message-content">
        ${displayContent}${typingIndicator}
      </div>
      ${
        sources && sources.length > 0
          ? `
        <div class="open-docs-rag-widget-message-sources">
          <small>Sources:</small>
          ${sources.map((source) => `<a href="${source}" target="_blank">${source}</a>`).join(', ')}
        </div>
      `
          : ''
      }
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
          ${message.content}
        </div>
        ${
          message.sources
            ? `
          <div class="open-docs-rag-widget-message-sources">
            <small>Sources:</small>
            ${message.sources.map((source) => `<a href="${source}" target="_blank">${source}</a>`).join(', ')}
          </div>
        `
            : ''
        }
      </div>
    `
      )
      .join('');

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
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
