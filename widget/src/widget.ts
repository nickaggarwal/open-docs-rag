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
    this.container = this.createContainer();
    this.init();
  }

  private init(): void {
    document.body.appendChild(this.container);
    this.setupEventListeners();

    if (this.config.modalOpenByDefault) {
      this.open();
    }
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

    // Custom trigger elements
    if (this.config.modalOverrideOpenId) {
      const customTrigger = document.getElementById(this.config.modalOverrideOpenId);
      customTrigger?.addEventListener('click', () => this.open());
    }

    if (this.config.modalOverrideOpenClass) {
      const customTriggers = document.querySelectorAll(`.${this.config.modalOverrideOpenClass}`);
      customTriggers.forEach((trigger) => {
        trigger.addEventListener('click', () => this.open());
      });
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
    input.value = '';

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

    try {
      const response = await this.callAPI(content);
      const assistantMessage: ChatMessage = {
        id: this.generateId(),
        content: response.message,
        role: 'assistant',
        timestamp: new Date(),
        sources: response.sources
      };
      this.messages.push(assistantMessage);
      this.renderMessages();
    } catch (error) {
      console.error('OpenDocsRAGWidget: API call failed', error);
      const errorMessage: ChatMessage = {
        id: this.generateId(),
        content: 'Sorry, I encountered an error while processing your request. Please try again.',
        role: 'assistant',
        timestamp: new Date()
      };
      this.messages.push(errorMessage);
      this.renderMessages();
    }
  }

  private async callAPI(message: string): Promise<ApiResponse> {
    const response = await fetch(this.config.apiEndpoint!, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
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

    const result = await response.json();

    // Transform the response to match our expected format
    return {
      message: result.answer || result.response || result.message || 'No response from API',
      sources: result.sources || result.references || []
    };
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
