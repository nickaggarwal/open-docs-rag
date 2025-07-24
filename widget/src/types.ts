export interface WidgetConfig {
  // Required parameters
  websiteId: string;
  projectName: string;
  projectColor: string;
  projectLogo: string;

  // Modal configuration
  modalTitle?: string;
  modalDisclaimer?: string;
  modalExampleQuestions?: string;
  modalExampleQuestionsTitle?: string;
  modalAskAiInputPlaceholder?: string;

  // Button configuration
  buttonHide?: boolean;
  buttonHeight?: string;
  buttonWidth?: string;
  buttonImageHeight?: string;
  buttonImageWidth?: string;
  buttonTextFontSize?: string;
  buttonImage?: string;

  // Styling
  fontFamily?: string;
  modalDisclaimerBgColor?: string;
  modalDisclaimerFontSize?: string;
  modalDisclaimerTextColor?: string;
  modalTitleFontSize?: string;
  exampleQuestionButtonPaddingX?: string;
  exampleQuestionButtonPaddingY?: string;
  exampleQuestionButtonBorder?: string;
  exampleQuestionButtonTextColor?: string;
  exampleQuestionButtonFontSize?: string;
  exampleQuestionButtonHeight?: string;
  exampleQuestionButtonWidth?: string;

  // Modal behavior
  modalOverrideOpenId?: string;
  modalOverrideOpenClass?: string;
  modalOpenByDefault?: boolean;
  modalOpenOnCommandK?: boolean;

  // API configuration
  apiEndpoint?: string;
  apiKey?: string;
}

export interface WidgetInstance {
  open: () => void;
  close: () => void;
  destroy: () => void;
  updateConfig: (config: Partial<WidgetConfig>) => void;
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  sources?: string[];
}

export interface ApiResponse {
  message: string;
  sources?: string[];
  error?: string;
}
