import { WidgetConfig } from './types';

/**
 * Extracts configuration from data attributes on the script tag
 */
export function parseConfigFromScript(): WidgetConfig | null {
  console.log('Looking for widget script tag...');

  // Try multiple selectors to find the widget script
  let scripts = document.querySelectorAll('script[src*="open-docs-rag-widget"]');
  console.log('Scripts with open-docs-rag-widget:', scripts.length);

  // Fallback: look for script with data-website-id (our required attribute)
  if (scripts.length === 0) {
    scripts = document.querySelectorAll('script[data-website-id]');
    console.log('Scripts with data-website-id:', scripts.length);
  }

  // Debug: log all script tags
  if (scripts.length === 0) {
    const allScripts = document.querySelectorAll('script');
    console.log('All script tags found:', allScripts.length);
    allScripts.forEach((script, index) => {
      console.log(`Script ${index}:`, {
        src: (script as HTMLScriptElement).src,
        hasDatasetWebsiteId: !!(script as HTMLScriptElement).dataset.websiteId,
        dataset: (script as HTMLScriptElement).dataset
      });
    });
  }

  if (scripts.length === 0) {
    console.warn('OpenDocsRAGWidget: No script tag found with open-docs-rag-widget src or data-website-id');
    return null;
  }

  const script = scripts[scripts.length - 1] as HTMLScriptElement;
  const dataset = script.dataset;

  // Check for required parameters
  const requiredParams = ['websiteId', 'projectName', 'projectColor', 'projectLogo'];
  for (const param of requiredParams) {
    if (!dataset[param]) {
      console.error(`OpenDocsRAGWidget: Missing required parameter: data-${camelToKebab(param)}`);
      return null;
    }
  }

  const config: WidgetConfig = {
    // Required parameters
    websiteId: dataset.websiteId!,
    projectName: dataset.projectName!,
    projectColor: dataset.projectColor!,
    projectLogo: dataset.projectLogo!,

    // Optional parameters
    modalTitle: dataset.modalTitle,
    modalDisclaimer: dataset.modalDisclaimer,
    modalExampleQuestions: dataset.modalExampleQuestions,
    modalExampleQuestionsTitle: dataset.modalExampleQuestionsTitle,
    modalAskAiInputPlaceholder: dataset.modalAskAiInputPlaceholder,

    buttonHide: dataset.buttonHide === 'true',
    buttonHeight: dataset.buttonHeight,
    buttonWidth: dataset.buttonWidth,
    buttonImageHeight: dataset.buttonImageHeight,
    buttonImageWidth: dataset.buttonImageWidth,
    buttonTextFontSize: dataset.buttonTextFontSize,
    buttonImage: dataset.buttonImage,

    fontFamily: dataset.fontFamily,
    modalDisclaimerBgColor: dataset.modalDisclaimerBgColor,
    modalDisclaimerFontSize: dataset.modalDisclaimerFontSize,
    modalDisclaimerTextColor: dataset.modalDisclaimerTextColor,
    modalTitleFontSize: dataset.modalTitleFontSize,
    exampleQuestionButtonPaddingX: dataset.exampleQuestionButtonPaddingX,
    exampleQuestionButtonPaddingY: dataset.exampleQuestionButtonPaddingY,
    exampleQuestionButtonBorder: dataset.exampleQuestionButtonBorder,
    exampleQuestionButtonTextColor: dataset.exampleQuestionButtonTextColor,
    exampleQuestionButtonFontSize: dataset.exampleQuestionButtonFontSize,
    exampleQuestionButtonHeight: dataset.exampleQuestionButtonHeight,
    exampleQuestionButtonWidth: dataset.exampleQuestionButtonWidth,

    modalOverrideOpenId: dataset.modalOverrideOpenId,
    modalOverrideOpenClass: dataset.modalOverrideOpenClass,
    modalOpenByDefault: dataset.modalOpenByDefault === 'true',
    modalOpenOnCommandK: dataset.modalOpenOnCommandK === 'true',

    apiEndpoint: dataset.apiEndpoint,
    apiKey: dataset.apiKey
  };

  return config;
}

/**
 * Converts camelCase to kebab-case
 */
function camelToKebab(str: string): string {
  return str.replace(/([a-z0-9]|(?=[A-Z]))([A-Z])/g, '$1-$2').toLowerCase();
}

/**
 * Default configuration values
 */
export const defaultConfig: Partial<WidgetConfig> = {
  modalTitle: 'AI Assistant',
  modalExampleQuestionsTitle: 'Suggested questions:',
  modalAskAiInputPlaceholder: 'Ask me a question...',
  buttonHeight: '54px',
  buttonWidth: '48px',
  buttonImageHeight: '20px',
  buttonImageWidth: '20px',
  buttonTextFontSize: '12px',
  modalDisclaimerBgColor: '#F1FDD2',
  modalDisclaimerFontSize: '14px',
  modalDisclaimerTextColor: '#3F4254',
  modalTitleFontSize: '20px',
  exampleQuestionButtonPaddingX: '16px',
  exampleQuestionButtonPaddingY: '8px',
  exampleQuestionButtonBorder: '1px solid #DEE2E6',
  exampleQuestionButtonTextColor: '#3F4254',
  exampleQuestionButtonFontSize: '16px',
  exampleQuestionButtonHeight: '100%',
  exampleQuestionButtonWidth: '100%',
  fontFamily: '-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif',
  apiEndpoint: 'http://100.28.126.254:8000/question'
};
