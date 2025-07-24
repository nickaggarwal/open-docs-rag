module.exports = {
  parser: '@typescript-eslint/parser',
  extends: ['eslint:recommended', '@typescript-eslint/recommended'],
  plugins: ['@typescript-eslint'],
  env: {
    browser: true,
    es2021: true,
    node: true
  },
  parserOptions: {
    ecmaVersion: 2021,
    sourceType: 'module'
  },
  rules: {
    '@typescript-eslint/no-unused-vars': 'warn',
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/no-explicit-any': 'warn',
    'prefer-const': 'error',
    'no-var': 'error'
  },
  ignorePatterns: ['dist/', 'node_modules/']
};
