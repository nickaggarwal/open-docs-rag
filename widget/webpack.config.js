const path = require('path');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';

  return {
    entry: './src/bundle.ts',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'open-docs-rag-widget.bundle.js',
      library: 'OpenDocsRAGWidget',
      libraryTarget: 'umd',
      clean: true
    },
    resolve: {
      extensions: ['.ts', '.js', '.css']
    },
    module: {
      rules: [
        {
          test: /\.ts$/,
          use: 'ts-loader',
          exclude: /node_modules/
        },
        {
          test: /\.css$/,
          use: [isProduction ? MiniCssExtractPlugin.loader : 'style-loader', 'css-loader']
        }
      ]
    },
    plugins: [
      ...(isProduction
        ? [
            new MiniCssExtractPlugin({
              filename: 'open-docs-rag-widget.bundle.css'
            })
          ]
        : []),
      new HtmlWebpackPlugin({
        template: './src/example.html',
        filename: 'example.html',
        inject: 'body'
      })
    ],
    devServer: {
      static: {
        directory: path.join(__dirname, 'dist')
      },
      compress: true,
      port: 3000,
      open: true
    },
    optimization: {
      minimize: isProduction
    }
  };
};
