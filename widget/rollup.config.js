const typescript = require('@rollup/plugin-typescript');
const { nodeResolve } = require('@rollup/plugin-node-resolve');
const commonjs = require('@rollup/plugin-commonjs');
const dts = require('rollup-plugin-dts');
const postcss = require('rollup-plugin-postcss');

module.exports = [
  // ES modules build with CSS extraction
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/index.esm.js',
      format: 'es'
    },
    plugins: [
      postcss({
        extract: 'styles.css', // Extract CSS to separate file
        inject: false, // Don't inject when extracting
        minimize: true
      }),
      nodeResolve(),
      commonjs(),
      typescript({
        tsconfig: './tsconfig.json',
        outputToFilesystem: true
      })
    ],
    external: ['react', 'react-dom'] // Add any peer dependencies here
  },

  // CommonJS build
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/index.js',
      format: 'cjs'
    },
    plugins: [
      postcss({
        extract: false, // Keep inline for CommonJS to avoid duplicate CSS files
        inject: function (varname, id) {
          return `
var style = document.createElement('style');
style.textContent = ${varname};
document.head.appendChild(style);`;
        },
        minimize: true
      }),
      nodeResolve(),
      commonjs(),
      typescript({
        tsconfig: './tsconfig.json',
        outputToFilesystem: false
      })
    ],
    external: ['react', 'react-dom'] // Add any peer dependencies here
  },
  // Type definitions
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/index.d.ts',
      format: 'es'
    },
    plugins: [dts.default()],
    external: [/\.css$/] // Exclude CSS from type generation
  },
  // CSS only build for manual import
  {
    input: 'src/styles.ts',
    output: {
      file: 'dist/styles-export.js',
      format: 'es'
    },
    plugins: [
      postcss({
        extract: 'styles-manual.css',
        minimize: true
      }),
      nodeResolve(),
      commonjs(),
      typescript({
        tsconfig: './tsconfig.json',
        outputToFilesystem: false
      })
    ]
  }
];
