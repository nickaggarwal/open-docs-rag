# FAISS Vector Store Improvements for Code Block Indexing

## Overview

The FAISS vector store has been significantly enhanced to better handle code blocks and provide improved indexing and search capabilities for technical documentation. These improvements enable more accurate retrieval of code examples, function definitions, and programming-related content.

## Key Improvements

### 1. Enhanced Code Block Detection

**Programming Language Recognition**: Automatic detection of programming languages using pattern matching for:
- Python (functions, classes, imports)
- JavaScript/TypeScript (functions, variables, modules)
- Java (classes, methods, packages)
- C++ (headers, main functions, std namespace)
- Bash/Shell (scripts, commands, variables)
- SQL (queries, DDL statements)
- YAML/JSON (configuration files)
- Dockerfile (container definitions)
- HTML/CSS (markup and styling)

**Multiple Detection Patterns**:
- Markdown code blocks with language specification (```python)
- HTML `<pre>` and `<code>` tags with language classes
- Indented code blocks (4+ spaces or tabs)

### 2. Specialized Code Embeddings

**Context-Aware Embeddings**: Code blocks are embedded with surrounding context:
- Up to 200 characters before and after the code block
- Complete sentences for better context understanding
- Enhanced text format: "Context before: ... Code (language): ... Context after: ..."

**Rich Metadata for Code Blocks**:
- `content_type`: "code_block" or "mixed"
- `programming_language`: Detected language (python, javascript, etc.)
- `code_block_type`: Detection method (markdown_code_block, html_pre_block, indented_code_block)
- `raw_code`: Original code content without context
- `context_before`/`context_after`: Surrounding text for context
- `is_code_block`: Boolean flag for easy filtering

### 3. Improved Document Chunking

**Dual Document Strategy**: Each original document is processed into:
- **Main Document**: Original content with code block metadata
- **Code Block Documents**: Separate documents for each code block with enhanced context

**Benefits**:
- Code blocks maintain their integrity (no splitting across chunks)
- Better context preservation for code understanding
- Separate indexing allows for code-specific search optimization

### 4. Code-Aware Search Enhancement

**Auto-Detection of Code Queries**: Automatic identification of code-focused queries based on:
- Programming language keywords (python, javascript, etc.)
- Code-related terms (function, method, class, variable, etc.)
- Code patterns (parentheses, brackets, dots, underscores)
- Keyword density analysis (>20% code-related words)

**Intelligent Scoring**:
- **Code-focused queries**: Significant boost for code blocks (+0.4), language matching (+0.2)
- **Text-focused queries**: Preference for mixed content, slight penalty for pure code blocks
- **Function/class matching**: Enhanced scoring for code blocks with matching identifiers
- **Context relevance**: Boost for relevant surrounding text

**Result Grouping**: Smart grouping prevents overwhelming results with multiple code blocks from the same document:
- Groups by main document ID
- Selects best result per document
- Prioritizes content type based on query focus

### 5. Enhanced Metadata and Context

**Code Block Metadata**:
```json
{
  "content_type": "code_block",
  "programming_language": "python",
  "code_block_type": "markdown_code_block",
  "code_block_index": 0,
  "raw_code": "def hello_world():\n    print('Hello, World!')",
  "context_before": "Here's a simple example:",
  "context_after": "This function prints a greeting.",
  "is_code_block": true
}
```

**Document Metadata**:
```json
{
  "content_type": "mixed",
  "has_code_blocks": true,
  "code_block_count": 3
}
```

## Usage Examples

### Basic Search
```python
# Auto-detection of query type
results = await vector_store.search("python function example", k=5)
```

### Explicit Code-Focused Search
```python
# Force code-focused search
results = await vector_store.search("authentication middleware", k=5, code_focused=True)
```

### Text-Focused Search
```python
# Force text-focused search (avoid code blocks)
results = await vector_store.search("authentication concepts", k=5, code_focused=False)
```

### Adding Documents with Code Blocks
```python
document = {
    "text": """
# Authentication Guide

Here's how to implement authentication:

```python
def authenticate_user(username, password):
    # Check credentials
    if verify_credentials(username, password):
        return generate_token(username)
    return None
```

This function handles user authentication securely.
    """,
    "metadata": {
        "url": "https://docs.example.com/auth",
        "title": "Authentication Guide"
    }
}

doc_id = await vector_store.add_document(document)
```

## Search Result Enhancements

### Enhanced Scoring Factors

1. **Embedding Similarity**: Base similarity score from vector search
2. **Content Type Matching**: Boost/penalty based on query focus and content type
3. **Language Matching**: Additional boost for programming language matches
4. **Keyword Matching**: Exact term matches in content
5. **Code Symbol Matching**: Function/class/variable name matches in raw code
6. **Context Relevance**: Surrounding text relevance for code blocks
7. **Document Structure**: Heading matches and hierarchy levels

### Result Format

Each search result includes:
```json
{
  "text": "Enhanced text with context for code blocks",
  "metadata": {
    "url": "source_url",
    "title": "Document Title",
    "content_type": "code_block",
    "programming_language": "python",
    "raw_code": "actual code content",
    "context_before": "preceding context",
    "context_after": "following context"
  },
  "score": 0.95,
  "id": "doc_id_code_block_0"
}
```

## Performance Improvements

1. **Intelligent Retrieval**: Retrieve 4x requested results for better re-ranking
2. **Parallel Processing**: Batch embedding generation for multiple documents
3. **Efficient Grouping**: Prevent result duplication from same source document
4. **Smart Caching**: Embedding cache to avoid re-processing identical content

## Migration from Previous Version

The enhanced vector store is backward compatible:
- Existing indexes continue to work
- Previous documents are treated as "mixed" content type
- Enhanced features apply to newly added documents
- Search API maintains compatibility with optional new parameters

## Language Detection Patterns

The system recognizes patterns for each supported language:

- **Python**: `def`, `class`, `import`, `from...import`, `__name__ == "__main__"`
- **JavaScript**: `function`, `const`, `let`, `var`, `console.log`, `=>`
- **TypeScript**: `interface`, `type`, type annotations (`:string`, `:number`)
- **Java**: `public class`, `public static void main`, `System.out.println`
- **C++**: `#include`, `int main`, `std::`, `using namespace std`
- **Bash**: shebang (`#!/bin/bash`), variables (`$var`), commands (`echo`, `chmod`)
- **SQL**: `SELECT`, `FROM`, `WHERE`, `INSERT INTO`, `CREATE TABLE`
- **YAML**: key-value pairs, lists, version specifications
- **JSON**: object/array syntax, quoted keys and values
- **Dockerfile**: `FROM`, `RUN`, `COPY`, `WORKDIR`, `EXPOSE`

## Best Practices

1. **Include Context**: Ensure code blocks have explanatory text before/after
2. **Use Language Tags**: Specify language in markdown code blocks when possible
3. **Meaningful Queries**: Use specific function/class names in search queries
4. **Balance Results**: Use appropriate `k` values to get diverse results
5. **Query Types**: Let auto-detection work or explicitly set `code_focused` for specialized searches

This enhanced FAISS vector store provides a significant improvement in code block handling, making it ideal for technical documentation, API references, and programming tutorials.