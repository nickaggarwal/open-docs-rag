# SSE Streaming Widget Test Instructions

## 🎯 What's Updated

The widget has been updated to handle Server-Sent Events (SSE) streaming responses with the following features:

### ✅ Completed Features

1. **SSE Response Handling**: Widget now processes `text/event-stream` responses
2. **Real-time Streaming**: Text appears word-by-word as it streams from the server
3. **Sources Display**: Sources appear immediately when received
4. **Typing Indicator**: Shows animated dot (●) while streaming
5. **Send Button Disable**: Prevents multiple requests during streaming

## 🧪 How to Test

### Step 1: Start the Widget Test Server

```bash
cd /Users/piperguy/Projects/Inferless/open-docs-rag/widget
python3 -m http.server 8080
```

### Step 2: Open Test Page

Navigate to: `http://localhost:8080/debug-test.html`

### Step 3: Test SSE Streaming

1. Click "Test Button (Should trigger widget)"
2. Ask: "What is an input schema?"
3. Watch for:
   - ✅ Sources appear immediately
   - ✅ Text streams word by word
   - ✅ Typing indicator (●) while streaming
   - ✅ Send button disabled during stream
   - ✅ Complete response when done

## 🔍 Expected SSE Response Format

The widget now expects this SSE format from the backend:

```
data: {"type": "sources", "content": ["url1", "url2", "url3"]}

data: {"type": "answer_chunk", "content": "An"}

data: {"type": "answer_chunk", "content": " "input"}

data: {"type": "answer_chunk", "content": " schema""}

...more chunks...

data: [DONE]
```

## 🛠 Backend Requirements

To enable SSE streaming, the backend `/question` endpoint should:

1. Set `Content-Type: text/event-stream`
2. Send sources first: `data: {"type": "sources", "content": [...]}`
3. Stream answer chunks: `data: {"type": "answer_chunk", "content": "..."}`
4. End with: `data: [DONE]`

## 🐛 Debugging

### Browser Console Messages

- Look for: "OpenDocsRAGWidget: ..." initialization messages
- SSE parsing warnings/errors will appear if format is incorrect
- Stream reading errors indicate connection issues

### Widget Behavior

- **Sources appear immediately**: SSE sources working
- **Text streams smoothly**: Answer chunks working
- **Typing indicator animates**: CSS and streaming logic working
- **Button re-enables**: Stream completion working

## 📝 Implementation Details

### New Methods Added:

- `callAPIStream()`: Handles SSE responses
- `renderStreamingMessage()`: Updates UI during streaming
- `updateMessageInArray()`: Maintains message state

### CSS Added:

- `.typing-indicator`: Animated dot during streaming
- `@keyframes pulse`: Animation for typing indicator

### Error Handling:

- Malformed SSE data logged as warnings
- Stream interruption handled gracefully
- Network errors show user-friendly message
