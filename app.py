import os
import logging
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from google import genai

# 1. Configure logging to write to both terminal and app.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

# 2. Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    logging.error("GEMINI_API_KEY is missing from environment variables!")
else:
    logging.info("GEMINI_API_KEY loaded successfully.")

# 3. Initialize Flask app and Google GenAI client
app = Flask(__name__)
client = genai.Client(api_key=api_key)

# Modern Dark-Mode UI Template
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini AI Assistant</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-main: #0f172a;
            --bg-chat: #1e293b;
            --bg-user: #2563eb;
            --bg-bot: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .chat-container {
            width: 100%;
            max-width: 800px;
            height: 92vh;
            background-color: var(--bg-chat);
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }
        .chat-header {
            padding: 18px 24px;
            background-color: rgba(15, 23, 42, 0.8);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.15rem;
            font-weight: 600;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            background-color: #22c55e;
            border-radius: 50%;
            box-shadow: 0 0 8px #22c55e;
        }
        .clear-btn {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: 0.2s;
        }
        .clear-btn:hover { color: white; border-color: #64748b; }
        #chat-box {
            flex: 1;
            padding: 24px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 78%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
            font-size: 0.95rem;
            word-wrap: break-word;
        }
        .user-msg {
            background-color: var(--bg-user);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 2px;
        }
        .bot-msg {
            background-color: var(--bg-bot);
            color: var(--text-primary);
            align-self: flex-start;
            border-bottom-left-radius: 2px;
        }
        .bot-msg pre {
            background: #0f172a;
            padding: 10px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 8px 0;
        }
        .bot-msg code { font-family: monospace; font-size: 0.9em; }
        .bot-msg p { margin-bottom: 8px; }
        .bot-msg p:last-child { margin-bottom: 0; }
        .bot-msg ul, .bot-msg ol { margin-left: 20px; margin-bottom: 8px; }
        .typing {
            font-style: italic;
            color: var(--text-secondary);
            font-size: 0.85rem;
            align-self: flex-start;
            display: none;
        }
        .input-area {
            padding: 16px 24px;
            background-color: rgba(15, 23, 42, 0.8);
            border-top: 1px solid var(--border-color);
            display: flex;
            gap: 12px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-chat);
            color: white;
            font-size: 0.95rem;
            outline: none;
        }
        input[type="text"]:focus { border-color: var(--bg-user); }
        button#send-btn {
            padding: 12px 24px;
            background-color: var(--bg-user);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.2s;
        }
        button#send-btn:hover { background-color: #1d4ed8; }
        button#send-btn:disabled { background-color: #475569; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="header-title">
                <span class="status-dot"></span>
                <span>Gemini AI Agent</span>
            </div>
            <button class="clear-btn" onclick="clearChat()">Clear Chat</button>
        </div>
        <div id="chat-box">
            <div class="message bot-msg">Hello! I am your Gemini AI Assistant. How can I help you today?</div>
        </div>
        <div class="typing" id="typing-indicator" style="padding: 0 24px 8px;">Gemini is thinking...</div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Ask me anything..." onkeypress="if(event.key === 'Enter') sendMessage()">
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const inputField = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const typingIndicator = document.getElementById('typing-indicator');

        function clearChat() {
            chatBox.innerHTML = '<div class="message bot-msg">Chat cleared. How can I help you?</div>';
        }

        async function sendMessage() {
            const message = inputField.value.trim();
            if (!message) return;

            // Add user message
            const userDiv = document.createElement('div');
            userDiv.className = 'message user-msg';
            userDiv.textContent = message;
            chatBox.appendChild(userDiv);

            inputField.value = '';
            inputField.disabled = true;
            sendBtn.disabled = true;
            typingIndicator.style.display = 'block';
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                const data = await response.json();
                
                const botDiv = document.createElement('div');
                botDiv.className = 'message bot-msg';
                
                if (data.reply) {
                    botDiv.innerHTML = marked.parse(data.reply);
                } else {
                    botDiv.style.color = '#f87171';
                    botDiv.textContent = 'Error: ' + (data.error || 'Something went wrong');
                }
                chatBox.appendChild(botDiv);
            } catch (err) {
                const errDiv = document.createElement('div');
                errDiv.className = 'message bot-msg';
                errDiv.style.color = '#f87171';
                errDiv.textContent = 'Error: Could not connect to the server.';
                chatBox.appendChild(errDiv);
            } finally {
                typingIndicator.style.display = 'none';
                inputField.disabled = false;
                sendBtn.disabled = false;
                inputField.focus();
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    logging.info("Home route accessed.")
    return render_template_string(HOME_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        logging.info(f"Received user message: {user_message}")

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_message,
        )
        
        bot_reply = response.text
        logging.info("Successfully generated Gemini response.")
        
        return jsonify({"reply": bot_reply})

    except Exception as e:
        logging.error(f"Error occurred during generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)