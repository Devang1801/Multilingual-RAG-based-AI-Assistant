document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    
    // Generate a unique thread ID for the session
    const threadId = 'session_' + Math.random().toString(36).substr(2, 9);

    function appendMessage(sender, text, type="normal") {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('chat-message');
        if (type === "error") {
            msgDiv.classList.add('error');
        } else {
            msgDiv.classList.add(sender === 'user' ? 'user' : 'bot');
        }
        msgDiv.innerText = text;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        appendMessage('user', text);
        chatInput.value = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text, thread_id: threadId })
            });
            const data = await response.json();
            
            if (response.ok) {
                appendMessage('bot', data.answer);
            } else {
                appendMessage('bot', data.detail || data.error || 'An error occurred.', 'error');
            }
        } catch (error) {
            appendMessage('bot', 'Network error. Please try again later.', 'error');
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
