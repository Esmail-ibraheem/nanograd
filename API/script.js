const API_URL = 'http://localhost:8000';
// const API_TOKEN = "secret";  // Ensure this matches the token in the backend


let userApiToken = "";  // Store the token globally (you can also store it in localStorage)

// Function to set the token
function setApiToken() {
    const tokenInput = document.getElementById('apiTokenInput').value;

    if (tokenInput.trim() !== "") {
        userApiToken = tokenInput;
        document.getElementById('apiTokenStatus').textContent = "Token set successfully!";
        document.getElementById('apiTokenStatus').style.color = "green";
    } else {
        document.getElementById('apiTokenStatus').textContent = "Please enter a valid token.";
        document.getElementById('apiTokenStatus').style.color = "red";
    }
}


// Log function to append messages to the log section
let isSidebarOpen = false;

// Log function to append messages to the log section
function logMessage(message) {
    const logOutput = document.getElementById('logOutput');
    const logEntry = document.createElement('p');
    logEntry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
    logOutput.appendChild(logEntry);
    logOutput.scrollTop = logOutput.scrollHeight; // Auto scroll to the bottom
}

// Toggle the log sidebar
document.getElementById('toggleLogsButton').addEventListener('click', function(event) {
    event.preventDefault(); // Prevent any default button behavior
    const logSidebar = document.getElementById('logSidebar');

    // Toggle the sidebar
    if (isSidebarOpen) {
        logSidebar.classList.remove('active');
    } else {
        logSidebar.classList.add('active');
    }
    
    isSidebarOpen = !isSidebarOpen; // Toggle the state
});
// Update slider values
document.getElementById('cfgScale').addEventListener('input', function() {
    document.getElementById('cfgScaleValue').textContent = this.value;
});

document.getElementById('numSteps').addEventListener('input', function() {
    document.getElementById('numStepsValue').textContent = this.value;
});

async function generateImage() {
    const prompt = document.getElementById('imagePrompt').value;
    const cfgScale = document.getElementById('cfgScale').value;
    const numSteps = document.getElementById('numSteps').value;
    const sampler = document.getElementById('sampler').value;

    const imageOutput = document.getElementById('imageOutput');
    imageOutput.innerHTML = '<p>Generating image...</p>';
    logMessage('Image generation started.');

    // Ensure the user has entered a token
    if (!userApiToken) {
        document.getElementById('apiTokenStatus').textContent = "Please enter your API token first.";
        document.getElementById('apiTokenStatus').style.color = "red";
        return;
    }

    try {
        const response = await fetch(`${API_URL}/generate_image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${userApiToken}`  // Use the user-provided token
            },
            body: JSON.stringify({
                prompt: prompt,
                cfg_scale: parseFloat(cfgScale),
                num_inference_steps: parseInt(numSteps),
                sampler: sampler
            }),
        });

        const data = await response.json();
        imageOutput.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Generated Image">`;
        logMessage('Image generated successfully.');
    } catch (error) {
        imageOutput.innerHTML = `<p>Error: ${error.message}</p>`;
        logMessage(`Error during image generation: ${error.message}`);
    }
}

async function generateText() {
    const model = document.getElementById('ollamaModel').value;
    const prompt = document.getElementById('textPrompt').value;

    const textOutput = document.getElementById('textOutput');
    textOutput.innerHTML = '<p>Generating text...</p>';
    logMessage('Text generation started.');

    // Ensure the user has entered a token
    if (!userApiToken) {
        document.getElementById('apiTokenStatus').textContent = "Please enter your API token first.";
        document.getElementById('apiTokenStatus').style.color = "red";
        return;
    }

    try {
        const response = await fetch(`${API_URL}/chat_with_ollama`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${userApiToken}`  // Use the user-provided token
            },
            body: JSON.stringify({
                model_name: model,
                prompt: prompt
            }),
        });

        const data = await response.json();
        textOutput.innerHTML = `<p>${data.response}</p>`;
        logMessage('Text generated successfully.');
    } catch (error) {
        textOutput.innerHTML = `<p>Error: ${error.message}</p>`;
        logMessage(`Error during text generation: ${error.message}`);
    }
}

async function applyBlueprint() {
    const blueprint = document.getElementById('blueprintSelect').value;

    const blueprintOutput = document.getElementById('blueprintOutput');
    blueprintOutput.innerHTML = '<p>Applying blueprint...</p>';
    logMessage('Applying blueprint started.');

    try {
        const response = await fetch(`${API_URL}/apply_blueprint`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                blueprint_name: blueprint
            }),
        });

        const data = await response.json();
        blueprintOutput.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        logMessage('Blueprint applied successfully.');
    } catch (error) {
        blueprintOutput.innerHTML = `<p>Error: ${error.message}</p>`;
        logMessage(`Error during blueprint application: ${error.message}`);
    }
}

async function tokenizeText() {
    const text = document.getElementById('tokenizeInput').value;

    const tokenizeOutput = document.getElementById('tokenizeOutput');
    tokenizeOutput.innerHTML = '<p>Tokenizing...</p>';
    logMessage('Tokenization started.');

    try {
        const response = await fetch(`${API_URL}/tokenize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        tokenizeOutput.innerHTML = `<pre>${JSON.stringify(data.tokens, null, 2)}</pre>`;
        logMessage('Tokenization completed successfully.');
    } catch (error) {
        tokenizeOutput.innerHTML = `<p>Error: ${error.message}</p>`;
        logMessage(`Error during tokenization: ${error.message}`);
    }
}

async function askArabicChatbot() {
    const question = document.getElementById('arabicQuestion').value;

    const arabicChatbotOutput = document.getElementById('arabicChatbotOutput');
    arabicChatbotOutput.innerHTML = '<p>Processing...</p>';
    logMessage('Arabic chatbot request started.');

    try {
        const response = await fetch(`${API_URL}/chatbot_arabic`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        arabicChatbotOutput.innerHTML = `<p>${data.response}</p>`;
        logMessage('Arabic chatbot response received.');
    } catch (error) {
        arabicChatbotOutput.innerHTML = `<p>Error: ${error.message}</p>`;
        logMessage(`Error during Arabic chatbot interaction: ${error.message}`);
    }
}
