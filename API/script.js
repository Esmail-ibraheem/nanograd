const API_URL = 'http://localhost:8000';

let userApiToken = "";  // Store the token globally (you can also store it in localStorage)


async function generateToken() {
    const username = document.getElementById('usernameInput').value;
    const password = document.getElementById('passwordInput').value;

    if (username.trim() === "" || password.trim() === "") {
        document.getElementById('tokenGenerationStatus').textContent = "Please enter both username and password.";
        document.getElementById('tokenGenerationStatus').style.color = "red";
        return;
    }

    try {
        const response = await fetch(`${API_URL}/token`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
        });

        const data = await response.json();

        if (response.ok) {
            userApiToken = data.access_token;
            document.getElementById('tokenInput').value = userApiToken;
            document.getElementById('tokenStatus').textContent = "Token set successfully!";
            document.getElementById('tokenStatus').style.color = "green";
            document.getElementById('tokenGenerationStatus').textContent = "Token generated and set successfully!";
            document.getElementById('tokenGenerationStatus').style.color = "green";
            logMessage('Token generated and set successfully.');
        } else {
            document.getElementById('tokenGenerationStatus').textContent = data.detail || "Token generation failed.";
            document.getElementById('tokenGenerationStatus').style.color = "red";
            logMessage(`Token generation failed: ${data.detail || 'Unknown error'}`);
        }
    } catch (error) {
        document.getElementById('tokenGenerationStatus').textContent = `Error: ${error.message}`;
        document.getElementById('tokenGenerationStatus').style.color = "red";
        logMessage(`Error during token generation: ${error.message}`);
    }
}


function setApiToken() {
    const tokenInput = document.getElementById('tokenInput').value;

    if (tokenInput.trim() !== "") {
        userApiToken = tokenInput;
        document.getElementById('tokenStatus').textContent = "Token set successfully!";
        document.getElementById('tokenStatus').style.color = "green";
        logMessage('Token set successfully.');
    } else {
        document.getElementById('tokenStatus').textContent = "Please enter a valid token.";
        document.getElementById('tokenStatus').style.color = "red";
        logMessage('Failed to set token: Invalid token.');
    }
}

// Function to login and get the token
async function login() {
    const username = document.getElementById('usernameInput').value;
    const password = document.getElementById('passwordInput').value;

    if (username.trim() === "" || password.trim() === "") {
        document.getElementById('loginStatus').textContent = "Please enter both username and password.";
        document.getElementById('loginStatus').style.color = "red";
        return;
    }

    try {
        const response = await fetch(`${API_URL}/token`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
        });

        const data = await response.json();

        if (response.ok) {
            userApiToken = data.access_token;
            document.getElementById('loginStatus').textContent = "Login successful!";
            document.getElementById('loginStatus').style.color = "green";
            logMessage('User logged in successfully.');
        } else {
            document.getElementById('loginStatus').textContent = data.detail || "Login failed.";
            document.getElementById('loginStatus').style.color = "red";
            logMessage(`Login failed: ${data.detail || 'Unknown error'}`);
        }
    } catch (error) {
        document.getElementById('loginStatus').textContent = `Error: ${error.message}`;
        document.getElementById('loginStatus').style.color = "red";
        logMessage(`Error during login: ${error.message}`);
    }
}


// The rest of the JavaScript code remains the same as in the previous assistant's response, ensuring that all fetch requests include the JWT token in the Authorization header.


// Log function to append messages to the log section
let isSidebarOpen = false;

function logMessage(message) {
    const logOutput = document.getElementById('logOutput');
    const logEntry = document.createElement('p');
    logEntry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
    logOutput.appendChild(logEntry);
    logOutput.scrollTop = logOutput.scrollHeight; // Auto scroll to the bottom
}

// Toggle the log sidebar
document.getElementById('toggleLogsButton').addEventListener('click', function(event) {
    event.preventDefault();
    const logSidebar = document.getElementById('logSidebar');

    if (isSidebarOpen) {
        logSidebar.classList.remove('active');
    } else {
        logSidebar.classList.add('active');
    }

    isSidebarOpen = !isSidebarOpen;
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
    const cfgScale = parseFloat(document.getElementById('cfgScale').value);
    const numSteps = parseInt(document.getElementById('numSteps').value);
    const sampler = document.getElementById('sampler').value;

    const imageOutput = document.getElementById('imageOutput');
    imageOutput.innerHTML = '<p>Generating image...</p>';
    logMessage('Image generation started.');

    // Ensure the user is logged in
    if (!userApiToken) {
        document.getElementById('loginStatus').textContent = "Please login first.";
        document.getElementById('loginStatus').style.color = "red";
        return;
    }

    try {
        const response = await fetch(`${API_URL}/generate_image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${userApiToken}`
            },
            body: JSON.stringify({
                prompt: prompt,
                cfg_scale: cfgScale,
                num_inference_steps: numSteps,
                sampler: sampler
            }),
        });

        const data = await response.json();

        if (response.ok) {
            imageOutput.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Generated Image">`;
            logMessage('Image generated successfully.');
        } else {
            imageOutput.innerHTML = `<p>Error: ${data.detail || 'Unknown error'}</p>`;
            logMessage(`Error during image generation: ${data.detail || 'Unknown error'}`);
        }
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

    // Ensure the user is logged in
    if (!userApiToken) {
        document.getElementById('loginStatus').textContent = "Please login first.";
        document.getElementById('loginStatus').style.color = "red";
        return;
    }

    try {
        const response = await fetch(`${API_URL}/chat_with_ollama`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${userApiToken}`
            },
            body: JSON.stringify({
                model_name: model,
                prompt: prompt
            }),
        });

        const data = await response.json();

        if (response.ok) {
            textOutput.innerHTML = `<p>${data.response}</p>`;
            logMessage('Text generated successfully.');
        } else {
            textOutput.innerHTML = `<p>Error: ${data.detail || 'Unknown error'}</p>`;
            logMessage(`Error during text generation: ${data.detail || 'Unknown error'}`);
        }
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

    // Ensure the user is logged in
    if (!userApiToken) {
        document.getElementById('loginStatus').textContent = "Please login first.";
        document.getElementById('loginStatus').style.color = "red";
        return;
    }

    try {
        const response = await fetch(`${API_URL}/apply_blueprint`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${userApiToken}`
            },
            body: JSON.stringify({
                blueprint_name: blueprint
            }),
        });

        const data = await response.json();

        if (response.ok) {
            blueprintOutput.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            logMessage('Blueprint applied successfully.');
        } else {
            blueprintOutput.innerHTML = `<p>Error: ${data.detail || 'Unknown error'}</p>`;
            logMessage(`Error during blueprint application: ${data.detail || 'Unknown error'}`);
        }
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

    // Ensure the user is logged in
    if (!userApiToken) {
        document.getElementById('loginStatus').textContent = "Please login first.";
        document.getElementById('loginStatus').style.color = "red";
        return;
    }

    try {
        const response = await fetch(`${API_URL}/tokenize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${userApiToken}`
            },
            body: JSON.stringify({ text: text }),
        });

        const data = await response.json();

        if (response.ok) {
            tokenizeOutput.innerHTML = `<pre>${JSON.stringify(data.tokens, null, 2)}</pre>`;
            logMessage('Tokenization completed successfully.');
        } else {
            tokenizeOutput.innerHTML = `<p>Error: ${data.detail || 'Unknown error'}</p>`;
            logMessage(`Error during tokenization: ${data.detail || 'Unknown error'}`);
        }
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

    // Ensure the user is logged in
    if (!userApiToken) {
        document.getElementById('loginStatus').textContent = "Please login first.";
        document.getElementById('loginStatus').style.color = "red";
        return;
    }

    try {
        const response = await fetch(`${API_URL}/chatbot_arabic`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${userApiToken}`
            },
            body: JSON.stringify({ question: question }),
        });

        const data = await response.json();

        if (response.ok) {
            arabicChatbotOutput.innerHTML = `<p>${data.response}</p>`;
            logMessage('Arabic chatbot response received.');
        } else {
            arabicChatbotOutput.innerHTML = `<p>Error: ${data.detail || 'Unknown error'}</p>`;
            logMessage(`Error during Arabic chatbot interaction: ${data.detail || 'Unknown error'}`);
        }
    } catch (error) {
        arabicChatbotOutput.innerHTML = `<p>Error: ${error.message}</p>`;
        logMessage(`Error during Arabic chatbot interaction: ${error.message}`);
    }
}
