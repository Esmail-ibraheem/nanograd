const apiBaseUrl = 'http://localhost:8000';  // Adjust this based on your backend deployment

// Install Ollama
function installOllama() {
    fetch(`${apiBaseUrl}/install/ollama`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("install-status").innerText = data.message;
        })
        .catch(error => {
            document.getElementById("install-status").innerText = "Error: " + error;
        });
}

// Run GPT Model
function runGPT() {
    fetch(`${apiBaseUrl}/run_gpt`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("gpt-status").innerText = data.message;
        })
        .catch(error => {
            document.getElementById("gpt-status").innerText = "Error: " + error;
        });
}

// Run LLaMA Model
function runLLaMA() {
    fetch(`${apiBaseUrl}/run_llama`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("llama-status").innerText = data.message;
        })
        .catch(error => {
            document.getElementById("llama-status").innerText = "Error: " + error;
        });
}

// Run Ollama Model
function runOllama() {
    const modelName = document.getElementById("ollama-model").value;
    fetch(`${apiBaseUrl}/run/${modelName}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("ollama-status").innerText = data.message;
        })
        .catch(error => {
            document.getElementById("ollama-status").innerText = "Error: " + error;
        });
}

// Run Stable Diffusion
function runStableDiffusion() {
    fetch(`${apiBaseUrl}/run_diffusion`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("sd-status").innerText = data.message;
        })
        .catch(error => {
            document.getElementById("sd-status").innerText = "Error: " + error;
        });
}
