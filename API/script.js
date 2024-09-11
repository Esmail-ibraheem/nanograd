const API_URL = 'http://localhost:8000';

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

    try {
        const response = await fetch(`${API_URL}/generate_image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
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
    } catch (error) {
        imageOutput.innerHTML = `<p>Error: ${error.message}</p>`;
    }
}

async function generateText() {
    const model = document.getElementById('ollamaModel').value;
    const prompt = document.getElementById('textPrompt').value;

    const textOutput = document.getElementById('textOutput');
    textOutput.innerHTML = '<p>Generating text...</p>';

    try {
        const response = await fetch(`${API_URL}/chat_with_ollama`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: model,
                prompt: prompt
            }),
        });

        const data = await response.json();
        textOutput.innerHTML = `<p>${data.response}</p>`;
    } catch (error) {
        textOutput.innerHTML = `<p>Error: ${error.message}</p>`;
    }
}

async function applyBlueprint() {
    const blueprint = document.getElementById('blueprintSelect').value;

    const blueprintOutput = document.getElementById('blueprintOutput');
    blueprintOutput.innerHTML = '<p>Applying blueprint...</p>';

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
    } catch (error) {
        blueprintOutput.innerHTML = `<p>Error: ${error.message}</p>`;
    }
}

async function tokenizeText() {
  const text = document.getElementById('tokenizeInput').value;

  const tokenizeOutput = document.getElementById('tokenizeOutput');
  tokenizeOutput.innerHTML = '<p>Tokenizing...</p>';

  try {
      const response = await fetch(`${API_URL}/tokenize`, {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: text }),  // Change this line
      });

      if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      tokenizeOutput.innerHTML = `<pre>${JSON.stringify(data.tokens, null, 2)}</pre>`;
  } catch (error) {
      tokenizeOutput.innerHTML = `<p>Error: ${error.message}</p>`;
      console.error('Error:', error);
  }
}


async function askArabicChatbot() {
  const question = document.getElementById('arabicQuestion').value;

  const arabicChatbotOutput = document.getElementById('arabicChatbotOutput');
  arabicChatbotOutput.innerHTML = '<p>Processing...</p>';

  try {
      const response = await fetch(`${API_URL}/chatbot_arabic`, {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify({ question: question }),  // Change this line
      });

      if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      arabicChatbotOutput.innerHTML = `<p>${data.response}</p>`;
  } catch (error) {
      arabicChatbotOutput.innerHTML = `<p>Error: ${error.message}</p>`;
      console.error('Error:', error);
  }
}
