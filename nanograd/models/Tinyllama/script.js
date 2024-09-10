document.getElementById('model-form').addEventListener('submit', async function (event) {
    event.preventDefault();
  
    const checkpoint = document.getElementById('checkpoint').value;
    const temperature = parseFloat(document.getElementById('temperature').value);
    const steps = parseInt(document.getElementById('steps').value);
    const prompt = document.getElementById('prompt').value;
  
    const modelInput = {
      checkpoint: checkpoint,
      temperature: temperature,
      steps: steps,
      prompt: prompt
    };
  
    const response = await fetch('http://127.0.0.1:8000/run_model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(modelInput),
    });
  
    const result = await response.json();
  
    document.getElementById('output').textContent = result.result;
  });
  
