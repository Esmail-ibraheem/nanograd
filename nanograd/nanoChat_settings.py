import gradio as gr
import requests
import json
import platform
import os

# Configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "llama3"

# Available models in your system
AVAILABLE_MODELS = [
    "llama3:latest",
    "aya:latest",
    "codellama:latest"
]

def get_hostname():
    if platform.system() == 'Windows':
        return os.getenv('COMPUTERNAME')
    return os.uname().nodename

def generate_text(prompt, server_url, model_name, temperature=0.7, max_tokens=1000):
    """Generate text using Ollama model"""
    generate_url = f"{server_url}/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    try:
        response = requests.post(
            generate_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True
        )
        response.raise_for_status()
        
        result = ""
        for line in response.iter_lines():
            if line:
                json_line = json.loads(line)
                result += json_line.get("response", "")
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

def pull_model(model_name, server_url):
    """Pull a new model from Ollama"""
    pull_url = f"{server_url}/api/pull"
    
    try:
        response = requests.post(
            pull_url,
            headers={"Content-Type": "application/json"},
            json={"name": model_name},
            stream=True
        )
        response.raise_for_status()
        
        status = ""
        for line in response.iter_lines():
            if line:
                json_line = json.loads(line)
                status += json_line.get("status", "") + "\n"
        return status
        
    except Exception as e:
        return f"Error pulling model: {str(e)}"

"""Build the Gradio interface"""
with gr.Blocks() as demo:
    gr.Markdown("# nanoChat-Advanced Settings")
    
    with gr.Row():
        server_url = gr.Textbox(
            label="Ollama Server URL",
            value=DEFAULT_OLLAMA_URL
        )
        model_name = gr.Dropdown(
            label="Model",
            choices=AVAILABLE_MODELS,
            value=DEFAULT_MODEL_NAME + ":latest",
            allow_custom_value=True
        )

    with gr.Tabs():
        # Text Generation Tab
        with gr.Tab("Text Generation"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        lines=4,
                        label="Input Prompt",
                        placeholder="Enter your prompt here..."
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=2000,
                        value=1000,
                        step=50,
                        label="Max Tokens"
                    )
                    generate_btn = gr.Button("Generate")
                
                with gr.Column():
                    output_text = gr.Textbox(
                        lines=10,
                        label="Generated Output"
                    )

        # Model Management Tab
        with gr.Tab("Model Management"):
            pull_btn = gr.Button("Pull Model")
            pull_status = gr.Textbox(
                lines=5,
                label="Pull Status"
            )

    # Set up event handlers
    generate_btn.click(
        generate_text,
        inputs=[prompt, server_url, model_name, temperature, max_tokens],
        outputs=output_text
    )
    
    pull_btn.click(
        pull_model,
        inputs=[model_name, server_url],
        outputs=pull_status
    )


if __name__ == "__main__":
    demo.launch()