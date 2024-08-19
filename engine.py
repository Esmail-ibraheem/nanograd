import gradio as gr
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import subprocess
import os

from nanograd.models.stable_diffusion import model_loader, pipeline

# Configure devices
DEVICE = "cpu"
ALLOW_CUDA = True 
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Load Stable Diffusion model
tokenizer_vocab_path = Path("C:\\nanograd\\nanograd\\models\\stable_diffusion\\data\\tokenizer_vocab.json")
tokenizer_merges_path = Path("C:\\nanograd\\nanograd\\models\\stable_diffusion\\data\\tokenizer_merges.txt")
model_file = Path("C:\\nanograd\\nanograd\\models\\stable_diffusion\\data\\v1-5-pruned-emaonly.ckpt")

tokenizer = CLIPTokenizer(str(tokenizer_vocab_path), merges_file=str(tokenizer_merges_path))
models = model_loader.preload_models_from_standard_weights(str(model_file), DEVICE)

# Blueprints for image generation and text generation
blueprints = {
    "Visual Story": {
        "sd_prompt": "A futuristic city skyline at dusk, flying cars, neon lights, cyberpunk style",
        "sd_cfg_scale": 9,
        "sd_num_inference_steps": 60,
        "sd_sampler": "ddpm",
        "ollama_prompt": "Describe a futuristic city that blends natural elements with advanced technology.",
        "ollama_model": "llama3"
    },
    "Nature & Poetry": {
        "sd_prompt": "A peaceful mountain landscape at sunrise, photorealistic, serene",
        "sd_cfg_scale": 7,
        "sd_num_inference_steps": 40,
        "sd_sampler": "ddpm",
        "ollama_prompt": "Write a short poem about a tranquil sunrise over the mountains.",
        "ollama_model": "aya"
    }
}

# Define functions for each feature
def generate_image(prompt, cfg_scale, num_inference_steps, sampler):
    uncond_prompt = ""
    do_cfg = True
    input_image = None
    strength = 0.9
    seed = 42

    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    output_image = Image.fromarray(output_image)
    return output_image

def apply_blueprint(blueprint_name):
    if blueprint_name in blueprints:
        bp = blueprints[blueprint_name]
        return (
            bp["sd_prompt"], bp["sd_cfg_scale"], bp["sd_num_inference_steps"], bp["sd_sampler"], 
            bp["ollama_model"], bp["ollama_prompt"]
        )
    return "", 7, 20, "ddpm", "aya", ""  # Default values

def download_checkpoint(checkpoint):
    try:
        # Run the litgpt download command
        command = ["litgpt", "download", checkpoint]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()
        if process.returncode == 0:
            return f"Checkpoint '{checkpoint}' downloaded successfully.\n{output}"
        else:
            return f"Error downloading checkpoint '{checkpoint}':\n{error}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def chat_with_ollama(model_name, prompt):
    command = ['ollama', 'run', model_name, prompt]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

def install_ollama():
    try:
        # Command to install Ollama silently
        installer_path = "OllamaSetup.exe"
        if not os.path.exists(installer_path):
            # Download the installer if not already available
            subprocess.run(["curl", "-o", installer_path, "https://ollama.com/download/OllamaSetup.exe"], check=True)

        # Run the installer silently
        subprocess.run([installer_path, "/S"], check=True)
        return "Ollama installed successfully."
    except Exception as e:
        return f"Installation failed: {str(e)}"

# Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        with gr.Tab("nanograd Engine", interactive=True):
            with gr.Row():
                with gr.Column(scale=1): 
                    # Text Generation with Ollama
                    gr.Markdown("### Generate Text with Ollama")
                    ollama_model_name = gr.Dropdown(label="Select Ollama Model", choices=["aya", "llama3", "codellama"], value="aya")
                    ollama_prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
                    ollama_output = gr.Textbox(label="Output", placeholder="Model output will appear here", interactive=True)
                    ollama_btn = gr.Button("Generate", variant="primary")
                    ollama_btn.click(fn=chat_with_ollama, inputs=[ollama_model_name, ollama_prompt], outputs=ollama_output)

                    gr.Markdown("### GPT Checkpoints Management")
                    checkpoint_dropdown = gr.Dropdown(label="Select Checkpoint", choices=["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "microsoft/phi-2", "codellama/CodeLlama-13b-hf"], value="EleutherAI/gpt-neo-125M")
                    download_btn = gr.Button("Download Checkpoint", variant="primary")
                    checkpoint_status = gr.Textbox(label="Download Status", placeholder="Status will appear here", interactive=True)
                    download_btn.click(fn=download_checkpoint, inputs=checkpoint_dropdown, outputs=checkpoint_status)

                    gr.Markdown("### Install Ollama")
                    install_ollama_btn = gr.Button("Install Ollama", variant="primary")
                    installation_status = gr.Textbox(label="Installation Status", placeholder="Status will appear here", interactive=True)
                    install_ollama_btn.click(fn=install_ollama, outputs=installation_status)

                with gr.Column(scale=1):
                    gr.Markdown("### Stable Diffusion Image Generation")
                    
                    # Blueprint Dropdown
                    blueprint_dropdown = gr.Dropdown(label="Select Blueprint", choices=list(blueprints.keys()), value=list(blueprints.keys())[0])
                    prompt_input = gr.Textbox(label="Prompt", placeholder="A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution")
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=20, value=7, step=1)
                    num_inference_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=100, value=20, step=5)
                    sampler = gr.Radio(label="Sampling Method", choices=["ddpm", "Euler a", "Euler", "LMS", "Heun", "DPM2 a", "PLMS"], value="ddpm")
                    generate_img_btn = gr.Button("Generate", variant="primary")
                    output_image = gr.Image(label="Output", show_label=False, height=700, width=750)

                    # Update fields when a blueprint is selected
                    blueprint_dropdown.change(fn=apply_blueprint, inputs=blueprint_dropdown, outputs=[prompt_input, cfg_scale, num_inference_steps, sampler])

                    generate_img_btn.click(fn=generate_image, inputs=[prompt_input, cfg_scale, num_inference_steps, sampler], outputs=output_image)
        with gr.Tab("Blueprints"):
            with gr.Row():
                blueprint_dropdown = gr.Dropdown(label="Select Blueprint", choices=list(blueprints.keys()), value=list(blueprints.keys())[0])
                load_blueprint_btn = gr.Button("Load Blueprint", variant="primary")
                
                # Blueprint Outputs
                sd_prompt_output = gr.Textbox(label="Stable Diffusion Prompt", interactive=True)
                sd_cfg_output = gr.Slider(label="CFG Scale", minimum=1, maximum=20, value=7, step=1, interactive=True)
                sd_steps_output = gr.Slider(label="Sampling Steps", minimum=10, maximum=100, value=20, step=5, interactive=True)
                sd_sampler_output = gr.Radio(label="Sampling Method", choices=["ddpm", "Euler a", "Euler", "LMS", "Heun", "DPM2 a", "PLMS"], value="ddpm", interactive=True)

                # ollama_model_name = gr.Dropdown(label="Select Ollama Model", choices=["aya", "llama3", "codellama"], value="aya")
                ollama_model_output = gr.Textbox(label="Ollama Model", interactive=True)
                ollama_prompt_output = gr.Textbox(label="Ollama Prompt", interactive=True)

                load_blueprint_btn.click(fn=apply_blueprint, inputs=blueprint_dropdown, outputs=[sd_prompt_output, sd_cfg_output, sd_steps_output, sd_sampler_output, ollama_model_output, ollama_prompt_output])

        
    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
