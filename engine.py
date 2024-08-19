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
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Load Stable Diffusion model
tokenizer_vocab_path = Path("C:\\nanograd\\nanograd\\models\\stable_diffusion\data\\tokenizer_vocab.json")
tokenizer_merges_path = Path("C:\\nanograd\\nanograd\\models\\stable_diffusion\data\\tokenizer_merges.txt")
model_file = Path("C:\\nanograd\\nanograd\\models\\stable_diffusion\\data\\v1-5-pruned-emaonly.ckpt")

tokenizer = CLIPTokenizer(str(tokenizer_vocab_path), merges_file=str(tokenizer_merges_path))
models = model_loader.preload_models_from_standard_weights(str(model_file), DEVICE)

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
        with gr.Tab("nanograd Engine"):
            with gr.Row():
                # Left Column: Text Generation with GPT and Ollama
                with gr.Column(scale=1):
                    

                    gr.Markdown("### Generate Text with Ollama")
                    ollama_model_name = gr.Dropdown(
                        label="Select Ollama Model", 
                        choices=["aya", "llama3", "codellama"], 
                        value="aya"
                    )
                    ollama_prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
                    ollama_output = gr.Textbox(label="Output", placeholder="Model output will appear here", interactive=False)
                    ollama_btn = gr.Button("Generate", variant="primary")

                    ollama_btn.click(fn=chat_with_ollama, inputs=[ollama_model_name, ollama_prompt], outputs=ollama_output)
                    
                    gr.Markdown("### GPT Checkpoints Management")
                    checkpoint_dropdown = gr.Dropdown(
                        label="Select Checkpoint", 
                        choices=["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "microsoft/phi-2", "codellama/CodeLlama-13b-hf"], 
                        value="EleutherAI/gpt-neo-125M"
                    )
                    download_btn = gr.Button("Download Checkpoint", variant="primary")
                    checkpoint_status = gr.Textbox(label="Download Status", placeholder="Status will appear here", interactive=False)

                    download_btn.click(fn=download_checkpoint, inputs=checkpoint_dropdown, outputs=checkpoint_status)

                    gr.Markdown("### Install Ollama")
                    install_ollama_btn = gr.Button("Install Ollama", variant="primary")
                    installation_status = gr.Textbox(label="Installation Status", placeholder="Status will appear here", interactive=False)

                    install_ollama_btn.click(fn=install_ollama, outputs=installation_status)

                # Right Column: Stable Diffusion
                with gr.Column(scale=1):
                    gr.Markdown("### Stable Diffusion Image Generation")
                    prompt_input = gr.Textbox(label="Prompt", placeholder="A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution")
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=20, value=7, step=1)
                    num_inference_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=100, value=20, step=5)
                    sampler = gr.Radio(label="Sampling Method", choices=["ddpm", "Euler a", "Euler", "LMS", "Heun", "DPM2 a", "PLMS"], value="ddpm")
                    generate_img_btn = gr.Button("Generate", variant="primary")
                    output_image = gr.Image(label="Output", show_label=False, height=700, width=750)

                    generate_img_btn.click(fn=generate_image, inputs=[prompt_input, cfg_scale, num_inference_steps, sampler], outputs=output_image)

    demo.launch()

if __name__ == "__main__":
    gradio_interface()
