from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import subprocess
import os
import random
import base64
import io

from nanograd.models.stable_diffusion import model_loader, pipeline
from nanograd.models.GPT.tokenizer import tokenize
from examples import ollama_prompted

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware




# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure devices
DEVICE = "cpu"
ALLOW_CUDA = False 
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
        "sd_prompts": [
            "A futuristic city skyline at dusk, flying cars, neon lights, cyberpunk style",
            "A bustling marketplace in a futuristic city, holograms, diverse crowd",
            "A serene park in a futuristic city with advanced technology blending with nature"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Describe a futuristic city that blends natural elements with advanced technology.",
            "Write about an advanced cityscape with unique technological elements.",
            "Imagine a futuristic metropolis where nature and technology harmoniously coexist."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
    },
    # ... (other blueprints)
}

class ImageGenerationRequest(BaseModel):
    prompt: str
    cfg_scale: float
    num_inference_steps: int
    sampler: str

class TextGenerationRequest(BaseModel):
    model_name: str
    prompt: str

class BlueprintRequest(BaseModel):
    blueprint_name: str

@app.post("/generate_image")
async def generate_image(request: ImageGenerationRequest):
    uncond_prompt = ""
    do_cfg = True
    input_image = None
    strength = 0.9
    seed = 42

    output_image = pipeline.generate(
        prompt=request.prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=request.cfg_scale,
        sampler_name=request.sampler,
        n_inference_steps=request.num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    output_image = Image.fromarray(output_image)
    
    # Convert PIL Image to base64 string
    buffered = io.BytesIO()
    output_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"image": img_str}

@app.post("/chat_with_ollama")
async def chat_with_ollama(request: TextGenerationRequest):
    command = ['ollama', 'run', request.model_name, request.prompt]
    result = subprocess.run(command, capture_output=True, text=True)
    return {"response": result.stdout}

@app.post("/apply_blueprint")
async def apply_blueprint(request: BlueprintRequest):
    if request.blueprint_name in blueprints:
        bp = blueprints[request.blueprint_name]
        sd_prompts = random.choice(bp["sd_prompts"])
        sd_cfg_scale = random.choice(bp["sd_cfg_scales"])
        sd_num_inference_steps = random.choice(bp["sd_num_inference_steps"])
        sd_sampler = random.choice(bp["sd_samplers"])
        ollama_prompts = random.choice(bp["ollama_prompts"])
        ollama_model = random.choice(bp["ollama_models"])
        return {
            "sd_prompt": sd_prompts,
            "sd_cfg_scale": sd_cfg_scale,
            "sd_num_inference_steps": sd_num_inference_steps,
            "sd_sampler": sd_sampler,
            "ollama_model": ollama_model,
            "ollama_prompt": ollama_prompts
        }
    raise HTTPException(status_code=404, detail="Blueprint not found")

@app.post("/tokenize")
async def tokenize_text(text: str):
    return {"tokens": tokenize(text)}

@app.post("/chatbot_arabic")
async def chatbot_arabic(question: str):
    response = ollama_prompted.run(question)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
