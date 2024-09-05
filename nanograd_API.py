from fastapi import FastAPI, HTTPException
import os
import subprocess
import gradio as gr

app = FastAPI()

def install_ollama():
    script_path = os.path.join(os.path.dirname(__file__), 'ollama_install.sh')
    subprocess.run([script_path], check=True)

def download_llama():
    script_path = os.path.join(os.path.dirname(__file__), 'download.sh')
    subprocess.run([script_path], check=True)

def generate_dataset():
    script_path = os.path.join(os.path.dirname(__file__), 'generate_dataset.py')
    subprocess.run(['python', script_path], check=True)

def download_checkpoint(repo_id):
    command = ['litgpt', 'download', repo_id]
    subprocess.run(command, check=True)

def list_supported_models():
    command = ['litgpt', 'download', 'list']
    subprocess.run(command, check=True)

def pretrain_model(model_name, initial_checkpoint_dir, tokenizer_dir, out_dir, data_dir, train_data_path, lr_warmup_steps, lr):
    command = [
        'litgpt', 'pretrain', model_name,
        '--initial_checkpoint_dir', initial_checkpoint_dir,
        '--tokenizer_dir', tokenizer_dir,
        '--out_dir', out_dir,
        '--data', data_dir,
        '--data.train_data_path', train_data_path,
        '--train.lr_warmup_steps', str(lr_warmup_steps),
        '--optimizer.lr', str(lr),
    ]
    subprocess.run(command, check=True)

def run_gpt():
    script_path = os.path.join(os.path.dirname(__file__), 'models', 'GPT', 'inference_gpt.py')
    subprocess.run(['python', script_path], check=True)

def run_llama():
    script_path = os.path.join(os.path.dirname(__file__), 'models', 'llama', 'inference_llama.py')
    subprocess.run(['python', script_path], check=True)

def run_ollama(model_name):
    command = ['ollama', 'run', model_name]
    subprocess.run(command, check=True)

def run_stable_diffusion():
    script_path = os.path.join(os.path.dirname(__file__), 'models', 'stable_diffusion', 'sd_inference.py')
    subprocess.run(['python', script_path], check=True)

def run_stable_diffusion_with_interface():
    script_path = os.path.join(os.path.dirname(__file__), 'models', 'stable_diffusion', 'sd_gradio.py')
    subprocess.run(['python', script_path], check=True)

from fastapi.responses import FileResponse
import os

@app.get("/")
def read_root():
    return {"message": "Nanograd API is running!"}

# Route to serve the frontend
@app.get("/frontend")
def serve_frontend():
    return FileResponse(os.path.join(os.getcwd(), "index.html"))

@app.get("/install/{package}")
def api_install(package: str):
    if package == 'ollama':
        try:
            install_ollama()
            return {"message": f"{package} installed successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown package: {package}")

@app.get("/generate/{dataset}")
def api_generate_dataset(dataset: str):
    if dataset == 'dataset':
        try:
            generate_dataset()
            return {"message": "Dataset generated successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")

@app.get("/download/{type}/{repo_id}")
def api_download(type: str, repo_id: str = None):
    if type == 'llama':
        download_llama()
        return {"message": "LLaMA model downloaded successfully."}
    elif type == 'checkpoints':
        if repo_id:
            download_checkpoint(repo_id)
            return {"message": "Checkpoint downloaded successfully."}
        else:
            raise HTTPException(status_code=400, detail="repo_id is required for downloading checkpoints.")
    elif type == 'list':
        list_supported_models()
        return {"message": "Supported models listed successfully."}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {type}")

@app.post("/pretrain")
def api_pretrain_model(
    model_name: str, initial_checkpoint_dir: str, tokenizer_dir: str, out_dir: str,
    data_dir: str, train_data_path: str, lr_warmup_steps: int, lr: float):
    try:
        pretrain_model(model_name, initial_checkpoint_dir, tokenizer_dir, out_dir, data_dir, train_data_path, lr_warmup_steps, lr)
        return {"message": f"Model {model_name} pretraining initiated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/run_gpt")
def api_run_gpt():
    try:
        run_gpt()
        return {"message": "GPT model inference initiated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/run_llama")
def api_run_llama():
    try:
        run_llama()
        return {"message": "LLaMA model inference initiated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/run/{model_name}")
def api_run_ollama(model_name: str):
    try:
        run_ollama(model_name)
        return {"message": f"Ollama model {model_name} inference initiated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/run_diffusion")
def api_run_diffusion(interface: bool = False):
    try:
        if interface:
            run_stable_diffusion_with_interface()
        else:
            run_stable_diffusion()
        return {"message": "Stable Diffusion model run initiated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Launch Gradio Interface for Ollama models
@app.get("/gradio_interface/{model_name}")
def api_gradio_interface(model_name: str):
    try:
        def run_ollama_interface(prompt):
            command = ['ollama', 'run', model_name, prompt]
            result = subprocess.run(command, capture_output=True, text=True)
            return result.stdout

        iface = gr.Interface(
            fn=run_ollama_interface,
            inputs="text",
            outputs="text",
            title=f"Run {model_name} Ollama models using nanograd",
            description=f"Enter a prompt to generate text using the {model_name} model with Ollama."
        )
        iface.launch()
        return {"message": "Gradio interface launched."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

