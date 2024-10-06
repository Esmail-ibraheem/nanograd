import gradio as gr
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch 
import subprocess
import os  
import random
import io
import sys
import matplotlib.pyplot as plt  

from nanograd.models.stable_diffusion import model_loader, pipeline

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
tokenizer_vocab_path = Path("C:\\Users\\Esmail\\Desktop\\nanograd\\nanograd\\models\\stable_diffusion\\sd_data\\tokenizer_vocab.json")
tokenizer_merges_path = Path("C:\\Users\\Esmail\\Desktop\\nanograd\\nanograd\\models\\stable_diffusion\\sd_data\\tokenizer_merges.txt")
model_file = Path("C:\\Users\\Esmail\\Desktop\\nanograd\\nanograd\\models\\stable_diffusion\\sd_data\\v1-5-pruned-emaonly.ckpt")

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
    # Other blueprints with similar structure...
    "Nature & Poetry": {
        "sd_prompts": [
            "A peaceful mountain landscape at sunrise, photorealistic, serene",
            "A tranquil lake surrounded by autumn trees, soft light, misty atmosphere",
            "A hidden waterfall in a dense jungle, lush greenery, crystal clear water"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Write a short poem about a tranquil sunrise over the mountains.",
            "Describe the beauty of a hidden waterfall in a jungle.",
            "Compose a poetic reflection on the serenity of a lake at dawn."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
    },
    # Additional blueprints with multiple prompts...
    "Dreamscape": {
        "sd_prompts": [
            "A surreal dreamscape with floating islands and bioluminescent creatures",
            "An endless horizon of strange landscapes, blending day and night",
            "A fantastical world with floating rocks and neon-lit skies"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Describe a dreamlike world filled with wonder and mystery.",
            "Write about a place where time doesn't exist, only dreams.",
            "Create a story where reality and fantasy blur together."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
    },
    "Abstract Art": {
        "sd_prompts": [
            "Abstract painting with vibrant colors and dynamic shapes",
            "A digital artwork with chaotic patterns and bold contrasts",
            "Geometric abstraction with a focus on form and color"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Write a short description of an abstract painting.",
            "Describe a piece of modern art that defies traditional norms.",
            "Imagine a world where art is created by emotions, not hands."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
    },
    "Fashion Design": {
        "sd_prompts": [
            "A high-fashion model wearing a futuristic outfit, neon colors, catwalk pose",
            "A chic ensemble blending classic elegance with modern flair",
            "Avant-garde fashion with bold textures and unconventional shapes"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Describe a unique and innovative fashion design.",
            "Write about a new fashion trend inspired by nature.",
            "Imagine a clothing line that combines style with sustainability."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
    },
    "Food & Recipe": {
        "sd_prompts": [
            "Abstract painting with vibrant colors and dynamic shapes",
            "A digital artwork with chaotic patterns and bold contrasts",
            "Geometric abstraction with a focus on form and color"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Write a short description of an abstract painting.",
            "Describe a piece of modern art that defies traditional norms.",
            "Imagine a world where art is created by emotions, not hands."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
    },
    "Interior Design": {
        "sd_prompts": [
            "A modern living room with sleek furniture, minimalist design, and natural light",
            "A cozy study room with rich textures, warm colors, and elegant decor",
            "An open-plan kitchen with contemporary appliances and stylish finishes"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Describe an interior design that combines modern and classic elements.",
            "Write about a space that enhances productivity and relaxation through design.",
            "Imagine a luxurious interior design for a high-end apartment."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
    },
    "Historical Fiction": {
        "sd_prompts": [
            "A bustling Victorian-era street with horse-drawn carriages and period architecture",
            "A grand historical ballroom with opulent decor and elegantly dressed guests",
            "An ancient battlefield with detailed historical accuracy and dramatic scenery"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Describe a significant historical event as if it were a scene in a novel.",
            "Write about a character navigating the challenges of a historical setting.",
            "Imagine a historical figure interacting with modern technology."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
    },
    "Science Fiction": {
        "sd_prompts": [
            "A futuristic cityscape with flying cars, neon lights, and towering skyscrapers",
            "An alien planet with unique landscapes, strange flora, and advanced technology",
            "A space station with cutting-edge design and high-tech equipment"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Describe a futuristic world where technology has reshaped society.",
            "Write about an encounter with an alien civilization.",
            "Imagine a story set in a distant future with advanced technology and space exploration."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
    },
    "Character Design": {
        "sd_prompts": [
            "A detailed fantasy character with elaborate costumes and accessories",
            "A sci-fi hero with futuristic armor and high-tech gadgets",
            "A historical figure portrayed with accurate attire and realistic features"
        ],
        "sd_cfg_scales": [9, 8, 7],
        "sd_num_inference_steps": [60, 50, 45],
        "sd_samplers": ["ddpm", "k_euler_ancestral", "euler"],
        "ollama_prompts": [
            "Describe a unique character from a fantasy novel, focusing on their appearance and personality.",
            "Write about a futuristic character with advanced technology and a compelling backstory.",
            "Imagine a historical figure as a character in a modern setting."
        ],
        "ollama_models": ["llama3", "aya", "codellama"]
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
        sd_prompts = random.choice(bp["sd_prompts"])
        sd_cfg_scale = random.choice(bp["sd_cfg_scales"])
        sd_num_inference_steps = random.choice(bp["sd_num_inference_steps"])
        sd_sampler = random.choice(bp["sd_samplers"])
        ollama_prompts = random.choice(bp["ollama_prompts"])
        ollama_model = random.choice(bp["ollama_models"])
        return (
            sd_prompts, sd_cfg_scale, sd_num_inference_steps, sd_sampler, 
            ollama_model, ollama_prompts
        )
    return "", 7, 20, "ddpm", "aya", ""

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

def welcome(name):
    return f"Welcome to nanograd Engine, {name}!"

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to nanograd Engine!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""

def show_intro():
    return """
    ## Welcome to nanograd Engine!
    nanograd is your comprehensive tool for AI-powered engine your Unreal engine but for AI.
    - **20+ LLMs**: Leverage the Ollama model for flexible, multi-language text outputs.
    - **Stable Diffusion**: Generate stunning images using Stable Diffusion with advanced customization options.
    - **Vision Transformer**: your multimodal chatbot section, Arabic chatbot, upload images
    - **Voice To Text**: your multimodal chatbot section, Arabic chatbot, upload images
    - **Auto-Trainer using LLaMAFactory**: supporting more than 10+ models, dataset to train, and finetune.

    Click "Get Started" to begin using the interface!
    """

# Function to hide the intro popup
def dismiss_intro():
    return gr.update(visible=False), gr.update(visible=False)


import ollama

# Function to run the chatbot with user input and a customizable prompt
default_prompt = '''الان الموضوع كالتالي اريدك ان تجيب على اسئلتي و التالي سوف تكون عن اي موضوع متعلق بالطيران او السفر او شركة الطيران مثل اريد انا اقطع جواز سفر الى اين اذهب بالضبط من الشركة او اريد انا اقطع فيزه للسفر مثلا الى اسبانيا و هكذا دواليك , 
شروط الاجابه هي : 1- اولا حاول التحدث و كأنك موظف في شركة الطيران , 2- ثانيا حاول ان تجيب على الاسئله باللهجة المصرية , 3- ثالثا حاول ان تعطي حلول اخرى اذا لم تعجبني مثلا طريقة قطع الجواز مثل انه تقول لي اذهب الى كذا و كذا 
بالمختصر حاول ان تكون مساعدي الشخصي. شارة البدايه عندما اقول لك ابداء و انت ابداء بقول اهلا عزيزي المستخدم كيف يمكنني ان اساعدك هنا في شركة الطيران , طبعا تخيل ان شركة الطيران هذه يمنيه'''

# Define the run function to handle chatbot responses
def run(user_input, custom_prompt, tone, response_style, personality, response_language):
    # Construct the prompt based on user selections
    custom_prompt += f"\n\nTone: {tone}. Response Style: {response_style}. Personality: {personality}. Language: {response_language}."

    # Initialize the chat with the custom or default prompt
    messages = [{'role': 'user', 'content': custom_prompt}]

    # Add user input to the messages
    messages.append({'role': 'user', 'content': user_input})

    # Get the model response
    response = ollama.chat(model='aya', messages=messages)
    ai_response = response['message']['content']

    # Add the model response to the messages
    messages.append({'role': 'assistant', 'content': ai_response})

    return ai_response


def describe_image(image: Image.Image) -> str:
    # Placeholder logic: You can replace this with actual Vision Transformer logic
    return "This is a placeholder description for the uploaded image."



# Define a function to execute the code and capture output
def execute_code(code):
    # Create an environment to execute the code in
    local_env = {}
    # Redirect standard output to capture print statements
    output_capture = io.StringIO()
    sys.stdout = output_capture
    try:
        # Execute the code in the local environment
        exec(code, {}, local_env)
        # Get the output from the captured stdout
        output = output_capture.getvalue()
        if output.strip() == "":
            return "Code executed successfully, but no output was produced."
        return output
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        # Reset stdout to the default
        sys.stdout = sys.__stdout__


# Gradio interface
def gradio_interface():
    with gr.Blocks(theme='ParityError/Interstellar', js=js) as demo:
        gr.HTML(
                '<h3><center>Visit <a href="https://github.com/Esmail-ibraheem/nanograd-Engine" target="_blank">'
                "nanograd Ecosystem</a> for details.</center></h3>"
            )
        with gr.Row(visible=True) as intro_popup:
            intro_md = gr.Markdown(show_intro(), visible=True)
            dismiss_button = gr.Button("Get Started")
            dismiss_button.click(dismiss_intro, [], [intro_md, intro_popup])

        with gr.Tab("Stories"):
            with gr.Row():
                with gr.Column(scale=1): 
                    # Text Generation with Ollama
                    gr.Markdown("### Generate Text with Ollama")
                    ollama_model_name = gr.Dropdown(label="Select Ollama Model", choices=
                    ["aya", "llama3", "codellama", "gemma2", "qwen2.5"
                    "phi3.5", "mistral-small", "mistral-nemo","mistral",
                    "mixtral", "codegemma", "llava", "llama3", "gemma", "qwen",
                    "llama2", "nomic-embed-text", "deepseek-coder", "starcoder2",
                    "llava-llama3", "tinyllama", "codestral", "wizard-vicuna-uncensored"], value="aya")
                    ollama_prompts = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
                    ollama_output = gr.Textbox(label="Output", placeholder="Model output will appear here", interactive=True)
                    ollama_btn = gr.Button("Generate", variant="primary")
                    ollama_btn.click(fn=chat_with_ollama, inputs=[ollama_model_name, ollama_prompts], outputs=ollama_output)

                    image_folder = "C:\\Users\\Esmail\\Desktop\\nanograd\\nanograd\\models\\stable_diffusion\\output"

                    cheetahs = [
                        os.path.join(image_folder, "c.png"),
                        os.path.join(image_folder, "d.png"),
                        os.path.join(image_folder, "output.png"),
                        os.path.join(image_folder, "output_image.png"),
                        os.path.join(image_folder, "R.png"),
                        os.path.join(image_folder, "s.png"),
                        os.path.join(image_folder, "test.png"),
                        os.path.join(image_folder, "generated_image.png"),
                        os.path.join(image_folder, "llama_3.jpg"),
                        os.path.join(image_folder, "omniverse.jpg"),
                        os.path.join(image_folder, "realistic_cat_in_animation_style.png"),
                        os.path.join(image_folder, "realistic_panda_in_animation_style.png"),
                        os.path.join(image_folder, "realistic_small_panda_in_animation_style.png"),
                        os.path.join(image_folder, "4373d3fd-5442-4499-9a77-9da589c94a68.jpg"),
                    ]

                   
                    gr.Gallery(value=cheetahs, columns=4)

                    gr.Markdown("### GPT Checkpoints Management")
                    checkpoint_dropdown = gr.Dropdown(label="Select Checkpoint", choices=["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "microsoft/phi-2", "codellama/CodeLlama-13b-hf"
                                                                                          "codellama/CodeLlama-13b-Python-hf", "databricks/dolly-v2-3b", "garage-bAInd/Camel-Platypus2-13B",
                                                                                          "google/gemma-2-9b", "lmsys/longchat-13b-16k", "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-v0.1",
                                                                                          "tiiuae/falcon-180B", "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"], value="EleutherAI/gpt-neo-125M")
                    download_btn = gr.Button("Download Checkpoint", variant="primary")
                    checkpoint_status = gr.Textbox(label="Download Status", placeholder="Status will appear here", interactive=True)
                    download_btn.click(fn=download_checkpoint, inputs=checkpoint_dropdown, outputs=checkpoint_status)

                    gr.Markdown("### Install Ollama")
                    install_ollama_btn = gr.Button("Install Ollama", variant="primary")
                    installation_status = gr.Textbox(label="Installation Status", placeholder="Status will appear here", interactive=True)
                    install_ollama_btn.click(fn=install_ollama, outputs=installation_status)

                with gr.Column(scale=1):
                    gr.Markdown("### Stable Diffusion Image Generation")
                    
                    prompt_input = gr.Textbox(label="Prompt", placeholder="A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution")
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=20, value=7, step=1)
                    num_inference_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=100, value=20, step=5)
                    sampler = gr.Radio(label="Sampling Method", choices=["ddpm", "Euler a", "Euler", "LMS", "Heun", "DPM2 a", "PLMS"], value="ddpm")
                    generate_img_btn = gr.Button("Generate", variant="primary")
                    output_image = gr.Image(label="Output", show_label=False, height=700, width=750)

                    generate_img_btn.click(fn=generate_image, inputs=[prompt_input, cfg_scale, num_inference_steps, sampler], outputs=output_image)
                    # # Define the folder path where your images are stored
                   
            with gr.Tab("Blueprints"):
                with gr.Row():
                    blueprint_dropdown = gr.Dropdown(label="Select Blueprint", choices=list(blueprints.keys()), value=list(blueprints.keys())[0])
                    load_blueprint_btn = gr.Button("Load Blueprint", variant="primary")
                    
                    # Blueprint Outputs
                    sd_prompt_output = gr.Textbox(label="SD Prompt", interactive=True)
                    sd_cfg_output = gr.Slider(label="SD CFG Scale", minimum=1, maximum=20, step=1, interactive=True)
                    sd_steps_output = gr.Slider(label="SD Sampling Steps", minimum=10, maximum=100, step=5, interactive=True)
                    sd_sampler_output = gr.Radio(label="SD Sampler", choices=["ddpm", "Euler a", "Euler", "LMS", "Heun", "DPM2 a", "PLMS"], value="ddpm", interactive=True)
                    ollama_model_output = gr.Dropdown(label="Ollama Model", choices=["aya", "llama3.1", "codellama"], value="aya", interactive=True)
                    ollama_prompt_output = gr.Textbox(label="Ollama Prompt", interactive=True)

                    def load_blueprint(blueprint_name):
                        if blueprint_name in blueprints:
                            bp = blueprints[blueprint_name]
                            sd_prompts = random.choice(bp["sd_prompts"])
                            sd_cfg_scale = random.choice(bp["sd_cfg_scales"])
                            sd_num_inference_steps = random.choice(bp["sd_num_inference_steps"])
                            sd_sampler = random.choice(bp["sd_samplers"])
                            ollama_prompts = random.choice(bp["ollama_prompts"])
                            ollama_model = random.choice(bp["ollama_models"])
                            return (
                                sd_prompts, sd_cfg_scale, sd_num_inference_steps, sd_sampler, 
                                ollama_model, ollama_prompts
                            )
                        return "", 7, 20, "ddpm", "aya", ""

                    def apply_loaded_blueprint(prompt, cfg_scale, num_inference_steps, sampler, model, ollama_prompts):
                        return (
                            gr.update(value=prompt), 
                            gr.update(value=cfg_scale), 
                            gr.update(value=num_inference_steps), 
                            gr.update(value=sampler), 
                            gr.update(value=model), 
                            gr.update(value=ollama_prompts)
                        )

                    load_blueprint_btn.click(fn=load_blueprint, inputs=blueprint_dropdown, outputs=[sd_prompt_output, sd_cfg_output, sd_steps_output, sd_sampler_output, ollama_model_output, ollama_prompt_output])
                    load_blueprint_btn.click(fn=apply_loaded_blueprint, inputs=[sd_prompt_output, sd_cfg_output, sd_steps_output, sd_sampler_output, ollama_model_output, ollama_prompt_output], outputs=[prompt_input, cfg_scale, num_inference_steps, sampler, ollama_model_name, ollama_prompts])

        with gr.Tab("Chatbot-Prompts"):
            with gr.Row():
                with gr.Column(scale=1):
                    from nanograd.models.GPT.bpe_tokenizer import tokenize
                    gr.Markdown("<h1><center>BPE Tokenizer</h1></center>")
                    text_input = gr.Textbox(label="Input Text", placeholder="Type or paste your text here...")

                    # Output components
                    output_json = gr.JSON(label="Tokenization Output")
                    output_table = gr.Dataframe(label="Tokenization Visualization", headers=["Token", "Token Bytes", "Token Translated", "Token Merged", "Token Index"])
                    
                    # Button to run the tokenizer
                    btn = gr.Button("Tokenize")
                    
                    def run_tokenizer(text):
                        result = tokenize(text)
                        # Return structured output for JSON and DataFrame
                        return result, result['Visualization Data']
                    
                    btn.click(run_tokenizer, inputs=text_input, outputs=[output_json, output_table])

                with gr.Column(scale=1):
                    gr.Markdown("<h1><center>Chatbot (لغة عربية)</h1></center>")
                    
                    user_input = gr.Textbox(lines=1, placeholder="Ask a question about travel or airlines")
                    
                    # Add customization fields for tone, style, and personality
                    tone = gr.Dropdown(choices=["Friendly", "Formal", "Professional"], label="Tone", value="Friendly")
                    response_style = gr.Dropdown(choices=["Concise", "Elaborate", "Creative"], label="Response Style", value="Elaborate")
                    personality = gr.Dropdown(choices=["Helpful Travel Agent", "Friendly Assistant", "Strict Professional"], label="Personality", value="Helpful Travel Agent")
                    response_language = gr.Dropdown(choices=["Egyptian Arabic", "Modern Standard Arabic", "Yemeni Arabic"], label="Response Language", value="Egyptian Arabic")

                    custom_prompt = gr.Code(value=default_prompt, language="python", label="Customize Prompt")

                    ai_output = gr.Textbox(label="Aya's response")
                    
                    submit_button = gr.Button("Submit")
                    
                    # Pass all new inputs to the run function
                    submit_button.click(run, inputs=[user_input, custom_prompt, tone, response_style, personality, response_language], outputs=ai_output)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("<h1><center>Vision Transformer Image Description</h1></center>")
                    
                    # Input for image upload
                    image_input = gr.Image(label="Upload an image", type="pil")
                    
                    # Output for image description
                    image_description_output = gr.Textbox(label="Image Description")

                    # Button to trigger the image description function
                    describe_button = gr.Button("Describe Image")

                    # Link button to function for generating image description
                    describe_button.click(describe_image, inputs=image_input, outputs=image_description_output)

        with gr.Tab("Trainer-LlamaFactory"):
            from nanograd.trainer.src.llamafactory.webui.interface import create_ui
            create_ui().queue()
        
        with gr.Tab("AutoCoder"):
            interface = gr.Interface(
            fn=execute_code,  # Function to execute the code
            inputs=gr.Code(language="python", label="Code Editor"),  # Realistic code editor with Python syntax highlighting
            outputs="text",  # Output is displayed as text in the interface
            title="Code Editor",
            description="Write and execute Python code directly in the browser. Output will be displayed below."
            )
            with gr.Column(scale=1): 
                    # Text Generation with Ollama
                    gr.Markdown("### AutoCoder")
                    ollama_model_name = gr.Dropdown(label="Select Ollama Model", choices=
                    ["codellama",  "codegemma",  "deepseek-coder", "starcoder2",
                    "tinyllama", "codestral"], value="codellama")
                    ollama_prompts = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
                    ollama_output = gr.Textbox(label="Output", placeholder="Model output will appear here", interactive=True)
                    ollama_btn = gr.Button("Generate", variant="primary")
                    ollama_btn.click(fn=chat_with_ollama, inputs=[ollama_model_name, ollama_prompts], outputs=ollama_output)
    
    demo.launch(server_name="0.0.0.0", server_port=7860)

# Run the Gradio interface
gradio_interface()
