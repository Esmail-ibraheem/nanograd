import argparse
import os
import subprocess
import gradio as gr

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
    command = [
        'litgpt', 'download',
        '--repo_id', repo_id
    ]
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
    script_path = os.path.join(os.path.dirname(__file__), 'C:\\Users\\Esmail\\Desktop\\nanograd\\nanograd\\models\\stable_diffusion\\sd_inference.py')
    subprocess.run(['python', script_path], check=True)

def gradio_interface(model_name):
    def run_ollama_interface(prompt):
        command = ['ollama', 'run', model_name, prompt]
        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout

    iface = gr.Interface(
        fn=run_ollama_interface,
        inputs="text",
        outputs="text",
        title=f"Run {model_name} with Ollama using nanograd",
        description=f"Enter a prompt to generate text using the {model_name} model with Ollama."
    )
    iface.launch()

def main():
    parser = argparse.ArgumentParser(description="Nanograd CLI")
    subparsers = parser.add_subparsers(dest='command', help="Sub-commands")

    install_parser = subparsers.add_parser('install', help="Install dependencies")
    install_parser.add_argument('package', type=str, help="Name of the package to install (e.g., 'ollama')")

    generate_parser = subparsers.add_parser('generate', help="Generate datasets")
    generate_parser.add_argument('dataset', type=str, help="Specify 'dataset' to generate dataset")

    download_parser = subparsers.add_parser('download', help="Download checkpoints or llama")
    download_parser.add_argument('type', type=str, choices=['checkpoints', 'llama'], help="Specify 'checkpoints' or 'llama' to download")
    download_parser.add_argument('repo_id', type=str, nargs='?', help="Hugging Face URL of the checkpoint (required for 'checkpoints')")

    pretrain_parser = subparsers.add_parser('pretrain', help="Pretrain a model")
    pretrain_parser.add_argument('model_name', type=str, help="Model name for pretraining")
    pretrain_parser.add_argument('--initial_checkpoint_dir', type=str, required=True, help="Initial checkpoint directory")
    pretrain_parser.add_argument('--tokenizer_dir', type=str, required=True, help="Tokenizer directory")
    pretrain_parser.add_argument('--out_dir', type=str, required=True, help="Output directory for new checkpoints")
    pretrain_parser.add_argument('--data', type=str, required=True, help="Data directory")
    pretrain_parser.add_argument('--data_train_data_path', type=str, required=True, help="Training data path")
    pretrain_parser.add_argument('--train_lr_warmup_steps', type=int, required=True, help="Learning rate warmup steps")
    pretrain_parser.add_argument('--optimizer_lr', type=float, required=True, help="Optimizer learning rate")

    run_gpt_parser = subparsers.add_parser('run_gpt', help="Run GPT inference")
    run_llama_parser = subparsers.add_parser('run_llama', help="Run LLaMA inference")
    
    run_stable_diffusion_parser = subparsers.add_parser('run_diffusion', help="Run Stable Diffusion")
    run_stable_diffusion_parser.add_argument('stable_diffusion', type=str, help="generate images with stable diffusion using nanograd")


    run_parser = subparsers.add_parser('run', help="Run model inference")
    run_parser.add_argument('model_name', type=str, help="Model name to run (e.g., 'llama2', 'aya')")
    run_parser.add_argument('--interface', action='store_true', help="Run with Gradio interface")

    args = parser.parse_args()

    if args.command == 'install':
        if args.package == 'ollama':
            install_ollama()
        else:
            print(f"Unknown package: {args.package}")
    elif args.command == 'generate':
        if args.dataset == 'dataset':
            generate_dataset()
        else:
            print(f"Unknown dataset: {args.dataset}")
    elif args.command == 'download':
        if args.type == 'llama':
            download_llama()
        elif args.type == 'checkpoints':
            if args.repo_id:
                download_checkpoint(args.repo_id)
            else:
                print("URL is required for downloading checkpoints.")
        else:
            print(f"Unknown or missing argument for download: {args.type}, {args.url}")
    elif args.command == 'pretrain':
        pretrain_model(
            args.model_name,
            args.initial_checkpoint_dir,
            args.tokenizer_dir,
            args.out_dir,
            args.data,
            args.data_train_data_path,
            args.train_lr_warmup_steps,
            args.optimizer_lr
        )
    elif args.command == 'run_gpt':
        run_gpt()
    elif args.command == 'run_llama':
        run_llama()
    elif args.command == 'run_diffusion':
        run_stable_diffusion()
    elif args.command == 'run':
        if args.interface:
            gradio_interface(args.model_name)
        else:
            run_ollama(args.model_name)
    else:
        parser.print_help()

nanograd = [
    "##     ##     ###     ##      ##  ########     ######   ########       ###       ########", 
    "###    ##   ##   ##   ###     ## ##      ##   ##   ##   ##     ##    ##   ##     ##      ##",
    "####   ##  ##     ##  ####    ## ##      ##  ##         ##     ##   ##      ##   ##       ##",
    "## ##  ## ##       ## ##  ##  ## ##      ##  ##  ####   ########   ##        ##  ##       ##",     
    "##   #### ########### ##    #### ##      ##  ##    ##   ##    ##   ############  ##       ##", 
    "##    ### ##       ## ##      ## ##      ##  ##    ##   ##    ##   ##        ##  ##      ##",
    "##     ## ##       ## ##      ##  ########    ######    ##     ##  ##        ##  ########"     
]

for line in nanograd:
    print(line)

if __name__ == '__main__':
    main()
