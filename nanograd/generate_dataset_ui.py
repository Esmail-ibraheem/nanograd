import gradio as gr
import torch
import os
from nanograd.generate_dataset import ModelConfig, Transformer, create_datasets, generate

# Load the model and datasets
work_dir = 'out'
device = 'cpu'

# Function to create the model and datasets based on user input
def setup_model(input_file):
    train_dataset, _ = create_datasets(input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                         n_layer=4, n_head=4, n_embd=64, n_embd2=64)
    model = Transformer(config)
    model.load_state_dict(torch.load(os.path.join(work_dir, 'model.pt'), map_location=device))
    model.to(device)
    model.eval()
    
    return model, train_dataset

# Function to generate names
def generate_names(input_file, num_names, temperature, top_k):
    model, train_dataset = setup_model(input_file)
    
    X_init = torch.zeros(num_names, 1, dtype=torch.long).to(device)
    top_k = top_k if top_k > 0 else None
    steps = train_dataset.get_output_length() - 1
    X_samp = generate(model, X_init, steps, temperature=temperature, top_k=top_k, do_sample=True).to('cpu')
    
    generated_names = []
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        name = train_dataset.decode(row)
        generated_names.append(name)
    
    return "\n".join(generated_names)

def gradio_interface(input_file, num_names, temperature, top_k):
    return generate_names(input_file, num_names, temperature, top_k)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Path to Input File", placeholder="Enter the path to the text file"),
        gr.Slider(1, 50, 10, step=1, label="Number of Names"),
        gr.Slider(0.1, 2.0, 1.0, step=0.1, label="Temperature"),
        gr.Slider(0, 100, 0, step=1, label="Top-K (0 for no limit)"),
    ],
    outputs="text",
    title="AI Name Generator",
    description="Generate unique names using a transformer model trained on a dataset of names.",
)

if __name__ == "__main__":
    iface.launch()
