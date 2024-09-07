import gradio as gr
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset 
import pandas as pd
import json
import io 

# List of available models
MODEL_OPTIONS = [
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased",
    "gpt2"
]

def prepare_local_dataset(file):
    try:
        # Read the file into a pandas DataFrame
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        # Convert to JSON
        json_data = df.to_json(orient='records')
        # Save JSON to a file
        json_file_path = file.name.replace('.csv', '.json')
        with open(json_file_path, 'w') as f:
            f.write(json_data)
        # Load dataset from JSON file
        dataset = load_dataset('json', data_files={'train': json_file_path})
        return dataset['train'], json_file_path
    except Exception as e:
        return str(e), None

def prepare_hf_dataset(dataset_url):
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_url)
        return dataset['train'], dataset
    except Exception as e:
        return str(e), None

def train_model(model_name, output_dir, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def tokenize_function(examples):
        if 'text' not in examples:
            raise ValueError("The dataset must contain a 'text' column.")
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    
    trainer.train()
    return f"Model trained and saved to {output_dir}"

def finetune_model(model_name, output_dir, dataset):
    return train_model(model_name, output_dir, dataset)

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Hugging Face Model Training Interface", elem_id="header")
        
        dataset = gr.State(value=None)  # Initialize a State for the dataset
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Select Dataset Source")
                dataset_source = gr.Radio(
                    choices=["Upload Local CSV", "Hugging Face Dataset"], 
                    label="Choose Dataset Source", 
                    value="Upload Local CSV"
                )
                
                # Local CSV upload section
                with gr.Row(visible=True) as local_upload:
                    dataset_file = gr.File(label="Upload CSV File")
                    prepared_dataset_button = gr.Button("Prepare Local Dataset", variant="primary")
                    prepared_dataset_output = gr.Textbox(label="Dataset Output", placeholder="Dataset ready for training will appear here.")
                    
                    def prepare_and_set_local_dataset(file):
                        result, json_file_path = prepare_local_dataset(file)
                        if json_file_path:
                            return f"Dataset prepared and saved to {json_file_path}. Total samples: {len(result)}", result
                        else:
                            return f"Error: {result}", None
                    
                    prepared_dataset_button.click(
                        prepare_and_set_local_dataset, 
                        inputs=dataset_file, 
                        outputs=[prepared_dataset_output, dataset]
                    )
                
                # Hugging Face dataset section
                with gr.Row(visible=True) as hf_select:
                    hf_dataset_url = gr.Textbox(label="Hugging Face Dataset URL", placeholder="Enter dataset URL (e.g., 'imdb')")
                    prepare_hf_button = gr.Button("Load Hugging Face Dataset", variant="primary")
                    hf_dataset_output = gr.Textbox(label="Dataset Output", placeholder="Dataset ready for training will appear here.")
                    
                    def prepare_and_set_hf_dataset(dataset_url):
                        result, dataset = prepare_hf_dataset(dataset_url)
                        if dataset:
                            return f"Dataset loaded from Hugging Face. Total samples: {len(result)}", dataset
                        else:
                            return f"Error: {result}", None
                    
                    prepare_hf_button.click(
                        prepare_and_set_hf_dataset, 
                        inputs=hf_dataset_url, 
                        outputs=[hf_dataset_output, dataset]
                    )
                
                # Update visibility based on dataset source selection
                dataset_source.change(
                    lambda choice: local_upload.update(visible=choice == "Upload Local CSV") or hf_select.update(visible=choice == "Hugging Face Dataset"),
                    inputs=dataset_source
                )

            with gr.Column():
                gr.Markdown("### Model Training")
                model_name = gr.Dropdown(label="Select Model", choices=MODEL_OPTIONS, value=MODEL_OPTIONS[0])
                output_dir = gr.Textbox(label="Output Directory", placeholder="Enter directory for saving model")
                train_button = gr.Button("Train Model", variant="primary")
                train_output = gr.Textbox(label="Training Output", placeholder="Training results will appear here.")
                
                train_button.click(
                    train_model, 
                    inputs=[model_name, output_dir, dataset], 
                    outputs=train_output
                )

                gr.Markdown("### Fine-tune Model")
                model_name_finetune = gr.Dropdown(label="Select Model", choices=MODEL_OPTIONS, value=MODEL_OPTIONS[0])
                output_dir_finetune = gr.Textbox(label="Output Directory", placeholder="Enter directory for saving model")
                finetune_button = gr.Button("Fine-tune Model", variant="primary")
                finetune_output = gr.Textbox(label="Fine-tuning Output", placeholder="Fine-tuning results will appear here.")
                
                finetune_button.click(
                    finetune_model, 
                    inputs=[model_name_finetune, output_dir_finetune, dataset], 
                    outputs=finetune_output
                )

    demo.launch()

gradio_interface()
