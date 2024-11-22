import gradio as gr
import sqlite3
import subprocess
import requests
import json
from datetime import datetime

def initialize_database():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    # Create the session table
    c.execute('''
        CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Create the conversations table
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            model_name TEXT,
            user_input TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES session(id)
        )
    ''')
    conn.commit()
    conn.close()

# Function to get a list of available models from the command line
def get_available_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        models = [line.split()[0] for line in lines[1:]]  # Skip the header line and get the model name
        return models
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving model list: {e}")
        return []

# Function to generate a response from the selected model
def generate_response(model_name, user_input):
    url = f'http://localhost:11434/api/generate'
    headers = {'Content-Type': 'application/json'}
    data = {
        'model': model_name,
        'prompt': user_input,
        'stream': False
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json().get('response', '')
    else:
        print('Error communicating with the model.')
        return ''

# Function to save conversation to the database
def save_conversation(session_id, model_name, user_input, bot_response):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO conversations (session_id, model_name, user_input, bot_response)
        VALUES (?, ?, ?, ?)
    ''', (session_id, model_name, user_input, bot_response))
    conn.commit()
    conn.close()

# Function to create a new session and return its ID
def create_new_session(name):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO session (name)
        VALUES (?)
    ''', (name,))
    conn.commit()
    session_id = c.lastrowid
    conn.close()
    return session_id

# Function to load all sessions
def load_sessions():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT id, name FROM session ORDER BY timestamp DESC')
    sessions = c.fetchall()
    conn.close()
    return sessions

# Function to load conversation history by session ID
def load_conversation_history(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        SELECT user_input, bot_response, model_name
        FROM conversations
        WHERE session_id = ?
        ORDER BY timestamp DESC
    ''', (session_id,))
    conversation = c.fetchall()
    conn.close()
    return conversation

# Initialize the database
initialize_database()

# Load initial data
models = get_available_models()
if not models:
    models = ["No models found. Please check your Ollama installation."]
session_list = load_sessions()
session_names = [name for id, name in session_list]

# Define Gradio app
with gr.Blocks() as demo:
    gr.Markdown('# Ask nanoChat with Ollama models')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('## Select Model')
            model_dropdown = gr.Dropdown(label='Model', choices=models, value=models[0] if models else None)

            gr.Markdown('## Create New Session')
            session_name_input = gr.Textbox(label='Enter new session name')
            create_session_button = gr.Button('Create New Session')

            gr.Markdown('## Session History')
            session_radio = gr.Radio(label='Session History', choices=session_names)

        with gr.Column(scale=3):
            current_session_name_display = gr.Markdown('## Chat - No Session Selected')
            chat_display = gr.Chatbot()
            user_input = gr.Textbox(label='You:', lines=2)
            send_button = gr.Button('Send')

    # Initialize states
    session_list_state = gr.State(value=session_list)
    conversation_history_state = gr.State(value=[])
    current_session_id = gr.State(value=None)
    current_session_name = gr.State(value='')

    # Function to add a new session
    def add_new_session(session_name, session_list_state):
        if session_name:
            session_id = create_new_session(session_name)
            # Update session_list_state
            session_list_state.append((session_id, session_name))
            # Update session names
            session_names = [name for id, name in session_list_state]
            # Update current session display
            current_session_name_display_value = gr.update(value=f'## Chat - {session_name}')
            return (
                gr.update(choices=session_names, value=session_name),
                session_list_state,
                session_id,
                session_name,
                [],
                current_session_name_display_value,
                gr.update(value=''),
                gr.update(value=[])
            )
        else:
            return (
                gr.update(),
                session_list_state,
                None,
                '',
                [],
                gr.update(value='## Chat - No Session Selected'),
                gr.update(value=''),
                gr.update(value=[])
            )

    # Function to select a session
    def select_session(session_name, session_list_state):
        session_id = None
        for id, name in session_list_state:
            if name == session_name:
                session_id = id
                break
        if session_id:
            conversation = load_conversation_history(session_id)
            messages = []
            for user_msg, bot_msg, model_name in reversed(conversation):
                messages.insert(0, (user_msg, f"{model_name}: {bot_msg}"))
            # Update current session display
            current_session_name_display_value = gr.update(value=f'## Chat - {session_name}')
            return (
                messages,
                session_id,
                session_name,
                messages,
                gr.update(value=''),
                current_session_name_display_value
            )
        else:
            return (
                [],
                None,
                '',
                [],
                gr.update(value=''),
                gr.update(value='## Chat - No Session Selected')
            )

    # Function to send a message
    def send_message(user_message, model_name, current_session_id, conversation_history_state):
        if user_message and current_session_id:
            bot_response = generate_response(model_name, user_message)
            if bot_response:
                save_conversation(current_session_id, model_name, user_message, bot_response)
                # Append to conversation_history_state
                conversation_history_state.append((user_message, f"{model_name}: {bot_response}"))
                return conversation_history_state, '', conversation_history_state
        return conversation_history_state, '', conversation_history_state

    # Bind functions to events
    create_session_button.click(
        fn=add_new_session,
        inputs=[session_name_input, session_list_state],
        outputs=[
            session_radio,  # Updated from session_dropdown
            session_list_state,
            current_session_id,
            current_session_name,
            conversation_history_state,
            current_session_name_display,
            user_input,
            chat_display
        ]
    )

    session_radio.change(  # Updated from session_dropdown
        fn=select_session,
        inputs=[session_radio, session_list_state],
        outputs=[
            chat_display,
            current_session_id,
            current_session_name,
            conversation_history_state,
            user_input,
            current_session_name_display
        ]
    )

    send_button.click(
        fn=send_message,
        inputs=[user_input, model_dropdown, current_session_id, conversation_history_state],
        outputs=[chat_display, user_input, conversation_history_state]
    )

    # Clear user input on enter key
    user_input.submit(
        fn=send_message,
        inputs=[user_input, model_dropdown, current_session_id, conversation_history_state],
        outputs=[chat_display, user_input, conversation_history_state]
    )

if __name__ == "__main__":
    demo.launch()
