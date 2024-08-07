def chat_with_models(model1, model2, initial_message, max_turns=5):
    """
    This function is designed to make two models talk to each other. 
    In my case, those models are Aya and Llama3. You can download 
    them to your computer by entering the command 'nanograd install ollama', 
    and then run the commands 'ollama run aya' and 'ollama run llama3', of course, 
    after downloading nanograd.
    """
    import ollama
    conversation1 = [{'role': 'user', 'content': initial_message}]
    conversation2 = []
    current_model = model1

    for _ in range(max_turns):
        response = ollama.chat(model=current_model, messages=conversation1 if current_model == model1 else conversation2)
        message_content = response['message']['content']
        
        print(f"{current_model} says: {message_content}")

        if current_model == model1:
            conversation1.append({'role': 'assistant', 'content': message_content})
            conversation2.append({'role': 'user', 'content': message_content})
            current_model = model2
        else:
            conversation2.append({'role': 'assistant', 'content': message_content})
            conversation1.append({'role': 'user', 'content': message_content})
            current_model = model1


def chat_models():
    # Initial message for the aya model
    initial_message = 'ابدأ المحادثة هنا.'

    # Start the conversation
    chat_with_models('aya', 'llama3', initial_message)

if __name__ == "__main__":
    chat_models()