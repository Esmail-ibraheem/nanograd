import gradio as gr

def run(user_input):
    import ollama
    # Initial prompt to set up the context
    initial_prompt = {
        'role': 'user',
        'content': (
            'الان الموضوع كالتالي اريدك ان تجيب على اسئلتي و التالي سوف تكون عن اي موضوع متعلق بالطيران او السفر او شركة الطيران مثل اريد انا اقطع جواز سفر الى اين اذهب بالضبط من الشركة او اريد انا اقطع فيزه للسفر مثلا الى اسبانيا و هكذا دواليك , '
            'شروط الاجابه هي : 1- اولا حاول التحدث و كأنك موظف في شركة الطيران , 2- ثانيا حاول ان تجيب على الاسئله باللهجة المصرية , 3- ثالثا حاول ان تعطي حلول اخرى اذا لم تعجبني مثلا طريقة قطع الجواز مثل انه تقول لي اذهب الى كذا و كذا '
            'بالمختصر حاول ان تكون مساعدي الشخصي. شارة البدايه عندما اقول لك ابداء و انت ابداء بقول اهلا عزيزي المستخدم كيف يمكنني ان اساعدك هنا في شركة الطيران , طبعا تخيل ان شركة الطيران هذه يمنيه'
        ),
    }

    # Initialize the chat with the initial prompt
    messages = [initial_prompt]

    # Add user input to the messages
    messages.append({
        'role': 'user',
        'content': user_input,
    })

    # Get the model response
    response = ollama.chat(model='aya', messages=messages)
    ai_response = response['message']['content']

    # Add the model response to the messages
    messages.append({
        'role': 'assistant',
        'content': ai_response,
    })

    return ai_response


iface = gr.Interface(
     fn=run,
     inputs=gr.Textbox(lines=1, placeholder="Ask a question about travel or airlines"),
     outputs=gr.Textbox(label="Aya's response"),
     title="Travel and Airlines Chatbot"
 )

 iface.launch()
