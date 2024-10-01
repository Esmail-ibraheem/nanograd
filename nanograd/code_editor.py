import gradio as gr
import io
import sys

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

# Create the Gradio interface
interface = gr.Interface(
    fn=execute_code,  # Function to execute the code
    inputs=gr.Code(language="python", label="Python Code Editor"),  # Realistic code editor with Python syntax highlighting
    outputs="text",  # Output is displayed as text in the interface
    title="Python Code Executor",
    description="Write and execute Python code directly in the browser. Output will be displayed below."
)

# Launch the interface
interface.launch()
