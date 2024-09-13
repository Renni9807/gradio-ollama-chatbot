import gradio as gr
import ollama
import traceback
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

# Initialize Ollama client
ollama_client = ollama.Client()

# Global variables
selected_model = None

def get_available_models():
    try:
        models = ollama_client.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        print(f"Error getting models: {e}")
        return ["Error fetching models"]

def set_model(model_name):
    global selected_model
    if model_name == "Error fetching models":
        return "Unable to set model. Please check Ollama server."
    selected_model = model_name
    return f"Model set to {model_name}"

def chat(message, history):
    global selected_model
    if not selected_model:
        return "Please select a model first."
    
    try:
        response = ollama_client.chat(model=selected_model, messages=[
            {"role": "user", "content": message}
        ])
        return response['message']['content']
    except Exception as e:
        error_msg = f"An error occurred during chat: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot with Model Selection")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Type your message here")
            with gr.Row():
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")
                cancel = gr.Button("Cancel")
        
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(choices=get_available_models(), label="Select Model")
            model_status = gr.Textbox(label="Model Status")
    
    def handle_chat(message, history):
        bot_message = chat(message, history)
        history.append((message, bot_message))
        return "", history
    
    model_dropdown.change(set_model, model_dropdown, model_status)
    submit_click = submit.click(handle_chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    # Cancel functionality
    cancel.click(lambda: None, None, None, cancels=[submit_click])

if __name__ == "__main__":
    print("Starting the chatbot...")
    demo.launch(debug=True)
