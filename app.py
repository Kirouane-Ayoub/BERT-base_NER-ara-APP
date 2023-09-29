import gradio as gr
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("ayoubkirouane/BERT-base_NER-ar")
tokenizer = AutoTokenizer.from_pretrained("ayoubkirouane/BERT-base_NER-ar")

# Create a function to perform NER
def perform_ner(text):
    # Tokenize the input text
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))

    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Perform NER inference
    with torch.no_grad():
        outputs = model(torch.tensor([input_ids]))

    # Get the predicted labels for each token
    predicted_labels = outputs[0].argmax(dim=2).cpu().numpy()[0]

    # Map label IDs to human-readable labels
    predicted_labels = [model.config.id2label[label_id] for label_id in predicted_labels]

    # Create a list of entities and their labels
    entities = [{"entity": token, "label": label} for token, label in zip(tokens, predicted_labels)]

    return entities

# Create a Gradio interface
iface = gr.Interface(
    fn=perform_ner,
    inputs="text",
    outputs="json",
    live=True,
    title="Arabic Named Entity Recognition Using BERT-base_NER-ar",
    description="Enter Arabic text to extract named entities (e.g., names of people, locations, organizations).",
)

# Launch the Gradio app
iface.launch(debug=True) # share = True 
