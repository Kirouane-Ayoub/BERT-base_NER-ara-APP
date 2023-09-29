

## Model Name: BERT-base_NER-ar

### Model Description : 

**BERT-base_NER-ar** is a fine-tuned **BERT** multilingual base model for Named Entity Recognition (NER) in Arabic. The base model was pretrained on a diverse set of languages and fine-tuned specifically for the task of NER using the "wikiann" dataset. This model is case-sensitive, distinguishing between different letter cases, such as "english" and "English." [**https://huggingface.co/ayoubkirouane/BERT-base_NER-ar**]

### Dataset
The model was fine-tuned on the **wikiann** dataset, which is a multilingual named entity recognition dataset. It contains Wikipedia articles annotated with three types of named entities: LOC (location), PER (person), and ORG (organization). The annotations are in the IOB2 format. The dataset supports 176 of the 282 languages from the original WikiANN corpus.

### Supported Tasks and Leaderboards
The primary supported task for this model is named entity recognition (NER) in Arabic. However, it can also be used to explore the zero-shot cross-lingual capabilities of multilingual models, allowing for NER in various languages.

### Use Cases
+ **Arabic Named Entity Recognition**: *BERT-base_NER-ar* can be used to extract named entities (such as names of people, locations, and organizations) from Arabic text. This is valuable for information retrieval, text summarization, and content analysis in Arabic language applications.

+ **Multilingual NER**: The model's multilingual capabilities enable it to perform NER in other languages supported by the "wikiann" dataset, making it versatile for cross-lingual NER tasks.

### Limitations

+ **Language Limitation**: While the model supports multiple languages, it may not perform equally well in all of them. Performance could vary depending on the quality and quantity of training data available for specific languages.

+ **Fine-Tuning Data**: The model's performance is dependent on the quality and representativeness of the fine-tuning data (the "wikiann" dataset in this case). If the dataset is limited or biased, it may affect the model's performance.


## Usage : 

```python 
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch 
# Load the fine-tuned model
model = AutoModelForTokenClassification.from_pretrained("ayoubkirouane/BERT-base_NER-ar")
tokenizer = AutoTokenizer.from_pretrained("ayoubkirouane/BERT-base_NER-ar")

# Tokenize your input text
text = "أبو ظبي هي عاصمة دولة الإمارات العربية المتحدة."
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

# Print the tokenized text and its associated labels
for token, label in zip(tokens, predicted_labels):
    print(f"Token: {token}, Label: {label}")

```

## Gradio APP 

```
pip install -r requirements.txt
python app.py
```

+ You can check The Demo from here: **https://huggingface.co/spaces/ayoubkirouane/BERT-base_NER-ar**

![Screenshot at 2023-09-29 13-16-40](https://github.com/Kirouane-Ayoub/BERT-base_NER-ara-APP/assets/99510125/79b986b3-b976-4c37-a075-972fd10b917c)


+ developed by **Kirouane Ayoub**
