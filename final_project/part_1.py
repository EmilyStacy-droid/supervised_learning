#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# Project: Create Notes from texts in Bible
# Data source: https://github.com/christos-c/bible-corpus
# Description: Develop a supervised learning model that identifies and extracts named entities from Bible texts to create notes
# 
# 

# EDA Step 1: Parse the original xml file and save the texts into a txt file

# In[3]:


import xml.etree.ElementTree as ET
import re
tree = ET.parse('English.xml')
root = tree.getroot()
texts = []
all_punctuation = "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~"
for seg in root.findall(".//seg"):
    text = seg.text
    text = text.replace('\t','')
    text =  re.sub(rf"[{all_punctuation}]", '', text)
    texts.append(text)

with open('bible_eng_2.txt', 'w+') as f:
    f.writelines(texts)   


# EDA Step 2: Categorize the data with Named Entity Recognition
# 1. tokenize each word
# 2. categorize word tokens
# 3. 
# 

# In[22]:


import pandas as pd
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf

# Load a pre-trained NER model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
model = TFAutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to process text batch by batch
def process_batch(batch):
    all_tokens = []
    all_labels = []
    all_notes = []
    
    # Tokenize the batch into words
    words = batch.split()
    
    # Encode the words using the tokenizer
    tokens = tokenizer(words, is_split_into_words=True, return_tensors="tf", truncation=True, padding=True)
    
    # Get model predictions
    outputs = model(tokens)
    predictions = tf.argmax(outputs.logits, axis=-1)
    
    # Convert input_ids to tokens and align labels
    tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    token_labels = [model.config.id2label[prediction.numpy()] for prediction in predictions[0]]
    
    aligned_tokens = []
    aligned_labels = []
    notes_labels = []
    
    for token, label in zip(tokens, token_labels):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            if token.startswith("##"):
                # Append to the last token if it is a subword token
                aligned_tokens[-1] = aligned_tokens[-1] + token[2:]
            else:
                aligned_tokens.append(token)
                aligned_labels.append(label)
                if label in {'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'}:
                    notes_labels.append(1)  # Potential note
                else:
                    notes_labels.append(0)  # Not a note
    
    all_tokens.extend(aligned_tokens)
    all_labels.extend(aligned_labels)
    all_notes.extend(notes_labels)
    
    return all_tokens, all_labels, all_notes

# Initialize lists to store the tokens, labels, and notes
all_tokens = []
all_labels = []
all_notes = []

# Read the text file line by line to handle large files
max_token_length = 512  # Maximum token length for BERT models
current_batch = ""

with open('bible_eng_2.txt', 'r', encoding='utf-8') as file:
    for line in file:
        current_batch += line.strip() + " "
        
        # Check if the current batch exceeds the maximum token length
        if len(tokenizer(current_batch.split(), is_split_into_words=True)["input_ids"]) > max_token_length:
            tokens, labels, notes = process_batch(current_batch)
            all_tokens.extend(tokens)
            all_labels.extend(labels)
            all_notes.extend(notes)
            current_batch = ""
    
    # Process any remaining lines in the current batch
    if current_batch:
        tokens, labels, notes = process_batch(current_batch)
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        all_notes.extend(notes)

# Create a DataFrame and save to CSV
df = pd.DataFrame({'Token': all_tokens, 'Label': all_labels, 'Notes': all_notes})
df.to_csv('bible_tokens.csv', index=False)

print("Tokenized data saved to bible_tokens.csv")


