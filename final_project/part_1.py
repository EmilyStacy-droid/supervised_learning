#!/usr/bin/env python
# coding: utf-8

# # Project: Bible-Related Keyword Extraction Model
# - Data source:
# <br>
#     &ensp;(1) **Bible Corpus**: The multilingual Bible translations in XML formats were obtained from [Bible Corpus](https://github.com/christos-c/bible-corpus).
#     <br>
#     &ensp;(2) **Sermons Collection**: Sermons were collected from the following online resources:
#     <br>
#          &emsp;- [The Kingdom Collective](https://www.thekingdomcollective.com/spurgeon/list/)
#          <br>
#          &emsp;- [St. Andrew's Enfield](https://www.standrewsenfield.com)
# 
# - Objective:
#     <br>
#     &ensp;(1) Develop a supervised learning model to identify and extract words relevant to the Bible.
#     <br>
#     &ensp;(2) Create highlights or keywords from sermons or other text related to biblical content.
# - Steps:
#     <br>
#     &ensp;(1) Data Cleansing and Parsing: 
#     <br>
#         &emsp;i.Gather relevant text data (sermons and biblical texts)
#     <br>
#         &emsp;ii.Clean and preprocess the data (remove noise, special characters)   
#         &emsp;iii.Tokenize the text into words or subword units
#     <br>
#     &ensp;(2) Label Assignment: Assign each word a binary label, 0: irrelevant to the Bible, 1: relevant to the Bible
#     <br>
#     &ensp;(3) Model Training: Transformer and SVM
#     <br>
#     &ensp;(4) Keyword extraction
# - Outcome: The resulting model can automatically identify and highlight Bible-related terms in sermons or other religious content
# 
# 
# 
# 

# **Original Data Source**
# <br>
# The data was initially in an XML format (English.xml) containing English Bible verses. This was parsed and converted into a TXT file (bible_eng_2.txt) for easier handling. The resulting dataset has 95,886 rows, with each row corresponding to a verse.
# <br>
# <br>
# **Initial Cleaning**
# The XML file was parsed to extract text data, which was then saved into a text file. This step is crucial as it transforms structured XML data into a format suitable for further processing.
# <br>
# <br>
# **Data Cleaning Steps**
# <br>
# a. Handling Missing Values:
# 
#     Feature Dropping: There were no features with NaN values that required dropping in this context, as the conversion focused on text extraction rather than feature-based datasets. But to proceed extra tags and punctuations need to be removed
# <br>
# b. Imputation:
# 
#     Not Applicable: Since the dataset consists of text verses, traditional imputation methods for missing values (such as using average values) are not applicable.
# <br>
# c. Feature Selection:
# 
#     Relevance: All features from the XML data were retained as they were directly relevant to the text analysis (i.e., the Bible verses themselves). No irrelevant features were present to remove. But later for business needs, some tokens with value 0 will be converted to 1, vice versa if necessary later
# <br>
# d. Outlier Removal:
# 
#     Outliers: There were no numeric outliers in the text dataset. The focus was on textual content rather than numerical values.
# 

# In[ ]:


import xml.etree.ElementTree as ET
import re
tree = ET.parse('English.xml')
root = tree.getroot()
texts = []
all_punctuation = "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~"
for seg in root.findall(".//seg"):
    text = seg.text
    text = text.replace('\n','').replace('\t','')
    text =  re.sub(rf"[{all_punctuation}]", '', text)
    texts.append(text)

with open('bible_eng.txt', 'w+') as f:
    f.writelines(texts)   


#  
# **Categorization and Labeling**
# <br>
# <br>
# Named Entity Recognition (NER):
# 
#     Tokenization: Words were tokenized using AutoTokenizer.
# 
#     Classification: The tokenized words were categorized using TFAutoModelForTokenClassification.
# 
#     Labeling: Added labels (0 and 1) were used to identify potential notes based on business needs.
# <br>
# <br>
# 
# **Model Development**
# <br>
# <br>
# BERT Model for NER:
# 
#     Pre-trained Model: Utilized dbmdz/bert-large-cased-finetuned-conll03-english, a pre-trained BERT model fine-tuned on the CoNLL-03 dataset for NER. This model was chosen for its effectiveness in recognizing entities in text.
#     
#     Fine-Tuning: The BERT model was fine-tuned to classify tokens into predefined categories relevant to the Bible texts, enhancing its ability to identify specific entities within the text.
# 
# Support Vector Machine (SVM):
# 
#     Feature Extraction: Features extracted from BERT’s token classification were used as input for the SVM model. This approach allows leveraging the contextual embeddings from BERT to improve classification performance.
#     
#     Model Training: An SVM classifier was trained to identify and extract meaningful words related to the Bible from the categorized tokens. Hyperparameters for the SVM were tuned to optimize performance.
# <br>
# <br>
# 
# **Visualizations**
# <br>
# 1. TF-IDF Heatmap:
# 
#     Purpose: Visualizes the importance of words (tokens) across different segments.
#     Implementation: Use seaborn to create a heatmap of the TF-IDF scores for each token in each segment.
# <br>
# 2. Classification Report and Confusion Matrix:
# 
#     Purpose: Evaluates the performance of the SVM model.
#     Implementation: Display precision, recall, and F1-score using a classification report and plot the confusion matrix.

# In[ ]:


import pandas as pd
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf

# model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
# model = TFAutoModelForTokenClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to process text batch by batch
# def process_batch(batch):
    # all_tokens = []
    # all_labels = []
    # all_notes = []
    
    # # Tokenize the batch into words
    # words = batch.split()
    
    # # Encode the words using the tokenizer
    # tokens = tokenizer(words, is_split_into_words=True, return_tensors="tf", truncation=True, padding=True)
    
    # # Get model predictions
    # outputs = model(tokens)
    # predictions = tf.argmax(outputs.logits, axis=-1)
    
    # # Convert input_ids to tokens and align labels
    # tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    # token_labels = [model.config.id2label[prediction.numpy()] for prediction in predictions[0]]
    
    # aligned_tokens = []
    # aligned_labels = []
    # notes_labels = []
    
    # for token, label in zip(tokens, token_labels):
    #     if token not in ["[CLS]", "[SEP]", "[PAD]"]:
    #         if token.startswith("##"):
    #             # Append to the last token if it is a subword token
    #             aligned_tokens[-1] = aligned_tokens[-1] + token[2:]
    #         else:
    #             aligned_tokens.append(token)
    #             aligned_labels.append(label)
    #             if label in {'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'}:
    #                 notes_labels.append(1)  # Potential note
    #             else:
    #                 notes_labels.append(0)  # Not a note
    
    # all_tokens.extend(aligned_tokens)
    # all_labels.extend(aligned_labels)
    # all_notes.extend(notes_labels)
    
    # return all_tokens, all_labels, all_notes

# Initialize lists to store the tokens, labels, and notes
# all_tokens = []
# all_labels = []
# all_notes = []

# Read the text file line by line to handle large files
# max_token_length = 512  # Maximum token length for BERT models
# current_batch = ""

# with open('bible_eng_2.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         current_batch += line.strip() + " "
        
#         # Check if the current batch exceeds the maximum token length
#         if len(tokenizer(current_batch.split(), is_split_into_words=True)["input_ids"]) > max_token_length:
#             tokens, labels, notes = process_batch(current_batch)
#             all_tokens.extend(tokens)
#             all_labels.extend(labels)
#             all_notes.extend(notes)
#             current_batch = ""
    
    # Process any remaining lines in the current batch
#     if current_batch:
#         tokens, labels, notes = process_batch(current_batch)
#         all_tokens.extend(tokens)
#         all_labels.extend(labels)
#         all_notes.extend(notes)

# # Create a DataFrame and save to CSV
# df = pd.DataFrame({'Token': all_tokens, 'Label': all_labels, 'Notes': all_notes})
# df.to_csv('bible_tokens.csv', index=False)


# In[1]:


# import pandas as pd
# df = pd.read_csv('bible_tokens.csv')
# tokens_to_change = {'gospel', 'saint', 'apostles'}
# df['Notes'] = df.apply(lambda row: 1 if row['Token'] in tokens_to_change else row['Notes'], axis=1)
# df.to_csv('modified_bible_tokens.csv', index=False)


# Model training: SVC
# With classifier 0 (not highlighted) and 1 (highlighted) from the bible text, use SVM to train a model that can extract meaningful words related to Bible from speeches.

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
import matplotlib.pyplot as plt
import re
import string

# Load the CSV document with Bible related tokens
bible_words_df = pd.read_csv('modified_bible_tokens.csv')

# Load the sermons text
with open('sermons.txt', 'r', encoding='utf-8') as file:
    large_document = file.read()

# Extract the tokens marked as 1 from the bible token document
marked_words = bible_words_df[bible_words_df['Marker'] == 1]['Token'].tolist()
marked_words = list(set(marked_words)) 

# Segment the sermons file to be paragraphs with length of 300
def segment_document(text, segment_size=300):
    words = text.split()
    segments = [' '.join(words[i:i+segment_size]) for i in range(0, len(words), segment_size)]
    return segments

segments = segment_document(large_document)

# Use TfidfVectorizer to calculate TF-IDF 
vectorizer = TfidfVectorizer(vocabulary=marked_words)
X = vectorizer.fit_transform(segments)
feature_names = vectorizer.get_feature_names_out()

# Create labels
y = [(1 if any(word in segment for word in marked_words) else 0) for segment in segments]

# Check labels 
print(f"Class distribution: {pd.Series(y).value_counts()}")

# If this paragraph only has either 1 or 0, create dummy words
if len(set(y)) < 2:
    dummy_segment_with_keywords = ' '.join(marked_words[:10])  
    dummy_segment_without_keywords = ' '.join(['dummyword']*10)  
    segments.extend([dummy_segment_with_keywords, dummy_segment_without_keywords])
    y.extend([1, 0])

# Create train and test data
X = vectorizer.fit_transform(segments)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVC
model = SVC(kernel='rbf', class_weight='balanced')
model.fit(X_train, y_train)

# Predict on all paragraphs
predictions = model.predict(X)

# test accuracy
y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, pos_label=1)
recall = recall_score(y_test, y_pred_test, pos_label=1)
f1 = f1_score(y_test, y_pred_test, pos_label=1)
print(f"Test Set Accuracy: {test_accuracy:.2f}")
print(f"Precision for 'Marked Words': {precision:.2f}")
print(f"Recall for 'Marked Words': {recall:.2f}")
print(f"F1-Score for 'Marked Words': {f1:.2f}")


# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, model.decision_function(X_test))
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

print(f"AUC: {roc_auc:.2f}")

# Only extract one markable word in each paragraph
notes = []
for segment, prediction in zip(segments, predictions):
    if prediction == 1 and "dummyword" not in segment.lower():
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', segment)
        for sentence in sentences:
            if any(word.lower() in sentence.lower() for word in marked_words):
                words = sentence.split()
                for i, word in enumerate(words):
                    if word.lower() in marked_words:
                        start = max(0, i - 1)
                        end = min(len(words), i + 2)
                        context_words = words[start:end]
                        notes.append(' '.join(context_words))
                        break  

with open('extracted_notes.txt', 'w', encoding='utf-8') as file:
    for note in notes:
        file.write(note + '\n')




# In[ ]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

# Convert TF-IDF matrix to DataFrame for better visualization
tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)
token_frequencies = tfidf_df.sum(axis=0)
token_freq_df = pd.DataFrame({'Token': feature_names, 'Frequency': token_frequencies}).sort_values(by='Frequency', ascending=False)

top_n = 50
top_tokens = token_freq_df.head(top_n)
top_tokens_list = top_tokens['Token'].tolist()

top_tfidf_df = tfidf_df[top_tokens_list]
# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(top_tfidf_df, cmap='YlGnBu', xticklabels=top_tfidf_df.columns, yticklabels=False)
plt.title('TF-IDF Heatmap of Segments')
plt.xlabel('Tokens')
plt.ylabel('Segments')
# plt.show()


print("Unique labels in y_test:", np.unique(y_test))
print("Unique labels in y_pred_test:", np.unique(y_pred_test))

# Confusion matrix
labels = [0, 1]  # Adjust based on your actual labels
cm = confusion_matrix(y_test, y_pred_test, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Marked Words', 'Marked Words'])
disp.plot(cmap='Blues')
# plt.show()



# Clean up the notes. Remove those that are marked as 0 in bible_tokens and repetitions

# In[ ]:


# with open('extracted_notes.txt', 'r', encoding='utf-8') as file:  
#     all_notes = file.readlines()  
#     print(all_notes)
#     cleaned_notes = []
#     for note in all_notes:
#         cleaned_words = [word for word in note.split() if word in marked_words]
#         cleaned_notes.extend(cleaned_words)

# # Remove duplicated notes
# cleaned_notes = list(set(cleaned_notes))


# with open('extracted_notes_2.txt', 'w', encoding='utf-8') as file:
#     for note in cleaned_notes:
#         file.write(note + '\n')

