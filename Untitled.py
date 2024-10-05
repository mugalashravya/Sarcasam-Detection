#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


# In[2]:


# Load your dataset
data = pd.read_csv('sarcasm.csv')  # Replace with your dataset path
print(data.columns)


# In[3]:


# Use the correct column names
comments = data['tweet']
labels = data['sarcastic']


# In[4]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(comments, labels, test_size=0.2, random_state=42)


# In[5]:


# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[6]:


X_train = X_train.astype(str)
X_test = X_test.astype(str)


# In[7]:


def tokenize(texts):
    return tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt')


# In[8]:


from transformers import BertTokenizer

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
def tokenize(texts):
    # Ensure input is a list of strings
    texts = list(texts)
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Apply tokenization
train_encodings = tokenize(X_train)
test_encodings = tokenize(X_test)


# In[9]:


import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments


# In[10]:


# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


# In[11]:


# Convert labels to torch tensors
train_labels = torch.tensor(y_train.values.tolist())
test_labels = torch.tensor(y_test.values.tolist())


# In[12]:


# Define the dataset class
class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# In[13]:


train_dataset = SarcasmDataset(train_encodings, train_labels)
test_dataset = SarcasmDataset(test_encodings, test_labels)


# In[14]:


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=1,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
)


# In[ ]:


# Initialize Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=test_dataset             
)

# Train model
trainer.train()


# In[2]:


import torch

def classify_input_text(model, tokenizer, text):
    """
    Function to classify whether the input text is sarcastic or not.
    """
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Put the model in evaluation mode
    model.eval()
    
    # Extract the logits (output scores)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()

    # Convert prediction to label
    if prediction == 1:  # Assuming '1' represents sarcasm
        return "This is a sarcastic comment."
    else:
        return "This is a genuine comment."




# In[3]:


# Take input from the user
user_input = input("Enter a comment to classify: ")

# Classify the user's input
result = classify_input_text(model, tokenizer, user_input)

# Print the result
print(result)


# In[ ]:




