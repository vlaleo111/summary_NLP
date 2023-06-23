import os
import time
import datetime
import pandas as pd

import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import warnings
warnings.filterwarnings("ignore")

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['Articulos'][idx]
        summary = self.data['Resumen'][idx]

        inputs = self.tokenizer.encode_plus(
            summary,
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': inputs['input_ids'].squeeze()
        }
    
def load_summary_dataset(tokenizer):
    file_path = "bbc_news_es.csv"
    df = pd.read_csv(file_path)[['Resumen', 'Articulos']]
    df = df[list(-(df["Resumen"].duplicated()))]
    # df = df[[0, 5]]
    # df.columns = ['Resumen', 'Articulos']
    # df = df.sample(300, random_state=1)
    # df = df.reset_index().drop(["index"], axis=1)

   # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        df['Articulos'].tolist(), df['Resumen'].tolist(), test_size=0.1, random_state=1)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.11, random_state=1)
    train_data = pd.DataFrame({'Articulos':list(X_train), 'Resumen':list(y_train)})
    test_data = pd.DataFrame({'Articulos':list(X_test), 'Resumen':list(y_test)})
    valid_data = pd.DataFrame({'Articulos':list(X_valid), 'Resumen':list(y_valid)})

    train_dataset = TextDataset(train_data, tokenizer, max_length=1024)
    valid_dataset = TextDataset(valid_data, tokenizer, max_length=1024)
    test_dataset = TextDataset(test_data, tokenizer, max_length=1024)

    return train_dataset, valid_dataset, test_dataset


# Cargar el modelo preentrenado y el tokenizer
model_name = 'DeepESP/gpt2-spanish'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Cargar y dividir el dataset en conjuntos de entrenamiento y prueba
train_dataset, valid_dataset, test_dataset = load_summary_dataset(tokenizer)

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='results',
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    save_strategy='epoch',
    logging_steps=10,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='logs'
)

# Definir el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Especifica la ruta de destino donde guardar el modelo y el tokenizer
ruta_destino = 'GPT_esp_summary_v4'

# Guarda el modelo y el tokenizer en la ruta de destino
model.save_pretrained(ruta_destino)
tokenizer.save_pretrained(ruta_destino)