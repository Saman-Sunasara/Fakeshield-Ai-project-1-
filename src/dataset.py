import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import os
import logging
from .preprocess import preprocess_dataframe

logger = logging.getLogger("FakeShield")

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name='distilbert-base-uncased', max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_dir='dataset'):
    """
    Loads Fake.csv and True.csv from data_dir.
    Creates dummy data if files don't exist for testing.
    """
    fake_path = os.path.join(data_dir, 'Fake.csv')
    true_path = os.path.join(data_dir, 'True.csv')
    
    if os.path.exists(fake_path) and os.path.exists(true_path):
        logger.info("Loading dataset from CSVs...")
        df_fake = pd.read_csv(fake_path)
        df_fake['label'] = 0
        df_true = pd.read_csv(true_path)
        df_true['label'] = 1
        
        df = pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
        # Assuming the text column is named 'text'
        df = preprocess_dataframe(df, text_col='text')
    else:
        logger.warning("Dataset not found! Creating dummy data for testing purposes.")
        os.makedirs(data_dir, exist_ok=True)
        # Create dummy data with unique strings so they aren't dropped as duplicates
        dummy_data = {
            'text': [f"This is dummy text number {i} for testing the FakeShield AI project." for i in range(1000)],
            'label': [i % 2 for i in range(1000)]
        }
        df = pd.DataFrame(dummy_data)
        df = preprocess_dataframe(df, text_col='text')
        
    return df['cleaned_text'].values, df['label'].values

def partition_data(texts, labels, num_clients=5, iid=True):
    """
    Partitions data among clients.
    """
    if iid:
        # Simple random split
        data_len = len(texts)
        client_data_len = data_len // num_clients
        indices = np.random.permutation(data_len)
        
        partitions = []
        for i in range(num_clients):
            start = i * client_data_len
            end = start + client_data_len if i < num_clients - 1 else data_len
            part_idx = indices[start:end]
            partitions.append((texts[part_idx], labels[part_idx]))
        return partitions
    else:
        # Non-IID simulation: Sort by label and then split
        sorted_indices = np.argsort(labels)
        data_len = len(texts)
        client_data_len = data_len // num_clients
        
        partitions = []
        for i in range(num_clients):
            start = i * client_data_len
            end = start + client_data_len if i < num_clients - 1 else data_len
            part_idx = sorted_indices[start:end]
            # Shuffle within partition to avoid ordered batches during training
            np.random.shuffle(part_idx)
            partitions.append((texts[part_idx], labels[part_idx]))
        return partitions

def create_dataloaders(texts, labels, batch_size=16, test_split=0.2):
    dataset = FakeNewsDataset(texts, labels)
    
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader
