import torch
import torch.nn as nn
from transformers import AutoModel

class FakeNewsModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2):
        super(FakeNewsModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # DistilBERT outputs last_hidden_state (batch_size, seq_len, hidden_size)
        # BERT outputs (last_hidden_state, pooler_output)
        self.is_distilbert = 'distilbert' in model_name
        
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.is_distilbert:
            # DistilBERT doesn't have a pooler, use the CLS token (first token)
            pooled_output = outputs.last_hidden_state[:, 0]
        else:
            # BERT has a pooler output
            pooled_output = outputs.pooler_output
            
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits
