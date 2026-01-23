import torch
import torch.nn as nn

class SentimentBERT(nn.Module):
    def __init__ (self, model, sentiment_out, type_out, dropout_rate=0.4):
        super(SentimentBERT, self).__init__()
        # bert as base model & feature extractor
        self.model = model
        hidden_dim = self.model.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        # classifiers head
        self.sentiment = nn.Linear(hidden_dim, sentiment_out)
        self.type_ = nn.Linear(hidden_dim, type_out)

    def forward(self, ids, attention_mask):
        x = self.model(input_ids=ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :] # extract [batch_size, hidden_state]

        sentiment_out = self.sentiment(x)
        type_out = self.type_(x)

        return sentiment_out, type_out