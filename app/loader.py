import torch
import json
import joblib

from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

from models.bert.model import SentimentBERT
from scripts.utils import *

def model_setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('./models/bert/config.json', 'r') as f:
        config=json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    base_model = AutoModel.from_pretrained(config['model_name'])

    model = SentimentBERT(
        model=base_model,
        sentiment_out=config['sentiment_out'],
        type_out=config['type_out'],
        dropout_rate=config['dropout_rate']
    )

    states = load_file('./models/bert/model.safetensors')
    model.load_state_dict(states)
    model.to(device)

    sentiment_decoder = joblib.load('./models/sentiment_encoder.pkl')
    type_decoder = joblib.load('./models/type_encoder.pkl')
    return model, tokenizer, sentiment_decoder, type_decoder, device