# -*- coding: utf-8 -*-
"""Zendesk Generator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fqcKvmj376PXj1Yifu5dQ6gJq_hsqH9l
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install transformers
# !pip install zenpy

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, BertForMaskedLM, \
    BertTokenizer
import torch.nn as nn
import torch.optim as optim
import torch
import requests
from torch.utils.data import Dataset, DataLoader, IterableDataset
import re
import pandas as pd
from collections import Counter
from pprint import pprint
from zenpy import Zenpy
from zenpy.lib.exception import RecordNotFoundException
import datetime
from zenpy.lib.api_objects import Comment, Ticket
from random import random
from pprint import pprint

ZENDESK_USER = os.environ['ZENDESK_USER']
ZENDESK_TOKEN = os.environ['ZENDESK_TOKEN']

ZENPY_CLIENT = Zenpy(subdomain='iliff', email=ZENDESK_USER,
                     token=ZENDESK_TOKEN)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = model.to(device)

# find a pretrained model for maskedtasks
test_model = BertForMaskedLM.from_pretrained('bert-large-uncased').to(device)
test_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


def get_metric(sentence):
    with torch.no_grad():
        tokens = test_tokenizer(sentence, return_tensors='pt').to(device)
        outputs = test_model(**tokens)

        softmax = torch.softmax(outputs.logits, dim=-1)[0]
        input_ids = tokens['input_ids'][0]
        probabilities = []
        for i,token in enumerate(input_ids):
            token_row = softmax[i]
            input_id = input_ids[i]
            token_probability = token_row[input_id]
            decoded = test_tokenizer.decode([input_id])
            probabilities.append((decoded, token_probability.item()))
        pprint(probabilities)
    return probabilities


def respond_to_ticket(ticket_id):
    ticket = ZENPY_CLIENT.tickets(id=ticket_id)

    user_comment = list(ZENPY_CLIENT.tickets.comments(ticket))[-1].plain_body
    with torch.no_grad():
        sentence = tokenizer(user_comment + ' <|endoftext|>', return_tensors='pt').to(device)
        outputs = model.generate(**sentence,
                                 max_length=500,
                                 min_length=10,
                                 do_sample=True,
                                 top_p=.8,
                                 top_k=3,
                                 num_return_sequences=1,
                                 repetition_penalty=1.2,
                                 temperature=0.7)
    output = outputs[0]
    decoded = tokenizer.decode(output)

    ticket_comment = decoded.split('<|endoftext|>')[1]
    get_metric(ticket_comment)
    ticket_comment = \
    re.split(r'[\n\r]\s*with\s+respect,?\s*[\n\r]|[\n\r]\s*best,?\s*[\n\r]|[\n\r]\s*thanks,?\s*[\n\r]', ticket_comment,
             flags=re.I)[0]
    ticket.comment = Comment(body=ticket_comment, public=False)
    ZENPY_CLIENT.tickets.update(ticket)


def needs_comment(ticket):
    user_comment = list(ZENPY_CLIENT.tickets.comments(ticket))[-1]
    if user_comment.author.email in ['eshafer@iliff.edu', 'mhemenway@iliff.edu', 'jbarber@iliff.edu', 'kvan@iliff.edu']:
        return False
    return True


def get_new_tickets():
    new_tickets = ZENPY_CLIENT.search(type='ticket', status='new')
    return list(new_tickets)


# respond_to_ticket(35362)

if __name__ == "__main__":

    model = torch.load('model_1.3485576329479791.pt')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

    model = model.to(device)

    respond_to_ticket(35362)