import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, sentences):
        print(sentences)
        tokens = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device)
        output = self.model(**tokens)
        print(output)
        next_token = torch.argmax(output.logits[:,-1,:], dim=-1)
        print(next_token)

if __name__ == "__main__":
    m = Model()
    m("this is a sentence")

