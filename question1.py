import streamlit as st
import numpy as np
import pandas as pd

import torch
from torch import nn

# Pytorch Model Code

class NextWord(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = self.lin2(x)
    return x


def load_model(block_size, stoi, emb_dim, device):



# Streamlit Code

st.set_page_config(
    page_title="Text Generator",
    page_icon="🧊",
    initial_sidebar_state="expanded",
)

st.title("Text Generation Using Next Word Prediction")

k = st.slider('**K** (Words to Predict)', 1, 10, 5)

prompt = st.text_input("**Enter some text to generate the next words**", placeholder="Type here")

if prompt:



# st.subheader('Embedding visualization')
# try:
#   st.pyplot(plot_emb(emb, itos))
# except:
#   pass
