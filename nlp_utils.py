# -*- coding: utf-8 -*-
"""Funções de NLP e a rede neural (compartilhadas por treino e chat)."""
import unicodedata

import nltk
import numpy as np
import torch.nn as nn
from nltk.stem.snowball import SnowballStemmer

for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

stemmer = SnowballStemmer("portuguese")
IGNORE_WORDS = ["?", "!", ".", ","]
ABREVIACOES = {"vc": "voce", "vcs": "voces", "pq": "porque", "tb": "tambem", "tbm": "tambem",
               "obg": "obrigado", "vlw": "valeu", "blz": "beleza", "flw": "falou", "n": "nao",
               "q": "que", "qdo": "quando", "sist": "sistema"}


def tokenize(sentence):
    return nltk.word_tokenize(sentence, language="portuguese")


def normalize(word):
    """minúsculas + sem acentos, para 'não' e 'nao' serem a mesma coisa."""
    w = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in w if unicodedata.category(c) != "Mn")


def stem(word):
    w = normalize(word)
    return stemmer.stem(ABREVIACOES.get(w, w))


def bag_of_words(tokenized_sentence, all_words):
    stems = {stem(w) for w in tokenized_sentence if w not in IGNORE_WORDS}
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in stems:
            bag[idx] = 1.0
    return bag


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        return self.l3(out)  # sem softmax: CrossEntropyLoss já aplica
