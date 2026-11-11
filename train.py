# -*- coding: utf-8 -*-
"""Treina o modelo e salva em data.pth.  Uso: python train.py"""
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nlp_utils import IGNORE_WORDS, NeuralNet, bag_of_words, stem, tokenize

INTENTS_FILE = "intents.json"
MODEL_FILE = "data.pth"

BATCH_SIZE = 8
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000


class ChatDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


def main():
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        intents = json.load(f)

    all_words, tags, xy = [], [], []
    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    all_words = sorted({stem(w) for w in all_words if w not in IGNORE_WORDS})
    tags = sorted(set(tags))

    x_train = np.array([bag_of_words(s, all_words) for s, _ in xy], dtype=np.float32)
    y_train = np.array([tags.index(t) for _, t in xy], dtype=np.int64)

    loader = DataLoader(ChatDataset(x_train, y_train), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(len(all_words), HIDDEN_SIZE, len(tags)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for words, labels in loader:
            words, labels = words.to(device), labels.to(device)
            loss = criterion(model(words), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "input_size": len(all_words),
        "hidden_size": HIDDEN_SIZE,
        "output_size": len(tags),
        "all_words": all_words,
        "tags": tags,
    }, MODEL_FILE)
    print(f"Treinamento completo. Arquivo salvo em {MODEL_FILE}")


if __name__ == "__main__":
    main()
