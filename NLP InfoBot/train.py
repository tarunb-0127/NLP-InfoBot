import json
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json','r') as file:
    intents = json.load(file)

all_words = []
tags=[]
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_chars = ['?','.',',','!']

all_words = [stem(w) for w in all_words if w not in ignore_chars]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)

X_train = []
y_train = []

for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)

    label =tags.index(tag)
    y_train.append(label)

X_train=np.array(X_train)
y_train = np.array(y_train)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()


class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self,idx):
            return self.x_data[idx],self.y_data[idx]
        
        def __len__(self):
            return self.n_samples

batch_size= 8 
hidden_size =8
output_size= len(tags)
input_size = len(X_train[0])
learning_rate =0.001
num_epochs= 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

device = torch.device('cpu')
model = NeuralNet(input_size,hidden_size,output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

for epoch in range(num_epochs):
    for (words,labels) in train_loader:
          words = words.to(device)
          labels = labels.to(device)

          outputs = model(words)
          loss = criterion(outputs,labels)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
    
    if (epoch+1)%100==0:
        print(f'epoch{epoch+1}/{num_epochs},loss={loss.item():.4f}')


data = {
         "model_state": model.state_dict(),
         "input_size" : input_size,
         "output_size" : output_size,
         "hidden_size" : hidden_size,
         "all_words" : all_words,
         "tags" : tags
    }

FILE = "data.pth"
torch.save(data,FILE)

print(f"Training Complete. File saved to {FILE}")
        
    