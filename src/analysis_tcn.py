import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from prepare import TCNDataset
from modelling import TemporalClassification, ClassificationEvaluation

## ANALYSIS SECTION: Combining function from other modules ##
### Read in cleaned data and defined reoccruing objects ###
df = pd.read_csv("../dataset/data_cleaned.csv")
sensor_columns = [
    col for col in df.columns[3:16] if "Linear" not in col
] 
label_columns = [col for col in df.columns[16:]]
label_columns.sort()
mapping = {i: label_columns[i] for i in range(len(label_columns))}

# Clean dataframe to required data
df = df[sensor_columns + label_columns]


# Define training parameters
batch_size = 256
input_channels = 1 # ten sensors
n_classes = 7 # seven classes 
hidden_units = 25
channel_sizes = [hidden_units] * 8
seq_length = int(10 / input_channels)

model = TemporalClassification(
    input_size = input_channels, 
    output_size = n_classes, 
    num_channels = channel_sizes, 
    kernel_size = 2, 
    dropout= 0.05
)

# Load in Data
train_set = TCNDataset(df, label_columns, split="train")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

def train(model, train_loader, epochs=20):
    optimizer = Adam(model.parameters(), lr=0.002)
    for epoch in range(epochs):
        steps = 0
        train_loss = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, input_channels, seq_length)
            # data.requires_grad, target.requires_grad = True, True
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss
            steps += seq_length

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss/steps))
    return model

tcn_fall = train(model, train_loader, epochs=1)


# Load in Data
test_set = TCNDataset(df, label_columns, split="test")
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

def evaluator(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    y_true, y_pred = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, input_channels, seq_length)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            y_true.extend(target.numpy())
            y_pred.extend(np.concatenate(pred.numpy(), axis=0))


        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return y_true, y_pred, test_loss

y_true, y_pred, test_loss = evaluator(tcn_fall, test_loader)


y_true = [mapping[i] for i in y_true]
y_pred = [mapping[i] for i in y_pred]

class_eval = ClassificationEvaluation()
cm = class_eval.confusion_matrix(y_true, y_pred, label_columns)
class_eval.confusion_matrix_visualize(cm, [col.split(" ")[1] for col in label_columns], "./cm_rf.png")
