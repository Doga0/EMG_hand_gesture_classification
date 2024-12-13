import pandas as pd
import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = "/content/drive/MyDrive/LMG/dataset2.csv"
df = pd.read_csv(dataset_path)

df.head()

gest_map = {
    "rest": 0,
    "fist": 1,
    "ring": 2
}

df.replace({"label": gest_map}, inplace=True)

df.head()

class LMG_Dataset(Dataset):
  def __init__(self, X, y):

    self.x = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)

  def __getitem__(self, idx):
    x = self.X[idx, :, :]
    y = self.y[idx, :]
    return x, y

  def __len__(self):
    return self.len

features = df.drop(["label"], axis=1)
target = df["label"]

def process_dataset(dataset, window_size, stride):

  X, y = [], []

  for i in range(0, len(dataset) - window_size + 1, stride):
      seq_x = dataset.iloc[i:i+window_size, :-1].values
      seq_y = dataset.iloc[i+window_size-1, -1]
      X.append(seq_x)
      y.append(seq_y)

  return X, y

import torch.nn as nn

class LSTMModel(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, num_layers, dropout, bidirectional):
    super().__init__()

    self.lstm = nn.LSTM(
          input_size,
          hidden_size,
          num_layers,
          dropout=dropout,
          bidirectional=bidirectional,
          batch_first=True
        )

    self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

  def forward(self, x):
    lstm_out, (h_n, c_n) = self.lstm(x)

    if self.lstm.bidirectional:
        lstm_out = lstm_out[:, -1, :].view(-1, self.lstm.hidden_size * 2)
    else:
        lstm_out = lstm_out[:, -1, :].view(-1, self.lstm.hidden_size)

    out = self.fc(lstm_out)

    return out

input_size = len(df.drop(["label"], axis=1).columns)
output_size = 3
hidden_size = 32
num_layers = 2
dropout = 0.2
bidirectional = True

model = LSTMModel(
      input_size,
      output_size,
      hidden_size,
      num_layers,
      dropout,
      bidirectional
    )

learning_rate = 0.01
epochs = 3
freq = 100
batch_size = 8
n_splits = 3
window_size = 50
stride = 10
weight_decay = 0.00001

optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()

def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def train_batch(model, train_loader, optimizer, loss_fn, freq, device):

  total_loss = 0.
  global_step = 0

  for x_batch, y_batch in train_loader:

      x_batch , y_batch = x_batch.to(device), y_batch.to(device)

      y_pred = model(x_batch)
      y_pred = y_pred.mean(dim=1).squeeze(-1)

      loss = loss_fn(y_pred, y_batch.type(torch.float32).to(device))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

      global_step += 1

      if global_step % freq == 0:
        last_loss = total_loss / freq

        print(f"  Batch: {global_step} | Loss: {last_loss:.2f}")
        total_loss = 0.

  return last_loss

def evaluate(model, val_loader, i, device):
  val_steps = 0
  total_val_loss = 0.

  preds, labels = [], []

  model.eval()

  with torch.no_grad():
    for x_batch, y_batch in val_loader:

      x_batch , y_batch = x_batch.to(device), y_batch.to(device)

      val_outputs = model(x_batch)

      if val_outputs.size(0) != y_batch.size(0):
        print(f"Shape mismatch: val_outputs size {val_outputs.size(0)}, y_batch size {y_batch.size(0)}")

      loss = loss_fn(val_outputs, y_batch.long().to(device))

      preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
      labels.extend(y_batch.cpu().numpy())

      total_val_loss += loss.item()
      val_steps += 1

  accuracy = accuracy_score(labels, preds)
  f1 = f1_score(labels, preds, average='weighted')

  return total_val_loss, val_steps, preds, labels, f1, accuracy

def train(model, df, optimizer, loss_fn, epochs, batch_size, n_splits, window_size, stride, freq, class_names):

  device = "cuda" if torch.cuda.is_available() else "cpu"

  model.to(device)

  X, y = process_dataset(df, window_size, stride)
  dataset_length = len(X)

  train_losses, val_losses = [], []
  all_preds, all_labels = [], []

  tscv = TimeSeriesSplit(n_splits)

  for i, (train_index, test_index) in enumerate(tscv.split(X)):

    if test_index[-1] >= dataset_length:
        print(f"Skipping split {i + 1}: Test index exceeds dataset bounds. {test_index[-1]}")
        continue

    X_train, X_test = np.array(X)[train_index, :], np.array(X)[test_index, :]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    train_dataset = torch.utils.data.TensorDataset(
      torch.tensor(X_train, dtype=torch.float32),
      torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = torch.utils.data.TensorDataset(
      torch.tensor(X_test, dtype=torch.float32),
      torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=None)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=None)

    print(f"\nSplit: {i + 1} | Total Batch: {len(train_loader)}\n")

    for epoch in range(epochs):
      print(f"EPOCH: {epoch + 1}")
      print("------------------------------------------------")

      model.train()

      avg_loss = train_batch(model, train_loader, optimizer, loss_fn, freq, device)

      total_val_loss, val_steps, preds, labels, f1, accuracy = evaluate(model, val_loader, i, device)

      all_preds.extend(preds)
      all_labels.extend(labels)

      avg_val_loss = total_val_loss / val_steps
      print(f"LOSS --> Train Loss: {avg_loss:.2f} | Valid Loss: {avg_val_loss:.2f} | F1: {f1:.2f} | Accuracy: {accuracy:.2f}\n")

      train_losses.append(avg_loss)
      val_losses.append(avg_val_loss)

  overall_accuracy = accuracy_score(all_labels, all_preds)
  overall_f1 = f1_score(all_labels, all_preds, average="weighted")

  print(f"Overall Accuracy: {overall_accuracy:.2f}")
  print(f"Overall F1 Score: {overall_f1:.2f}\n")

  plot_confusion_matrix(all_labels, all_preds, class_names)

  return train_losses, val_losses

def plot_losses(epochs, train_losses, val_losses):

  plt.figure()

  fig, ax = plt.subplots()

  ax.plot(epochs, train_losses, label="Train Loss")
  ax.plot(epochs, val_losses, linestyle="-.", label="Validation loss")

  ax.set_xlabel("Epochs")
  ax.set_ylabel("Loss")
  ax.legend(loc="upper right")

  fig.tight_layout()

class_names = list(gest_map.keys())

train_losses, val_losses = train(
      model,
      df,
      optimizer,
      loss_fn,
      epochs,
      batch_size,
      n_splits,
      window_size,
      stride,
      freq,
      class_names
    )

plot_losses(range(1, (epochs * n_splits)+1), train_losses, val_losses)