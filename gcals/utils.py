import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math


class GCALS(nn.Module):
    def __init__(self, label_smoothing_max=None, label_smoothing_min=0.00001, beta=0.95):
        super().__init__()
        self.label_smoothing_min = label_smoothing_min
        self.label_smoothing = label_smoothing_min
        self.label_smoothing_max = label_smoothing_max
        self.gamma = 0.0
        self.beta = beta

    def update_label_smoothing(self, logits, targets):
          with torch.no_grad():
              num_classes = logits.size(-1)
              softmaxes = F.softmax(logits, dim=-1)
              y_one_hot = F.one_hot(targets, num_classes=num_classes).float()

              G_1 = (1.0 - self.label_smoothing + self.label_smoothing / num_classes) * (softmaxes - y_one_hot)

              numerator = 1.0 - y_one_hot
              G_2 = (self.label_smoothing) * softmaxes * ((num_classes - 1) / num_classes) - (self.label_smoothing / (num_classes)) * numerator

              f_grad_norm = torch.norm(G_1, p=2, dim=-1).mean().item()
              f_reg_norm = torch.norm(G_2, p=2, dim=-1).mean().item()
              

              self.gamma = f_reg_norm / (f_grad_norm + 1e-8)

              target_label_smoothing = max(self.label_smoothing_min, self.label_smoothing_max * (1.0 - math.exp(-self.gamma)))

              self.label_smoothing = (1.0 - self.beta) * target_label_smoothing + self.beta * self.label_smoothing



    def forward(self, logits, targets):
        self.update_label_smoothing(logits, targets)
        loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        return loss


def get_resnet18(num_classes, stride):
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stride, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model


def train_track(num_epochs=None, criterion=None, train_loader=None, test_loader=None, mode=None, label_smoothing=0.0, num_classes=None, stride=None, device=None):
  model = get_resnet18(num_classes=num_classes, stride=stride)

  if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training!")
    model = nn.DataParallel(model)
    
  model = model.to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

  train_loss_log = []
  test_loss_log = []
  test_acc_log = []
  label_smoothing_log = []

  print(f"\n--- Launching {mode.upper()} Track ---")
  for epoch in range(num_epochs):

    model.train()
    running_train_loss = 0.0
    num_train_batches = 0

    for x_batch, y_batch in train_loader:

      x_batch, y_batch = x_batch.to(device), y_batch.to(device)

      optimizer.zero_grad()

      logits = model(x_batch)
      train_loss = criterion(logits, y_batch)

      train_loss.backward()
      optimizer.step()

      running_train_loss += train_loss.item()
      num_train_batches += 1

    epoch_train_loss = running_train_loss / num_train_batches


    model.eval()
    running_test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
      for x_test, y_test in test_loader:

        x_test, y_test = x_test.to(device), y_test.to(device)

        logits_test = model(x_test)

        loss_batch = F.cross_entropy(logits_test, y_test, reduction="sum")
        running_test_loss += loss_batch.item()

        _, predicted = logits_test.max(1)
        total += y_test.size(0)
        correct += predicted.eq(y_test).sum().item()

    epoch_test_loss = running_test_loss / total
    epoch_acc = 100.0 * correct / total

    current_eps = (
        criterion.label_smoothing
        if mode == "gcals"
        else (label_smoothing if mode == "static" else 0.0)
    )

    train_loss_log.append(epoch_train_loss)
    test_loss_log.append(epoch_test_loss)
    test_acc_log.append(epoch_acc)
    label_smoothing_log.append(current_eps)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Train Loss: {epoch_train_loss:.3f} | "
        f"Test Loss: {epoch_test_loss:.3f} | "
        f"Accuracy: {epoch_acc:.2f}% | "
        f"ε: {current_eps:.5f}"
    )


  return train_loss_log, test_loss_log, test_acc_log, label_smoothing_log


