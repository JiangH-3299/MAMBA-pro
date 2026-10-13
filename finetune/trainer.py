import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

WEIGHT_DECAY = 0.0001
KD_SCALE = 10


def _load(arr, device):
    """Load a batched tensor (from DataLoader) onto device."""
    arr = torch.squeeze(arr).float()
    arr = Variable(arr.to(device))
    return arr


def train_step1(model, data_loader, learning_rate, alpha, device):
    """Step 1 training: freeze GCN, Graphormer and GRP_linear, train MLP only."""
    for param in model.gcn.parameters():
        param.requires_grad = False
    for param in model.graphormer.parameters():
        param.requires_grad = False
    for param in model.GRP_linear.parameters():
        param.requires_grad = False

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=WEIGHT_DECAY,
    )

    epoch_loss = 0.0
    n_batches = 0

    for data in tqdm(data_loader):
        optimizer.zero_grad()
        _, _, _, LRP_wts, LRP_muts, GRP_wts, GRP_muts, graphs, biases, centralities, labels, teachers = data

        LRP_wts = _load(LRP_wts, device)
        LRP_muts = _load(LRP_muts, device)
        GRP_wts = _load(GRP_wts, device)
        GRP_muts = _load(GRP_muts, device)
        graphs = _load(graphs, device)
        biases = _load(biases, device)
        centralities = _load(centralities, device)
        labels = _load(labels, device)
        teachers = _load(teachers, device)

        y_true = (labels.float() + 15.0) / 30.0
        y_teac = (teachers.float() + 15.0) / 30.0

        y_pred = model(LRP_wts, LRP_muts, GRP_wts, GRP_muts, graphs, biases, centralities)
        y_pred = torch.squeeze(y_pred).float()

        label_loss = model.criterion(y_true, y_pred)
        kd_loss = KD_SCALE * torch.sqrt(F.mse_loss(y_pred, y_teac))
        loss = (alpha * label_loss + (1 - alpha) * kd_loss).float()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / n_batches


def train_step2(model, data_loader, learning_rate, alpha, device):
    """Step 2 training: all parameters are unfrozen."""
    for param in model.gcn.parameters():
        param.requires_grad = True
    for param in model.graphormer.parameters():
        param.requires_grad = True
    for param in model.GRP_linear.parameters():
        param.requires_grad = True

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=WEIGHT_DECAY,
    )

    epoch_loss = 0.0
    n_batches = 0

    for data in tqdm(data_loader):
        optimizer.zero_grad()
        _, _, _, LRP_wts, LRP_muts, GRP_wts, GRP_muts, graphs, biases, centralities, labels, teachers = data

        LRP_wts = _load(LRP_wts, device)
        LRP_muts = _load(LRP_muts, device)
        GRP_wts = _load(GRP_wts, device)
        GRP_muts = _load(GRP_muts, device)
        graphs = _load(graphs, device)
        biases = _load(biases, device)
        centralities = _load(centralities, device)
        labels = _load(labels, device)
        teachers = _load(teachers, device)

        y_true = (labels.float() + 15.0) / 30.0
        y_teac = (teachers.float() + 15.0) / 30.0

        y_pred = model(LRP_wts, LRP_muts, GRP_wts, GRP_muts, graphs, biases, centralities)
        y_pred = torch.squeeze(y_pred).float()

        label_loss = model.criterion(y_true, y_pred)
        kd_loss = KD_SCALE * torch.sqrt(F.mse_loss(y_pred, y_teac))
        loss = (alpha * label_loss + (1 - alpha) * kd_loss).float()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / n_batches


def evaluate(model, data_loader, alpha, device):
    """Evaluate model on a DataLoader."""
    model.eval()
    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_name = []
    valid_wt = []
    valid_mut = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            Ids, wts, muts, LRP_wts, LRP_muts, GRP_wts, GRP_muts, graphs, biases, centralities, labels, teachers = data

            LRP_wts = _load(LRP_wts, device)
            LRP_muts = _load(LRP_muts, device)
            GRP_wts = _load(GRP_wts, device)
            GRP_muts = _load(GRP_muts, device)
            graphs = _load(graphs, device)
            biases = _load(biases, device)
            centralities = _load(centralities, device)
            labels = _load(labels, device)
            teachers = _load(teachers, device)

            y_true = (labels.float() + 15.0) / 30.0
            y_teac = (teachers.float() + 15.0) / 30.0

            y_pred = model(LRP_wts, LRP_muts, GRP_wts, GRP_muts, graphs, biases, centralities)
            y_pred = torch.squeeze(y_pred).float()

            label_loss = model.criterion(y_true, y_pred)
            kd_loss = KD_SCALE * torch.sqrt(F.mse_loss(y_pred, y_teac))
            loss = (alpha * label_loss + (1 - alpha) * kd_loss).float()

            y_pred_list = y_pred.cpu().detach().numpy().tolist()
            y_true_list = y_true.cpu().detach().numpy().tolist()

            if isinstance(y_pred_list, float):
                y_pred_list = [y_pred_list]
            if isinstance(y_true_list, float):
                y_true_list = [y_true_list]

            valid_pred.extend(y_pred_list)
            valid_true.extend(y_true_list)
            valid_name.extend(Ids)
            valid_wt.extend(wts)
            valid_mut.extend(muts)

        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / n_batches, valid_true, valid_pred, valid_name, valid_wt, valid_mut
