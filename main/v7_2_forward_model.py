import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sympy.physics.units import length
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random
import os
import time
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from v7_main_model_functions import *
from v7_model_functions import *
from v7_1_get_data import *


def shuffle_data(seq, labels):
    """
    Shuffles the sequences and labels, maintaining alignment between them.
    """
    rand_state = np.random.RandomState(seed=torch.randint(0, 100000, (1,)).item())
    indices = np.arange(len(seq))
    rand_state.shuffle(indices)
    seq_shuffled = seq[indices]
    labels_shuffled = labels[indices]
    return seq_shuffled, labels_shuffled

def print_model_structure(model):
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

class ImprovedVAE_CNNModel_V2(nn.Module):
    def __init__(self, vectorSize, labelSize, w1, wd1, h1, w4, latent_dim_1, latent_dim_2, latent_dim_3, latent_dim_4):
        super(ImprovedVAE_CNNModel_V2, self).__init__()
        self.vectorSize = vectorSize
        self.h1 = h1
        self.wd1 = wd1
        self.w1 = w1
        self.w4 = w4
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2
        self.latent_dim_3 = latent_dim_3
        self.latent_dim_4 = latent_dim_4

        # Encoder
        self.conv1 = nn.Conv2d(1, w1, (1, wd1), padding='same')
        self.pool1 = nn.MaxPool2d((h1, 1), stride=(h1, 1))
        self.conv2 = nn.Conv2d(w1, w1 * 2, (1, wd1), padding='same')
        self.pool2 = nn.MaxPool2d((2, 1), stride=(2, 1))

        self.flat_size = self._get_flat_size(vectorSize)

        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flat_size, w4),
            nn.ReLU(),
            nn.BatchNorm1d(w4)
        )

        # Latent space
        self.fc_mu = nn.Linear(w4, latent_dim_1)
        self.fc_logvar = nn.Linear(w4, latent_dim_1)

        # Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim_1, w4),
            nn.ReLU(),
            nn.BatchNorm1d(w4),
            nn.Linear(w4, self.flat_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(w1 * 2, w1, (2, wd1), stride=(2, 1), padding=(0, wd1 // 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(w1, 1, (h1, wd1), stride=(h1, 1), padding=(0, wd1 // 2)),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim_1, latent_dim_2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim_2),
            nn.Dropout(0.3),
            nn.Linear(latent_dim_2, latent_dim_3),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim_3),
            nn.Dropout(0.3),
            nn.Linear(latent_dim_3, latent_dim_4),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim_4),
            nn.Linear(latent_dim_4, labelSize)
        )

    def _get_flat_size(self, vectorSize):
        x = torch.randn(1, 1, vectorSize, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        return x.numel()

    def encode(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)  # [B, 1, vectorSize, 1]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def get_conv1_output(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)  # [B, 1, vectorSize, 1]
        return self.conv1(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(-1, self.w1 * 2, self.vectorSize // (self.h1 * 2), 1)
        x = self.decoder(x)
        x = x.squeeze(1).squeeze(-1)
        x = F.interpolate(x.unsqueeze(1), size=self.vectorSize, mode='linear', align_corners=False).squeeze(1)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        classification = self.classifier(z)
        return x_recon, classification, mu, logvar

    def loss_function(self, recon_x, x, classification, labels, mu, logvar, beta=1.0, lambda_class=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        class_loss = F.binary_cross_entropy_with_logits(classification, labels.float(), reduction='sum')

        # 添加正则化损失
        l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())

        return recon_loss + beta * kl_loss + lambda_class * class_loss + 1e-5 * l2_reg, recon_loss, kl_loss, class_loss


def train_and_validate(model, train_loader, val_loader, num_epochs, learning_rate, beta, lambda_class):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            recon_batch, classification, mu, logvar = model(data)
            total_loss, recon_loss, kl_loss, class_loss = model.loss_function(
                recon_batch, data, classification, labels, mu, logvar, beta, lambda_class
            )
            loss = class_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                recon_batch, classification, mu, logvar = model(data)
                total_loss, recon_loss, kl_loss, class_loss = model.loss_function(
                    recon_batch, data, classification, labels, mu, logvar, beta, lambda_class
                )
                val_loss += class_loss.item()

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return model, train_losses, val_losses


def evaluate(model, test_loader, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []
    all_recons = []
    all_inputs = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            recon_batch, classification, _, _ = model(data)
            preds = (torch.sigmoid(classification) > threshold).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_recons.extend(recon_batch.cpu().numpy())
            all_inputs.extend(data.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_preds, average='samples', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=1)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 分析输入和重构的 RNA 序列之间的差异
    all_inputs = np.array(all_inputs)
    all_recons = np.array(all_recons)
    diff_positions = np.abs(all_inputs - all_recons) > 0.5  # 阈值为考虑一个位置不同
    # 计算每个位置的差异频率
    diff_frequency = np.mean(diff_positions, axis=0)
    # 找出差异最频繁的 N 个位置
    N = 10
    top_diff_positions = np.argsort(diff_frequency)[-N:][::-1]

    return precision, recall, f1, diff_frequency, top_diff_positions

def save_model(model, path="./model/model.pth"):
    torch.save(model, path)

def load_model(path="./model/model.pth"):
    model = torch.load(path)
    model.eval()
    return model

def extract_conv1_output(model, input_data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        conv1_output = model.get_conv1_output(input_data)
    return conv1_output


if __name__ == '__main__':
    basic_path = './'

    variant_list = ['alpha', 'beta', 'gamma', 'delta', 'omicron']

    learning_rate = 1e-4
    num_epochs = 200
    beta = 0.1
    lambda_class = 10.0

    generate_forward_number = 30
    train_model_number_forward = 1000

    forward_batchSize = 32
    keep_prob = 0.5

    if os.path.exists(f'{basic_path}vectorSize.csv'):
        vectorSize = pd.read_csv(f'{basic_path}vectorSize.csv', header=None).values[0][0]
    else:
        vectorSize = mixed_data(basic_path, delta_num=train_model_number_forward, other_percent=1)


    method = 'Pooling' # 'Pooling', 'Top', 'Combination'
    
    if method == 'Pooling':
        forward_method = 'Pooling'
        forward_max_Pooling_window_size = 148
    elif method == 'Top':
        forward_method = 'Top'
        forward_top_number = 175
        forward_max_Pooling_window_size = int(vectorSize / forward_top_number) + 1
    elif method == 'Combination':
        forward_method = 'Combination'
        forward_max_Pooling_window_size = 500
        forward_top_in_window = 10

    # check if the data is already prepared
    if os.path.exists(basic_path + 'index/train_sequence.csv'):
        pass
    else:
        train_valid_data(basic_path, n_splits=2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batchSize = forward_batchSize
    labelSize = len(variant_list)

    w1 = 12
    wd1 = 21
    h1 = forward_max_Pooling_window_size
    w4 = 256
    latent_dim_1 = 128
    latent_dim_2 = 64
    latent_dim_3 = 32
    latent_dim_4 = 16

    # check if the data is already prepared
    if os.path.exists(basic_path + 'index/train_dataset.pth'):
        train_dataset = torch.load(basic_path + 'index/train_dataset.pth')
        val_dataset = torch.load(basic_path + 'index/val_dataset.pth')
        test_dataset = torch.load(basic_path + 'index/test_dataset.pth')
    else:
        seq_T = pd.read_csv(basic_path + 'index/train_sequence.csv', header=None).values.ravel()
        label_T = pd.read_csv(basic_path + 'index/train_label.csv', header=None).values.ravel()

        seq_V = pd.read_csv(basic_path + 'index/valid_sequence.csv', header=None).values.ravel()
        label_V = pd.read_csv(basic_path + 'index/valid_label.csv', header=None).values.ravel()

        seq_Test = pd.read_csv(basic_path + 'index/train_sequence.csv', header=None).values.ravel()
        label_Test = pd.read_csv(basic_path + 'index/train_label.csv', header=None).values.ravel()

        seq_T, label_T = shuffle_data(seq_T, label_T)
        seq_V, label_V = shuffle_data(seq_V, label_V)
        seq_Test, label_Test = shuffle_data(seq_Test, label_Test)

        oneHot_labels_T = oneHot_pytorch(label_T, labelSize)
        oneHot_labels_V = oneHot_pytorch(label_V, labelSize)
        oneHot_labels_Test = oneHot_pytorch(label_Test, labelSize)

        seq_T = getBatch_pytorch(seq_T, seq_T.shape[0], vectorSize)
        seq_V = getBatch_pytorch(seq_V, seq_V.shape[0], vectorSize)
        seq_Test = getBatch_pytorch(seq_Test, seq_Test.shape[0], vectorSize)

        train_dataset = TensorDataset(seq_T, oneHot_labels_T)
        val_dataset = TensorDataset(seq_V, oneHot_labels_V)
        test_dataset = TensorDataset(seq_Test, oneHot_labels_Test)

        # save the variables
        torch.save(train_dataset, basic_path + 'index/train_dataset.pth')
        torch.save(val_dataset, basic_path + 'index/val_dataset.pth')
        torch.save(test_dataset, basic_path + 'index/test_dataset.pth')


    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n\n")

    # 检查是否有保存的模型
    if os.path.exists("./model/model.pth"):
        trained_model = load_model("./model/model.pth")
    else:
        model = ImprovedVAE_CNNModel_V2(vectorSize, labelSize, w1, wd1, h1, w4, latent_dim_1, latent_dim_2, latent_dim_3, latent_dim_4)
        trained_model, train_losses, val_losses = train_and_validate(model, train_loader, val_loader, num_epochs, learning_rate, beta, lambda_class)

        precision, recall, f1, diff_frequency, top_diff_positions = evaluate(trained_model, test_loader)

        # 打印结果
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Top different positions in RNA sequences:")
        for pos in top_diff_positions:
            print(f"Position {pos}: {diff_frequency[pos]:.4f}")

        # 保存模型
        save_model(trained_model, "./model/model.pth")


    # 从模型中提取第一层卷积层的输出
    # Generate forward primers
    for variant_Name in variant_list:
        print(f'Generating forward primers for {variant_Name} variant')
        path = f'{basic_path}{variant_Name}/forward/'

        seqName_file = f'{basic_path}seq_and_seqName/seqName_GISAID_{variant_Name}.csv'
        seq_file = f'{basic_path}seq_and_seqName/seq_GISAID_{variant_Name}.csv'

        seqName = pd.read_csv(seqName_file, header=None).values.ravel()
        sequence = pd.read_csv(seq_file, header=None).values.ravel()

        np.random.shuffle(sequence)
        np.random.shuffle(seqName)

        seqName = seqName[:generate_forward_number]
        sequence = sequence[:generate_forward_number]

        pd.DataFrame(sequence).to_csv(f'{path}filter_seq/filter_seq.csv', header=None, index=None)
        data = getBatch_pytorch(sequence, sequence.shape[0], vectorSize)

        print(f'\nStart the Filter files...')
        start_time = time.time()
        conv1_output = extract_conv1_output(trained_model, data).cpu().detach().numpy()
        end_time = time.time()
        print(f'Filter files complete!     Time cost: {end_time - start_time}\n')
        print(f'The first Conv1 layer size:      Conv1 = {conv1_output.shape}\n')

        for filterIndex in range(conv1_output.shape[1]):
            print(f'Variant {variant_Name} : Generating the {filterIndex} Filter file')

            Mat = conv1_output[:, filterIndex, :, 0]
            pd.DataFrame(Mat).to_csv(f'{path}filter/filter_{filterIndex}.csv', header=None, index=None)