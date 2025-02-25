import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import new_util
import torch.optim as optim
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class trainer():
    def __init__(self, scaler, batch_size, seq_x, seq_y, nhid , dropout, lrate, wdecay, device, supports,embedding_dim,adj):
        self.device = device
        self.model = TTG(seq_x, seq_y, dropout, supports=supports, batch_size=batch_size, residual_channels=nhid,embedding_dim=embedding_dim,adj=adj)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # 损失函数，使用自定义的 masked MAE 损失
        self.loss = new_util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, x_adj):
        self.model.train()

        self.optimizer.zero_grad()

        input = input.to(self.device)
        real_val = real_val.to(self.device)
        x_adj = x_adj.to(self.device)


        output = self.model(input,x_adj)

        output = output.transpose(1,3)

        real = torch.unsqueeze(real_val,dim=1)

        predict = self.scaler.inverse_transform(output.transpose(1,3)[...,0])
        predict = predict.unsqueeze(3).transpose(1,3)

        loss = self.loss(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        mape = new_util.masked_mape(predict,real,0.0).item()
        rmse = new_util.masked_rmse(predict,real,0.0).item()
        idmape =new_util.masked_idmape(predict,real,0.0).item()

        return loss.item(),mape,rmse,idmape

    def eval(self, input, real_val, x_adj):
        self.model.eval()

        input = input.to(self.device)
        real_val = real_val.to(self.device)
        x_adj = x_adj.to(self.device)

        output = self.model(input,x_adj)
        output = output.transpose(1,3)

        real = torch.unsqueeze(real_val,dim=1)

        predict = self.scaler.inverse_transform(output.transpose(1, 3)[..., 0])
        predict = predict.unsqueeze(3).transpose(1, 3)

        loss = self.loss(predict, real, 0.0)

        mape = new_util.masked_mape(predict,real,0.0).item()
        rmse = new_util.masked_rmse(predict,real,0.0).item()
        idmape = new_util.masked_idmape(predict,real,0.0).item()

        return loss.item(),mape,rmse, idmape

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def apply_attention(embedding):
    attn = torch.matmul(embedding, embedding.permute(0, 2, 1))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, embedding)

class TTG(nn.Module):
    def __init__(self, seq_x,
                 seq_y,
                 dropout=0.3,
                 supports=None,
                 batch_size=2,
                 residual_channels=32,
                 embedding_dim=64,
                 adj=False,

                 hidden_size_mlp =[256, 64],
                 if_activate_last= False,
                 batch_normalization=False,
                 ):

        super(TTG, self).__init__()
        self.seq_x = seq_x
        self.seq_y = seq_y
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.adj = adj
        self.dropout = dropout
        self.start_conv = nn.Conv2d(in_channels=2,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1)).to(device)



        self.supports = supports

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        hidden_size_mlp = [self.seq_x] + hidden_size_mlp
        layers2 = []
        for i in range(1, len(hidden_size_mlp)):
            layers2.append(nn.Linear(in_features=hidden_size_mlp[i - 1], out_features=hidden_size_mlp[i]))
            layers2.append(nn.ReLU())
            if batch_normalization:
                layers2.append(nn.BatchNorm1d(num_features=hidden_size_mlp[i]))
            if dropout > 0:
                layers2.append(nn.Dropout(p=dropout))
        layers2 = layers2 + [nn.Linear(in_features=hidden_size_mlp[-1], out_features=self.seq_y)]
        if if_activate_last:
            layers2.append((nn.ReLU()))


        self.mlp = nn.Sequential(*layers2).to(device)

        self.linear_q = nn.Linear(self.embedding_dim, 32).to(device)

        self.linear_k = nn.Linear(self.embedding_dim, 32).to(device)

        self.linear_v = nn.Linear(self.embedding_dim, 32).to(device)

        self.embedding_week = nn.Embedding(8, self.embedding_dim).to(device)

        self.embedding_position = nn.Embedding(14, self.embedding_dim).to(device)

        self.embedding_volume = nn.Embedding(73472, self.embedding_dim).to(device)

        self.reduce_conv_adj = nn.Conv2d(in_channels=residual_channels + (self.batch_size+1)*self.embedding_dim, out_channels=self.embedding_dim,
                                     kernel_size=(1, 1)).to(device)

        self.reduce_conv = nn.Conv2d(in_channels=residual_channels + self.embedding_dim, out_channels=self.embedding_dim,
                                     kernel_size=(1, 1)).to(device)


        self.feature_conv = torch.nn.Conv3d(in_channels=11, out_channels=1, kernel_size=1)

        self.time_conv = torch.nn.Conv2d(in_channels=self.seq_x, out_channels=1, kernel_size=(3, 1), padding='same')

    def forward(self, input,x_adj):
        input = input.to(device)
        x_adj = x_adj.to(device)

        x_adj = x_adj.permute(0, 4, 1, 2, 3)
        reduced_features = self.feature_conv(x_adj).squeeze(1)
        reduced_adj = self.time_conv(reduced_features)
        reduced_adj = (reduced_adj - reduced_adj.min()) / (reduced_adj.max() - reduced_adj.min())

        batch, feature_size,num_node, back_length = input.shape
        x = self.start_conv(input[:,:2,:,:])

        if self.supports is not None:
            self.supports[0] = F.softmax(F.relu(self.supports[0]), dim=1)
        new_supports_norm = [normalize_adj(support.cpu().detach().numpy()) for support in self.supports]

        if self.adj:
            new_supports_norm.extend([reduced_adj[b, 0, :, :].detach().cpu().numpy() for b in range(reduced_adj.shape[0])])

        spectral_embeddings = [
            SpectralEmbedding(n_components=self.embedding_dim, affinity='precomputed').fit_transform(adj)
            for adj in new_supports_norm]

        spectral_embed_expanded = [
            torch.tensor(embed).unsqueeze(0).repeat(batch, 1, 1).unsqueeze(3).repeat(1, 1, 1, back_length).to(device)
            for embed in spectral_embeddings]

        x_reshaped = x.permute(0, 2, 1, 3)
        Adj_with_embeddings = torch.cat([x_reshaped] + spectral_embed_expanded, dim=2).permute(0, 2, 1,3)

        Adj_embedding = (
            self.reduce_conv_adj(Adj_with_embeddings) if self.adj else self.reduce_conv(Adj_with_embeddings))
        Adj_embedding = Adj_embedding.permute(0, 2, 3, 1).reshape(self.batch_size * 82, self.seq_x, self.embedding_dim)


        attn = torch.matmul( Adj_embedding.permute(0, 2, 1), Adj_embedding)
        attn = torch.softmax(attn, dim=-1)
        Adj_embedding = torch.matmul(Adj_embedding,attn)


        weekday = input[:, 1, :, :].int().to(device)
        position = torch.LongTensor([i for i in range(14)]).repeat(batch, num_node, 1).to(device)
        volume = input[:, 2:, :, :].reshape(batch, num_node, back_length, -1).int().to(device)

        # 嵌入体积、星期几、位置
        volume_embedding = self.embedding_volume(volume)
        volume_embedding = torch.sum(volume_embedding, 3).reshape(-1, volume_embedding.shape[2], self.embedding_dim)
        weekday_embedding = self.embedding_week(weekday.reshape(-1, weekday.shape[2]))
        position_embedding = self.embedding_position(position.reshape(-1, position.shape[2]))


        volume_embedding = apply_attention(volume_embedding)
        weekday_embedding = apply_attention(weekday_embedding)
        position_embedding = apply_attention(position_embedding)

        embedding = volume_embedding + weekday_embedding + position_embedding + Adj_embedding

        q = self.linear_q(embedding)
        k = self.linear_k(embedding)
        v = self.linear_v(embedding)

        attn = torch.matmul(q, k.permute(0, 2, 1))
        attn = torch.softmax(attn, dim=-1)
        attn_value = torch.matmul(attn, v)

        x_temp = torch.mean(attn_value, dim=2)

        x_temp = input[:, 0, :, :] + x_temp.reshape(batch, num_node, -1)

        forecast = self.mlp(x_temp)

        return forecast.transpose(1,2).unsqueeze(3)



