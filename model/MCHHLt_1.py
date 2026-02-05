import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GINConv
from torch_geometric.nn import global_max_pool as gmp
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch_geometric.nn import HypergraphConv
import logging
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv, GATConv
import networkx as nx

EPS = torch.tensor(1e-15)


# 定义同类实体的消息传递层
class EntityGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EntityGCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 使用GCN进行同类图的消息传递
        x = self.conv(x, edge_index)
        return F.relu(x)


class EntityHGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EntityHGNNLayer, self).__init__()
        self.conv = HypergraphConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.relu(x)


# 定义通道内消息传递
class IntraMessagePassing(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IntraMessagePassing, self).__init__()
        self.drug_layer = EntityHGNNLayer(in_channels, out_channels)
        self.gene_layer = EntityHGNNLayer(in_channels, out_channels)
        self.disease_layer = EntityGCNLayer(in_channels, out_channels)

    def forward(self, drug_features, gene_features, disease_features, drug_edge_index, gene_edge_index,
                disease_edge_index):
        drug_edge_attr = torch.randn(148376, 768)
        gene_edge_attr = torch.randn(10143,768)
        disease_edge_attr = torch.randn(264,768)
        # 分别在药物、基因、疾病图中进行消息传递
        drug_features = self.drug_layer(drug_features, drug_edge_index)
        gene_features = self.gene_layer(gene_features, gene_edge_index)
        disease_features = self.disease_layer(disease_features, disease_edge_index)
        return drug_features, gene_features, disease_features


class DrugEncoder(nn.Module):
    def __init__(self, output):
        super(DrugEncoder, self).__init__()
        self.batch_dis1 = nn.BatchNorm1d(output)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.15)
        self.conv1 = GINConv(nn.Sequential(nn.Linear(78, output),
                                           nn.ReLU(),
                                           nn.Linear(output, output)))
        self.bn1 = torch.nn.BatchNorm1d(output)

        self.conv2 = GINConv(nn.Sequential(nn.Linear(output, output),
                                           nn.ReLU(),
                                           nn.Linear(output, output)))
        self.bn2 = torch.nn.BatchNorm1d(output)

        self.conv3 = GINConv(nn.Sequential(nn.Linear(output, output),
                                           nn.ReLU(),
                                           nn.Linear(output, output)))
        self.bn3 = torch.nn.BatchNorm1d(output)
        self.fc1_xd = nn.Linear(output, output)
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, dru_adj):
        # -----drug_train
        x_d, edge_index, batch = dru_adj.x, dru_adj.edge_index, dru_adj.batch  # 这里的批量，比如某个批量有3个，则为000，111，222
        x_d = self.relu(self.conv1(x_d, edge_index))
        x_d = self.bn1(x_d)
        x_d = self.relu(self.conv2(x_d, edge_index))
        x_d = self.bn2(x_d)
        x_d = self.relu(self.conv3(x_d, edge_index))
        x_d = self.bn3(x_d)
        x_d = gmp(x_d, batch)  # 全局最大池化
        x_d = self.relu(self.fc1_xd(x_d))
        x_d = self.drop_out(x_d)

        return x_d


class InterMessagePassing(nn.Module):
    def __init__(self, in_channels, dim_1):
        super(InterMessagePassing, self).__init__()
        self.conv1 = HypergraphConv(in_channels, dim_1)
        self.conv2 = HypergraphConv(dim_1, dim_1 // 2)
        self.conv3 = HypergraphConv(dim_1 // 2, dim_1 // 4)

    def forward(self, x, edge):
        # edges_transposed = edge.t()
        x = x
        x = self.conv1(x, edge)
        x = torch.relu(x)
        x = self.conv2(x, edge)
        x = torch.relu(x)
        x = self.conv3(x, edge)
        x = torch.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc23 = nn.Linear(in_channels // 4, in_channels // 8)
        self.batch23 = nn.BatchNorm1d(in_channels // 8)
        self.fc3 = nn.Linear(in_channels // 8, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, triple):
        h_0 = torch.stack([torch.cat((graph_embed[d], graph_embed[g], graph_embed[i]), dim=-1) for d, g, i in triple])
        h_1 = torch.relu(self.fc1(h_0))
        h_1 = self.batch1(h_1)
        h_2 = torch.relu(self.fc2(h_1))
        h_2 = self.batch2(h_2)
        h_23 = torch.tanh(self.fc23(h_2))
        h_23 = self.batch23(h_23)
        h_3 = self.fc3(h_23)
        return h_23, h_3.squeeze(dim=1)


class TextEncoder(nn.Module):
    def __init__(self, device, bert_model_name='bert-base-uncased'):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.device = device

    def forward(self, texts):
        # 对输入文本进行tokenize和编码
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.bert(**inputs)
        # 使用最后一个隐层的平均值作为文本嵌入
        return outputs.last_hidden_state.mean(dim=1)


class MCHHLT_1(nn.Module):
    def __init__(self, drug_encoder, inter_messagepassing, intra_messagepassing, decoder,out_channels):
        super(MCHHLT_1, self).__init__()
        self.inter_messagepassing = inter_messagepassing
        self.decoder = decoder
        self.drug_encoder = drug_encoder
        self.intra_messagepassing = intra_messagepassing
        self.weight = nn.Parameter(torch.Tensor(out_channels, out_channels))  # hidden_channels

    def discriminate(self, z, summary, sigmoid=True):
        # d_z = z.shape[1]  # z 的列数
        # d_w = summary.shape[0]  # summary 的行数
        # weight = nn.Parameter(torch.randn(d_z, d_w, device=z.device))
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def DGI_loss(self, pos, summary, neg):
        """
        Computes the mutual information maximization objective for a multi-class setting.

        Arguments:
        pos_z : Tensor
            The embeddings of positive samples (real graph).
        summary : Tensor
            The global summary (context) vector.
        *args : Tensor
            The embeddings of negative samples for different classes (negative samples for different categories).

        Returns:
        loss : Tensor
            The total loss computed from positive and negative samples.
        """
        # 正样本损失：最大化正样本与全局摘要的相似度
        pos_loss = -torch.log(self.discriminate(pos, summary) + EPS).mean()

        # 负样本损失：最小化负样本与全局摘要的相似度
        neg_loss = -torch.log(1 - self.discriminate(neg, summary) + EPS).mean()
        loss = pos_loss+neg_loss

        return loss

    def forward(self, drug_graph, drug_fea, gene_fea, disease_fea, hyperedge_index, triples_data, drug_edge_index,
                gene_edge_index, disease_edge_index,hypergraph_neg_edge):
        drug_graph = self.drug_encoder(drug_graph)
        drug_fea = (drug_fea + drug_graph) / 2
        drug_fea,gene_fea,disease_fea = self.intra_messagepassing(drug_fea,gene_fea,disease_fea,drug_edge_index,gene_edge_index,disease_edge_index)
        X = torch.concat((drug_fea, gene_fea, disease_fea), 0)
        hyper_output_pos = self.hyper_graph_encoder(X, hyperedge_index).squeeze()#(768,512)
        summary = torch.sigmoid(hyper_output_pos.mean(dim=0))
        embed_pos = torch.sigmoid(hyper_output_pos)
        hyper_output_neg = self.hyper_graph_encoder(X,hypergraph_neg_edge).squeeze()
        embed_neg = torch.sigmoid(hyper_output_neg)
        emb, logits = self.decoder(hyper_output_pos, triples_data)
        return emb, logits,embed_pos,embed_neg,summary
