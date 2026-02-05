import argparse
import random
from Utils_mine.process_smiles import *
import csv
import pandas as pd
from torch_geometric.data import DataLoader, Data



def random_init_graph(num_nodes=10, num_edges=20, feature_size=78):
    node_features = [torch.randn(feature_size).tolist() for _ in range(num_nodes)]
    edge_index = [[], []]
    for _ in range(num_edges):
        node1 = torch.randint(0, num_nodes, (1,)).item()
        node2 = torch.randint(0, num_nodes, (1,)).item()
        edge_index[0].append(node1)
        edge_index[1].append(node2)

    return num_nodes, node_features, edge_index


def smile_to_graph_checked(smile):
    try:
        g = smile_to_graph(smile)
        if g is None:
            raise ValueError("Invalid SMILES")
        return g
    except Exception as e:
        return random_init_graph()


def drug_fea_process(smiles_file, drug_num):
    reader = csv.reader(open(smiles_file))
    smile_graph = []
    for item in reader:
        smile = item[1]
        if smile and smile != 'N/A':
            g = smile_to_graph_checked(smile)
        else:
            g = random_init_graph()
        smile_graph.append(g)
    dru_data = GraphDataset_v(xc=smile_graph, cid=[i for i in range(drug_num + 1)])
    dru_data = torch.utils.data.DataLoader(dataset=dru_data, batch_size=drug_num, shuffle=False,
                                           collate_fn=collate)
    for step, batch_drug in enumerate(dru_data):
        drug_data = batch_drug
    return drug_data


def process_gene_descriptions(file_name):
    descriptions = []
    with open(file_name, 'r', encoding='gbk') as file:
        reader = csv.reader(file)
        for row in reader:
            descriptions.append(row[3])  # 假设第四列是描述
    return descriptions


def process_disease_descriptions(file_name):
    descriptions = []
    with open(file_name, 'r', encoding='gbk') as file:
        reader = csv.reader(file)
        for row in reader:
            descriptions.append(row[2])  # 假设第四列是描述
    return descriptions


def load_edge_index_from_csv(file_path):
    # 读取CSV文件，假设它没有列名，只有两列，代表源节点和目标节点
    df = pd.read_csv(file_path, header=None, names=['head', 'tail'])
    # 将head和tail两列转换为PyTorch张量
    edge_index = torch.tensor([df['head'].values, df['tail'].values], dtype=torch.long)

    return edge_index


def parameters_set():
    parser = argparse.ArgumentParser(description='using GAT process association matrix and GIN&Conv process feature')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='which GPU to use. Set -1 to use CPU.')
    parser.add_argument('--hypergraph_loss_ratio', type=float, default=0.8,
                        help='the weight of hypergraph_loss_ratio')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1,  # 0.005
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='the random seeds')
    parser.add_argument('--k_fold', type=int, default=5,
                        help='k_fold')
    parser.add_argument('--BATCH_SIZE', type=int, default=512,
                        help='BATCH_SIZE')
    parser.add_argument('--Task_Type', type=str, default='multi_classification',
                        help='Task_Type')
    parser.add_argument('--patience', type=int, default=15,
                        help='number of patience of early stopping mechanism')
    parser.add_argument('--train_neg_num', type=int, default=1,
                        help='number of negative samples')
    parser.add_argument('--test_neg_num', type=int, default=5,
                        help='number of negative samples')

    args = parser.parse_args()
    return args


def generate_negative_samples(data, num_drugs, num_genes, num_diseases, num_neg_samples=1):
    negative_samples = []
    #检查生成的三元组 (drug_idx, neg_gene_idx, neg_disease_idx) 是否已经存在于 data 中：
    for drug_idx, gene_idx, disease_idx in data:
        for _ in range(num_neg_samples):
            neg_gene_idx = np.random.randint(0, num_genes)
            neg_disease_idx = np.random.randint(0, num_diseases)
            while (drug_idx, neg_gene_idx, neg_disease_idx) in data:  # 确保是负样本
                neg_gene_idx = np.random.randint(0, num_genes)
                neg_disease_idx = np.random.randint(0, num_diseases)
            #将局部索引（相对基因或疾病）转换为全局索引：
            neg_gene_global_idx = num_drugs + neg_gene_idx
            neg_disease_global_idx = num_drugs + num_genes + neg_disease_idx
            negative_samples.append([drug_idx, neg_gene_global_idx, neg_disease_global_idx])
    return negative_samples


def build_hypergraph(data):
    pairs = np.array(data[:, 0:3]).reshape(1, -1)
    pairs_num = np.expand_dims(np.arange(len(data)), axis=-1)
    hyper_edge_num = np.concatenate((pairs_num, pairs_num, pairs_num), axis=1)
    hyper_edge_num = np.array(hyper_edge_num).reshape(1, -1)
    hyper_graph = np.concatenate((pairs, hyper_edge_num), axis=0)
    hyper_graph = torch.from_numpy(hyper_graph).type(torch.LongTensor)
    return hyper_graph
