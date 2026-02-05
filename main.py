import sys
import os
import logging
import json
import random
import numpy as np
import torch
import torch.backends.cudnn
import argparse  # Added for argument parsing
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Shared library imports
from utils.Tools2 import *
from metrics import *
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Specific imports for binary task
from utils.right_negative_sample_generate2 import *

# Model imports with aliases to prevent conflict
import model.MCHHLt_2 as m15
import model.MCHHLt_1 as m_bin  # For Task 1 (binary)

# Global Constants
num_drugs = 8076
num_genes = 8387
num_diseases = 3525
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def setup_logger(task_type, lr, epoch, log_dir="logs/"):
    task_log_dir = os.path.join(log_dir, task_type)
    if not os.path.exists(task_log_dir):
        os.makedirs(task_log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{task_log_dir}/{task_type}_lr{lr}_epoch{epoch}_{timestamp}.log"

    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def random_seed(sd):
    random.seed(sd)
    os.environ['PYTHONHASHSEED'] = str(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)


def early_stopping(val_accuracy, best_accuracy, patience, patience_counter, model, model_path):
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        patience_counter = 0
        torch.save(model.state_dict(), model_path)
        print(f"New best model saved with accuracy: {best_accuracy:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement in accuracy for {patience_counter} epoch(s)")

    if patience_counter > patience:
        print("Early stopping triggered!")
        return True, best_accuracy, patience_counter
    return False, best_accuracy, patience_counter


def get_hypergraph_edge(triples):
    hypergraph_edge = [[], []]
    for edge_idx, (drug_idx, gene_idx, disease_idx) in enumerate(triples):
        # Index conversion to global indices
        gene_global_idx = num_drugs + int(gene_idx)
        disease_global_idx = num_drugs + num_genes + int(disease_idx)

        hypergraph_edge[0].extend([drug_idx, gene_global_idx, disease_global_idx])
        hypergraph_edge[1].extend([edge_idx] * 3)
    hypergraph_edge = torch.tensor(hypergraph_edge).long()
    return hypergraph_edge


def convert_float32(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError("Type not serializable")


def get_negative_data(data):
    # Used in Multi
    triples = data
    negative_samples = generate_negative_samples(triples, num_drugs, num_genes, num_diseases)
    return negative_samples


# ==========================================
# Task 2 (Main 19) Specific Functions
# ==========================================

def get_train_val_data_task2(cv_data, cv_labels, index):
    triples = cv_data[index]
    labels = cv_labels[index]
    global_triples = []
    for edge_idx, (drug_idx, gene_idx, disease_idx) in enumerate(triples):
        gene_global_idx = num_drugs + int(gene_idx)
        disease_global_idx = num_drugs + num_genes + int(disease_idx)
        global_triples.append([int(drug_idx), gene_global_idx, disease_global_idx])
    labels = torch.tensor(labels).long()
    return global_triples, labels


def visualize_tsne(embeddings, labels, title="t-SNE Visualization", save_path=None, save_data_path=None):
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)

    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(embeddings, labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
    else:
        silhouette_avg = None
        print("Silhouette Score cannot be calculated with fewer than 2 clusters.")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=labels,
        palette=sns.color_palette("hsv", len(set(labels))),
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc='best', bbox_to_anchor=(1, 1), title="Labels")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    if save_data_path:
        save_data = {
            "tsne_results": tsne_results.tolist(),
            "labels": labels.tolist(),
            "silhouette_score": silhouette_avg,
            "title": title
        }
        with open(save_data_path, "w") as f:
            json.dump(save_data, f, default=convert_float32)
        print(f"t-SNE results and parameters saved to {save_data_path}")


# ==========================================
# Task 1 (Binary) Specific Functions
# ==========================================

def get_train_val_data_task1(cv_data, index):
    cv_labels = cv_data[3]
    cv_data_slice = cv_data[:3]
    triples = cv_data_slice[index]
    labels = cv_labels[index]
    global_triples = []
    for edge_idx, (drug_idx, gene_idx, disease_idx) in enumerate(triples):
        gene_global_idx = num_drugs + int(gene_idx)
        disease_global_idx = num_drugs + num_genes + int(disease_idx)
        global_triples.append([int(drug_idx), gene_global_idx, disease_global_idx])
    labels = torch.tensor(labels).long()
    return global_triples, labels


def series_num_task1(cv_data):
    cv_data = np.array(cv_data)
    triples = cv_data[:, :3]
    labels = cv_data[:, 3]
    global_triples = []
    for edge_idx, (drug_idx, gene_idx, disease_idx) in enumerate(triples):
        gene_global_idx = num_drugs + int(gene_idx)
        disease_global_idx = num_drugs + num_genes + int(disease_idx)
        global_triples.append([int(drug_idx), gene_global_idx, disease_global_idx])
    labels = torch.tensor(labels).long()
    return global_triples, labels


# ==========================================
# Execution Logics
# ==========================================

def run_task1_binary(args):
    print(">>> Executing TASK 1: Binary Classification Logic")
    random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_path = 'MCHHL_binary_model.pth'
    logger = setup_logger(args.Task_Type, args.lr, args.epochs)

    # Load Embeddings
    drug_embeddings = torch.squeeze(torch.load('model/drug_description.pt')).to(device)
    gene_embeddings = torch.squeeze(torch.load('model/gene_description.pt')).to(device)
    disease_embeddings = torch.squeeze(torch.load('model/disease_description.pt')).to(device)

    smiles_file = 'Data_mine/updated_drug_smile4.csv'
    gene_desc = 'Data_mine/gene_merged_cleaned.csv'
    disease_desc = 'Data_mine/disease.csv'

    # CPM Hyperedges
    drug_edge_index = pd.read_csv('Data_mine/cpm_drug_hyperedges.csv')
    drug_edge_index = torch.tensor(drug_edge_index.values.T, dtype=torch.long).to(device)
    gene_edge_index = pd.read_csv('Data_mine/cpm_gene_hyperedges.csv')
    gene_edge_index = torch.tensor(gene_edge_index.values.T, dtype=torch.long).to(device)
    disease_edge_index = pd.read_csv('Data_mine/disease_hyperedge_index.csv', header=None)
    disease_edge_index = torch.tensor(disease_edge_index.values.T, dtype=torch.long).to(device)

    batch_drug = drug_fea_process(smiles_file, 8076)
    batch_drug = batch_drug.to(device)

    # Load Relation Data
    if args.Task_Type == 'binary_classification':
        adj_data = np.loadtxt('Data_mine/adj_multiclassify.txt')
    elif args.Task_Type == 'multi_classification':
        adj_data = np.loadtxt('Data_mine/binary_adj_data_modified.txt')
    else:
        adj_data = np.loadtxt('Data_mine/binary_adj_data_modified.txt')

    cv_data = np.array(adj_data)
    kf = KFold(n_splits=5)
    fold_num = 0
    all_test_results = {
        'roc_auc': [], 'precision': [], 'recall': [], 'f1': [], 'pr_auc': [], 'accuracy': []
    }

    for train_val_index, test_index in kf.split(cv_data):
        best_results = {'accuracy': 0, 'roc_auc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'pr_auc': 0}
        patience = args.patience
        patience_counter = 0
        fold_num += 1

        train_val_data, test_data = cv_data[train_val_index], cv_data[test_index]
        train_index, val_index = train_test_split(np.arange(len(train_val_data)), test_size=0.2, random_state=args.seed)

        train_data_split = train_val_data[train_index]
        val_data_split = train_val_data[val_index]

        # Generate Negative Samples (Task 1 specific)
        train_neg, train_all = trainvaltest_neg_data_generate(train_data_split, args.train_neg_num, seed=1)
        val_neg, val_all = trainvaltest_neg_data_generate(val_data_split, args.test_neg_num, seed=1)
        test_neg, test_all = trainvaltest_neg_data_generate(test_data, args.test_neg_num, seed=1)

        train_data, train_labels = series_num_task1(train_all)
        val_data, val_labels = series_num_task1(val_all)
        test_data, test_labels = series_num_task1(test_all)

        hypergraph_edge = get_hypergraph_edge(train_val_data[:, :3])
        hypergraph_edge = hypergraph_edge.to(device)

        trainval_neg = np.array(train_neg + val_neg)
        hypergraph_neg_edge = get_hypergraph_edge(trainval_neg[:, :3])
        hypergraph_neg_edge = hypergraph_neg_edge.to(device)

        train_data = torch.tensor(train_data).to(device)
        train_labels = train_labels.to(device)
        val_data = torch.tensor(val_data).to(device)
        val_labels = val_labels.to(device)
        test_data = torch.tensor(test_data).to(device)
        test_labels = test_labels.to(device)

        train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=args.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=args.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=args.BATCH_SIZE, shuffle=True)

        # USE MODEL_BINARY
        model = m_bin.My_Model(
            m_bin.DrugEncoder(768),
            m_bin.HgnnEncoder(256, 512),
            m_bin.Decoder(384),
            m_bin.IntraMessagePassing(768, 256),
            128
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_losses = []
        val_losses = []
        logger.info(f'Fold {fold_num} started...')

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            all_labels = []
            all_logits = []

            for batch in train_loader:
                batch_data, batch_labels = batch
                optimizer.zero_grad()
                emb, logits, embed_pos, embed_neg, summary = model(
                    batch_drug, drug_embeddings, gene_embeddings, disease_embeddings, hypergraph_edge,
                    batch_data, drug_edge_index, gene_edge_index, disease_edge_index, hypergraph_neg_edge
                )
                batch_labels = batch_labels.float().cuda()
                logits = logits.float().cuda()

                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                all_logits.append(logits.detach().cpu())
                all_labels.extend(batch_labels.cpu().numpy())

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.tensor(all_labels).long()
            accuracy, precision, recall, f1, roc_auc, pr_auc = compute_binary_metrics(all_labels, all_logits)

            logger.info(f'Epoch [{epoch + 1}/{args.epochs}]')
            logger.info(f'Training Loss: {avg_loss:.4f}')
            logger.info(
                f'Traning:accuracy:{accuracy:.4f},Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}')

            model.eval()
            val_total_loss = 0
            val_all_logits = []
            val_all_labels = []
            with torch.no_grad():
                val_embeddings = []
                val_labels_list = []
                for step, val_batch in enumerate(val_loader):
                    val_data_batch, val_labels_batch = val_batch
                    val_emb, val_logits, embed_pos, embed_neg, summary = model(
                        batch_drug, drug_embeddings, gene_embeddings, disease_embeddings,
                        hypergraph_edge, val_data_batch, drug_edge_index, gene_edge_index,
                        disease_edge_index, hypergraph_neg_edge
                    )

                    val_labels_batch = val_labels_batch.float().cuda()
                    val_logits = val_logits.float().cuda()
                    val_loss = criterion(val_logits, val_labels_batch)
                    val_total_loss += val_loss.item()

                    val_all_logits.append(val_logits.detach().cpu())
                    val_all_labels.extend(val_labels_batch.cpu().numpy())
                    val_embeddings.append(val_emb.cpu())
                    val_labels_list.extend(val_labels_batch.cpu().numpy())

                val_avg_loss = val_total_loss / len(val_loader)
                val_losses.append(val_avg_loss)
                val_all_logits = torch.cat(val_all_logits, dim=0)
                val_all_labels = torch.tensor(val_all_labels).long()
                val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc = compute_binary_metrics(
                    val_all_labels, val_all_logits)

                logger.info(f'Validation Loss: {val_avg_loss:.4f}')
                logger.info(
                    f'Validation:accuracy:{val_accuracy:.4f},Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}, PR-AUC: {val_pr_auc:.4f}')

                stop_training, best_results['accuracy'], patience_counter = early_stopping(
                    val_accuracy, best_results['accuracy'], patience, patience_counter, model, model_path
                )

                if val_accuracy >= best_results['accuracy']:
                    best_results.update({
                        'precision': val_precision,
                        'recall': val_recall,
                        'f1': val_f1,
                        'pr_auc': val_pr_auc,
                        'ROC-AUC': val_roc_auc
                    })
                print(f"Epoch {epoch}: Learning rate: {optimizer.param_groups[0]['lr']}")
                val_embeddings = torch.cat(val_embeddings, dim=0).numpy()

                if stop_training:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}, Best accuracy: {best_results['accuracy']:.4f}")
                    break

        # Test Phase
        model.eval()
        test_all_logits = []
        test_all_labels = []
        with torch.no_grad():
            test_embeddings = []
            test_labels_list = []
            for test_batch in test_loader:
                test_data_batch, test_labels_batch = test_batch
                test_emb, test_logits, test_embed_pos, test_embed_neg, test_summary = model(
                    batch_drug, drug_embeddings, gene_embeddings, disease_embeddings,
                    hypergraph_edge, test_data_batch, drug_edge_index, gene_edge_index, disease_edge_index,
                    hypergraph_neg_edge
                )
                test_all_logits.append(test_logits.detach().cpu())
                test_all_labels.extend(test_labels_batch.cpu().numpy())
                test_embeddings.append(test_emb.cpu())
                test_labels_list.extend(test_labels_batch.cpu().numpy())

        test_all_logits = torch.cat(test_all_logits, dim=0)
        test_all_labels = torch.tensor(test_all_labels).long()
        test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc = compute_binary_metrics(
            test_all_labels, test_all_logits)
        test_embeddings = torch.cat(test_embeddings, dim=0).numpy()

        all_test_results['roc_auc'].append(test_roc_auc)
        all_test_results['precision'].append(test_precision)
        all_test_results['recall'].append(test_recall)
        all_test_results['f1'].append(test_f1)
        all_test_results['pr_auc'].append(test_pr_auc)
        all_test_results['accuracy'].append(test_accuracy)

        logger.info(
            f"Fold {fold_num} Test Results - Accuracy:{test_accuracy:.4f}, ROC-AUC: {test_roc_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, PR-AUC: {test_pr_auc:.4f}")

    logger.info(f'All folds best results:')
    for key, values in all_test_results.items():
        logger.info(f'{key}: {values}')

    logger.info(f'Mean Best Accuracy: {np.mean(all_test_results["accuracy"]):.4f}')
    print(f'Average Test ROC-AUC: {np.mean(all_test_results["roc_auc"]):.4f}')
    print(f'Average Test Precision: {np.mean(all_test_results["precision"]):.4f}')
    print(f'Average Test Recall: {np.mean(all_test_results["recall"]):.4f}')
    print(f'Average Test F1: {np.mean(all_test_results["f1"]):.4f}')
    print(f'Average Test PR-AUC: {np.mean(all_test_results["pr_auc"]):.4f}')


def run_task2_Multi(args):
    print(">>> Executing TASK 2: Multi Logic")
    random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    current_time = datetime.now().strftime("%Y-%m-%d_%H")
    model_path = f'MCHHL_model_{current_time}.pth'
    logger = setup_logger(args.Task_Type, args.lr, args.epochs)

    # Load Embeddings
    drug_embeddings = torch.squeeze(torch.load('model/drug_description.pt')).to(device)
    gene_embeddings = torch.squeeze(torch.load('model/gene_description.pt')).to(device)
    disease_embeddings = torch.squeeze(torch.load('model/disease_description.pt')).to(device)

    smiles_file = 'Data_mine/updated_drug_smile4.csv'
    gene_desc = 'Data_mine/gene_merged_cleaned.csv'
    disease_desc = 'Data_mine/disease.csv'

    # CPM Hyperedges
    drug_edge_index = pd.read_csv('Data_mine/cpm_drug_hyperedges.csv')
    drug_edge_index = torch.tensor(drug_edge_index.values.T, dtype=torch.long).to(device)
    gene_edge_index = pd.read_csv('Data_mine/cpm_gene_hyperedges.csv')
    gene_edge_index = torch.tensor(gene_edge_index.values.T, dtype=torch.long).to(device)
    disease_edge_index = pd.read_csv('Data_mine/disease_hyperedge_index.csv', header=None)
    disease_edge_index = torch.tensor(disease_edge_index.values.T, dtype=torch.long).to(device)

    batch_drug = drug_fea_process(smiles_file, 8076)
    batch_drug = batch_drug.to(device)

    # Load Relation Data
    if args.Task_Type == 'binary_classification':
        adj_data = np.loadtxt('Data_mine/adj_binaryclassify.txt')
    if args.Task_Type == 'multi_classification':
        adj_data = np.loadtxt('Data_mine/adj_multiclassify.txt')

    neg_data = np.loadtxt('Data_mine/hyper_neg_edge.txt')

    # Shuffle Data
    np.random.shuffle(adj_data)

    cv_data = [(row[0], row[1], row[2]) for row in adj_data]
    cv_data = np.array(cv_data)
    neg_data = [(row[0], row[1], row[2]) for row in neg_data]
    neg_data = np.array(neg_data)
    cv_labels = [int(row[3]) for row in adj_data]
    cv_labels = np.array(cv_labels)

    kf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    fold_num = 0
    print("learning rate = ", args.lr)

    all_test_results = {
        'roc_auc': [], 'precision': [], 'recall': [], 'f1': [], 'pr_auc': [], 'accuracy': []
    }

    for train_val_index, test_index in kf.split(cv_data, cv_labels):
        best_results = {'accuracy': 0, 'roc_auc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'pr_auc': 0}
        patience = args.patience
        patience_counter = 0
        fold_num += 1

        train_val_data, test_data = cv_data[train_val_index], cv_data[test_index]
        train_val_labels, test_labels = cv_labels[train_val_index], cv_labels[test_index]

        train_index, val_index = train_test_split(np.arange(len(train_val_data)), test_size=0.2, random_state=args.seed)
        train_data, train_labels = get_train_val_data_task2(train_val_data, train_val_labels, train_index)
        val_data, val_labels = get_train_val_data_task2(train_val_data, train_val_labels, val_index)
        test_data, test_labels = get_train_val_data_task2(test_data, test_labels, np.arange(len(test_data)))

        hypergraph_edge = get_hypergraph_edge(cv_data)
        hypergraph_edge = hypergraph_edge.to(device)
        hypergraph_neg_edge = get_hypergraph_edge(neg_data)
        hypergraph_neg_edge = hypergraph_neg_edge.to(device)

        train_data = torch.tensor(train_data).to(device)
        train_labels = train_labels.to(device)
        val_data = torch.tensor(val_data).to(device)
        val_labels = val_labels.to(device)
        test_data = torch.tensor(test_data).to(device)
        test_labels = test_labels.to(device)

        train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=args.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=args.BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=args.BATCH_SIZE)

        # USE MODEL15
        model = m15.My_Model(
            m15.DrugEncoder(768),
            m15.HgnnEncoder(256, 512),
            m15.Decoder(384, 4),
            m15.IntraMessagePassing(768, 256),
            128
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        train_losses = []
        val_losses = []
        logger.info(f'Fold {fold_num} started...')

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            all_labels = []
            all_logits = []
            for batch in train_loader:
                batch_data, batch_labels = batch
                optimizer.zero_grad()
                emb, logits, embed_pos, embed_neg, summary = model(
                    batch_drug, drug_embeddings, gene_embeddings, disease_embeddings, hypergraph_edge,
                    batch_data, drug_edge_index, gene_edge_index, disease_edge_index, hypergraph_neg_edge
                )
                loss_1 = criterion(logits.float().cuda(), batch_labels)
                loss_2 = model.DGI_loss(embed_pos, summary, embed_neg)
                loss = args.hypergraph_loss_ratio * loss_1 + (1 - args.hypergraph_loss_ratio) * loss_2
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                all_logits.append(logits.detach().cpu())
                all_labels.extend(batch_labels.cpu().numpy())

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.tensor(all_labels).long()
            accuracy, precision, recall, f1, roc_auc, pr_auc = compute_metrics(all_labels, all_logits)

            logger.info(f'Epoch [{epoch + 1}/{args.epochs}]')
            logger.info(f'Training Loss: {avg_loss:.4f}')
            logger.info(
                f'Traning:accuracy:{accuracy:.4f},Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}')

            model.eval()
            val_total_loss = 0
            val_all_logits = []
            val_all_labels = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_data_batch, val_labels_batch = val_batch
                    val_emb, val_logits, embed_pos, embed_neg, summary = model(
                        batch_drug, drug_embeddings, gene_embeddings, disease_embeddings,
                        hypergraph_edge, val_data_batch, drug_edge_index, gene_edge_index,
                        disease_edge_index, hypergraph_neg_edge
                    )
                    val_loss_1 = criterion(val_logits.float().cuda(), val_labels_batch)
                    val_loss_2 = model.DGI_loss(embed_pos, summary, embed_neg)
                    val_loss = args.hypergraph_loss_ratio * val_loss_1 + (1 - args.hypergraph_loss_ratio) * val_loss_2
                    val_total_loss += val_loss.item()
                    val_all_logits.append(val_logits.detach().cpu())
                    val_all_labels.extend(val_labels_batch.cpu().numpy())

                val_avg_loss = val_total_loss / len(val_loader)
                val_losses.append(val_avg_loss)
                val_all_logits = torch.cat(val_all_logits, dim=0)
                val_all_labels = torch.tensor(val_all_labels).long()
                val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc = compute_metrics(
                    val_all_labels, val_all_logits)

                logger.info(f'Validation Loss: {val_avg_loss:.4f}')
                logger.info(
                    f'Validation:accuracy:{val_accuracy:.4f},Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}, PR-AUC: {val_pr_auc:.4f}')

                stop_training, best_results['accuracy'], patience_counter = early_stopping(
                    val_accuracy, best_results['accuracy'], patience, patience_counter, model, model_path
                )
                if val_accuracy >= best_results['accuracy']:
                    best_results.update({
                        'precision': val_precision,
                        'recall': val_recall,
                        'f1': val_f1,
                        'pr_auc': val_pr_auc,
                        'ROC-AUC': val_roc_auc
                    })
                scheduler.step(val_avg_loss)
                print(f"Epoch {epoch}: Learning rate: {optimizer.param_groups[0]['lr']}")

                if stop_training:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}, Best accuracy: {best_results['accuracy']:.4f}")
                    break

        # Test Phase
        model.eval()
        test_all_logits = []
        test_all_labels = []
        with torch.no_grad():
            test_embeddings = []
            test_labels_list = []
            for test_batch in test_loader:
                test_data_batch, test_labels_batch = test_batch
                test_emb, test_logits, test_embed_pos, test_embed_neg, test_summary = model(
                    batch_drug, drug_embeddings, gene_embeddings, disease_embeddings,
                    hypergraph_edge, test_data_batch, drug_edge_index, gene_edge_index, disease_edge_index,
                    hypergraph_neg_edge
                )
                test_all_logits.append(test_logits.detach().cpu())
                test_all_labels.extend(test_labels_batch.cpu().numpy())
                test_embeddings.append(test_emb.cpu())
                test_labels_list.extend(test_labels_batch.cpu().numpy())

        test_all_logits = torch.cat(test_all_logits, dim=0)
        test_all_labels = torch.tensor(test_all_labels).long()
        test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc = compute_metrics(
            test_all_labels, test_all_logits)
        test_embeddings = torch.cat(test_embeddings, dim=0).numpy()

        picture_folder = "tsne_picture"
        json_folder = "tsne_json"
        if not os.path.exists(picture_folder): os.makedirs(picture_folder)
        if not os.path.exists(json_folder): os.makedirs(json_folder)

        visualize_tsne(
            test_embeddings,
            test_labels_list,
            title="t-SNE Visualization of Test Set",
            save_path=os.path.join(picture_folder, f"MCHLL_tsne_test_{current_time}_fold_{fold_num}.png"),
            save_data_path=os.path.join(json_folder, f"MCHLL_tsne_test_data_{current_time}_fold_{fold_num}.json")
        )

        all_test_results['roc_auc'].append(test_roc_auc)
        all_test_results['precision'].append(test_precision)
        all_test_results['recall'].append(test_recall)
        all_test_results['f1'].append(test_f1)
        all_test_results['pr_auc'].append(test_pr_auc)
        all_test_results['accuracy'].append(test_accuracy)

        logger.info(
            f"Fold {fold_num} Test Results - Accuracy:{test_accuracy:.4f}, ROC-AUC: {test_roc_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, PR-AUC: {test_pr_auc:.4f}")

    logger.info(f'All folds best results:')
    for key, values in all_test_results.items():
        logger.info(f'{key}: {values}')

    logger.info(f'Mean Best Accuracy: {np.mean(all_test_results["accuracy"]):.4f}')

    fold_model_path = f"MCHHL_model_{current_time}_fold_{fold_num}.pth"
    torch.save(model.state_dict(), fold_model_path)

    print(f'Average Test ROC-AUC: {np.mean(all_test_results["roc_auc"]):.4f}')
    print(f'Average Test Precision: {np.mean(all_test_results["precision"]):.4f}')
    print(f'Average Test Recall: {np.mean(all_test_results["recall"]):.4f}')
    print(f'Average Test F1: {np.mean(all_test_results["f1"]):.4f}')
    print(f'Average Test PR-AUC: {np.mean(all_test_results["pr_auc"]):.4f}')


if __name__ == '__main__':


    task_parser = argparse.ArgumentParser(add_help=False)  # Disable help here to avoid conflicts
    task_parser.add_argument('--task', type=str, default='task1', choices=['task1', 'task2'],
                             help='Choose task type: task1 (Binary) or task2 (Multi-class)')


    task_args, remaining_argv = task_parser.parse_known_args()


    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + remaining_argv


    args = parameters_set()


    sys.argv = original_argv


    args.task = task_args.task


    if args.task == 'task1':
        run_task1_binary(args)
    elif args.task == 'task2':
        run_task2_Multi(args)
    else:
        print("Invalid task specified.")