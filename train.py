import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from cora_loader import Cora
from citeseer_loader import CiteSeer
from pubmed_loader import PubMed
from coauthor_cs_loader import CoauthorCS

from sklearn.manifold import TSNE
import math
import os
import torch

from utils import UtilFunctions
from test import ModelEvaluation


class ModelTraining():

    def __init__(self):

        return

    def train(self, model, data_obj, adj, train_iter, opti, hidden_layers, device):

        model_path = os.getcwd() + "/saved_models/" + data_obj.name.lower() + "_" + str(hidden_layers) + "_layers_.pt"
        losses = []
        best_acc = 0.0

        # training loop
        for epoch in range(train_iter):

            model.train()
            opti.zero_grad()
            emb, pred = model(data_obj.node_features, adj)
            label = data_obj.node_labels
            pred = pred[data_obj.train_mask]
            label = label[data_obj.train_mask]
            pred = pred.to(device)
            label = label.to(device)
            loss = UtilFunctions.loss_fn(pred, label)
            loss.backward()
            opti.step()

            losses.append(loss)

            test_acc = ModelEvaluation().test(model, data_obj, adj, hidden_layers, model_path, device, is_validation = True)
            print(f"Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Val_acc: {test_acc:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), model_path)

        return model_path
