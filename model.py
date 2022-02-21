import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, device):
    
        super(GAT, self).__init__()
        self.dropout = dropout
        self.device = device
        self.attentions = nn.ModuleList()

        for head in range(nheads):
            self.attentions.append(GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True).to(self.device))

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False).to(self.device)

    def forward(self, x, adj):

        x = x.to(self.device)
        adj = adj.to(self.device)
        x = F.dropout(x, self.dropout, training = self.training)

        out_x = None
        for head in range(len(self.attentions)):
    
            attn_x = self.attentions[head](x, adj)
            
            if out_x == None:
                out_x = attn_x
            else:
                out_x = torch.cat([out_x, attn_x], dim = 1)

        x = F.elu(out_x)
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.out_att(x, adj)
        embedding = x
        x = F.elu(x)
        x = F.log_softmax(x, dim = 1)
        return embedding, x
