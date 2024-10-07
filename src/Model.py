import numpy
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
import numpy as np
from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge, adj_matrix_merge
from src.Decoupling_matrix_aggregation import coototensor



from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp




#from torch_geometric.nn.pool.topk_pool import TopKPooling,filter_adj
from torch_geometric.nn.pool import TopKPooling
from torch.nn import Parameter
from torch_geometric.nn.pool import SAGPooling





class Net2(torch.nn.Module):
    def __init__(self, nfeat, nhid, out, attn_vec_dim, pooling_ratio, dropout_ratio):
        super(Net2, self).__init__()
        #self.args = args
        self.nfeat = nfeat
        self.nhid = nhid
        self.out = out
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.attn_vec_dim = attn_vec_dim

        self.conv1 = GCNConv(self.nfeat, self.out)
        self.sig = torch.nn.Sigmoid()


    def forward(self, feature, edge_index):


        x = F.relu(self.conv1(feature, edge_index))

        h = x.mean(dim=0)
        #h = x.max(dim=0).values

        #h = F.relu(h)
        h = self.sig(h)

        return x, h





class HMGCN(nn.Module):
    def __init__(self, nfeat, nhid, out, pooling_ratio, dropout_ratio, attn_vec_dim):
        super(HMGCN, self).__init__()

        self.dropout_ratio = dropout_ratio
        self.nfeat = nfeat
        self.nhid = nhid
        self.out = out
        self.pooling_ratio = pooling_ratio
        self.attn_vec_dim = attn_vec_dim

        self.net1 = Net2(self.nfeat, self.nhid, self.out, self.attn_vec_dim, self.pooling_ratio, self.dropout_ratio)
        self.net2 = Net2(self.nfeat, self.nhid, self.out, self.attn_vec_dim, self.pooling_ratio, self.dropout_ratio)
        self.net3 = Net2(self.nfeat, self.nhid, self.out, self.attn_vec_dim, self.pooling_ratio, self.dropout_ratio)
        self.net4 = Net2(self.nfeat, self.nhid, self.out, self.attn_vec_dim, self.pooling_ratio, self.dropout_ratio)







        self.fc1 = nn.Linear(self.out, 1, bias=False)
        self.fc2 = nn.Linear(self.out, 1, bias=False)
        # weight initialization

        #nn.init.xavier_normal_(self.fc1.weight, gain=1.414)

    def forward(self, feature, A, node_types, use_relu=True):



        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        #feature = feature.to(device='cuda')
        feature = feature.float()




        embs = []
        beta0 = []
        beta1 = []
        a = A[0][0].tocoo()

        edge_index = torch.tensor(numpy.array([a.row, a.col]))
        try:
            edge_index = torch.tensor(edge_index.toarray())
        except:
            try:
                edge_index = torch.from_numpy(edge_index.toarray())
            except:
                pass

        #edge_index = edge_index.to(device='cuda')
        edge_index =edge_index.long()





        a = A[1][0].tocoo()
        edge_index1 = torch.tensor(numpy.array([a.row, a.col]))
        try:
            edge_index1 = torch.tensor(edge_index1.toarray())
        except:
            try:
                edge_index1 = torch.from_numpy(edge_index1.toarray())
            except:
                pass

        #edge_index1 = edge_index1.to(device='cuda')
        edge_index1 =edge_index1.long()



        a = A[2][0].tocoo()
        edge_index2 = torch.tensor(numpy.array([a.row, a.col]))
        try:
            edge_index2 = torch.tensor(edge_index2.toarray())
        except:
            try:
                edge_index2 = torch.from_numpy(edge_index2.toarray())
            except:
                pass

        #edge_index2 = edge_index2.to(device='cuda')
        edge_index2 = edge_index2.long()



        a = A[3][0].tocoo()
        edge_index3 = torch.tensor(numpy.array([a.row, a.col]))
        try:
            edge_index3 = torch.tensor(edge_index3.toarray())
        except:
            try:
                edge_index3 = torch.from_numpy(edge_index3.toarray())
            except:
                pass

        #edge_index3 = edge_index3.to(device='cuda')
        edge_index3 =edge_index3.long()






        embeds, x = self.net1(feature, edge_index)
        embs.append(embeds)

        fc = self.fc1(x)
        beta0.append(fc)
        fc = self.fc2(x)
        beta1.append(fc)




        embeds, x = self.net2(feature, edge_index1)
        embs.append(embeds)
        fc = self.fc1(x)
        beta0.append(fc)
        fc = self.fc2(x)
        beta1.append(fc)





        embeds, x = self.net3(feature, edge_index2)
        embs.append(embeds)
        fc = self.fc1(x)
        beta0.append(fc)
        fc = self.fc2(x)
        beta1.append(fc)






        embeds, x = self.net4(feature, edge_index3)
        embs.append(embeds)
        fc = self.fc1(x)
        beta0.append(fc)
        fc = self.fc2(x)
        beta1.append(fc)




        #beta0[0] = torch.tensor([1], dtype=torch.float64, requires_grad=True)
        #beta0[1] = torch.tensor([1], dtype=torch.float64, requires_grad=True)
        #beta0[2] = torch.tensor([1], dtype=torch.float64, requires_grad=True)
        #beta0[3] = torch.tensor([1], dtype=torch.float64, requires_grad=True)
        beta0 = torch.cat(beta0, dim=0)
        beta0 = F.softmax(beta0, dim=0)
        #beta1[0] = torch.tensor([1], dtype=torch.float64, requires_grad=True)
        #beta1[1] = torch.tensor([1], dtype=torch.float64, requires_grad=True)
        #beta1[2] = torch.tensor([1], dtype=torch.float64, requires_grad=True)
        #beta1[3] = torch.tensor([1], dtype=torch.float64, requires_grad=True)
        beta1 = torch.cat(beta1, dim=0)
        beta1 = F.softmax(beta1, dim=0)



        beta0 = beta0.tolist()
        beta1 = beta1.tolist()


        multiplied = []
        for s in range(0, len(embs)):
            node_indices = np.where(node_types == 0)[0]
            e = embs[s]
            e = e[node_indices, :]
            multiplied.append(beta0[s] * e)

        h0 = sum(multiplied)

        multiplied = []
        for s in range(0, len(embs)):
            node_indices = np.where(node_types == 1)[0]
            e = embs[s]
            e = e[node_indices, :]
            multiplied.append(beta1[s] * e)

        h1 = sum(multiplied)

        h = torch.cat((h0, h1), 0)






        return h

