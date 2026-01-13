# import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import static layers
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GATConv, TransformerConv, GINConv, GCN2Conv
# import pooling layers
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torch import Tensor
from typing import Optional, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
batch_size=10

# pre-calculated positional encoding
pe=torch.load("pe_128")
cat_pe=pe
for i in range(batch_size-1):
    cat_pe=torch.cat((cat_pe,pe),dim=0)
# print(cat_pe.shape)
pe=cat_pe
pe=pe.to(device)
# print(pe.shape)

silu=nn.SiLU()

def cal_batch(size):
    ans=[]
    for i in range(size):
        ans.append(torch.ones(84,)*i)
    ans=torch.stack(ans).reshape(-1)
    return ans.long().to(device)


# scaled dot product attention described in https://arxiv.org/abs/1706.03762
# implemented in https://github.com/sooftware/attentions
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class GNN_att(nn.Module):
    def __init__(self, conv_type, aggr_type):
        super(GNN_att, self).__init__()
        # record layer types
        self.c_type=conv_type
        self.a_type=aggr_type
        self.att=ScaledDotProductAttention(128)

        # graph layers
        if conv_type=="GCN":
            self.conv1=GCNConv(84,128)
            self.conv2=GCNConv(128,128)
        elif conv_type=="Cheb":
            self.conv1=ChebConv(84,128,K=4)
            self.conv2=ChebConv(128,128,K=4)
        elif conv_type=="SAGE":
            self.conv1=SAGEConv(84,128)
            self.conv2=SAGEConv(128,128)
        elif conv_type=="GAT":
            self.conv1=GATConv(84,128,heads=1,edge_dim=1)
            self.conv2=GATConv(128,128,heads=1,edge_dim=1)
        elif conv_type=="Transformer":
            self.conv1=TransformerConv(84,128,heads=1)
            self.conv2=TransformerConv(128,128,heads=1)
        elif conv_type=="GIN":
            self.conv1=GINConv(nn.Linear(84,128))
            self.conv2=GINConv(nn.Linear(128,128))
        else:
            print("Linear layer used in graph.")
            self.conv1=nn.Linear(84,128)
            self.conv2=nn.Linear(128,128)

        # linear layers
        if self.a_type=="local":
            self.fc1=nn.Linear(6144,512)
        else:
            self.fc1=nn.Linear(512,512)
        self.fc2=nn.Linear(512,32)
        self.fc3=nn.Linear(32,2)

        # dropout layer
        self.dp=nn.Dropout(p=0.2)

        # batch normalizations
        self.bn1=nn.BatchNorm1d(512)
        self.bn2=nn.BatchNorm1d(32)
    
    def forward(self, data):
        batch_num=data.x.shape[0]//1008
        if self.c_type=="SAGE":
            temp_x1=self.conv1(x=data.x,edge_index=data.edge_index)
            temp_x2=self.conv2(x=temp_x1,edge_index=data.edge_index)
        elif self.c_type=="GAT": 
            temp_x1=self.conv1(x=data.x,edge_index=data.edge_index,edge_attr=data.edge_attr)
            temp_x2=self.conv2(x=temp_x1,edge_index=data.edge_index,edge_attr=data.edge_attr)
        elif self.c_type=="Transformer" or self.c_type=="GIN":
            temp_x1=self.conv1(x=data.x,edge_index=data.edge_index)
            temp_x2=self.conv2(x=temp_x1,edge_index=data.edge_index)
        else:
            temp_x1=self.conv1(x=data.x,edge_index=data.edge_index,edge_weight=data.edge_attr)
            temp_x2=self.conv2(x=temp_x1,edge_index=data.edge_index,edge_weight=data.edge_attr)
        temp_x2+=pe
        temp_x2=temp_x2.reshape(-1,1008,128)
        temp_x2=self.att(temp_x2,temp_x2,temp_x2).reshape(-1,128)
        # batch*node*time X dim
        # print(temp_x2.shape)
        if self.a_type=="local":
            snapshot_size=12
            batch_size=data.x.shape[0]//1008
            aggr_list1=[[] for _ in range(snapshot_size)]
            aggr_list2=[[] for _ in range(snapshot_size)]
            for i in range(snapshot_size*batch_size):
                snap_num=i%snapshot_size
                segment1=temp_x1[84*i:84*i+84,:]
                segment2=temp_x2[84*i:84*i+84,:]
                aggr_list1[snap_num].append(segment1)
                aggr_list2[snap_num].append(segment2)
            aggr_1=[]
            aggr_2=[]
            for j in range(snapshot_size):
                aggr_1.append(torch.stack(aggr_list1[j]).reshape(-1,128))
                aggr_2.append(torch.stack(aggr_list2[j]).reshape(-1,128))
            aggr_all=[]
            temp_batch=cal_batch(batch_size)
            for k in range(snapshot_size):
                aggr_seg=torch.cat([gmp(aggr_1[k], temp_batch), gap(aggr_1[k], temp_batch), gmp(aggr_2[k], temp_batch), gap(aggr_2[k], temp_batch)], dim=1)
                aggr_all.append(aggr_seg)
            aggr_x=torch.stack(aggr_all).permute(1,0,2).reshape(-1,6144)
        # else:
        #     aggr_x=torch.cat([gmp(temp_x1, temp_batch), gap(temp_x1, temp_batch), gmp(temp_x2, temp_batch), gap(temp_x2, temp_batch)], dim=1)
        out=silu(self.fc1(aggr_x))
        out=self.bn1(out)
        out=self.dp(out)
        out=silu(self.fc2(out))
        out=self.bn2(out)
        out=self.dp(out)
        out=F.softmax(self.fc3(out),dim=1)
        return out


# cleaner implementation
class STNAGNN(nn.Module):
    def __init__(self, conv_type, aggr_type):
        super(STNAGNN, self).__init__()
        # constant from data
        self.roi_num=84
        self.window_num=12
        self.hidden_dim=128
        # record layer types
        self.c_type=conv_type
        self.a_type=aggr_type
        self.att=ScaledDotProductAttention(self.hidden_dim)

        # graph layers
        if conv_type=="GCN":
            self.conv1=GCNConv(self.roi_num,self.hidden_dim)
            self.conv2=GCNConv(self.hidden_dim,self.hidden_dim)
        elif conv_type=="Cheb":
            self.conv1=ChebConv(self.roi_num,self.hidden_dim,K=4)
            self.conv2=ChebConv(self.hidden_dim,self.hidden_dim,K=4)
        elif conv_type=="SAGE":
            self.conv1=SAGEConv(self.roi_num,self.hidden_dim)
            self.conv2=SAGEConv(self.hidden_dim,self.hidden_dim)
        elif conv_type=="GAT":
            self.conv1=GATConv(self.roi_num,self.hidden_dim,heads=1,edge_dim=1)
            self.conv2=GATConv(self.hidden_dim,self.hidden_dim,heads=1,edge_dim=1)
        elif conv_type=="Transformer":
            self.conv1=TransformerConv(self.roi_num,self.hidden_dim,heads=1)
            self.conv2=TransformerConv(self.hidden_dim,self.hidden_dim,heads=1)
        elif conv_type=="GIN":
            self.conv1=GINConv(nn.Linear(self.roi_num,self.hidden_dim))
            self.conv2=GINConv(nn.Linear(self.hidden_dim,self.hidden_dim))
        else:
            print("Linear layer used in graph.")
            self.conv1=nn.Linear(self.roi_num,self.hidden_dim)
            self.conv2=nn.Linear(self.hidden_dim,self.hidden_dim)

        # linear layers
        if self.a_type=="local":
            self.fc1=nn.Linear(self.hidden_dim*self.window_num*4,self.hidden_dim*4)
        else:
            self.fc1=nn.Linear(self.hidden_dim*4,self.hidden_dim*4)
        self.fc2=nn.Linear(self.hidden_dim*4,32)
        self.fc3=nn.Linear(32,2)

        # dropout layer
        self.dp=nn.Dropout(p=0.2)

        # batch normalizations
        self.bn1=nn.BatchNorm1d(self.hidden_dim*4)
        self.bn2=nn.BatchNorm1d(32)
    
    def forward(self, data):
        if self.c_type=="SAGE":
            temp_x1=self.conv1(x=data.x,edge_index=data.edge_index)
            temp_x2=self.conv2(x=temp_x1,edge_index=data.edge_index)
        elif self.c_type=="GAT": 
            temp_x1=self.conv1(x=data.x,edge_index=data.edge_index,edge_attr=data.edge_attr)
            temp_x2=self.conv2(x=temp_x1,edge_index=data.edge_index,edge_attr=data.edge_attr)
        elif self.c_type=="Transformer" or self.c_type=="GIN":
            temp_x1=self.conv1(x=data.x,edge_index=data.edge_index)
            temp_x2=self.conv2(x=temp_x1,edge_index=data.edge_index)
        else:
            temp_x1=self.conv1(x=data.x,edge_index=data.edge_index,edge_weight=data.edge_attr)
            temp_x2=self.conv2(x=temp_x1,edge_index=data.edge_index,edge_weight=data.edge_attr)
        temp_x2+=pe
        temp_x2=temp_x2.reshape(-1,self.roi_num*self.window_num,self.hidden_dim)
        temp_x2=self.att(temp_x2,temp_x2,temp_x2).reshape(-1,self.hidden_dim)
        # batch*node*time X dim
        # print(temp_x2.shape)
        if self.a_type=="local":
            snapshot_size=12
            batch_size=data.x.shape[0]//self.roi_num//self.window_num
            aggr_list1=[[] for _ in range(snapshot_size)]
            aggr_list2=[[] for _ in range(snapshot_size)]
            for i in range(snapshot_size*batch_size):
                snap_num=i%snapshot_size
                segment1=temp_x1[self.roi_num*i:self.roi_num*i+self.roi_num,:]
                segment2=temp_x2[self.roi_num*i:self.roi_num*i+self.roi_num,:]
                aggr_list1[snap_num].append(segment1)
                aggr_list2[snap_num].append(segment2)
            aggr_1=[]
            aggr_2=[]
            for j in range(snapshot_size):
                aggr_1.append(torch.stack(aggr_list1[j]).reshape(-1,self.hidden_dim))
                aggr_2.append(torch.stack(aggr_list2[j]).reshape(-1,self.hidden_dim))
            aggr_all=[]
            temp_batch=cal_batch(batch_size)
            for k in range(snapshot_size):
                aggr_seg=torch.cat([gmp(aggr_1[k], temp_batch), gap(aggr_1[k], temp_batch), gmp(aggr_2[k], temp_batch), gap(aggr_2[k], temp_batch)], dim=1)
                aggr_all.append(aggr_seg)
            aggr_x=torch.stack(aggr_all).permute(1,0,2).reshape(-1,self.hidden_dim*self.window_num*4)
        # else:
        #     aggr_x=torch.cat([gmp(temp_x1, temp_batch), gap(temp_x1, temp_batch), gmp(temp_x2, temp_batch), gap(temp_x2, temp_batch)], dim=1)
        out=silu(self.fc1(aggr_x))
        out=self.bn1(out)
        out=self.dp(out)
        out=silu(self.fc2(out))
        out=self.bn2(out)
        out=self.dp(out)
        out=F.softmax(self.fc3(out),dim=1)
        return out