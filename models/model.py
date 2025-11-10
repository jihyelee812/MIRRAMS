import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.attention import *


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x



class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5*dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:,i,:]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred



class Model(nn.Module):
    def __init__(self,
                 *,
                 categories,               
                 num_continuous,
                 cont_embeddings = 'MLP',
                 dim = 32,
                 num_heads = 16,
                 depth = 24,
                 dim_head = 16,
                 attn_dropout = 0.0,
                 ff_dropout = 0.0,
                 num_special_tokens = 0,
                 y_dim = 2):
        super().__init__()

        # --------------------------------------------------------------------------
        # Categorical feature specifics
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        self.num_categories = len(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        self.total_tokens = sum(categories) + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # --------------------------------------------------------------------------
        # Continuous feature specifics
        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.cont_embeddings = cont_embeddings
        
        
        # --------------------------------------------------------------------------
        self.nfeats = self.num_categories + self.num_continuous
        self.dim = dim
        self.num_heads = num_heads
        self.depth = depth
        self.y_dim = y_dim

        self.embeds_categ = nn.Embedding(self.total_tokens, dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories 


        self.encoder = Transformer(
            dim = dim,
            depth = depth,
            heads = num_heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )


        # --------------------------------------------------------------------------
        # missing
        cat_missing_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_missing_offset = cat_missing_offset.cumsum(dim = -1)[:-1]

        con_missing_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_missing_offset = con_missing_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_missing_offset', cat_missing_offset)
        self.register_buffer('con_missing_offset', con_missing_offset)

        self.missing_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.missing_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.nfeats, self.dim)

        self.mlp1 = sep_MLP(dim, self.num_categories, categories)
        self.mlp2 = sep_MLP(dim, self.num_continuous, np.ones(self.num_continuous).astype(int))


        self.mlpfory = simple_MLP([dim, 1000, y_dim])
        self.pt_mlp = simple_MLP([dim*self.nfeats ,6*dim*self.nfeats//5, dim*self.nfeats//2])
        self.pt_mlp2 = simple_MLP([dim*self.nfeats ,6*dim*self.nfeats//5, dim*self.nfeats//2])



    def forward(self, x_cat, x_con=None):
        '''
        Input is embedded.
        '''
        device = x_cat.device

        data = torch.cat((x_cat, x_con.to(device)),dim=1)
        data = self.encoder(data)

        return data
