import sys
import torch
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn as nn
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=FutureWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##---------------------------------------------------------------------------------------------#    
## Validation Loss


def clf_scores(model, dloader, device, y_dim, task):

    model.eval()
    softmax = nn.Softmax(dim=1)
    criterion_ce = nn.CrossEntropyLoss(reduction='sum')

    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    if task == 'binary':
        prob = torch.empty(0).to(device)
    else:
        prob = torch.empty(0, y_dim).to(device)
    loss = 0.0
    running_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dloader, 0):
            x_cat, x_con, y, x_cat_missing, x_con_missing = data[0].to(device), data[1].to(device),data[2].to(device), data[3].to(device), data[4].to(device)
            
            x_cat_emb, x_con_emb = embed_data_mask(x_cat, x_con.float(), x_cat_missing, x_con_missing, model)

            # Get model's outputs
            x_masked = model(x_cat_emb, x_con_emb)
            cls_token = x_masked[:, 0, :]
            y_out = model.mlpfory(cls_token)

            loss = criterion_ce(y_out, y.squeeze(1).long())

            y_pred = torch.cat([y_pred, torch.argmax(y_out, dim=1).float()],dim=0)
            y_test = torch.cat([y_test, y.flatten().float()],dim=0)

            if task == 'binary':
                prob = torch.cat([prob, softmax(y_out)[:,-1].float()],dim=0)
            else:
                prob = torch.cat([prob, softmax(y_out).float()],dim=0)

            running_loss += loss.item()

    running_loss /= y_test.shape[0]
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0] * 100

    prob = prob.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_test = y_test.cpu().numpy()

    if task == 'binary':
        auc = roc_auc_score(y_score=prob, y_true=y_test)
        f1 = f1_score(y_pred=y_pred, y_true=y_test)
    else:
        auc = roc_auc_score(y_score=prob, y_true=y_test, multi_class='ovr')
        f1 = f1_score(y_pred=y_pred, y_true=y_test, average='weighted')

    return acc.cpu().numpy(), auc, f1, running_loss



##---------------------------------------------------------------------------------------------#    
## Embedding
def embed_data_mask(x_cat, x_con, x_cat_missing, x_con_missing, model, mask_ratio=None):
    '''
    x_cat_missing, x_con_missing: "MissingValue"-0, Observed data-1
    '''

    device = x_con.device
    x_cat = x_cat + model.categories_offset.type_as(x_cat)
    x_cat_emb = model.embeds_categ(x_cat)
       
    if model.cont_embeddings == 'MLP':
        x_con_emb = torch.empty(*x_con.shape, model.dim, device=device)
        for i in range(model.num_continuous):
            x_con_emb[:,i,:] = model.simple_MLP[i](x_con[:,i])
    else:
        raise Exception('This case should not work!')

    if mask_ratio is not None:
        # Additional Masking
        x_cat_missing, x_con_missing = add_mask(x_cat_missing, x_con_missing, mask_ratio)


    cat_missing_temp = x_cat_missing + model.cat_missing_offset.type_as(x_cat_missing)
    con_missing_temp = x_con_missing + model.con_missing_offset.type_as(x_con_missing)

    cat_missing_temp = model.missing_embeds_cat(cat_missing_temp)
    con_missing_temp = model.missing_embeds_cont(con_missing_temp)

    x_cat_emb[x_cat_missing == 0] = cat_missing_temp[x_cat_missing == 0]
    x_con_emb[x_con_missing == 0] = con_missing_temp[x_con_missing == 0]

    return x_cat_emb, x_con_emb



def add_mask(x_cat_missing_old, x_con_missing_old, mask_ratio=0.1):

    device = x_cat_missing_old.device

    x_cat_mask = (torch.rand(*x_cat_missing_old.shape, device=device) > mask_ratio).long()
    x_con_mask = (torch.rand(*x_con_missing_old.shape, device=device) > mask_ratio).long()

    return torch.mul(x_cat_missing_old,x_cat_mask), torch.mul(x_con_missing_old,x_con_mask)

