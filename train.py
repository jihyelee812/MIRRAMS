import os

import torch
from torch import nn
from models.model import Model
from utils import count_parameters, clf_scores, embed_data_mask
from data_open import get_dataset, DataSetCatCon

import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import time

import numpy as np
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('--set_seed', default= 123 , type=int)
parser.add_argument('--ds_name', default='adult', type=str, help='The name of dataset')
parser.add_argument('--setting', default='test', type=str)
parser.add_argument('--task', default='binary', type=str, choices = ['binary','multiclass'])
parser.add_argument('--device_id', type=int, default=0, help='CUDA device number (default: 0)')
parser.add_argument('--run_name', default=None, type=str)
parser.add_argument('--model_load_path', type=str)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)

parser.add_argument('--mask_ratio', default=0.2, type=float)
parser.add_argument('--train_missing_ratio', default=0.1, type=float)
parser.add_argument('--test_missing_ratio', default=0.1, type=float)
parser.add_argument('--same_env', default=False, action = 'store_true')

parser.add_argument('--ssl', default=False, action = 'store_true')
parser.add_argument('--label_ratio', default=0.1, type=float)
parser.add_argument('--num_labeled_samples', default=None, type=int, choices = [50, 200, 500])

parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--depth', default=6, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str, choices = ['common','sep'])

parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--optimizer', default='Adam', type=str,choices = ['Adam','AdamW','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--train_iteration', type=int, default=15, help='Number of iteration per epoch')
parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive', 'denoising'])

parser.add_argument('--missing_type', default='MAR', choices=['MCAR', 'MAR', 'MNAR', 'MAR_y', 'MNAR_y'])
parser.add_argument('--lam_ce', default=1, type=float)
parser.add_argument('--lam_1', default=15, type=float)
parser.add_argument('--lam_2', default=15, type=float)
parser.add_argument('--tau', default=0.95, type=float)
# parser.add_argument('--rho', default=0.7, type=float)

opt = parser.parse_args()

def main():
    opt.run_name = f"{opt.setting}_{opt.mask_ratio}_{int(opt.lam_1)}_{int(opt.lam_2)}"

    device = torch.device(f"cuda:{opt.device_id}" if torch.cuda.is_available() else "cpu")

    print(f"\n=========={opt.ds_name}==========")
    print(f"Device: {opt.device_id}")
    print(f'Record Name: {opt.run_name}')
    print(f'lam_1: {opt.lam_1} | lam_2: {opt.lam_2}')
    print(f'Mask ratio: {opt.mask_ratio}')
    print(f'Missing Type: {opt.missing_type}')
    print(f'Train Missing ratio: {opt.train_missing_ratio}')
    print(f'Test Missing ratio: {opt.test_missing_ratio}')
    print(f'Epoch: {opt.epochs} | Batchsize: {opt.batchsize} | Train_iteration: {opt.train_iteration}')
    if opt.num_labeled_samples: print(f'# Labeled samples: {opt.num_labeled_samples}')

    modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.ds_name),opt.run_name)
    os.makedirs(modelsave_path, exist_ok=True)
    print(modelsave_path)

    if opt.ssl:
        result_folder = f'result_ssl_{1-opt.label_ratio:.2f}'
    else:
        result_folder = 'result'


    if not os.path.exists(f'{os.getcwd()}/{result_folder}/{opt.ds_name}/{opt.missing_type}/{opt.set_seed}/{opt.setting}'):
        result_save_path = os.path.join(os.getcwd(),result_folder,str(opt.ds_name),str(opt.missing_type),str(opt.set_seed),str(opt.setting))
        os.makedirs(result_save_path, exist_ok=True)
    else:
        result_save_path = os.path.join(os.getcwd(),result_folder,str(opt.ds_name),str(opt.missing_type),str(opt.set_seed),str(opt.setting))


    train_header = ['EPOCH', 'LOSS', 'LOSS_CE', 'LOSS_M', 'LOSS_F']
    val_header = ['EPOCH', 'VAL_LOSS', 'VAL_AUROC', 'VAL_ACCURACY', 'VAL_F1', 'TEST_LOSS', 'TEST_AUROC', 'TEST_ACCURACY', 'TEST_F1']
    train_results = []
    eval_results = []



    print('Downloading and processing the dataset, it might take some time.')



    ##### Dataset Load
    data_summary, X_train_lb, y_train_lb, X_train_ulb, y_train_ulb, X_valid, y_valid, X_test, y_test = get_dataset(opt.ds_name, device, opt.missing_type, opt.train_missing_ratio, opt.test_missing_ratio, opt.label_ratio, opt.set_seed, opt.same_env, opt.num_labeled_samples)
    

    nfeat = data_summary['nfeat']
    cat_dims = np.append(np.array([1]),np.array(data_summary['cat_dims'])).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.
    cat_idxs = data_summary['cat_idxs']
    con_idxs = data_summary['con_idxs']
    y_dim = data_summary['y_dim']


    ##### Setting some params based on inputs and dataset
    if nfeat > 100:
        opt.embedding_size = min(4,opt.embedding_size)
        opt.batchsize = min(64, opt.batchsize)

    if opt.ds_name in ['arrhythmia']:
        opt.depth = 1

    if opt.ds_name in ['arcene']:
        opt.num_heads = 1
        opt.depth = 4
    
    if opt.ssl:
        n_labeld_data = len(X_train_lb['data'])
        if n_labeld_data < opt.batchsize:
            opt.batchsize = n_labeld_data


    ##### Data Load for training
    train_labeled_ds = DataSetCatCon(X_train_lb, y_train_lb, cat_idxs, x_min=data_summary['x_min'], x_max=data_summary['x_max'])
    train_unlabeled_ds = DataSetCatCon(X_train_ulb, y_train_ulb, cat_idxs, x_min=data_summary['x_min'], x_max=data_summary['x_max'])
    valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, x_min=data_summary['x_min'], x_max=data_summary['x_max'])
    test_ds = DataSetCatCon(X_test, y_test, cat_idxs, x_min=data_summary['x_min'], x_max=data_summary['x_max'])

    labeled_trainloader = DataLoader(train_labeled_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4, drop_last=True)
    unlabeled_trainloader = DataLoader(train_unlabeled_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4, drop_last=True)
    validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4, drop_last=False)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4, drop_last=False)



    ##### Model
    model = Model(
        categories = tuple(cat_dims),
        num_continuous = len(con_idxs),
        cont_embeddings = opt.cont_embeddings,
        attn_dropout = opt.attention_dropout,
        ff_dropout = opt.ff_dropout,
        dim = opt.embedding_size,
        num_heads = opt.num_heads,
        depth = opt.depth,
        y_dim = y_dim
    ).to(device)

    model.to(device)



    #### Training
    print("\nTRAINING BEGINS!\n")
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),lr=opt.lr)
    best = {'val_auc' : 0, 'val_acc' : 0, 'val_f1'  : 0,
            'test_auc': 0, 'test_acc': 0, 'test_f1' : 0}

    running_time = 0
    total_running_time = 0
    patience = 0


    for epoch in range(opt.epochs):
        
        ## Train model
        if opt.ssl:
            running_loss, loss_ce, loss_m, loss_f, epoch_time = train_ssl(labeled_trainloader, unlabeled_trainloader, model, optimizer, opt, device)
        else:
            running_loss, loss_ce, loss_m, loss_f, epoch_time = train(unlabeled_trainloader, model, optimizer, opt, device)
        train_results.append([epoch + 1, running_loss, loss_ce, loss_m, loss_f])
        np.savetxt('%s/train_%s.txt' % (result_save_path, opt.run_name), 
                   np.vstack(np.array(train_results)), 
                   header=' '.join(train_header), 
                   comments='',
                   fmt=['%d', '%.3f', '%.3f', '%.3f', '%.3f'])
        running_time += epoch_time
    
        total_running_time += running_time

        print('[EPOCH %d/%d] LOSS: %.4f, LOSS_CE: %.4f, LOSS_M: %.4f, LOSS_F: %.4f, TIME: %.2f\n' %
            (epoch + 1, opt.epochs, running_loss, loss_ce, loss_m, loss_f, running_time))
        running_time = 0

        if epoch > 50:
            ## Validate model
            best, val_loss, accuracy, auroc, f1, test_loss, test_accuracy, test_auroc, test_f1, test_time, patience = validate(validloader, testloader, model, best, y_dim, opt, epoch, device, patience)
            eval_results.append([epoch + 1, val_loss, auroc, accuracy, f1,
                                            test_loss, test_auroc, test_accuracy, test_f1])
            np.savetxt('%s/valid_%s.txt' % (result_save_path, opt.run_name), 
                    np.vstack(np.array(eval_results)), 
                    header=' '.join(val_header), 
                    comments='',
                    fmt=['%d', '%.3f', '%.5f', '%.5f', '%.5f', '%.3f', '%.5f', '%.5f', '%.5f'])
            total_running_time += test_time
            
            if patience == 10:
                break


    model.eval()
    total_parameters = count_parameters(model)
    print('\nEND OF TRAINING!\n')
    print(f"Dataset: {opt.ds_name}")
    print(f"Device ID: {opt.device_id}")
    print(f'Record Name: {opt.run_name}\n')
    print(f'lam_1: {opt.lam_1} | lam_2: {opt.lam_2}')
    print(f'Mask ratio: {opt.mask_ratio}')
    print(f'Train Missing ratio: {opt.train_missing_ratio}')
    print(f'Test Missing ratio: {opt.test_missing_ratio}')
    print(f'Missingness Type: {opt.missing_type}')
    print(f'Epoch: {opt.epochs} | Batchsize: {opt.batchsize} | Train_iteration: {opt.train_iteration}')
    if opt.ssl:
        print(f'Ratio of Labeled data: {opt.label_ratio}')
    print(f'Total parameters: {total_parameters}\n')
    print(f'BEST VAL AUC:{round(best["val_auc"],5)}')
    print(f'BEST TEST AUC:{round(best["test_auc"],5)}\n')
    print(f'Done! Total Running time: {total_running_time / 3600}\n')




def train(dloader, model, optimizer, opt, device):

    model.train()
    softmax = nn.Softmax(dim=-1)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_m = 0.0
    running_loss_f = 0.0
    start_time = time.time()


    for batch_idx, data in enumerate(dloader, 0):
        optimizer.zero_grad()
        x_cat, x_con, y, x_cat_missing, x_con_missing = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device),data[4].to(device)

        # Embedding
        x_cat_emb, x_con_emb = embed_data_mask(x_cat, x_con.float(), x_cat_missing, x_con_missing, model)
        # Additional Masking
        m_x_cat_emb, m_x_con_emb = embed_data_mask(x_cat, x_con.float(), x_cat_missing, x_con_missing, model, opt.mask_ratio)
        
        org_masked = model(x_cat_emb, x_con_emb)
        masked_out = model(m_x_cat_emb, m_x_con_emb)

        cls_org = org_masked[:,0,:]
        cls_masked_out = masked_out[:,0,:]

        org_mlp = model.mlpfory(cls_org)
        mlp_masked_out = model.mlpfory(cls_masked_out)

        prop_m_tr = softmax(org_mlp)
        

        ## Loss
        ## 1. M_tr CrossEntropy
        loss_ce = opt.lam_ce * criterion_ce(org_mlp, y.squeeze(1).long())

        ## 2. M_m * M_tr CrossEntropy
        loss_m = opt.lam_1 * criterion_ce(mlp_masked_out, y.squeeze(1).long())

        ## 3. Only Unlabeled data
        max_prob, pseudo_label = torch.max(prop_m_tr.detach(), dim = 1) 
        ce_u = criterion_none(mlp_masked_out, pseudo_label)

        loss_f = opt.lam_2 * ((max_prob > opt.tau) * ce_u).mean()

        ## Total loss
        loss = loss_ce + loss_m + loss_f
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss_ce += loss_ce.item()
        running_loss_m += loss_m.item()
        running_loss_f += loss_f.item()

    end_time = time.time()
    epoch_time = end_time - start_time   

    return running_loss, running_loss_ce, running_loss_m, running_loss_f, epoch_time



def train_ssl(labeled_dloader, unlabeled_dloader, model, optimizer, opt, device):

    model.train()
    softmax = nn.Softmax(dim=-1)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_m = 0.0
    running_loss_f = 0.0
    start_time = time.time()

    labeled_train_iter = iter(labeled_dloader)
    unlabeled_train_iter = iter(unlabeled_dloader)

    for batch_idx in range(opt.train_iteration):
        optimizer.zero_grad()
        try:
            x_cat_lb, x_con_lb, y, x_cat_missing_lb, x_con_missing_lb = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_dloader)
            x_cat_lb, x_con_lb, y, x_cat_missing_lb, x_con_missing_lb = next(labeled_train_iter)
        try:
            x_cat_ulb, x_con_ulb, _, x_cat_missing_ulb, x_con_missing_ulb = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_dloader)
            x_cat_ulb, x_con_ulb, _, x_cat_missing_ulb, x_con_missing_ulb = next(unlabeled_train_iter)

        x_cat_lb, x_con_lb, y, x_cat_missing_lb, x_con_missing_lb = x_cat_lb.to(device), x_con_lb.to(device), y.to(device), x_cat_missing_lb.to(device), x_con_missing_lb.to(device)
        x_cat_ulb, x_con_ulb, x_cat_missing_ulb, x_con_missing_ulb = x_cat_ulb.to(device), x_con_ulb.to(device), x_cat_missing_ulb.to(device), x_con_missing_ulb.to(device)

        # Embedding
        x_cat_emb_lb, x_con_emb_lb = embed_data_mask(x_cat_lb, x_con_lb.float(), x_cat_missing_lb, x_con_missing_lb, model)
        m_x_cat_emb_lb, m_x_con_emb_lb = embed_data_mask(x_cat_lb, x_con_lb.float(), x_cat_missing_lb, x_con_missing_lb, model, opt.mask_ratio)
        m_x_cat_emb_ulb, m_x_con_emb_ulb = embed_data_mask(x_cat_ulb, x_con_ulb.float(), x_cat_missing_ulb, x_con_missing_ulb, model, opt.mask_ratio)

        # Get model's outputs
        org_masked = model(x_cat_emb_lb, x_con_emb_lb)
        masked_out_lb = model(m_x_cat_emb_lb, m_x_con_emb_lb)
        masked_out_ulb = model(m_x_cat_emb_ulb, m_x_con_emb_ulb)

        cls_org = org_masked[:,0,:]
        cls_masked_out_lb = masked_out_lb[:,0,:]
        cls_masked_out_ulb = masked_out_ulb[:,0,:]

        mlp_out = model.mlpfory(cls_org)
        mlp_masked_out_lb = model.mlpfory(cls_masked_out_lb)
        mlp_masked_out_ulb = model.mlpfory(cls_masked_out_ulb)

        prop_m_tr = softmax(mlp_masked_out_ulb)
        

        ## Loss
        ## 1. M_tr CrossEntropy
        loss_ce = opt.lam_ce * criterion_ce(mlp_out, y.squeeze(1).long())

        ## 2. M_m * M_tr CrossEntropy
        loss_m = opt.lam_1 * criterion_ce(mlp_masked_out_lb, y.squeeze(1).long())

        ## 3. Only Unlabeled data
        max_prob, pseudo_label = torch.max(prop_m_tr.detach(), dim = 1)    # y_hat: pseudo_label
        ce_u = criterion_none(mlp_masked_out_ulb, pseudo_label)            # f(M_ts): mlp_masked_out

        loss_f = opt.lam_2 * ((max_prob > opt.tau) * ce_u).mean()

        ## Total loss
        loss = loss_ce + loss_m + loss_f
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss_ce += loss_ce.item()
        running_loss_m += loss_m.item()
        running_loss_f += loss_f.item()

    end_time = time.time()
    epoch_time = end_time - start_time   

    return running_loss, running_loss_ce, running_loss_m, running_loss_f, epoch_time



def validate(validloader, testloader, model, best, y_dim, opt, epoch, device, patience):
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.ds_name),opt.run_name)

        accuracy, auroc, f1, loss = clf_scores(model, validloader, device, y_dim, opt.task)
        test_accuracy, test_auroc, test_f1, test_loss = clf_scores(model, testloader, device, y_dim, opt.task)
        running_time = time.time() - start_time

        print('[EPOCH %d] VALID AUROC: %.3f, VALID ACCURACY: %.3f, VALID F1: %.3f' %
            (epoch + 1, auroc, accuracy, f1))
        print('[EPOCH %d] TEST AUROC: %.3f, TEST ACCURACY: %.3f, TEST F1: %.3f,    / %.2fs\n' %
            (epoch + 1, test_auroc, test_accuracy, test_f1, running_time))
            
        if auroc > best['val_auc']:
            best['val_acc'], best['val_auc'], best['val_f1'] = accuracy, auroc, f1
            best['test_acc'], best['test_auc'], best['test_f1'] = test_accuracy, test_auroc, test_f1
            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
            print('--BEST AUROC[EPOCH %d]: VALID AUROC: %.5f / TEST AUROC: %.5f--\n' %
                (epoch + 1, auroc, test_auroc))
            patience = 0
        else:
            patience += 1
    
    return best, loss, accuracy, auroc, f1, test_loss, test_accuracy, test_auroc, test_f1, running_time, patience


if __name__ == '__main__': 
    opt.missing_type='MAR_y'
    main()
