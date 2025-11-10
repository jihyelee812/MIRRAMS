import torch
import pandas as pd
import numpy as np
import random
import pandas as pd


def add_missing_mar(x, missing_ratio, cat_idxs, con_idxs, seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    x_data = pd.DataFrame(x['data'])
    x_missing_old = x['missing']
    n_feat = len(cat_idxs) + len(con_idxs)

    n_mcar = int(n_feat*0.3)
    mcar_idxs = np.array(random.sample(range(n_feat), n_mcar), dtype=int)
    non_mcar_idxs = list(set(range(n_feat)) - set(mcar_idxs))

    mcar_mean = x_data.iloc[:, mcar_idxs].mean() 
    mcar_std = x_data.iloc[:, mcar_idxs].std().sum()

    N = x_data.shape[0]
    K = len(non_mcar_idxs)

    mcar_missing = (np.random.uniform(0, 1, N * n_mcar).reshape(N, n_mcar) > missing_ratio).astype(int)
    obs_data = x_data.iloc[:, mcar_idxs].where(mcar_missing==1)

    gamma = np.random.randn(len(mcar_idxs) * K).reshape(len(mcar_idxs), K) / mcar_std
    sigmoid_output = sigmoid((obs_data - mcar_mean).fillna(0) @ gamma)
    gamma_0 = missing_ratio / sigmoid_output.mean()

    unif = np.random.uniform(0, 1, N * K).reshape(N, K)
    tmp_missing = (unif > gamma_0.mul(sigmoid_output, axis=0)).astype(int)


    x_missing = np.zeros((N, n_feat), dtype=int)
    x_missing[:, mcar_idxs] = mcar_missing
    x_missing[:, non_mcar_idxs] = tmp_missing

    print('Marginal missing rate:', (1-x_missing).mean())
    print('Marginal missing sum:', (1-x_missing).sum())

    return x_missing_old * x_missing

def add_missing_mnar(x, missing_ratio, cat_idxs, con_idxs, seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    ## Categorical features are label-encoded
    
    x_data = pd.DataFrame(x['data'])
    x_missing_old = x['missing']
    n_feat = len(cat_idxs) + len(con_idxs)

    total_mean = x_data.mean()
    total_std = x_data.std().sum()

    n_mcar = int(n_feat*0.3)
    mcar_idxs = np.array(random.sample(range(n_feat), n_mcar), dtype=int)
    non_mcar_idxs = list(set(range(n_feat)) - set(mcar_idxs))

    N = x_data.shape[0]
    K = len(non_mcar_idxs)
    
    mcar_missing = (np.random.uniform(0, 1, N * n_feat).reshape(N, n_feat) > missing_ratio).astype(int)
    obs_data = x_data.where(mcar_missing==1) # mcar_missing==0 -> nan

    gamma = np.random.randn(n_feat * K).reshape(n_feat, K) / total_std
    sigmoid_output = sigmoid((obs_data - total_mean).fillna(0) @ gamma)
    gamma_0 = missing_ratio / sigmoid_output.mean()

    unif = np.random.uniform(0, 1, N * K).reshape(N, K)
    tmp_missing = (unif > gamma_0.mul(sigmoid_output, axis=0)).astype(int)
    
    x_missing = np.zeros((N, n_feat), dtype=int)
    x_missing[:, mcar_idxs] = mcar_missing[:, mcar_idxs]
    x_missing[:, non_mcar_idxs] = tmp_missing

    print('Marginal missing rate:', (1-x_missing).mean())

    return x_missing_old * x_missing


def add_missing_mcar(x, missing_ratio, cat_idxs, con_idxs, seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    x_data = pd.DataFrame(x['data'])
    x_missing_old = x['missing']
    n_feat = len(cat_idxs) + len(con_idxs)

    N = x_data.shape[0]

    unif = np.random.uniform(0, 1, N * n_feat).reshape(N, n_feat)
    x_missing = (unif > missing_ratio).astype(int)

    print('Marginal missing rate:', (1-x_missing).mean())

    return x_missing_old * x_missing


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def add_missing_mar_y(x, y, missing_ratio, cat_idxs, con_idxs, seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    x_data = pd.DataFrame(x['data'])
    y_data = pd.DataFrame(y['data'])
    x_missing_old = x['missing']
    y_missing_old = y['missing']

    data = pd.concat([x_data, y_data], axis=1)
    n_feat = len(cat_idxs) + len(con_idxs) + 1

    n_mcar = int(n_feat*0.3)
    mcar_idxs = np.array(random.sample(range(n_feat), n_mcar), dtype=int)
    non_mcar_idxs = list(set(range(n_feat)) - set(mcar_idxs))
    
    mcar_mean = data.iloc[:, mcar_idxs].mean() 
    mcar_std = data.iloc[:, mcar_idxs].std().sum()

    N = data.shape[0]
    K = len(non_mcar_idxs)

    mcar_missing = (np.random.uniform(0, 1, N * n_mcar).reshape(N, n_mcar) > missing_ratio).astype(int)
    obs_data = data.iloc[:, mcar_idxs].where(mcar_missing==1)

    gamma = np.random.randn(n_mcar * K).reshape(n_mcar, K) / mcar_std
    sigmoid_output = sigmoid((obs_data - mcar_mean).fillna(0) @ gamma)
    gamma_0 = missing_ratio / sigmoid_output.mean()

    unif = np.random.uniform(0, 1, N * K).reshape(N, K)
    tmp_missing = (unif > gamma_0.mul(sigmoid_output, axis=0)).astype(int)

    missing = np.zeros((N, n_feat), dtype=int)
    missing[:, mcar_idxs] = mcar_missing
    missing[:, non_mcar_idxs] = tmp_missing

    x_missing = missing[:, :-1]
    y_missing = missing[:, -1].reshape(-1, 1)

    return x_missing_old * x_missing, y_missing_old * y_missing

def add_missing_mnar_y(x, y, missing_ratio, cat_idxs, con_idxs, seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    ## Categorical features are label-encoded
    
    x_data = pd.DataFrame(x['data'])
    y_data = pd.DataFrame(y['data'])
    x_missing_old = x['missing']
    y_missing_old = y['missing']

    data = pd.concat([x_data, y_data], axis=1)
    n_feat = len(cat_idxs) + len(con_idxs) + 1

    total_mean = data.mean()
    total_std = data.std().sum()

    n_mcar = int(n_feat*0.3)
    mcar_idxs = np.array(random.sample(range(n_feat), n_mcar), dtype=int)
    non_mcar_idxs = list(set(range(n_feat)) - set(mcar_idxs))

    N = data.shape[0]
    K = len(non_mcar_idxs)
    
    mcar_missing = (np.random.uniform(0, 1, N * n_feat).reshape(N, n_feat) > missing_ratio).astype(int)
    obs_data = data.where(mcar_missing==1) # mcar_missing==0 -> nan

    gamma = np.random.randn(n_feat * K).reshape(n_feat, K) / total_std
    sigmoid_output = sigmoid((obs_data - total_mean).fillna(0) @ gamma)
    gamma_0 = missing_ratio / sigmoid_output.mean()

    unif = np.random.uniform(0, 1, N * K).reshape(N, K)
    tmp_missing = (unif > gamma_0.mul(sigmoid_output, axis=0)).astype(int)
    
    missing = np.zeros((N, n_feat), dtype=int)
    missing[:, mcar_idxs] = mcar_missing[:, mcar_idxs]
    missing[:, non_mcar_idxs] = tmp_missing

    x_missing = missing[:, :-1]
    y_missing = missing[:, -1].reshape(-1, 1)

    return x_missing_old * x_missing, y_missing_old * y_missing

