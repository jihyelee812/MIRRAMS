from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from dataset import task_dset_ids, dataset_open
from missing import *


def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def concat_data(X,y):
    return pd.concat([pd.DataFrame(X), pd.DataFrame(y[:,0].tolist(),columns=['target'])], axis=1)


def data_split(X, y, nan_missing, indices, missing_type, missing_ratio, cat_columns, con_columns, seed):

    x_d = {'data': X.values[indices],
           'missing': nan_missing.values[indices]}

    if x_d['data'].shape != x_d['missing'].shape:
        raise'Shape of data not same as that of nan missing!'
    
    if missing_type == 'MAR':
        x_d['missing'] = add_missing_mar(x_d, missing_ratio, cat_columns, con_columns, seed)
    elif missing_type == 'MNAR':
        x_d['missing'] = add_missing_mnar(x_d, missing_ratio, cat_columns, con_columns, seed)
    else:
        x_d['missing'] = add_missing_mcar(x_d, missing_ratio, cat_columns, con_columns, seed)

    y_d = {'data': y[indices].reshape(-1, 1)}     

    return x_d, y_d



def get_dataset(ds_name, device, missing_type, train_missing_ratio, test_missing_ratio, label_ratio = 0.1, seed = 123, same_env = False, num_labeled_samples = None):

    if same_env:
        seed_tr = seed
        seed_ts = seed
    else:
        seed_tr = seed
        seed_ts = seed + 100
        
    datasplit=[.65, .15, .2]
    # Base Dataset load
    dataset, categorical_indicator, _, target = dataset_open(ds_name)
    # print(dataset.head())

    np.random.seed(seed)
        
    y = dataset.iloc[:,-1].copy()
    X = dataset.iloc[:,:-1].copy()

    # cat / con
    cat_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    con_columns = list(set(X.columns.tolist()) - set(cat_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
    nfeat = len(cat_idxs) + len(con_idxs)
    
    # Data Preprocessing
    ## Categorical data
    for col in cat_columns:
        X[col] = X[col].astype("object")

    temp = X.fillna("MissingValue")
    nan_missing = temp.ne("MissingValue").astype(int)

    cat_dims = []
    for col in cat_columns:
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))

    ## Split Dataset
    train_labeled_idxs, train_unlabeled_idxs, val_idxs, test_idxs = train_val_split(y, label_ratio, num_labeled_samples, datasplit)


    ## Continuous data
    for col in con_columns:
        X.fillna(X.loc[train_unlabeled_idxs, col].mean(), inplace=True)
    y = y.values
    l_enc_for_y = LabelEncoder()
    y = l_enc_for_y.fit_transform(y)

    y_dim = len(np.unique(y))

    # Split data
    X_train_lb, y_train_lb = data_split(X, y, nan_missing, train_labeled_idxs, missing_type, train_missing_ratio, cat_idxs, con_idxs, seed_tr)
    X_train_ulb, y_train_ulb = data_split(X, y, nan_missing, train_unlabeled_idxs, missing_type, train_missing_ratio, cat_idxs, con_idxs, seed_tr)
    X_valid, y_valid = data_split(X, y, nan_missing, val_idxs, missing_type, train_missing_ratio, cat_idxs, con_idxs, seed_tr)
    X_test, y_test = data_split(X, y, nan_missing, test_idxs, missing_type, test_missing_ratio, cat_idxs, con_idxs, seed_ts)


    x_min = np.min(X_train_ulb['data'][:, con_idxs], axis=0).reshape(1, -1).astype(np.float32)
    x_max = np.max(X_train_ulb['data'][:, con_idxs], axis=0).reshape(1, -1).astype(np.float32)


    # Classification -> add label feature
    print(f'y_dim: {y_dim}')

    data_summary = {
        'cat_dims': cat_dims,
        'y_dim': y_dim,
        'target': target,
        'nfeat': nfeat,

        'cat_idxs': cat_idxs,
        'con_idxs': con_idxs,
        'x_min': x_min,
        'x_max': x_max
    }

    print(f'#Total_Data: {len(X)}')
    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    print(f'#Feature:{nfeat}, #Categorical:{len(cat_idxs)}, #Continuous:{len(con_idxs)}')

    return data_summary, X_train_lb, y_train_lb, X_train_ulb, y_train_ulb, X_valid, y_valid, X_test, y_test



def train_val_split(labels, label_ratio=0.1, num_labeled_samples=False, datasplit=[.65, .15, .2]):
    """
    labels: the values of y
    n_labels: the number of labeled data
    """
    
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    test_idxs = []
    
    for i in unique_labels:
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)

        if num_labeled_samples:
            n_labeled_per_class = int(num_labeled_samples * len(idxs) / labels.size)
        else:
            n_labeled_per_class = int(datasplit[0] * len(idxs) * label_ratio)

        n_train = int(datasplit[0] * len(idxs))
        n_val = int(datasplit[1] * len(idxs))

        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[:n_train])
        val_idxs.extend(idxs[n_train:n_train + n_val])
        test_idxs.extend(idxs[n_train + n_val:])

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs, test_idxs
    


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, x_min=None, x_max=None):
        X_missing =  X['missing'].copy()
        X = X['data'].copy()

        cat_cols = list(cat_cols)
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))

        self.X_cat = X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X_con = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X_cat_missing = X_missing[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X_con_missing = X_missing[:,con_cols].copy().astype(np.int64) #numerical columns

        self.y = Y['data']

        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)

        if x_min is not None and x_max is not None:
            scale = x_max - x_min
            epsilon = 1e-8
            zero_mask = scale == 0
            if np.any(zero_mask):
                print("[Warning] Detected x_max == x_min at indices:", np.where(zero_mask)[0])
                scale[zero_mask] = epsilon
                x_min[zero_mask] -= epsilon/2
            self.X_con = 2 * ((self.X_con - x_min) / scale) - 1

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return np.concatenate((self.cls[idx], self.X_cat[idx])), self.X_con[idx], self.y[idx], np.concatenate((self.cls_mask[idx], self.X_cat_missing[idx])), self.X_con_missing[idx]
