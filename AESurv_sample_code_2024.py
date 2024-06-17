import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch
from torch import nn
import torch.nn.functional as F
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from pycox.evaluation import EvalSurv
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc


class NetAESurv(nn.Module):
    def __init__(self, in_features,h1,latent_size,out_features,drop_rate):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, h1), nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(h1, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, h1), nn.ReLU(),
            nn.Linear(h1, in_features),
        )
        self.surv_net = nn.Sequential(
            nn.Linear(latent_size, out_features),#directpredict,noMLP
        )

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        phi = self.surv_net(encoded)
        return phi, decoded

    def predict(self, input):
        encoded = self.encoder(input)
        return self.surv_net(encoded).view(-1)
    def predict_emb(self, input):
        return self.encoder(input)

class LossAECoxph(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need `alpha` in [0, 1].'
        self.alpha = alpha
        self.loss_surv = CoxPHLoss()
        self.loss_ae = nn.MSELoss()
        
    def forward(self, phi, decoded, target_surv, target_ae):
        idx_durations, events = target_surv
        loss_surv = self.loss_surv(phi, idx_durations, events)
        loss_ae = self.loss_ae(decoded, target_ae)
        return self.alpha * loss_surv + (1 - self.alpha) * loss_ae

def Kfold_c(length,fold):
    size = np.arange(length).tolist()
    train_index = []
    val_index = []
    rest = length % fold
    fold_size = int(length/fold)
    temp_fold_size = fold_size
    for i in range(fold):
        temp_train = []
        temp_val = []
        if rest>0:
            temp_fold_size = fold_size+1
            rest = rest -1
            temp_val = size[i*temp_fold_size:+i*temp_fold_size+temp_fold_size]
            temp_train = size[0:i*temp_fold_size] + size[i*temp_fold_size+temp_fold_size:]
        else:
            temp_val = size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size
                            :(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size]
            temp_train = size[0:(length % fold)*temp_fold_size+(i-(length % fold))*fold_size] + size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size:]
        train_index.append(temp_train)
        val_index.append(temp_val)
    return (train_index,val_index)

#Data could be obtained through request as described in 'Data Availability" section at Shen et al., Briefings in Bioinformatics (in revision)
DNAm_sites=np.load('data.npy') #This is DNA methylation data
n_sample=DNAm_sites.shape[0] #This is to extract number of participants
CHD_label_binary=np.load('data.npy').reshape(n_sample,1) #This is CHD binary label data
CHD_label_timetoevent=np.load('data.npy').reshape(n_sample,1) #This is CHD time-to-event label data


site_count = DNAm_sites.shape[1]
site_name = []
for i in range(site_count):
    site_name.append('x'+str(i))

col_name = site_name + ['duration','event']

data_combine = np.concatenate((DNAm_sites,CHD_label_timetoevent,CHD_label_binary),1)

df_all = pd.DataFrame(data_combine,columns=col_name)

time_to_event_label = np.zeros(n_sample, dtype={'names':('cens','time'),'formats':('?', '<f8')})
time_to_event_label['cens'] = CHD_label_binary.reshape(-1)
time_to_event_label['time'] = CHD_label_timetoevent.reshape(-1)

AESur_Cindex=[]
AESur_mean_auc_all=[]
AESur_auc_all=[]


for epoch in range(5):
    print('repeat run',epoch)
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)

    train_split_index,test_split_index = Kfold_c(n_sample,5)

    splits = 5
    concordance_score_all = []
    mean_auc_all = []
    auc_time_all = []
    time_all = []
    
    embedding_test_all=[]
    embedding_train_all=[]
    embedding_val_all=[]
    embedding_test_label_all=[]
    embedding_train_label_all=[]
    embedding_val_label_all=[]    
    
    for k in range(splits):

        print('batch is ',k)
        train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
        valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
        test_index = test_split_index[k]

        train_id = [total_id[i] for i in train_index]
        valid_id = [total_id[i] for i in valid_index]
        test_id = [total_id[i] for i in test_index]
        
        cols_standardize = site_name
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        x_mapper = DataFrameMapper(standardize) 

        df_train = df_all.iloc[train_id]
        df_val = df_all.iloc[valid_id]
        df_test = df_all.iloc[test_id]

        x_train = x_mapper.fit_transform(df_train).astype('float32')
        print('x_train.shape',x_train.shape)
        x_val = x_mapper.transform(df_val).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')

        forAUC_train = np.array([time_to_event_label[i] for i in train_id])
        forAUC_test = np.array([time_to_event_label[i] for i in test_id])

        
        get_target = lambda df: (df['duration'].values, df['event'].values)
        y_train = get_target(df_train)
        y_val = get_target(df_val)
        durations_test, events_test = get_target(df_test)

        
        train = tt.tuplefy(x_train, (y_train, x_train))
        val = tt.tuplefy(x_val, (y_val, x_val))
        
        in_features = x_train.shape[1]
        out_features = 1
        h1= 512
        latent_size=32
        drop_rate = 0.5
        batch_size = 128
        AEmodel = NetAESurv(in_features, h1,latent_size, out_features,drop_rate)
        
        loss_combined = LossAECoxph(0.5)
        
        model = CoxPH(net=AEmodel, optimizer=tt.optim.Adam(0.0001,weight_decay=0.0001), loss=loss_combined)
        
        #print loss in metrics
        metrics = dict(loss_surv = LossAECoxph(1),loss_ae = LossAECoxph(0))
        callbacks = [tt.cb.EarlyStopping()]
        epochs = 100
        verbose=True
        
        #%%time
        log = model.fit(*train, batch_size, epochs, callbacks, verbose,val_data=val, val_batch_size=batch_size,metrics=metrics)
        _ = model.compute_baseline_hazards(input = x_train,target=y_train)
        logits = model.predict(x_test)
        
        #embeddings
        embedding_test=model.net.predict_emb(torch.Tensor(x_test))
        embedding_val = model.net.predict_emb(torch.Tensor(x_val))
        embedding_train = model.net.predict_emb(torch.Tensor(x_train))
        
        
        hr_pred = -logits
        hr_pred_1 = -np.exp(logits)
        c_ind = concordance_index(durations_test,hr_pred,events_test)
        c_ind_1 = concordance_index(durations_test,hr_pred_1,events_test)

        concordance_score_all.append(c_ind)

        
        time_int = np.array(model.compute_baseline_hazards(input = x_train,target=y_train).keys())
        est = -hr_pred_1
        est_T = np.array(est).T
        measure_times = np.linspace(1, 27, 300)
        auc_time,mean_auc = cumulative_dynamic_auc(forAUC_train, forAUC_test, est_T, measure_times)
        auc_time_all.append(auc_time)
        mean_auc_all.append(mean_auc)
        
        embedding_test_all.append(embedding_test)
        embedding_train_all.append(embedding_train)
        embedding_val_all.append(embedding_val)   
        
        embedding_test_label_all.append(events_test)
        embedding_train_label_all.append(y_train)
        embedding_val_label_all.append(y_val)       
    
    AESur_Cindex.append(concordance_score_all)
    AESur_mean_auc_all.append(mean_auc_all)
    AESur_auc_all.append(auc_time_all)

    

d1 = {'AESur_c-index':AESur_Cindex,'AESur_meanauroc':AESur_mean_auc_all,      'AESur_c-index mean':np.mean(AESur_Cindex), 'AESur_meanauroc mean':np.mean(AESur_mean_auc_all),      'AESur_c-index std':np.std(AESur_Cindex), 'AESur_meanauroc std':np.std(AESur_mean_auc_all),      'AESur_c-index_':str(np.mean(AESur_Cindex).round(3))+u"\u00B1"+str(np.std(AESur_Cindex).round(3)),      'AESur_meanauroc_':str(np.mean(AESur_mean_auc_all).round(3))+u"\u00B1"+str(np.std(AESur_mean_auc_all).round(3)),     }


