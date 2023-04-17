# import os
import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm

from model import MTLModels
import warnings
warnings.filterwarnings('ignore') 
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()

def calculate_metrics(pred, targ, pos_label=1, average='binary'):
    return {
            'f1':metrics.f1_score(targ, pred, pos_label=pos_label, average=average), 
            'acc':metrics.accuracy_score(targ, pred, pos_label=pos_label, average=average), 
            'recall':metrics.recall_score(targ, pred, pos_label=pos_label, average=average), 
            'precision':metrics.precision_score(targ, pred, pos_label=pos_label, average=average)
            }

# def run(df_train, df_val, max_len, task, transformer, batch_size, drop_out, lr, df_results):
def run(df_train, df_val, model_name, heads, max_len, transformer, batch_size, drop_out, lr, df_results):
    
    
    train_dataset = dataset.TransformerDataset(
        text=df_train['text_col'].values,
        target=df_train['label_col'].values,
        max_len=max_len,
        transformer=transformer
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers = config.TRAIN_WORKERS
    )

    val_dataset = dataset.TransformerDataset(
        text=df_val['text_col'].values,
        target=df_val['label_col'].values,
        max_len=max_len,
        transformer=transformer
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        num_workers=config.VAL_WORKERS
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MTLModels(transformer, drop_out, number_of_classes=df_train['label_col'].max()+1)
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / batch_size * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    for epoch in range(1, config.EPOCHS+1):
        pred_train, targ_train, loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        train_metrics = calculate_metrics(pred_train, targ_train)
        
        pred_val, targ_val, loss_val = engine.eval_fn(val_data_loader, model, device)
        val_metrics = calculate_metrics(pred_val, targ_val)

        
        df_new_results = pd.DataFrame({'model':model_name, #list(model_info.keys())[0]
                            'data': heads, #TODO add index for the dataset or dataset name directly [???]
                            'epoch':epoch,
                            'transformer':transformer,
                            'max_len':max_len,
                            'batch_size':batch_size,
                            'lr':lr,
                            'dropout':drop_out,
                            'accuracy_train':train_metrics['acc'],
                            'f1-macro_train':train_metrics['f1'],
                            'recall_train':train_metrics['recall'],
                            'precision_train':train_metrics['precision'],
                            'loss_train':loss_train,
                            'accuracy_val':val_metrics['acc'],
                            'f1-score_val':val_metrics['f1'],
                            'recall_val':val_metrics['recall'],
                            'precision_val':val_metrics['precision'],
                            'loss_val':loss_val
                        }, index=[0]
        ) 
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        tqdm.write("Epoch {}/{} f1-macro_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f} f1-macro_val = {:.3f}  accuracy_val = {:.3f}  loss_val = {:.3f}".format(epoch, 
                                                                                                                                                                                    config.EPOCHS, 
                                                                                                                                                                                    train_metrics['f1'], train_metrics['acc'], loss_train, 
                                                                                                                                                                                    val_metrics['f1'], val_metrics['acc'], loss_val))

    return df_results

if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    skf = StratifiedKFold(n_splits=config.SPLITS, shuffle=True, random_state=config.SEED)
    

    #TODO: confidence interval for the evaluation metrics "df_new_results" & "df_results"
    df_results = pd.DataFrame(columns=['model', 
                                        'data',
                                        'epoch',
                                        'transformer',
                                        'max_len',
                                        'batch_size',
                                        'lr',
                                        'dropout',
                                        'accuracy_train',
                                        'f1-macro_train',
                                        'recall_train',
                                        'precision_train'
                                        'loss_train',
                                        'accuracy_val',
                                        'f1-score_val',
                                        'recall_val',
                                        'precision_val',
                                        'loss_val'
                                    ]
        )
    
    #COMMENT: To think about tqdm code
    #COMMENT: add feature layers encoder, feature layers decoder
    inter = len(config.TRANSFORMERS) * len(config.MAX_LEN) * len(config.BATCH_SIZE) * len(config.DROPOUT) * len(config.LR) * config.SPLITS   
    grid_search_bar = tqdm(total=inter, desc='GRID SEARCH', position=2)

    # get model_name such as 'STL', 'MTL0' and etc & parameters
    for model_name, model_characteristics in config.MODELS.items():
        
        # start model -> get datasets/heads
        #COMMENT: Here I can get the other models characteristics for the MTL models
        for group_heads in model_characteristics['decoder']['heads']:
            
            # Model script starts Here!
            for head in group_heads.split('-'):
                data_dict = dict()
                
                # load datasets & create StratifiedKFold splitter
                data_dict[head]['merge'] = pd.read_csv(config.DATA_PATH + '/' + config.INFO_DATA[head]['datasets']['train'][:-4] + '_merge' + '_processed.csv', nrows=config.N_ROWS)
                data_dict[head]['data_split'] = skf.split(data_dict[head]['merge'][config.INFO_DATA[head]['text_col']], data_dict[head]['merge'][config.INFO_DATA[head]['label_col']])
                
            
            # grid search
            # for transformer in tqdm(config.TRANSFORMERS, desc='TRANSFORMERS', position=0):
            for transformer in config.TRANSFORMERS:
                for max_len in config.MAX_LEN:
                    for batch_size in config.BATCH_SIZE:
                        for drop_out in config.DROPOUT:
                            for lr in config.LR:
                                
                                # split data
                                for fold, indexes in enumerate(zip(*[data_dict[d]['data_split'] for d in sorted(data_dict.keys())])):
                                    for data, index in zip(sorted(data_dict.keys()), indexes):
                                        data_dict[data]['train'] = data_dict[data]['merge'].loc[index[0]]
                                        data_dict[data]['val'] = data_dict[data]['merge'].loc[index[1]]
                                        
                                        #TODO: move code below out of last for loop
                                        #TODO: run must receice data_dict instead of data_dict[data]['train'] or data_dict[data]['val']
                                        tqdm.write(f'\nModel: {model_name} Heads: {group_heads} Max_len: {max_len} Batch_size: {batch_size} Dropout: {drop_out} lr: {lr} Fold: {fold+1}/{config.SPLITS}')
                                        
                                        #TODO: I shouldn't pass "head" or "data" to run function
                                        df_results = run(data_dict[data]['train'],
                                                            data_dict[data]['val'],
                                                            model_name,
                                                            group_heads,
                                                            max_len, 
                                                            transformer, 
                                                            batch_size, 
                                                            drop_out,
                                                            lr,
                                                            df_results
                                        )
                
                                        grid_search_bar.update(1)
                    
                                
                                df_results = df_results.groupby(['model',
                                                                'data',
                                                                'epoch',
                                                                'transformer',
                                                                'max_len',
                                                                'batch_size',
                                                                'lr',
                                                                'dropout'], as_index=False, sort=False)['accuracy_train',
                                                                                                    'f1-macro_train',
                                                                                                    'recall_train',
                                                                                                    'precision_train'
                                                                                                    'loss_train',
                                                                                                    'accuracy_val',
                                                                                                    'f1-macro_val',
                                                                                                    'recall_val',
                                                                                                    'precision_val',
                                                                                                    'loss_val'].mean()
                                
                                df_results.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '.csv', index=False)
        
#TODO may remove save models
    

# pre_trained_model = "bert-base-uncased"
# transformer = AutoModel.from_pretrained(pre_trained_model)
# tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)

# max_len = 15
# Example1 = "Angel table home car"
# Example2 = "bhabha char roofing house get"
# Example3 = "I wan to go to the beach for surfing"

# pt_batch = tokenizer(
#     [Example1, Example2, Example3],
#     padding=True,
#     truncation=True,
#     add_special_tokens=True,
#     max_length=max_len,
#     return_tensors="pt")

# print(pt_batch)






# if __name__ == "__main__":
#     random.seed(config.SEED)
#     np.random.seed(config.SEED)
#     torch.manual_seed(config.SEED)
#     torch.cuda.manual_seed_all(config.SEED)

#     dfx = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TRAIN, sep='\t', nrows=config.N_ROWS).fillna("none")
#     skf = StratifiedKFold(n_splits=config.SPLITS, shuffle=True, random_state=config.SEED)

#     df_results = pd.DataFrame(columns=['task',
#                                         'epoch',
#                                         'transformer',
#                                         'max_len',
#                                         'batch_size',
#                                         'lr',
#                                         'dropout',
#                                         'accuracy_train',
#                                         'f1-macro_train',
#                                         'loss_train',
#                                         'accuracy_val',
#                                         'f1-macro_val',
#                                         'loss_val'
#             ]
#     )
    
#     inter = len(config.LABELS) * len(config.TRANSFORMERS) * len(config.MAX_LEN) * len(config.BATCH_SIZE) * len(config.DROPOUT) * len(config.LR) * config.SPLITS
#     grid_search_bar = tqdm(total=inter, desc='GRID SEARCH', position=2)
    
#     for task in tqdm(config.LABELS, desc='TASKS', position=1):
#         df_grid_search = dfx.loc[dfx[task]>=0].reset_index(drop=True)
#         for transformer in tqdm(config.TRANSFORMERS, desc='TRANSFOMERS', position=0):
#             for max_len in config.MAX_LEN:
#                 for batch_size in config.BATCH_SIZE:
#                     for drop_out in config.DROPOUT:
#                         for lr in config.LR:
                            
#                             for fold, (train_index, val_index) in enumerate(skf.split(df_grid_search[config.DATASET_TEXT_PROCESSED], df_grid_search[task])):
#                                 df_train = df_grid_search.loc[train_index]
#                                 df_val = df_grid_search.loc[val_index]
                                
#                                 tqdm.write(f'\nTask: {task} Transfomer: {transformer.split("/")[-1]} Max_len: {max_len} Batch_size: {batch_size} Dropout: {drop_out} lr: {lr} Fold: {fold+1}/{config.SPLITS}')
                                
#                                 df_results = run(df_train,
#                                                     df_val, 
#                                                     max_len, 
#                                                     task, 
#                                                     transformer, 
#                                                     batch_size, 
#                                                     drop_out,
#                                                     lr,
#                                                     df_results
#                                 )
                            
#                                 grid_search_bar.update(1)
                            
#                             df_results = df_results.groupby(['task',
#                                                             'epoch',
#                                                             'transformer',
#                                                             'max_len',
#                                                             'batch_size',
#                                                             'lr',
#                                                             'dropout'], as_index=False, sort=False)['accuracy_train',
#                                                                                                 'f1-macro_train',
#                                                                                                 'loss_train',
#                                                                                                 'accuracy_val',
#                                                                                                 'f1-macro_val',
#                                                                                                 'loss_val'].mean()
                            
#                             df_results.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '.csv', index=False)