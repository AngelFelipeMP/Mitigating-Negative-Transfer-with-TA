from json import decoder
import dataset
# import engine
import engine_dataparallelism as engine
import torch
import pandas as pd
import numpy as np
import math
import random
import config
from utils import *
from tqdm import tqdm

from torch.utils.data.dataset import ConcatDataset
from samplers import BatchSamplerTrain, BatchSamplerValidation
# from model import MTLModels
from model_dataparallelism import MTLModels
import warnings
warnings.filterwarnings('ignore') 
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()


class CrossValidation(MetricTools, StatisticalTools):
    def __init__(self, model_name, heads, data_dict, max_len, transformer, batch_size, drop_out, lr, df_results, fold, num_efl, num_dfl):
        super(CrossValidation, self).__init__(model_name, heads, transformer, max_len, batch_size, lr, drop_out, num_efl, num_dfl)
        self.model_name = model_name
        self.data_dict = data_dict
        self.heads = sorted(heads.split('-'))
        self.max_len = max_len
        self.transformer = transformer
        self.batch_size = batch_size
        self.drop_out = drop_out
        self.lr = lr
        self.df_results = df_results if isinstance(df_results, pd.DataFrame) else super().create_df_results()
        self.fold = fold
        self.num_efl = num_efl
        self.num_dfl = num_dfl
        
    # def calculate_metrics(self, output_train, average='macro'):
    #     metrics_dict = {head:{} for head in self.heads}
    #     for head in self.heads:
    #         # macro average
    #         metrics_dict[head]['f1'] = metrics.f1_score(output_train[head]['targets'], output_train[head]['predictions'], average=average)
    #         metrics_dict[head]['acc'] = metrics.accuracy_score(output_train[head]['targets'], output_train[head]['predictions'])
    #         metrics_dict[head]['recall'] = metrics.recall_score(output_train[head]['targets'], output_train[head]['predictions'], average=average) 
    #         metrics_dict[head]['precision'] = metrics.precision_score(output_train[head]['targets'], output_train[head]['predictions'], average=average)
            
    #         metrics_dict[head]['f1_weighted'] = metrics.f1_score(output_train[head]['targets'], output_train[head]['predictions'], average='weighted')
    #         metrics_dict[head]['recall_weighted'] = metrics.recall_score(output_train[head]['targets'], output_train[head]['predictions'], average='weighted') 
    #         metrics_dict[head]['precision_weighted'] = metrics.precision_score(output_train[head]['targets'], output_train[head]['predictions'], average='weighted')
        
    #     return metrics_dict
    
    def calculate_metrics(self, output_train, average='binary'):
        metrics_dict = {head:{} for head in self.heads}
        for head in self.heads:
            metrics_dict[head]['acc'] = metrics.accuracy_score(output_train[head]['targets'], output_train[head]['predictions'])
            
            metrics_dict[head]['f1'] = metrics.f1_score(output_train[head]['targets'], output_train[head]['predictions'], average=average)
            metrics_dict[head]['recall'] = metrics.recall_score(output_train[head]['targets'], output_train[head]['predictions'], average=average) 
            metrics_dict[head]['precision'] = metrics.precision_score(output_train[head]['targets'], output_train[head]['predictions'], average=average)
            
            metrics_dict[head]['f1_macro'] = metrics.f1_score(output_train[head]['targets'], output_train[head]['predictions'], average='macro')
            metrics_dict[head]['recall_macro'] = metrics.recall_score(output_train[head]['targets'], output_train[head]['predictions'], average='macro') 
            metrics_dict[head]['precision_macro'] = metrics.precision_score(output_train[head]['targets'], output_train[head]['predictions'], average='macro')
            
            metrics_dict[head]['f1_weighted'] = metrics.f1_score(output_train[head]['targets'], output_train[head]['predictions'], average='weighted')
            metrics_dict[head]['recall_weighted'] = metrics.recall_score(output_train[head]['targets'], output_train[head]['predictions'], average='weighted') 
            metrics_dict[head]['precision_weighted'] = metrics.precision_score(output_train[head]['targets'], output_train[head]['predictions'], average='weighted')
        
        return metrics_dict
    
    def longer_dataset(self):
        bigger = 0
        for head in self.data_dict.keys():
            if self.data_dict[head]['rows'] > bigger:
                bigger = self.data_dict[head]['rows']
                dataset = head
        return dataset
        
    def task_text_input(self, head, step):
        if 'task-identification-text' in config.MODELS[self.model_name]['encoder']['input']:
            return self.data_dict[head][step]['task'].values
        else:
            return None
        
    def run(self):
        self.concat = {'train_datasets':[], 'val_datasets':[]}
        # loading datasets
        for head in self.heads:
            self.concat['train_datasets'].append(dataset.TransformerDataset(
                text=self.data_dict[head]['train'][config.INFO_DATA[head]['text_col']].values,
                text_par= self.task_text_input(head, 'train'),
                target=self.data_dict[head]['train'][config.INFO_DATA[head]['label_col']].values,
                max_len=self.max_len,
                transformer=self.transformer
                )
            )
            self.concat['val_datasets'].append(dataset.TransformerDataset(
                text=self.data_dict[head]['val'][config.INFO_DATA[head]['text_col']].values,
                text_par= self.task_text_input(head, 'val'),
                target=self.data_dict[head]['val'][config.INFO_DATA[head]['label_col']].values,
                max_len=self.max_len,
                transformer=self.transformer
                )
            )
            
        # concat datasets
        concat_train = ConcatDataset(self.concat['train_datasets'])
        concat_val = ConcatDataset(self.concat['val_datasets'])
        
        # creating dataloaders
        train_data_loader = torch.utils.data.DataLoader(
            dataset=concat_train,
            sampler=BatchSamplerTrain(dataset=concat_train,batch_size=self.batch_size),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.TRAIN_WORKERS
        )

        val_data_loader = torch.utils.data.DataLoader(
            dataset=concat_val,
            batch_sampler=BatchSamplerValidation(dataset=concat_val,batch_size=self.batch_size),
            shuffle=False,
            num_workers=config.VAL_WORKERS
        )
        
        device = config.DEVICE if config.DEVICE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Mymodel = MTLModels(self.transformer, self.drop_out, self.heads, self.data_dict, self.model_name, self.num_efl, self.num_dfl) 
        model = torch.nn.DataParallel(Mymodel, device_ids=[0, 1])
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

        num_train_steps = int(len(self.data_dict[self.longer_dataset()]['train']) / self.batch_size * config.EPOCHS)
        optimizer = AdamW(optimizer_parameters, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )
        
        # create obt for save preds class
        manage_preds = PredTools(self.data_dict, self.model_name, self.heads, self.drop_out, self.num_efl, self.num_dfl, self.lr, self.batch_size, self.max_len, self.transformer)
        
        for epoch in range(1, config.EPOCHS+1):
            output_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, self.heads)
            train_metrics = self.calculate_metrics(output_train)
            
            output_val = engine.eval_fn(val_data_loader, model, device, self.heads)
            val_metrics = self.calculate_metrics(output_val)
            
            # save epoch preds
            manage_preds.hold_epoch_preds(output_val, epoch)
            
            # create a list of dataframes with last caculated metrics and then add them to df_results
            list_new_results = super().new_lines_df(epoch, train_metrics, output_train, val_metrics, output_val)
            self.df_results = pd.concat([self.df_results, *list_new_results], ignore_index=True)

            tqdm.write("Epoch {}/{}".format(epoch,config.EPOCHS))
            for head in self.heads:
                tqdm.write("    Head: {:<8} f1-score_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f} f1-score_val = {:.3f}  accuracy_val = {:.3f}  loss_val = {:.3f}".format(head,
                                                                                                                                                                                        train_metrics[head]['f1'], 
                                                                                                                                                                                        train_metrics[head]['acc'], 
                                                                                                                                                                                        output_train[head]['loss'], 
                                                                                                                                                                                        val_metrics[head]['f1'], 
                                                                                                                                                                                        val_metrics[head]['acc'], 
                                                                                                                                                                                        output_val[head]['loss']))
        
        # save a fold preds
        manage_preds.concat_fold_preds()
            
        # avg and save logs
        if self.fold == config.SPLITS:
            self.df_results = super().avg_results(self.df_results)
            self.df_results = super().add_margin_of_error(self.data_dict, self.heads, self.df_results)
            super().save_results(self.df_results)
            
            # save all folds preds "gridsearch"
            manage_preds.save_preds()

        return self.df_results
    
if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    #rename old log files adding date YMD-HMS
    rename_logs()

    # create progress bar
    grid_search_bar = tqdm(total=tdqm_gridsearch(), desc='GRID SEARCH', position=0)

    # metric results dataset
    df_results = None

    # get model_name/framework_name such as 'STL', 'MTL0' and etc & parameters
    for model_name, model_characteristics in config.MODELS.items():
        
        # start model -> get datasets/heads
        for group_heads in model_characteristics['decoder']['heads']:
            
            # Model script starts Here!
            data_dict = dict()
            for head in sorted(group_heads.split('-')):
                
                # load datasets & create StratifiedKFold splitter
                data_dict[head] = {}
                data_dict[head]['merge'] = pd.read_csv(config.DATA_PATH + '/' + str(config.INFO_DATA[head]['datasets']['train'].split('_')[0]) + '_merge' + '_processed.csv', nrows=config.N_ROWS)
                data_dict[head]['num_class'] = len(data_dict[head]['merge'][config.INFO_DATA[head]['label_col']].unique().tolist())
                data_dict[head]['rows'] = data_dict[head]['merge'].shape[0] 
                data_dict[head]['skf'] = StratifiedKFold(n_splits=config.SPLITS, shuffle=True, random_state=config.SEED)
            
            # grid search
            for num_efl in parameters(model_name)['task-identification-vector']:
                for num_dfl in parameters(model_name)['deep-classifier']:
                    for transformer in config.TRANSFORMERS:
                        for max_len in config.MAX_LEN:
                            for batch_size in config.BATCH_SIZE:
                                for drop_out in config.DROPOUT:
                                    for lr in config.LR:
                                        
                                        # split data
                                        for fold, indexes in enumerate(zip(*[data_dict[d]['skf'].split(data_dict[d]['merge'][config.INFO_DATA[d]['text_col']], data_dict[d]['merge'][config.INFO_DATA[d]['label_col']]) for d in sorted(data_dict.keys())]), start=1):
                                            
                                            for data, index in zip(sorted(data_dict.keys()), indexes):
                                                data_dict[data]['train'] = data_dict[data]['merge'].loc[index[0]]
                                                data_dict[data]['val'] = data_dict[data]['merge'].loc[index[1]]
                                                
                                            tqdm.write(f'\nModel: {model_name} Heads: {group_heads} Encode-feature-layers: {num_efl} Decoder-feature-layers: {num_dfl} Dropout: {drop_out} lr: {lr} Max_len: {max_len} Batch_size: {batch_size} Fold: {fold}/{config.SPLITS}')
                                            
                                            cv = CrossValidation(model_name, 
                                                                group_heads,
                                                                data_dict, 
                                                                max_len, 
                                                                transformer, 
                                                                batch_size, 
                                                                drop_out,
                                                                lr,
                                                                df_results,
                                                                fold,
                                                                num_efl,
                                                                num_dfl
                                            )
                                            
                                            df_results = cv.run()
                                            grid_search_bar.update(1)
                                            