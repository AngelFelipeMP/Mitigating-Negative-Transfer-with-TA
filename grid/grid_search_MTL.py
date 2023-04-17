import dataset
import engine
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
from model import MTLModels
import warnings
warnings.filterwarnings('ignore') 
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()


#COMMENT: the CrossValidation need to receive model_characteristics because super().save_preds() needs it@
class CrossValidation(MetricTools, StatisticalTools):
    def __init__(self, model_name, heads, data_dict, max_len, transformer, batch_size, drop_out, lr, df_results, fold):
        super(CrossValidation, self).__init__(model_name, heads, transformer, max_len, batch_size, lr, drop_out)
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
        
    def calculate_metrics(self, output_train, pos_label=1, average='micro'):
        metrics_dict = {head:{} for head in self.heads}
        for head in self.heads:
            metrics_dict[head]['f1'] = metrics.f1_score(output_train[head]['targets'], output_train[head]['predictions'], pos_label=pos_label, average=average)
            metrics_dict[head]['acc'] = metrics.accuracy_score(output_train[head]['targets'], output_train[head]['predictions'])
            metrics_dict[head]['recall'] = metrics.recall_score(output_train[head]['targets'], output_train[head]['predictions'], pos_label=pos_label, average=average) 
            metrics_dict[head]['precision'] = metrics.precision_score(output_train[head]['targets'], output_train[head]['predictions'], pos_label=pos_label, average=average)
        
        return metrics_dict
    
    def longer_dataset(self):
        bigger = 0
        for head in self.data_dict.keys():
            if self.data_dict[head]['rows'] > bigger:
                bigger = self.data_dict[head]['rows']
                dataset = head
        return dataset
        
    def run(self):
        self.concat = {'train_datasets':[], 'val_datasets':[]}
        # loading datasets
        for head in self.heads:
            self.concat['train_datasets'].append(dataset.TransformerDataset(
                text=self.data_dict[head]['train'][config.INFO_DATA[head]['text_col']].values,
                target=self.data_dict[head]['train'][config.INFO_DATA[head]['label_col']].values,
                max_len=self.max_len,
                transformer=self.transformer
                )
            )
            
            self.concat['val_datasets'].append(dataset.TransformerDataset(
                text=self.data_dict[head]['val'][config.INFO_DATA[head]['text_col']].values,
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
            sampler=BatchSamplerTrain(dataset=concat_train,batch_size=batch_size),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.TRAIN_WORKERS
        )

        val_data_loader = torch.utils.data.DataLoader(
            dataset=concat_val,
            batch_sampler=BatchSamplerValidation(dataset=concat_val,batch_size=batch_size),
            shuffle=False,
            num_workers=config.VAL_WORKERS
        )
        
        device = config.DEVICE if config.DEVICE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MTLModels(self.transformer, self.drop_out, self.heads, self.data_dict)
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
        manage_preds = PredTools(self.data_dict, self.model_name, self.heads, self.drop_out, self.lr, self.batch_size, self.max_len, self.transformer)
        
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

    #COMMENT: add feature layers encoder, feature layers decoder @
    inter_parameters = len(config.TRANSFORMERS) * len(config.MAX_LEN) * len(config.BATCH_SIZE) * len(config.DROPOUT) * len(config.LR) * config.SPLITS
    inter_models =  len(config.MODELS.keys()) * math.prod([len(items['decoder']['heads']) for items in config.MODELS.values()])
    grid_search_bar = tqdm(total=(inter_parameters*inter_models), desc='GRID SEARCH', position=0)
    
    # metric results dataset
    df_results = None

    # get model_name/framework_name such as 'STL', 'MTL0' and etc & parameters
    #COMMENT: I should send everything "model_name, model_characteristics" together to "CrossValidation()"@
    #COMMENT: related comment below@
    for model_name, model_characteristics in config.MODELS.items():
        
        # start model -> get datasets/heads
        #COMMENT: Here I can get the other models characteristics for the MTL models @
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
                                        
                                    tqdm.write(f'\nModel: {model_name} Heads: {group_heads} Transformer: {transformer.split("/")[-1]} Max_len: {max_len} Batch_size: {batch_size} Dropout: {drop_out} lr: {lr} Fold: {fold}/{config.SPLITS}')
                                    
                                    cv = CrossValidation(model_name, 
                                                        group_heads, #COMMENT: I shouldn't pass "heads" to function I could get it from data_dict if I add it as first key to data_dict["EXIST-DETOXIS-HatEval"] @
                                                        data_dict, 
                                                        max_len, 
                                                        transformer, 
                                                        batch_size, 
                                                        drop_out,
                                                        lr,
                                                        df_results,
                                                        fold
                                    )
                                    
                                    df_results = cv.run()
                                    grid_search_bar.update(1)









        ## TOOLKIT
        # print('@'*100)
        # print(self.df_train.columns)
        
        # TASKS:tes
        #     1) check COMMENTS [DONE]
        #     2) check paper and drive notes [DONE]
        #     3) check test [DONE]
        #     4) reading DataLoader MTL web article [DONE]
        #     5) plan the MTL implementation [DONE]
        #     6) Start implementation [DONE]
        #     7) Write dataloader script [DONE]
        #     8) add heads in the dataloader output[X]
        #     9) check model.py, engine.py and dataloader.py [DONE]
        #     10) Adapt new_grid_search for MTL - Dataset/DataLoader [DONE]
        #     11) Adapt new_grid_search for MTL - remaining [DONE]
        #     12) Check adaptation of new_grid.py [DONE]
        #     13) Run script and fix problems new_grid.py [X]
                    # - Run script and fix errors [X]
                    # - check logs/tables --> Bug skf.split --> check logs --> [X]
                    # - print import output - add resuts, avg and so on [X]
                    # - check backpropagation [X]
                    # remove unnecessary commented lines [X]
        #     14) Break the script into utils.py and grid_search.py [X]
        #     15) Move part of the run code to a new class or func[X]
        #     16) Double check utils.py and grid_search.py [X]
        #     17) Run utils.py and grid_search.py [X]
        #     18) Check the results from the experiment that I let running [X]
        #     19) Run middle lgth test with the code adapted to MTL [X]
                # - machine 2 GRIUs
                # - STL & MTL separeted

## PLAN WHAT I WILL DO NEXT !!!!!!!


        # If I don't obtain the expected results
        #     x1) Modify engine.py and model.py to be able to print model structure