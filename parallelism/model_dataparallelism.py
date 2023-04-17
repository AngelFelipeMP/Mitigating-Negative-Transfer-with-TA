import config
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MTLModels(nn.Module):
    def __init__(self, transformer, drop_out, heads, data_dict, model_name, num_efl, num_dfl):
        super(MTLModels, self).__init__()
        self.num_efl = num_efl
        self.num_dfl = num_dfl
        self.data_dict = data_dict
        self.model_name = model_name
        self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
        self.transformer = AutoModel.from_pretrained(transformer)
        self.dropout = nn.Dropout(drop_out)
        self.heads = heads
        self.classifiers = nn.ModuleDict()
        for head in self.heads:
            self.classifiers[head] = nn.ModuleList()
            self.classifiers[head].append(nn.Linear(self.embedding_size * 2, self.data_dict[head]['num_class']))
        
        if ('task-identification-vector' in config.MODELS[self.model_name]['encoder']['input']) & (self.num_efl > 0):
            self.encoder_feature_layers = nn.ModuleList()
            for layer,_ in enumerate(range(num_efl), start=1):
                input_size = self.embedding_size * 2 + len(self.heads) if layer == 1 else self.embedding_size * 2
                self.encoder_feature_layers.append(nn.Linear(input_size, self.embedding_size * 2))
                self.encoder_feature_layers.append(nn.ReLU(self.embedding_size * 2))

        if ('deep-classifier' in config.MODELS[self.model_name]['decoder']['model']) & (self.num_dfl > 0):
            for head in self.heads:
                for _ in range(num_dfl):
                    self.classifiers[head].insert(0, nn.ReLU(self.embedding_size * 2))
                    self.classifiers[head].insert(0, nn.Linear(self.embedding_size * 2, self.embedding_size * 2))
    
    def forward(self, iputs, head):
        transformer_output  = self.transformer(**iputs)
        mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
        max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
        cat = torch.cat((mean_pool,max_pool), 1)
        out = self.dropout(cat)
        
        if ('task-identification-vector' in config.MODELS[self.model_name]['encoder']['input']) & (self.num_efl > 0):
            task_ident_vector = [0] * len(self.heads)
            task_ident_vector[self.heads.index(head)] = 1
            # task_ident_vector = torch.tensor([task_ident_vector] * out.shape[0], dtype=torch.long).to(config.DEVICE)
            task_ident_vector = torch.tensor([task_ident_vector] * out.shape[0], dtype=torch.long).to(out.get_device())
            out = torch.cat((out,task_ident_vector), 1)
            for layer in self.encoder_feature_layers:
                out = layer(out)

        if ('deep-classifier' in config.MODELS[self.model_name]['decoder']['model']) & (self.num_dfl > 0):
            for layer in self.classifiers[head][:-1]:
                out = layer(out)
        
        return self.classifiers[head][-1](out)