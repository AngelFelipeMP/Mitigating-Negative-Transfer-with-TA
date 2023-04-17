import torch
from transformers import AutoTokenizer

class TransformerDataset:
    def __init__(self, text, text_par,target, max_len, transformer):
        self.text = text
        self.text_par = text_par
        self.target = target
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(transformer) 

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text_par =  None if (self.text_par is None) else str(self.text_par[item])

        inputs = self.tokenizer.encode_plus(
            text,
            text_par,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation='only_first'
        )
        
        inputs = {k:torch.tensor(v, dtype=torch.long) for k,v in inputs.items()}
        inputs['targets'] = torch.tensor(self.target[item], dtype=torch.long)
        
        return inputs