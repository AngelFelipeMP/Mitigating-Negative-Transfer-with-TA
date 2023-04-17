#TODO: delet this script as soon as I finished code adaptaion for MTL
import math
import torch
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import RandomSampler


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.smallest_dataset_size = min([len(cur_dataset.samples) for cur_dataset in dataset.datasets])
        self.interactions = math.ceil(self.smallest_dataset_size / self.batch_size)

    def __len__(self):
        return self.batch_size * math.ceil(self.smallest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        datasets_size = []
        batch_size = []
        start_interactions = []
        
        for dataset_idx in range(self.number_of_datasets):
            
            cur_dataset = self.dataset.datasets[dataset_idx]
            datasets_size.append(len(cur_dataset ))
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)
            
            batch = datasets_size[dataset_idx]//self.interactions
            print('#'*50)
            print(batch)
            batch_size.append(batch)
            sample = batch*self.interactions
            print(sample)
            remainder = datasets_size[dataset_idx] - sample
            print(remainder)
            start_interactions.append(self.interactions - remainder)
            print(self.interactions - remainder)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        # print('!'*75)
        # print(push_index_val)
        # step = self.batch_size * self.number_of_datasets # I don't need it 
        # epoch_samples = self.smallest_dataset_size * self.number_of_datasets
        final_samples_list = []  # this is a list of indexes from the combined dataset
        
        # for j, _ in enumerate(range(0, epoch_samples, step)):  # I don't need it  # interations
        print('@'*50)
        print(self.interactions)
        
        ########### !!!!!!!!!!!!!!! ANALISE the following code !!!!!!!!!!!!!!!!!!!! ##############
        ########### !!!!!!!!!!!!!!! ANALISE inside loop !!!!!!!!!!!!!!!!!!!! ##############
        for j in range(self.interactions):
            print('*'*50)
            print(j)
            
            for i in range(self.number_of_datasets):
                print('&'*50)
                print(start_interactions[i])
                
                samples_to_grab = batch_size[i] if j < start_interactions[i] else batch_size[i] + 1
                
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    print('^'*50)
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                        # print(cur_samples)
                    except StopIteration:
                        print('BREAK')
                        break
                final_samples_list.append(cur_samples)
        
        print(final_samples_list)
        
        return iter(final_samples_list)
    
    
    
class MyFirstDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((-torch.ones(6), torch.ones(7)))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]


class MySecondDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((torch.ones(10) * 5, torch.ones(7) * -5))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]
    
    
class MyThirdDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((torch.ones(9) * 8, torch.ones(11) * -8))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]    

if __name__ == "__main__":
    first_dataset = MyFirstDataset()
    second_dataset = MySecondDataset()
    third_dataset = MyThirdDataset()
    concat_dataset = ConcatDataset([first_dataset, second_dataset, third_dataset])
    
    

    batch_size = 4

    # dataloader with BatchSchedulerSampler
    dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                            batch_sampler=BatchSchedulerSampler(dataset=concat_dataset,
                                                                        batch_size=batch_size),
                                            shuffle=False)

    for inputs in dataloader:
        print(inputs)