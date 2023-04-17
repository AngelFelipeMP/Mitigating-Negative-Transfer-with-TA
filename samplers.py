import math
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler

class BatchSamplerTrain(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)




class BatchSamplerValidation(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a batch per task in each mini-batch 
    with different length for each task.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.smallest_dataset_size = min([len(cur_dataset) for cur_dataset in dataset.datasets])
        self.interactions = math.ceil(self.smallest_dataset_size / self.batch_size)

    def __len__(self):
        # return self.batch_size * math.ceil(self.smallest_dataset_size / self.batch_size) * len(self.dataset.datasets)
        return math.ceil(self.smallest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        datasets_size = []
        batch_size = []
        start_interactions = []
        
        for dataset_idx in range(self.number_of_datasets):
            
            cur_dataset = self.dataset.datasets[dataset_idx]
            datasets_size.append(len(cur_dataset ))
            sampler = SequentialSampler(cur_dataset)
            # sampler = RandomSampler(cur_dataset) #TODO: check if it is possible to use random
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)
            
            batch = datasets_size[dataset_idx]//self.interactions
            batch_size.append(batch)
            sample = batch*self.interactions
            remainder = datasets_size[dataset_idx] - sample
            start_interactions.append(self.interactions - remainder)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        final_samples_list = []  # this is a list of indexes from the combined dataset

        for j in range(self.interactions):
            
            for i in range(self.number_of_datasets):
                samples_to_grab = batch_size[i] if j < start_interactions[i] else batch_size[i] + 1
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + push_index_val[i]
                    cur_samples.append(cur_sample)

                final_samples_list.append(cur_samples)
                
        return iter(final_samples_list)