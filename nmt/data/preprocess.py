from torchtext import data
import torch


class MyIterator(data.Iterator):
    """Defines an iterator that loads batches of data from a Dataset.

       Attributes:
           dataset: The Dataset object to load Examples from.
           batch_size: Batch size.
           batch_size_f: Function of three arguments (new example to add, current
               count of examples in the batch, and current effective batch size)
               that returns the new effective batch size resulting from adding
               that example to a batch. This is useful for dynamic batching, where
               this function would add to the current effective batch size the
               number of tokens in the new example.
           sort_key: A key to use for sorting examples in order to batch together
               examples with similar lengths and minimize padding. The sort_key
               provided to the Iterator constructor overrides the sort_key
               attribute of the Dataset, or defers to it if None.
           train: Whether the iterator represents a train set.
           repeat: Whether to repeat the iterator for multiple epochs. Default: False.
           shuffle: Whether to shuffle examples between epochs.
           sort: Whether to sort examples according to self.sort_key.
               Note that shuffle and sort default to train and (not train).
           sort_within_batch: Whether to sort (in descending order according to
               self.sort_key) within each batch. If None, defaults to self.sort.
               If self.sort is True and this is False, the batch is left in the
               original (ascending) sorted order.
           device (str or `torch.device`): A string or instance of `torch.device`
               specifying which device the Variables are going to be created on.
               If left as default, the tensors will be created on cpu. Default: None.
       """
    # def __init__(self, dataset, batch_size, sort_key=None, device=None,
    #              batch_size_f=None, train=True,
    #              repeat=False, shuffle=None, sort=None,
    #              sort_within_batch=None):
    #     super().__init__(dataset=dataset,
    #                      batch_size=batch_size,
    #                      sort_key=sort_key,
    #                      device=device,
    #                      batch_size_fn=batch_size_f,
    #                      train=train,
    #                      repeat=repeat,
    #                      shuffle=shuffle,
    #                      sort=sort,
    #                      sort_within_batch=sort_within_batch)
    #     self.batches = []

    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))
