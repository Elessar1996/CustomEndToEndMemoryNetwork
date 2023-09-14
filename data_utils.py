import torch
from dataset import BagOfTheWordsAggregate
import os

def convert_to_torch_dataset(list_of_tuples):
    '''
    this functions converts the tuples of ndarrays to torch datasets

    :param list_of_tuples: a tuple consisting of numpy ndarrays of stories, queries and answers
    :return: torch datasets

    '''

    torch_like_data = [(torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), torch.from_numpy(np.array(z).copy()))
                       for x, y, z in list_of_tuples]

    torch_like_stories, torch_like_query, torch_like_answer = zip(*torch_like_data)

    torch_like_stories = torch.stack(torch_like_stories, dim=0)
    torch_like_query = torch.stack(torch_like_query, dim=0)
    torch_like_answer = torch.stack(torch_like_answer, dim=0)

    torch_like_dataset = torch.utils.data.TensorDataset(torch_like_stories, torch_like_query, torch_like_answer)

    return torch_like_dataset


def prepare_data(data_directory='data/data/en', task_ids=[1]):
    """

    this function gets the data directory and taks ids and
    returns the corresponding torch datasets

    :param data_directory: data directory
    :param task_ids: a list of task ids
    :return: a tuple cotaining train dataset and test dataset
    """

    dataset = BagOfTheWordsAggregate(
        data_directory=data_directory,
        task_id=task_ids
    )
    train_data, test_data = dataset.get_idx_data()

    train_torch_dataset = convert_to_torch_dataset(train_data)
    test_torch_dataset = convert_to_torch_dataset(test_data)

    return train_torch_dataset, test_torch_dataset


def get_dataloaders(train_torch_dataset, test_torch_dataset, batch_size=32):

    """
    this function gets the torch dataset and convert it to torch dataloader

    :param torch_train_dataset: train torch dataset that wanted to be converted to torch dataloader
    :param torch_test_dataset: test torch dataset that wanterd to be converted to torch dataloader
    :param batch_size:batch_size
    :return:
    dataloader with specified batch size
    """

    train_dataloader = torch.utils.data.DataLoader(train_torch_dataset, batch_size=32, shuffle=True,
                                                   num_workers=os.cpu_count())
    test_dataloader = torch.utils.data.DataLoader(test_torch_dataset, batch_size=32, shuffle=False,
                                                  num_workers=os.cpu_count())

    return train_dataloader, test_dataloader
