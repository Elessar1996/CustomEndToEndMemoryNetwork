
import nltk
nltk.download('punkt')
import torch
from dataset import BagOfTheWordsAggregate
from data_utils import *
from tqdm.auto import tqdm
from model import EndToEndMemoryNetworkHops
from engine import train

epochs = 200

batch_size = 32

num_hops = 10

embedding_size = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_directory = 'data/data/en'

task_ids = [i for i in range(1, 2)]

train_torch_dataset, test_torch_dataset = prepare_data(data_directory, task_ids)

train_dataloader, test_torch_dataset = get_dataloaders(train_torch_dataset, test_torch_dataset, batch_size=batch_size)

model = EndToEndMemoryNetworkHops(embedding_size=embedding_size,
                                         vocabulary_size=train_torch_dataset.vocabulary_size,
                                         memory_size=100,
                                         sentence_size=dataset.MAX_STORY_LENGTH,
                                         num_hops=num_hops)
optimizer = torch.optim.Adam(params=model.parameters(),lr = 1e-3, weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)




training_result = train(model=model,
                        optimizer=optimizer,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_torch_dataset,
                        loss_fn=criterion,
                        device=device)





