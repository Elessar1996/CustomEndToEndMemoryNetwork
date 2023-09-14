import torch
import torchmetrics


def train_for_one_epoch(model: torch.nn.Module,
                        dataloader: torch.nn.utils.DataLoader,
                        criterion: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        device: torch.device):
    model.train()
    acc, l = 0, 0
    for story, query, answer in dataloader:
        story = story.to(device)

        query = query.to(device)

        story = story.type(torch.float32)

        query = query.type(torch.float32)

        answer = answer.long()
        answer = answer.to(device)

        a_hat = model(story, query)

        loss = criterion(a_hat.squeeze(),
                         torch.nn.functional.one_hot(answer.long(), num_classes=model.vocabulary_size).float())
        l += loss
        accuracy = torchmetrics.functional.precision(a_hat.squeeze().argmax(dim=-1),
                                                     torch.nn.functional.one_hot(answer).argmax(dim=-1),
                                                     task="multiclass", num_classes=model.vocabulary_size)
        acc += accuracy
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40)
    acc /= len(dataloader)
    l /= len(dataloader)

    return acc, l


def test_for_one_epoch(model: torch.nn.Module,
                       dataloader: torch.nn.utils.DataLoader,
                       criterion: torch.nn.Module,
                       device: torch.device):
    model.eval()

    acc, l = 0, 0

    for story, query, answer in dataloader:
        story = story.to(device)

        query = query.to(device)

        story = story.type(torch.float32)

        query = query.type(torch.float32)

        answer = answer.long()
        answer = answer.to(device)
        with torch.inference_mode():
            a_hat = model(story, query)

        loss = criterion(a_hat.squeeze(),
                         torch.nn.functional.one_hot(answer.long(), num_classes=model.vocabulary_size).float())

        l += loss

        accuracy = torchmetrics.functional.precision(a_hat.squeeze().argmax(dim=-1),
                                                     torch.nn.functional.one_hot(answer).argmax(dim=-1),
                                                     task="multiclass", num_classes=model.vocabulary_size)
        acc += accuracy

    acc /= len(dataloader)
    l /= len(dataloader)

    return acc, l


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler: torch.optim.lr_scheduler.LinearLR):
    training_result = []

    model.to(device)

    for epoch in range(epochs):

        print(f'epoch: {epoch}')

        train_acc, train_l = train_step(model=model,
                                        dataloader=train_dataloader,
                                        criterion=loss_fn,
                                        optimizer=optimizer,
                                        device=device)

        test_acc, test_l = test_step(model=model,
                                     dataloader=test_dataloader,
                                     criterion=loss_fn,
                                     device=device)

        print(f'train accuracy: {train_acc} || train loss: {train_l} || test accuracy: {test_acc} || test loss: {test_l}')
        print('---------------------------------------------------------')
        training_result.append((train_acc, train_l, test_acc, test_l))

        scheduler.step()

    torch.save(obj=model.state_dict(),
             f='./saved_model.pth')

    return training_result
