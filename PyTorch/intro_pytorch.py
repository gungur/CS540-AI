import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training=True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if training:
        train_set = datasets.FashionMNIST('./ data', train=True, download=True, transform=custom_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
        return train_loader
    elif not training:
        test_set = datasets.FashionMNIST('./ data', train=False, transform=custom_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
        return test_loader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * labels.size(0)

        print(
            f'Train Epoch: {epoch}\tAccuracy: {correct}/{total}({correct / total * 100:.2f}%)\tLoss: {running_loss / total:.3f}')


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """

    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0

        for data in test_loader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * labels.size(0)

        if show_loss:
            print(f'Average loss: {running_loss / total:.4f}')
            print(f'Accuracy: {correct / total * 100:.2f}%')
        elif not show_loss:
            print(f'Accuracy: {correct / total * 100:.2f}%')


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle Boot']
    logits = model(test_images[index])
    prob = F.softmax(logits, dim=1, dtype=torch.float64)
    order = torch.argsort(prob, descending=True, stable=True)
    index_1 = order[0][0]
    index_2 = order[0][1]
    index_3 = order[0][2]

    print(str(class_names[index_1]) + ': ' + f'{prob[0][index_1] * 100:.2f}%')
    print(str(class_names[index_2]) + ': ' + f'{prob[0][index_2] * 100:.2f}%')
    print(str(class_names[index_3]) + ': ' + f'{prob[0][index_3] * 100:.2f}%')


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to examine the correctness of your functions. 
    Note that this part will not be graded.
    '''

    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)

    model = build_model()
    print(model)

    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, 5)

    evaluate_model(model, test_loader, criterion, show_loss=False)
    evaluate_model(model, test_loader, criterion, show_loss=True)

    test_images = next(iter(test_loader))[0]
    predict_label(model, test_images, 1)
