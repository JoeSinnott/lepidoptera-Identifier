import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

def train_binary_model():
    # dataset paths
    data_dir = 'training_images/bootstrapping'
    
    # standard imagenet stats and augmentations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # init dataloaders
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"training on {dataset_sizes['train']} images, validating on {dataset_sizes['val']}...")

    # load pretrained resnet
    model = models.resnet18(pretrained=True)
    
    # freeze backbone weights
    for param in model.parameters():
        param.requires_grad = False

    # replace head for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # loss and optimizer (optimize head only)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # train loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}')

    # save weights
    torch.save(model.state_dict(), 'life_stage_classifier.pth')
    print("model saved as life_stage_classifier.pth")

if __name__ == '__main__':
    train_binary_model()