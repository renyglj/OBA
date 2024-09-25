import torch
import torchvision
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from options import args_parser
from Googlenet_cifar import GoogLeNet
from ALexnet_cifar import AlexNet
from vgg_16_cifar import VGG

class MTL_Model():
    def __init__(self, shared_layers, learning_rate, lr_decay, lr_decay_epoch, momentum, weight_decay):
        self.shared_layers = shared_layers
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.momentum = momentum
        self.weight_decay = weight_decay

        param_dict = [{"params": self.shared_layers.parameters()}]
        self.optimizer = optim.SGD(params=param_dict,
                                   lr=learning_rate,
                                   momentum=momentum,
                                   weight_decay=weight_decay)
        self.optimizer_state_dict = self.optimizer.state_dict()  # state_dict()将每一层与它的参数建立映射关系
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5, last_epoch=-1)

    def exp_lr_sheduler(self, epoch):
        if (epoch + 1) % self.lr_decay_epoch:
            return None
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_decay
            # print(f"Epoch {epoch + 1}: Reducing learning rate to {param_group['lr']}")

    def optimize_model(self, input_batch, label_batch):
        self.shared_layers.train(True)
        output_batch = self.shared_layers(input_batch)
        self.optimizer.zero_grad()
        batch_loss = self.criterion(output_batch, label_batch)
        # batch_loss.backward()
        self.shared_layers.backward(output_batch)
        self.optimizer.step()
        return batch_loss.item()

    def test_model(self, input_batch):
        self.shared_layers.train(False)
        with torch.no_grad():
            output = self.shared_layers(input_batch)
        self.shared_layers.train(True)
        return output

    def update_model(self, new_shared_layers):
        self.shared_layers.load_state_dict(new_shared_layers)




def initialize_model(args, device):
    if args.dataset == 'cifar10':
        # if args.model == 'AlexNet':
        if args.model == 'vgg-16':
            # shared_layers = GoogLeNet()
            # shared_layers = AlexNet()
            shared_layers = VGG()

        else:
            raise ValueError('Model is not implemented for CIFAR-10')
    else:
        raise ValueError('The dataset is not implemented')
    if args.cuda:
        shared_layers = shared_layers.cuda(device)


    model = MTL_Model(shared_layers=shared_layers,
                      learning_rate=args.lr,
                      lr_decay=args.lr_decay,
                      lr_decay_epoch=args.lr_decay_epoch,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)
    return model

def main():
    args = args_parser()
    device = 'cpu'
    model = initialize_model(args, device)

    # import os
    # from torchvision import transforms, datasets
    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    #     "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
    #                                transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))# get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
    #                                      transform=data_transform["train"])
    # batch_size = 32
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))
    #
    # trainloader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size, shuffle=True,
    #                                            num_workers=nw)
    #
    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
    #                                         transform=data_transform["val"])
    # val_num = len(validate_dataset)
    # testloader = torch.utils.data.DataLoader(validate_dataset,
    #                                               batch_size=4, shuffle=True,
    #                                               num_workers=nw)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    # trainset = torchvision.datasets.CIFAR10(root='./data_set/flower_data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    # testset = torchvision.datasets.CIFAR10(root='./data_set/flower_data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    for epoch in tqdm(range(1)):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss = model.optimize_model(input_batch=inputs, label_batch=labels)
            running_loss += loss
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model.test_model(input_batch=images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    main()




