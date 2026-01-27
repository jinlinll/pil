from models import ResNet18, ResNet50, VGG19, DenseNet121, MobileNetV2
from tools import get_dataset, load_set, SplitMergeDataset
from trainer import train_model, train_adv_model
from augmentations import cross_entropy, Cutout,CutMix,MixUp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

augs = {
        'none':
        transforms.Compose([
            transforms.ToTensor(),
        ]),
        'basic':
            transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'rotation':
            transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'perspective':
            transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'grayscale':
            transforms.Compose([
            transforms.Grayscale(3),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'channelshuffle':
            transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[torch.randperm(x.size(0)), :, :])
        ]),
        'cutout':
            transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            Cutout(16)
        ]),
        'cutmix':
            transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
        ]),
        'mixup':
            transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
        ]),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PIL attack.')

    parser.add_argument('--model', default = 'resnet18', type=str, help='attacked model type, ' \
                        'choose `resnet18`, `resnet50`, `vgg19`, `densenet121`, `mobilenetv2`.')
    parser.add_argument('--dataset',default='cifar10',type=str, help='dataset, choose `svhn`, `cifar10` or `cifar100`. ' \
                        'For other datasets, please modify the code directly. This argument is used to load clean train set and test set.')
    parser.add_argument('--unlearnable_path', default = './data/ue/unlearnable_cifar10.pt', type=str, help='path to unlearnable datasets.')

    parser.add_argument('--partial_perturb', action='store_true', help='only perturb a proportion of the dataset.')
    parser.add_argument('--perturb_ratio', default=1.0, type=float, help='percentage ([0,1]) of perturbed data.')

    parser.add_argument('--augmentation', default = 'basic', type=str, help=f"augmentations during train, \
                        choose one from {', '.join([f'`{aug_name}`'  for aug_name,_ in augs.items()])}.")

    parser.add_argument('--AT', action='store_true', help='PGD-7 Adversarial training.')
    parser.add_argument('--AT_eps', default=8/255, type=float, help='Adversarial training radius.')

    parser.add_argument('--lr', default = 0.1, type=float, help='learning rate for training the model.')
    parser.add_argument('--iter', default = 100, type = int, help = ' number of iteration used to train the model.')
    parser.add_argument('--gc', default = None, type=int, help='grad clip threshold for training the model.')

    args = parser.parse_args()

    num_classes=10 if args.dataset != 'cifar100' else 100
    if args.augmentation not in augs.keys():
        raise ValueError(f'No augmentation called {args.augmentation}.')
    
    criterion = nn.CrossEntropyLoss()
    if 'mix' in args.augmentation:
        criterion = cross_entropy

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    unlearnable_set = load_set(path=args.unlearnable_path, transform=augs[args.augmentation])
    test_set = get_dataset(name=args.dataset, train=False, transform=transform_test)

    train_set = unlearnable_set
    if args.partial_perturb:
        clean_set = get_dataset(name=args.dataset, train=True, transform=augs[args.augmentation])
        train_set = SplitMergeDataset(unlearnable_set, clean_set, args.perturb_ratio)

    if args.augmentation == 'cutmix':
        train_set = CutMix(train_set, num_class=num_classes)
    elif args.augmentation == 'mixup':
        train_set = MixUp(train_set, num_class=num_classes)

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, pin_memory=True, num_workers=8)
    model_map = {
        'resnet18': ResNet18,
        'resnet50': ResNet50,
        'vgg19': VGG19,
        'densenet121': DenseNet121,
        'mobilenetv2': MobileNetV2
    }
    model = model_map[args.model](num_classes=num_classes)
    if args.AT :
        r = 8
        train_adv_model(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader, 
                    num_epochs=args.iter,
                    learning_rate=args.lr,
                    grad_clip=args.gc,
                    criterion = criterion,
                    epsilon= args.AT_eps,
                    alpha= args.AT_eps/4,
                    num_steps=7)
    else:
        train_model(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader, 
                    num_epochs=args.iter,
                    learning_rate=args.lr,
                    grad_clip=args.gc,
                    criterion = criterion)
    


