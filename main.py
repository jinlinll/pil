from models import LinearModel, ResNet18, ResNet50, VGG19, DenseNet121, MobileNetV2
from attacks import PILAdversary
from tools import get_free_gpu, get_dataset, load_set
from trainer import train_model
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PIL attack.')
    parser.add_argument('--dataset',default='cifar10',type=str, help='dataset, choose `svhn`, `cifar10` or `cifar100`. ' \
                        'For other datasets, please modify the code directly.')
    
    parser.add_argument('--lr_train', default = 0.003, type=float, help='learning rate for pretraining the linear model.')
    parser.add_argument('--pretrain_iter', default = 30, type = int, help = ' number of iteration used to pertrain the linear model.')
    parser.add_argument('--gc_train', default = None, type=int, help='grad clip threshold for pretraining the linear model.')

    parser.add_argument('--alpha', default = 8/2550, type = float, help = 'attack step size of PIL.')
    parser.add_argument("--eps", default= 8/255, type=float, help="epsilon. Perturbation budget of PIL")
    parser.add_argument('--lmd', default = 0.9, type = float, help = 'balancing factor of PIL.')
    parser.add_argument('--attack_iter', default = 30, type = int, help = ' number of iteration used to generate unlearnable data.')
    

    parser.add_argument('--save_path', default = './data/ue', type=str, help='path to save perturbed dataset.')

    parser.add_argument('--show_clean_test', action='store_true', help='train the attacked model on a clean dataset for comparison.')

    parser.add_argument('--attacked_model', default = 'resnet18', type=str, help='attacked model type, ' \
                        'choose `resnet18`, `resnet50`, `vgg19`, `densenet121`, `mobilenetv2`.')
    parser.add_argument('--lr_test', default = 0.1, type=float, help='learning rate for training the attacked model.')
    parser.add_argument('--test_iter', default = 30, type = int, help = ' number of iteration used to train the attacked model.')
    parser.add_argument('--gc_test', default = None, type=int, help='grad clip threshold for training the attacked model.')

    args = parser.parse_args()

    num_classes=10 if args.dataset != 'cifar100' else 100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = get_dataset(name=args.dataset, train=True, transform = transform_test)
    test_set = get_dataset(name=args.dataset, train=False, transform = transform_test)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=False, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, pin_memory=True, num_workers=8)


    linear_model = LinearModel(num_classes= num_classes)
    print("Pertraining the linear model...")
    train_model(model=linear_model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=args.pretrain_iter,
                learning_rate=args.lr_train,
                grad_clip=args.gc_train,
                weight_decay=0)


    print("Generating the PIL attack...")
    adversary = PILAdversary(
        model=linear_model,
        epsilon=args.eps,
        alpha=args.alpha,
        steps=args.attack_iter,
        lmd=args.lmd,
        num_classes=num_classes,
        device= get_free_gpu()
    )
    adversary.generate(train_loader, save_path=os.path.join(args.save_path,f'unlearnable_{args.dataset}.pt'))

    unlearnable_set = load_set(path=os.path.join(args.save_path,f'unlearnable_{args.dataset}.pt'), transform=transform_train)
    unlearnable_loader = DataLoader(unlearnable_set, batch_size=512, shuffle=True, pin_memory=True, num_workers=8)

    model_map = {
        'resnet18': ResNet18,
        'resnet50': ResNet50,
        'vgg19': VGG19,
        'densenet121': DenseNet121,
        'mobilenetv2': MobileNetV2
    }
    target_model = model_map[args.attacked_model](num_classes=num_classes)

    if args.show_clean_test:
        clean_model = model_map[args.attacked_model](num_classes=num_classes)
        print(f"Training {args.attacked_model} on clean dataset...")
        clean_acc=train_model(model=clean_model,
                    train_loader=train_loader,
                    test_loader=test_loader, 
                    num_epochs=args.test_iter,
                    learning_rate=args.lr_test,
                    grad_clip=args.gc_test)
    
    print(f"Training {args.attacked_model} on unlearnable dataset...")
    unlearnable_acc = train_model(model=target_model,
                train_loader=unlearnable_loader,
                test_loader=test_loader,
                num_epochs=args.test_iter,
                learning_rate=args.lr_test,
                grad_clip=args.gc_test)
    if args.show_clean_test:
        print(f'Accuracy drop: {100*(clean_acc-unlearnable_acc)}%')