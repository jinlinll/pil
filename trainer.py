from tools import TrainingMetrics, get_free_gpu
import torch
from torch import nn
from tqdm import tqdm

def train_model(model, train_loader, test_loader, num_epochs=30, learning_rate=0.03, 
                device=get_free_gpu(), criterion = nn.CrossEntropyLoss(), weight_decay=1e-4, momentum = 0.9, grad_clip = None):

    model.to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr= learning_rate, weight_decay=weight_decay, momentum= momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    train_metric = TrainingMetrics()
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_metric.reset()
        pbar = tqdm(train_loader, total=len(train_loader))
        head = 0
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            batch_size = len(labels)
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=grad_clip, 
                    norm_type=2 
                )
            optimizer.step()
            train_metric.update(loss, logits, labels)
            train_loss, train_acc = train_metric.get_epoch_stats()
            pbar.set_description(f"Epoch {epoch+1} Acc {train_acc * 100 :.2f} Loss: {train_loss :.2f}" )
        scheduler.step()
        head+=batch_size
        # Eval
        model.eval()
        correct, total = 0, 0
        for _, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total
        tqdm.write(f'Test Accuracy {test_acc * 100 :.2f}\n')
    return test_acc



def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.003, num_steps=20):
    images = images.clone().detach().requires_grad_(True)
    original_images = images.clone().detach()
    
    for _ in range(num_steps):
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        grad = images.grad.data
        images = images + alpha * torch.sign(grad)
        
        images = torch.max(torch.min(images, original_images + epsilon), original_images - epsilon)
        images = torch.clamp(images, 0, 1)
        
        images = images.clone().detach().requires_grad_(True)
    
    return images

def train_adv_model(model, train_loader, test_loader, num_epochs=30, learning_rate=0.03, 
                device=get_free_gpu(), criterion = nn.CrossEntropyLoss(), weight_decay=1e-4, momentum = 0.9, epsilon=0.031, alpha=0.007, num_steps=7, grad_clip = None):

    model.to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr= learning_rate, weight_decay=weight_decay, momentum= momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    train_metric = TrainingMetrics()
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_metric.reset()
        pbar = tqdm(train_loader, total=len(train_loader))
        head = 0
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            batch_size = len(labels)
            model.zero_grad()
            optimizer.zero_grad()
            adv_inputs = pgd_attack(model, images, labels, epsilon, alpha, num_steps)
            logits = model(adv_inputs)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=grad_clip, 
                    norm_type=2 
                )
            optimizer.step()
            train_metric.update(loss, logits, labels)
            train_loss, train_acc = train_metric.get_epoch_stats()
            pbar.set_description(f"Epoch {epoch+1} Acc {train_acc * 100 :.2f} Loss: {train_loss :.2f}" )
        scheduler.step()
        head+=batch_size
        # Eval
        model.eval()
        correct, total = 0, 0
        for _, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total
        tqdm.write(f'Test Accuracy {test_acc * 100 :.2f}\n')
    return test_acc
