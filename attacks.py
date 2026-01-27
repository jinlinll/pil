import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

class PILAdversary:
    def __init__(self, model, num_classes = 100, epsilon=8/255, alpha=8/2550, lmd = 0.9, steps=30, device = 'cuda'):
        self.model = model.to(device).eval()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.lmd = lmd
        self.steps = steps
        self.device = device
        
    def attack(self, images, labels):
        orig_images = images.clone().detach().to(self.device)
        unl_images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)

        # init noise
        noise = torch.zeros_like(unl_images).uniform_(-self.epsilon, self.epsilon)

        # optimize noise
        for _ in range(self.steps):
            noise.requires_grad = True
            clean_outputs = self.model(unl_images - noise)
            delta_outputs = self.model(noise)
            ce_loss = F.cross_entropy(delta_outputs,labels)
            p_clean = F.softmax(clean_outputs, dim = 1)
            kl_loss = F.kl_div(p_clean.log(), torch.tensor([1/self.num_classes]*self.num_classes, dtype=torch.float32).to(p_clean.device), reduction='batchmean')
            loss = self.lmd*ce_loss + (1-self.lmd)*kl_loss

            self.model.zero_grad()
            loss.backward()
            grad = noise.grad.data
            noise = torch.clamp(noise.detach() - self.alpha * grad.sign(), 
                               min=-self.epsilon, max=self.epsilon)
        unl_images = torch.clamp(orig_images - noise, 0, 1).detach()
            
        return unl_images
    
    def generate(self, dataloader, save_path):
        """
        Generate unlearnable samples for an entire dataset and save them.

        Args:
            dataloader: DataLoader for the clean training set.
            save_path: Path to save the poisoned dataset (recommended format: .pt).
        """
        all_images = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc='Generating PIL Data'):
            poisoned_batch = self.attack(images, labels)
            all_images.append(poisoned_batch.cpu())
            all_labels.append(labels.cpu())

        final_images = torch.cat(all_images, dim=0)
        final_labels = torch.cat(all_labels, dim=0)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save({
            'data': final_images,
            'labels': final_labels
        }, save_path)

        print(f'Poisoned dataset saved to: {save_path}')
    
    