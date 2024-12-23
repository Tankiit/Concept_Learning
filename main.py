import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict
from rotated_mnist import RotatedMNIST, get_dataset
from causal_vae import ConceptVAE
import numpy as np
import torchvision.transforms as transforms
from config import get_config,TrainingConfig
from datetime import datetime
from pathlib import Path
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR,LinearLR
from torch.optim.lr_scheduler import SequentialLR
from matplotlib import gridspec





def custom_collate(batch):
    """Custom collate function to handle dictionary batches"""
    elem = batch[0]
    batch_dict = {}
    
    for key in elem:
        try:
            if key == 'image':
                # Ensure all images are tensors
                images = [d[key] if torch.is_tensor(d[key]) else transforms.ToTensor()(d[key]) 
                         for d in batch]
                batch_dict[key] = torch.stack(images)
            elif key in ['digit', 'angle']:
                # Convert to tensor for labels
                batch_dict[key] = torch.tensor([d[key] for d in batch])
            elif key in ['invariant_concepts', 'variant_concepts']:
                # Concepts should already be tensors
                batch_dict[key] = torch.stack([d[key] for d in batch])
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            print(f"Type of first element: {type(batch[0][key])}")
            raise
    
    return batch_dict


def main():
    config = get_config()
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{config.wandb.name}_{timestamp}"
    config.training.checkpoint_dir = f"checkpoints/{exp_name}"
    
    # Setup wandb
    wandb.init(
        project=config.wandb.project,
        name=exp_name,
        tags=config.wandb.tags,
        notes=config.wandb.notes,
        config=config.to_dict()
    )
    
    # Set random seed
    torch.manual_seed(config.training.seed)
    
    # Get datasets separately
    train_dataset = get_dataset(
        root_dir=config.data.root_dir,
        train=True,
        angles=config.data.angles, 
        concept_type=config.data.concept_type
    )
    
    val_dataset = get_dataset(
        root_dir=config.data.root_dir,
        train=False,
        angles=config.data.angles,
        concept_type=config.data.concept_type
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=custom_collate,
        pin_memory=config.training.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=custom_collate,
        pin_memory=config.training.pin_memory
    )
    
    # Initialize model
    model = ConceptVAE(
        config=config,
        input_dim=config.model.input_dim,
        zy_dim=config.model.zy_dim,
        zd_dim=config.model.zd_dim,
        concept_dim=config.model.concept_dim,
        n_classes=config.model.n_classes,
        n_domains=config.model.n_domains
    ).to(config.training.device)
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    
    def get_scheduler(optimizer, config):
        """Initialize learning rate scheduler
        
        Args:
            optimizer: The optimizer to schedule
            config:object
            
        Returns:
            scheduler: The configured learning rate scheduler
        """
        # First, create the main scheduler based on config type
        if config.training.scheduler_type == 'step':
            main_scheduler = StepLR(
                optimizer,
                step_size=config.training.step_size,
                gamma=config.training.gamma
            )
        elif config.training.scheduler_type == 'cosine':
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.training.epochs - config.training.warmup_epochs,
                eta_min=config.training.min_lr
            )
        elif config.training.scheduler_type == 'linear':
            main_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=config.training.min_lr / config.training.lr,
                total_iters=config.training.epochs - config.training.warmup_epochs
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config.training.scheduler_type}")
        
        # Add warmup if specified
        if config.training.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.training.warmup_epochs
            )
            
            scheduler = SequentialLR(
                optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config.training.warmup_epochs]
            )
        else:
            scheduler = main_scheduler
    
        return scheduler
    


    # Setup scheduler
    scheduler = get_scheduler(optimizer, config)
   
     # Training loop
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    print(f"Training on device: {config.training.device}")
    print(f"Total epochs: {config.training.epochs}")
    print(f"Scheduler type: {config.training.scheduler_type}")
    print(f"Warmup epochs: {config.training.warmup_epochs}")
    print(f"Initial learning rate: {config.training.lr}")
    print(f"Checkpoints will be saved to: {config.training.checkpoint_dir}")
    
    
    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch+1}/{config.training.epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_metrics = train_epoch(
            model, train_loader, optimizer, 
            config, epoch
        )
        print_metrics(epoch, train_metrics, mode='Train')
        
        # Validation phase
        model.eval()
        val_metrics = evaluate(
            model, val_loader, 
            config, epoch
        )
        print_metrics(epoch, val_metrics, mode='Validation')
        
        #Update learning rate scheduler
        scheduler.step()
        
        # Update best model
        is_best = False
        if val_metrics['val_acc'] > best_val_acc:
            best_val_acc = val_metrics['val_acc']
            is_best = True
            patience_counter = 0
        elif val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            is_best = True
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            {**train_metrics, **val_metrics},
            config, is_best
        )
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            **train_metrics,
            **val_metrics
        })
        
        # Print best results so far
        print(f"\nBest results so far:")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
            break
        
        next_lr=optimizer.param_groups[0]['lr']
        print(f"Learning rate: {next_lr:.6f}")
    # Final evaluation
    print("\nTraining completed!")
    print(f"Loading best model from {config.training.checkpoint_dir}/best_model.pt")
    
    # Load best model
    checkpoint = torch.load(f"{config.training.checkpoint_dir}/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    model.eval()
    final_metrics = evaluate_generalization(model, val_loader)
    print("\nFinal Evaluation Results:")
    print_metrics(epoch, final_metrics, mode='Final')

def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False):
    """Save model checkpoint"""
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.to_dict()
    }
    
    # Save latest checkpoint
    latest_path = checkpoint_dir / 'latest_checkpoint.pt'
    torch.save(checkpoint, latest_path)
    
    # Save best model if this is the best so far
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"\nNew best model saved at epoch {epoch}!")
    
    # Save periodic checkpoint
    if epoch % config.training.save_freq == 0:
        periodic_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, periodic_path)

def print_metrics(epoch, metrics, mode='Train'):
    """Print metrics in a formatted way"""
    print(f"\n{mode} Metrics - Epoch {epoch}:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")
    print("=" * 50)



def evaluate(model, dataloader, config, epoch):
    """Standard evaluation for validation during training"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    losses = defaultdict(float)
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            images = batch['image'].to(config.training.device)
            digits = batch['digit'].to(config.training.device)
            angles = batch['angle'].to(config.training.device)
            inv_concepts = batch['invariant_concepts'].to(config.training.device)
            var_concepts = batch['variant_concepts'].to(config.training.device)
            
            # Forward pass
            outputs = model(images,digit=digits,angle=angles)
            
            # Calculate losses (same as training)
            recon_loss = F.mse_loss(outputs['recon'], images)
            digit_loss = F.cross_entropy(outputs['digit_pred'], digits)
            domain_loss = F.cross_entropy(outputs['domain_pred'], angles)
            
            # Concept losses
            inv_concept_loss = F.mse_loss(
                outputs['invariant_concepts'],
                inv_concepts
            )
            var_concept_loss = F.mse_loss(
                outputs['variant_concepts'],
                var_concepts
            )
            
            # KL divergence for both latent spaces
            kl_y = -0.5 * torch.sum(
                1 + outputs['zy_logvar'] - outputs['zy_mean'].pow(2) - outputs['zy_logvar'].exp()
            )
            kl_d = -0.5 * torch.sum(
                1 + outputs['zd_logvar'] - outputs['zd_mean'].pow(2) - outputs['zd_logvar'].exp()
            )
            kl_loss = kl_y + kl_d
            
            # Total loss
            loss = (recon_loss + 
                   digit_loss + 
                   config.training.domain_weight * domain_loss +
                   config.training.concept_weight * (inv_concept_loss + var_concept_loss) +
                   config.training.kl_weight * kl_loss)
            
            # Update metrics
            total_loss += loss.item()
            pred = outputs['digit_pred'].argmax(dim=1)
            correct += pred.eq(digits).sum().item()
            total += digits.size(0)
            
            # Store component losses
            losses['recon'] += recon_loss.item()
            losses['digit'] += digit_loss.item()
            losses['domain'] += domain_loss.item()
            losses['concept'] += (inv_concept_loss.item() + var_concept_loss.item())
            losses['kl'] += kl_loss.item()
            
            # Store concepts for analysis
            if total % 100 == 0:  # Every 100 samples
                wandb.log({
                    'concept_similarity': F.cosine_similarity(
                        outputs['invariant_concepts'].mean(0),
                        outputs['variant_concepts'].mean(0),
                        dim=0
                    ).item()
                })
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    metrics = {
        'val_loss': avg_loss,
        'val_acc': accuracy,
        'val_recon_loss': losses['recon'] / len(dataloader),
        'val_digit_loss': losses['digit'] / len(dataloader),
        'val_domain_loss': losses['domain'] / len(dataloader),
        'val_concept_loss': losses['concept'] / len(dataloader),
        'val_kl_loss': losses['kl'] / len(dataloader)
    }
    
    # Additional analysis
    if epoch % 5 == 0:  # Every 5 epochs
        concept_metrics = analyze_concepts(model, dataloader,config)
        metrics.update(concept_metrics)
    
    return metrics

def analyze_concepts(model, dataloader,config):
    """Analyze concept learning quality"""
    model.eval()
    device = config.training.device
    # Initialize as None
    # all_inv_concepts = None
    # all_var_concepts = None
    # all_digit_labels = None
    # all_angle_labels = None

    # # Convert tensor to list if needed
    # if isinstance(inv_concepts, torch.Tensor):
    #     inv_concepts = inv_concepts.cpu().numpy().tolist()
    
    # Now initialize as empty lists
    inv_concepts = None
    var_concepts = None
    digit_labels = None
    angle_labels = None
    
    with torch.no_grad():
        for batch_idx,data in  enumerate(dataloader):
            # Move data to device
            images = data['image'].to(device)
            digits = data['digit'].to(device)
            angles = data['angle'].to(device)
            
            # Get model outputs
            outputs = model(images, digit=digits, angle=angles)

            if inv_concepts is None:
                inv_concepts = outputs['invariant_concepts']
                var_concepts = outputs['variant_concepts']
                digit_labels = digits
                angle_labels = angles
            else:
                inv_concepts = torch.cat([inv_concepts, outputs['invariant_concepts']],dim=0)
                var_concepts = torch.cat([var_concepts, outputs['variant_concepts']],dim=0)
                digit_labels = torch.cat([digit_labels, digits],dim=0)
                angle_labels = torch.cat([angle_labels, angles],dim=0)

        metrics = {
            'inv_concept_norm': inv_concepts.norm(dim=1).mean().item(),
            'var_concept_norm': var_concepts.norm(dim=1).mean().item(),
            'inv_concept_std': inv_concepts.std(dim=1).mean().item(),
            'var_concept_std': var_concepts.std(dim=1).mean().item()
        }
        return metrics
            
            # # Now we can safely append
            # inv_concepts.append(outputs['invariant_concepts'])
            # var_concepts.append(outputs['variant_concepts'])
            # digit_labels.append(digits)
            # angle_labels.append(angles)

    # # Initialize lists to store concepts
    # inv_concepts = []
    # var_concepts = []
    # digit_labels = []
    # angle_labels = []

    
    
    # with torch.no_grad():
    #     for batch_idx,data in  enumerate(dataloader):
    #         # Move data to device
    #         images = data['image'].to(device)
    #         digits = data['digit'].to(device)
    #         angles = data['angle'].to(device)
            
    #         # Get model outputs
    #         outputs = model(images, digit=digits, angle=angles)
            
    #         # Convert numpy arrays to tensors if needed and append
    #         if isinstance(outputs['invariant_concepts'], np.ndarray):
    #             inv_concepts.append(torch.from_numpy(outputs['invariant_concepts']).to(device))
    #         else:
    #             inv_concepts.append(outputs['invariant_concepts'])
                
    #         if isinstance(outputs['variant_concepts'], np.ndarray):
    #             var_concepts.append(torch.from_numpy(outputs['variant_concepts']).to(device))
    #         else:
    #             var_concepts.append(outputs['variant_concepts'])
                
    #         digit_labels.append(digits)
    #         angle_labels.append(angles)
    
    # Stack all tensors (using stack instead of cat since they're individual tensors)
    inv_concepts = torch.stack(inv_concepts)
    var_concepts = torch.stack(var_concepts)
    digit_labels = torch.cat(digit_labels)
    angle_labels = torch.cat(angle_labels)
    
    # Calculate concept metrics
    metrics = {
        'inv_concept_norm': inv_concepts.norm(dim=1).mean().item(),
        'var_concept_norm': var_concepts.norm(dim=1).mean().item(),
        'inv_concept_std': inv_concepts.std(dim=1).mean().item(),
        'var_concept_std': var_concepts.std(dim=1).mean().item()
    }
    
    # Optional: Add more detailed analysis
    if config.training.analyze_concepts:
        # Compute concept similarity matrix
        inv_sim = torch.mm(inv_concepts.view(-1, inv_concepts.size(-1)), 
                          inv_concepts.view(-1, inv_concepts.size(-1)).t())
        var_sim = torch.mm(var_concepts.view(-1, var_concepts.size(-1)), 
                          var_concepts.view(-1, var_concepts.size(-1)).t())
        
        metrics.update({
            'inv_concept_sim': inv_sim.mean().item(),
            'var_concept_sim': var_sim.mean().item()
        })
    
    return metrics
    
    # with torch.no_grad():
    #     inv_concepts = []
    #     var_concepts = []
    #     digits = []
    #     angles = []
        
    #     # Collect concepts
    #     for batch in dataloader:
    #         outputs = model(batch['image'].to(config.training.device))
    #         inv_concepts.append(outputs['invariant_concepts'].cpu().detach().numpy())
    #         var_concepts.append(outputs['variant_concepts'].cpu().detach().numpy())
    #         digits.append(batch['digit'])
    #         angles.append(batch['angle'])
        
    #     # Concatenate all
    #     inv_concepts = torch.cat(inv_concepts)
    #     var_concepts = torch.cat(var_concepts)
    #     digits = torch.cat(digits)
    #     angles = torch.cat(angles)
        
    #     # 1. Concept Disentanglement
    #     concept_metrics['disentanglement'] = 1 - torch.abs(
    #         F.cosine_similarity(inv_concepts.mean(0), var_concepts.mean(0))
    #     ).item()
        
    #     # 2. Invariance Score (how well invariant concepts ignore rotation)
    #     inv_by_angle = {}
    #     for angle in torch.unique(angles):
    #         mask = angles == angle
    #         inv_by_angle[angle.item()] = inv_concepts[mask].mean(0)
        
    #     inv_scores = []
    #     for a1 in inv_by_angle:
    #         for a2 in inv_by_angle:
    #             if a1 < a2:
    #                 sim = F.cosine_similarity(
    #                     inv_by_angle[a1],
    #                     inv_by_angle[a2],
    #                     dim=0
    #                 )
    #                 inv_scores.append(sim.item())
    #     concept_metrics['invariance_score'] = np.mean(inv_scores)
        
    #     # 3. Concept Consistency
    #     digit_centroids = {}
    #     for d in torch.unique(digits):
    #         mask = digits == d
    #         digit_centroids[d.item()] = inv_concepts[mask].mean(0)
        
    #     consistency_scores = []
    #     for d1 in digit_centroids:
    #         for d2 in digit_centroids:
    #             if d1 < d2:
    #                 sim = F.cosine_similarity(
    #                     digit_centroids[d1],
    #                     digit_centroids[d2],
    #                     dim=0
    #                 )
    #                 consistency_scores.append(1 - sim.item())  # Lower similarity is better
    #     concept_metrics['concept_consistency'] = np.mean(consistency_scores)
    
    # return concept_metrics
 

def train_epoch(model, dataloader, optimizer, config, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_concept_loss = 0
    total_domain_loss = 0
    correct_digits = 0
    correct_domains = 0
    total_samples = 0
    
    # Metrics for different components
    losses = defaultdict(float)
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch} Training"):
        optimizer.zero_grad()
        
        # Move data to GPU
        images = batch['image'].to(config.training.device)
        digits = batch['digit'].to(config.training.device)
        angles = batch['angle'].to(config.training.device)
        inv_concepts = batch['invariant_concepts'].to(config.training.device)
        var_concepts = batch['variant_concepts'].to(config.training.device)
        
        # Forward pass
        outputs = model(images,digit=digits,angle=angles)
        
        # Calculate losses
        recon_loss = F.mse_loss(outputs['recon'], images)
        digit_loss = F.cross_entropy(outputs['digit_pred'], digits)
        domain_loss = F.cross_entropy(outputs['domain_pred'], angles)
        
        # Concept losses
        inv_concept_loss = F.mse_loss(
            outputs['invariant_concepts'],
            inv_concepts
        )
        var_concept_loss = F.mse_loss(
            outputs['variant_concepts'],
            var_concepts
        )

        # KL divergence for both latent spaces
        kl_y = -0.5 * torch.sum(
            1 + outputs['zy_logvar'] - outputs['zy_mean'].pow(2) - outputs['zy_logvar'].exp()
        )
        kl_d = -0.5 * torch.sum(
            1 + outputs['zd_logvar'] - outputs['zd_mean'].pow(2) - outputs['zd_logvar'].exp()
        )
        kl_loss = kl_y + kl_d

        
        # # KL divergence
        # kl_loss = -0.5 * torch.sum(
        #     1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
        # )
        
        # Total loss
        loss = (recon_loss + 
                digit_loss + 
                config.training.domain_weight * domain_loss +
                config.training.concept_weight * (inv_concept_loss + var_concept_loss) +
                config.training.kl_weight * kl_loss)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        pred = outputs['digit_pred'].argmax(dim=1)
        correct_digits += pred.eq(digits).sum().item()
        correct_domains += pred.eq(angles).sum().item()
        total_samples += digits.size(0)
        
        # Store component losses
        losses['recon'] += recon_loss.item()
        losses['digit'] += digit_loss.item()
        losses['domain'] += domain_loss.item()
        losses['concept'] += (inv_concept_loss.item() + var_concept_loss.item())
        losses['kl'] += kl_loss.item()
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    accuracy_digits = 100. * correct_digits / total_samples
    accuracy_domains = 100. * correct_domains / total_samples
    
    return {
        'train_loss': avg_loss,
        'train_acc_digits': accuracy_digits,
        'train_acc_domains': accuracy_domains,
        'train_recon_loss': losses['recon'] / len(dataloader),
        'train_digit_loss': losses['digit'] / len(dataloader),
        'train_domain_loss': losses['domain'] / len(dataloader),
        'train_concept_loss': losses['concept'] / len(dataloader),
        'train_kl_loss': losses['kl'] / len(dataloader)
    }

def evaluate_generalization(model, dataloader,config):
    """Evaluate domain generalization performance"""
    model.eval()
    results = defaultdict(lambda: defaultdict(float))
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(config.training.device)
            digits = batch['digit'].to(config.training.device)
            angles = batch['angle'].to(config.training.device)
            
            outputs = model(images,category=digits,domain=angles)
            pred = outputs['digit_pred'].argmax(dim=1)
            
            # Per-domain accuracy
            for angle in torch.unique(angles):
                mask = angles == angle
                if mask.any():
                    correct = pred[mask].eq(digits[mask]).sum().item()
                    total = mask.sum().item()
                    results[angle.item()]['correct'] += correct
                    results[angle.item()]['total'] += total
    
    # Calculate metrics
    domain_accs = {}
    for angle in results:
        acc = 100. * results[angle]['correct'] / results[angle]['total']
        domain_accs[f'acc_{angle}deg'] = acc
    
    # Overall accuracy
    total_correct = sum(results[a]['correct'] for a in results)
    total_samples = sum(results[a]['total'] for a in results)
    overall_acc = 100. * total_correct / total_samples
    
    # Domain generalization gap
    accs = list(domain_accs.values())
    gen_gap = max(accs) - min(accs)
    
    return {
        'test_acc': overall_acc,
        'domain_gen_acc': sum(accs) / len(accs),
        'domain_gen_gap': gen_gap,
        **domain_accs
    }

def visualize_concepts(model, dataset, config,num_samples=10):
    """Visualize concept learning and interpretability"""
    model.eval()
    fig = plt.figure(figsize=(20, 10))
    
    # Create subplots grid
    gs = gridspec.GridSpec(3, num_samples)
    
    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(config.training.device)
            
            # Get model outputs
            outputs = model(image)
            
            # Original image
            ax1 = plt.subplot(gs[0, i])
            ax1.imshow(sample['image'].squeeze().cpu(), cmap='gray')
            ax1.set_title(f"Digit: {sample['digit']}\nAngle: {sample['angle']}°")
            ax1.axis('off')
            
            # Reconstruction
            ax2 = plt.subplot(gs[1, i])
            ax2.imshow(outputs['recon'].squeeze().cpu(), cmap='gray')
            ax2.set_title('Reconstruction')
            ax2.axis('off')
            
            # Concept visualization
            ax3 = plt.subplot(gs[2, i])
            inv_concepts = outputs['invariant_concepts'][0]
            var_concepts = outputs['variant_concepts'][0]
            concept_vis = torch.cat([inv_concepts, var_concepts]).cpu().numpy()
            im = ax3.imshow(concept_vis.reshape(-1, 1), aspect='auto', cmap='viridis')
            ax3.set_title('Concepts\nInv | Var')
            ax3.axis('off')
    
    plt.tight_layout()
    return fig

def visualize_results(model, val_loader, config, num_samples=10):
    """Visualize model results after training
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        config: Configuration object
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Create directory for visualizations
    vis_dir = Path(config.training.checkpoint_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # Get a batch of data
        batch = next(iter(val_loader))
        images = batch['image'][:num_samples].to(config.training.device)
        digits = batch['digit'][:num_samples]
        angles = batch['angle'][:num_samples]
        
        # Get model outputs
        outputs = model(images,category=digits,domain=angles)
        
        # Create figure
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(4, num_samples, height_ratios=[1, 1, 0.5, 0.5])
        
        for i in range(num_samples):
            # Original image
            ax1 = plt.subplot(gs[0, i])
            ax1.imshow(images[i].cpu().squeeze(), cmap='gray')
            ax1.set_title(f'Original\nDigit: {digits[i]}\nAngle: {angles[i]}°')
            ax1.axis('off')
            
            # Reconstruction
            ax2 = plt.subplot(gs[1, i])
            ax2.imshow(outputs['recon'][i].cpu().squeeze(), cmap='gray')
            pred_digit = outputs['digit_pred'][i].argmax().item()
            pred_angle = outputs['domain_pred'][i].argmax().item()
            ax2.set_title(f'Reconstructed\nPred: {pred_digit}\nAngle: {pred_angle}°')
            ax2.axis('off')
            
            # Invariant concepts
            ax3 = plt.subplot(gs[2, i])
            inv_concepts = outputs['invariant_concepts'][i].cpu()
            im3 = ax3.imshow(inv_concepts.unsqueeze(0), aspect='auto', cmap='viridis')
            ax3.set_title('Invariant Concepts')
            ax3.axis('off')
            
            # Variant concepts
            ax4 = plt.subplot(gs[3, i])
            var_concepts = outputs['variant_concepts'][i].cpu()
            im4 = ax4.imshow(var_concepts.unsqueeze(0), aspect='auto', cmap='viridis')
            ax4.set_title('Variant Concepts')
            ax4.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(vis_dir / 'model_results.png', dpi=300, bbox_inches='tight')
        wandb.log({"final_visualization": wandb.Image(fig)})
        plt.close()
        
        # Create concept similarity matrix
        plt.figure(figsize=(10, 8))
        sim_matrix = torch.cosine_similarity(
            outputs['invariant_concepts'].unsqueeze(1),
            outputs['variant_concepts'].unsqueeze(0),
            dim=2
        ).cpu()
        plt.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.title('Concept Similarity Matrix')
        plt.xlabel('Variant Concepts')
        plt.ylabel('Invariant Concepts')
        plt.savefig(vis_dir / 'concept_similarity.png', dpi=300, bbox_inches='tight')
        wandb.log({"concept_similarity": wandb.Image(plt.gcf())})
        plt.close()
        
        return vis_dir

def load_best_model(config):
    """Load the best model from checkpoint
    
    Args:
        config: Configuration object
        
    Returns:
        model: Best model
    """
    # Initialize model architecture
    model = ConceptVAE(
        input_dim=config.model.input_dim,
        zy_dim=config.model.zy_dim,
        zd_dim=config.model.zd_dim,
        concept_dim=config.model.concept_dim,
        n_classes=config.model.n_classes,
        n_domains=config.model.n_domains
    ).to(config.device)
    
    # Load best model weights
    checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    return model

# # Add this to your main function after training or use separately
# def visualize_trained_model(config_path=None):
#     """Visualize results of a trained model
    
#     Args:
#         config_path: Optional path to config file
#     """
#     # Load config
#     config = get_config()
#     if config_path:
#         # Load specific config if provided
#         pass
    
#     # Get validation dataset
#     val_dataset = get_dataset(train=False)
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.training.batch_size,
#         shuffle=False,
#         num_workers=config.training.num_workers,
#         collate_fn=custom_collate,
#         pin_memory=config.training.device != 'cpu'
#     )
    
#     # Load best model
#     model = load_best_model(config)
#     model.eval()
    
#     # Create visualizations
#     vis_dir = visualize_results(model, val_loader, config)
#     print(f"\nVisualizations saved to {vis_dir}")
    
#     # Optional: Run additional analysis
#     analyze_concepts(model, val_loader, config)

if __name__ == "__main__":
    main()
