
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_config,get_kl_weight
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms.functional as TF
from collections import defaultdict
from tqdm import tqdm

config = get_config()


class ConceptExplainer:
    def __init__(self, config):
        self.config = config
        
        # Templates for explanations
        self.digit_templates = {
            'high_conf': "This is clearly a {digit} because {reason}",
            'medium_conf': "This appears to be a {digit}, as {reason}",
            'low_conf': "This might be a {digit}, but {uncertainty}"
        }
        
        self.rotation_templates = {
            'high_conf': "The rotation is {angle}° because {reason}",
            'medium_conf': "The rotation appears to be {angle}°, as {reason}",
            'low_conf': "The rotation might be {angle}°, but {uncertainty}"
        }
        
        # Concept interpretation mappings
        self.digit_concepts = {
            0: {"primary": "closed loop", "key_features": ["continuous curve", "hollow center"]},
            1: {"primary": "vertical line", "key_features": ["single stroke", "upright stance"]},
            2: {"primary": "curved hook with base", "key_features": ["top curve", "diagonal sweep"]},
            3: {"primary": "double curve", "key_features": ["connected loops", "rightward curves"]},
            4: {"primary": "crossed lines", "key_features": ["vertical stem", "horizontal cross"]},
            5: {"primary": "top bar with hook", "key_features": ["horizontal top", "curved base"]},
            6: {"primary": "curved loop", "key_features": ["top hook", "bottom circle"]},
            7: {"primary": "angled line", "key_features": ["top bar", "diagonal stroke"]},
            8: {"primary": "double loop", "key_features": ["stacked circles", "center crossing"]},
            9: {"primary": "loop with tail", "key_features": ["top circle", "descending line"]}
        }
        
        self.rotation_concepts = {
            0: {"primary": "upright", "key_features": ["standard orientation", "vertical alignment"]},
            90: {"primary": "right-tilted", "key_features": ["horizontal flip", "rightward tilt"]},
            180: {"primary": "inverted", "key_features": ["upside down", "complete flip"]},
            270: {"primary": "left-tilted", "key_features": ["horizontal flip", "leftward tilt"]}
        }
    
    def explain_prediction(self, outputs, batch, batch_idx=None):
        """Generate explanation for model predictions"""
        # Get predictions and confidences
        digit_probs = torch.softmax(outputs['digit_pred'], dim=1)
        domain_probs = torch.softmax(outputs['domain_pred'], dim=1)
        
        # Get actual labels
        true_digits = batch['digit']
        true_angles = batch['angle']
        
        # Process each sample in the batch
        explanations = []
        for idx in range(len(true_digits)):
            # Get predicted and true labels
            pred_digit = outputs['digit_pred'][idx].argmax().item()
            pred_angle = outputs['domain_pred'][idx].argmax().item() * 90
            true_digit = true_digits[idx].item()
            true_angle = true_angles[idx].item() * 90
            
            # Get confidences
            digit_conf = digit_probs[idx, pred_digit].item()
            angle_conf = domain_probs[idx, pred_angle//90].item()
            
            # Get concepts
            inv_concepts = outputs['invariant_concepts'][idx]
            var_concepts = outputs['variant_concepts'][idx]
            
            # Generate explanation
            explanation = {
                'sample_idx': batch_idx * len(true_digits) + idx if batch_idx is not None else idx,
                'prediction': {
                    'digit': pred_digit,
                    'angle': pred_angle,
                    'digit_confidence': digit_conf,
                    'angle_confidence': angle_conf
                },
                'true_labels': {
                    'digit': true_digit,
                    'angle': true_angle
                },
                'explanation': self._generate_explanation(
                    pred_digit, pred_angle, digit_conf, angle_conf,
                    inv_concepts, var_concepts
                ),
                'concepts': {
                    'invariant': inv_concepts.cpu().tolist(),
                    'variant': var_concepts.cpu().tolist()
                }
            }
            
            # Add correctness
            explanation['correct'] = {
                'digit': pred_digit == true_digit,
                'angle': pred_angle == true_angle
            }
            
            explanations.append(explanation)
        
        return explanations
    
    def _generate_explanation(self, digit, angle, digit_conf, angle_conf, inv_concepts, var_concepts):
        """Generate human-readable explanation based on concepts"""
        try:
            # Get concept info
            digit_info = self.digit_concepts[digit]
            angle_info = self.rotation_concepts[angle]
        
            # Generate digit explanation based on confidence
            if digit_conf > 0.9:
               digit_template = self.digit_templates['high_conf']
               digit_reason = f"it shows a {digit_info['primary']} with {' and '.join(digit_info['key_features'])}"
               digit_explanation = digit_template.format(digit=digit, reason=digit_reason)
            elif digit_conf > 0.7:
                 digit_template = self.digit_templates['medium_conf']
                 digit_reason = f"it has a {digit_info['primary']} structure"
                 digit_explanation = digit_template.format(digit=digit, reason=digit_reason)
            else:
                digit_template = self.digit_templates['low_conf']
                digit_uncertainty = "the features are not clearly distinguishable"
                digit_explanation = digit_template.format(digit=digit, uncertainty=digit_uncertainty)
        
            # Generate rotation explanation based on confidence
            if angle_conf > 0.9:
               angle_template = self.rotation_templates['high_conf']
               angle_reason = f"it shows {angle_info['primary']} features with {' and '.join(angle_info['key_features'])}"
               angle_explanation = angle_template.format(angle=angle, reason=angle_reason)
            elif angle_conf > 0.7:
                 angle_template = self.rotation_templates['medium_conf']
                 angle_reason = f"it shows {angle_info['primary']} characteristics"
                 angle_explanation = angle_template.format(angle=angle, reason=angle_reason)
            else:
                angle_template = self.rotation_templates['low_conf']
                angle_uncertainty = "the orientation is ambiguous"
                angle_explanation = angle_template.format(angle=angle, uncertainty=angle_uncertainty)
        
            return {
                  'digit': digit_explanation,
                  'angle': angle_explanation,
                  'confidence': {
                  'digit': digit_conf,
                   'angle': angle_conf
                    }
            }
        
        except Exception as e:
               print(f"Error generating explanation for digit {digit}, angle {angle}: {str(e)}")
               return {
                      'digit': f"Unable to explain digit {digit}",
                      'angle': f"Unable to explain angle {angle}",
                      'confidence': {
                      'digit': 0.0,
                      'angle': 0.0
                       }
                       }
          
def validate_with_explanations(model, val_loader, config, explainer,epoch):
    """Validation with detailed explanations"""
    model.eval()
    all_explanations = []
    metrics = defaultdict(float)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
            # Move data to device
            images = batch['image'].to(config.training.device)
            digits = batch['digit'].to(config.training.device)
            angles = batch['angle'].to(config.training.device)
            
            # Forward pass
            outputs = model(images)
            
            # Get explanations
            batch_explanations = explainer.explain_prediction(outputs, batch, batch_idx)
            all_explanations.extend(batch_explanations)
            
            # Compute metrics
            loss, batch_metrics = compute_loss(outputs, images, digits, angles, config,epoch)
            
            # Update metrics
            for k, v in batch_metrics.items():
                metrics[k] += v
    
    # Average metrics
    metrics = {k: v / len(val_loader) for k, v in metrics.items()}
    
    return metrics, all_explanations


class ConceptVAE(nn.Module):
    def __init__(self, config, input_dim, zy_dim, zd_dim, concept_dim, n_classes, n_domains):
        super(ConceptVAE, self).__init__()
        
        # Configuration
        self.input_channels = config.model.input_channels
        self.img_size = config.model.img_size
        self.zy_dim = zy_dim  # invariant dimension
        self.zd_dim = zd_dim  # variant dimension
        self.concept_dim = concept_dim
        
        # Domain-invariant encoder
        self.invariant_encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Domain-variant encoder
        self.variant_encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened dimension
        self.flat_dim = 64 * (self.img_size // 4) * (self.img_size // 4)
        
        # Latent projections
        self.fc_zy = nn.Linear(self.flat_dim, zy_dim * 2)  # invariant: mean and logvar
        self.fc_zd = nn.Linear(self.flat_dim, zd_dim * 2)  # variant: mean and logvar
        
        # Concept projections
        self.concept_proj_y = nn.Sequential(
            nn.Linear(zy_dim, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, concept_dim)
        )
        
        self.concept_proj_d = nn.Sequential(
            nn.Linear(zd_dim, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, concept_dim)
        )
        
        # Classifier heads
        self.digit_classifier = nn.Linear(concept_dim, n_classes)
        self.domain_classifier = nn.Linear(concept_dim, n_domains)
        
        # Decoder
        self.decoder_input = nn.Linear(zy_dim + zd_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, self.img_size // 4, self.img_size // 4)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.input_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode_invariant(self, x):
        """Encode domain-invariant features"""
        h = self.invariant_encoder(x)
        params = self.fc_zy(h)
        mu, logvar = torch.chunk(params, 2, dim=1)
        return mu, logvar
    
    def encode_variant(self, x):
        """Encode domain-variant features"""
        h = self.variant_encoder(x)
        params = self.fc_zd(h)
        mu, logvar = torch.chunk(params, 2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, zy, zd):
        """Decode from both latent spaces"""
        z = torch.cat([zy, zd], dim=1)
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def get_concepts(self, zy, zd):
        """Extract concepts from latent representations"""
        invariant_concepts = self.concept_proj_y(zy)
        variant_concepts = self.concept_proj_d(zd)
        return invariant_concepts, variant_concepts
    
    def forward(self, x):
        # Encode
        zy_mu, zy_logvar = self.encode_invariant(x)
        zd_mu, zd_logvar = self.encode_variant(x)
        
        # Sample latent vectors
        zy = self.reparameterize(zy_mu, zy_logvar)
        zd = self.reparameterize(zd_mu, zd_logvar)
        
        # Get concepts
        invariant_concepts, variant_concepts = self.get_concepts(zy, zd)
        
        # Get predictions
        digit_pred = self.digit_classifier(invariant_concepts)
        domain_pred = self.domain_classifier(variant_concepts)
        
        # Reconstruction
        recon = self.decode(zy, zd)
        
        return {
            'recon': recon,
            'zy_mu': zy_mu,
            'zy_logvar': zy_logvar,
            'zd_mu': zd_mu,
            'zd_logvar': zd_logvar,
            'invariant_concepts': invariant_concepts,
            'variant_concepts': variant_concepts,
            'digit_pred': digit_pred,
            'domain_pred': domain_pred
        }

def compute_loss(outputs, x, digit_labels, domain_labels, config, epoch):
    """Compute balanced loss components"""
    # Reconstruction loss (with normalized MSE)
    recon_loss = F.mse_loss(outputs['recon'], x, reduction='mean')
    
    # KL divergence with annealing
    kl_y = -0.5 * torch.mean(1 + outputs['zy_logvar'] - outputs['zy_mu'].pow(2) - outputs['zy_logvar'].exp())
    kl_d = -0.5 * torch.mean(1 + outputs['zd_logvar'] - outputs['zd_mu'].pow(2) - outputs['zd_logvar'].exp())
    kl_loss = kl_y + kl_d
    
    # Classification losses with label smoothing
    digit_loss = F.cross_entropy(outputs['digit_pred'], digit_labels, label_smoothing=0.1)
    domain_loss = F.cross_entropy(outputs['domain_pred'], domain_labels, label_smoothing=0.1)
    
    # Concept separation loss with cosine similarity
    concept_similarity = F.cosine_similarity(
        F.normalize(outputs['invariant_concepts'], dim=-1),
        F.normalize(outputs['variant_concepts'], dim=-1)
    ).abs().mean()
    concept_separation_loss = 1.0 - concept_similarity
    
    # Get annealed KL weight
    kl_weight = get_kl_weight(config, epoch)
    
    # Combined loss with balanced weights
    total_loss = (
        config.training.recon_weight * recon_loss +
        kl_weight * kl_loss +
        config.training.digit_weight * digit_loss +
        config.training.domain_weight * domain_loss +
        config.training.concept_sep_weight * concept_separation_loss
    )
    
    # Calculate accuracies
    with torch.no_grad():
        digit_acc = (outputs['digit_pred'].argmax(dim=1) == digit_labels).float().mean()
        domain_acc = (outputs['domain_pred'].argmax(dim=1) == domain_labels).float().mean()
    
    return total_loss, {
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'digit_loss': digit_loss.item(),
        'domain_loss': domain_loss.item(),
        'concept_separation_loss': concept_separation_loss.item(),
        'total_loss': total_loss.item(),
        'digit_acc': digit_acc.item(),
        'domain_acc': domain_acc.item(),
        'kl_weight': kl_weight
    }

        
