import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, average_precision_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from ood_expts import ConceptBasedOODDetector
from config import get_config
from generalization_evaluation import GeneralizationAnalyzer
from rotated_mnist import RotatedMNISTv3
from torch.utils.data import DataLoader

def get_analyzer():
    config = get_config()
    analyzer = GeneralizationAnalyzer(model_path='new_checkpoints/best_model.pt', config=config)
    return analyzer,config
    

def analyze_model_performance(results, true_digits, split_name, angle):
    """Comprehensive analysis of model performance for a specific angle"""
    # Concatenate predictions and true labels
    all_preds = torch.cat([r['predictions'] for r in results]).cpu().numpy()
    all_conf = torch.cat([r['confidences'] for r in results]).cpu().numpy()
    all_true = true_digits.cpu().numpy()
    
    # Basic metrics
    accuracy = accuracy_score(all_true, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true, all_preds, average=None, labels=range(10)
    )
    conf_matrix = confusion_matrix(all_true, all_preds, labels=range(10))
    
    # Per-class metrics
    per_class_metrics = {}
    for digit in range(10):
        digit_mask = (all_true == digit)
        if digit_mask.sum() > 0:
            per_class_metrics[digit] = {
                'accuracy': accuracy_score(all_true[digit_mask], all_preds[digit_mask]),
                'precision': precision[digit],
                'recall': recall[digit],
                'f1': f1[digit],
                'support': support[digit],
                'confidence': all_conf[digit_mask].mean(),
                'confidence_std': all_conf[digit_mask].std(),
                'misclassified_as': Counter(all_preds[digit_mask][all_preds[digit_mask] != digit])
            }
    
    # Concept similarity analysis
    all_inv_sims = torch.cat([r['concept_similarities']['invariant'] 
                             for r in results]).cpu().numpy()
    all_var_sims = torch.cat([r['concept_similarities']['variant'] 
                             for r in results]).cpu().numpy()
    
    concept_metrics = {
        'invariant_mean': all_inv_sims.mean(),
        'invariant_std': all_inv_sims.std(),
        'variant_mean': all_var_sims.mean(),
        'variant_std': all_var_sims.std(),
    }
    
    # Per-class concept analysis
    per_class_concepts = {}
    for digit in range(10):
        digit_mask = (all_true == digit)
        if digit_mask.sum() > 0:
            per_class_concepts[digit] = {
                'invariant_mean': all_inv_sims[digit_mask].mean(),
                'invariant_std': all_inv_sims[digit_mask].std(),
                'variant_mean': all_var_sims[digit_mask].mean(),
                'variant_std': all_var_sims[digit_mask].std(),
            }
    
    # Additional analysis: Confidence calibration
    confidence_bins = np.linspace(0, 1, 11)
    calibration_stats = {
        'bin_accuracies': [],
        'bin_confidences': [],
        'bin_counts': []
    }
    
    for i in range(len(confidence_bins) - 1):
        bin_mask = (all_conf >= confidence_bins[i]) & (all_conf < confidence_bins[i + 1])
        if bin_mask.sum() > 0:
            bin_acc = accuracy_score(all_true[bin_mask], all_preds[bin_mask])
            bin_conf = all_conf[bin_mask].mean()
            calibration_stats['bin_accuracies'].append(bin_acc)
            calibration_stats['bin_confidences'].append(bin_conf)
            calibration_stats['bin_counts'].append(bin_mask.sum())
    
    return {
        'overall_accuracy': accuracy,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': conf_matrix,
        'concept_metrics': concept_metrics,
        'per_class_concepts': per_class_concepts,
        'calibration_stats': calibration_stats
    }

def plot_confusion_matrix(conf_matrix, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def plot_concept_similarities(per_class_concepts, save_path):
    digits = list(per_class_concepts.keys())
    inv_means = [per_class_concepts[d]['invariant_mean'] for d in digits]
    var_means = [per_class_concepts[d]['variant_mean'] for d in digits]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(digits))
    width = 0.35
    
    plt.bar(x - width/2, inv_means, width, label='Invariant')
    plt.bar(x + width/2, var_means, width, label='Variant')
    
    plt.xlabel('Digit')
    plt.ylabel('Similarity')
    plt.title('Concept Similarities by Digit')
    plt.xticks(x, digits)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curve(calibration_stats, save_path):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.scatter(
        calibration_stats['bin_confidences'],
        calibration_stats['bin_accuracies'],
        s=[c/50 for c in calibration_stats['bin_counts']],  # Size proportional to count
        alpha=0.6
    )
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def analyze_failure_cases(results, true_digits, metrics, n_samples=5):
    """Analyze specific failure cases"""
    all_preds = torch.cat([r['predictions'] for r in results]).cpu()
    all_conf = torch.cat([r['confidences'] for r in results]).cpu()
    all_true = true_digits.cpu()
    
    failure_cases = {
        'high_confidence_mistakes': [],
        'low_confidence_correct': [],
        'systematic_errors': defaultdict(list)
    }
    
    # Find high confidence mistakes
    mistakes = (all_preds != all_true)
    if mistakes.sum() > 0:
        mistake_conf = all_conf[mistakes]
        high_conf_mistakes = mistake_conf.argsort(descending=True)[:n_samples]
        for idx in high_conf_mistakes:
            failure_cases['high_confidence_mistakes'].append({
                'true': all_true[idx].item(),
                'predicted': all_preds[idx].item(),
                'confidence': all_conf[idx].item()
            })
    
    # Find systematic errors (common misclassification patterns)
    for true_digit in range(10):
        digit_mask = (all_true == true_digit)
        if digit_mask.sum() > 0:
            wrong_preds = all_preds[digit_mask][all_preds[digit_mask] != true_digit]
            if len(wrong_preds) > 0:
                common_mistakes = Counter(wrong_preds.tolist()).most_common(3)
                failure_cases['systematic_errors'][true_digit] = common_mistakes
    
    return failure_cases

def collect_angle_results(detector, test_loader, angle):
    """Collect results for a specific angle"""
    results = []
    true_digits = []
    
    for batch in tqdm(test_loader, desc=f"Angle {angle}°"):
        result = detector.detect_ood_angle(batch, angle)
        results.append(result)
        true_digits.append(batch['digit'])
    
    # Concatenate results
    all_preds = torch.cat([r['predictions'] for r in results]).cpu().numpy()
    all_conf = torch.cat([r['confidences'] for r in results]).cpu().numpy()
    all_true = torch.cat(true_digits).cpu().numpy()
    all_inv_sims = torch.cat([r['concept_similarities']['invariant'] for r in results]).cpu().numpy()
    all_var_sims = torch.cat([r['concept_similarities']['variant'] for r in results]).cpu().numpy()
    
    # Calculate metrics
    angle_results = {
        'accuracy': (all_preds == all_true).mean(),
        'confidence': all_conf.mean(),
        'invariant_sim': all_inv_sims.mean(),
        'variant_sim': all_var_sims.mean(),
        'confusion_matrix': confusion_matrix(all_true, all_preds),
        'per_digit': {}
    }
    
    # Per-digit metrics
    for digit in range(10):
        digit_mask = (all_true == digit)
        if digit_mask.sum() > 0:
            angle_results['per_digit'][digit] = {
                'accuracy': (all_preds[digit_mask] == all_true[digit_mask]).mean(),
                'confidence': all_conf[digit_mask].mean(),
                'fpr': (all_preds[digit_mask] != all_true[digit_mask]).mean(),
                'invariant_sim': all_inv_sims[digit_mask].mean(),
                'variant_sim': all_var_sims[digit_mask].mean()
            }
    
    return angle_results


def main():
    # Setup
    save_dir = Path('detailed_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Get analyzer and detector
    analyzer,config = get_analyzer()
    config.training.device = torch.device('cpu')
    config.training.pin_memory = False
    config.training.num_workers = 0  # Disable multiprocessing
    
    detector = ConceptBasedOODDetector(analyzer)
     # Create test dataset
    test_dataset = RotatedMNISTv3(
        config=config,
        train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    trained_angles = [0, 90, 180, 270]
    ood_angles = [45, 135, 225]
    
    # Initialize results storage
    results = {
        'trained': defaultdict(list),
        'ood': defaultdict(list)
    }
    true_digits = {
        'trained': {},
        'ood': {}
    }
    
    # Collect results
    print("\nCollecting results...")
    for split, angles in [('trained', trained_angles), ('ood', ood_angles)]:
        print(f"\nAnalyzing {split} angles...")
        for angle in tqdm(angles):
            angle_true_digits = []
            for batch in test_loader:
                result = detector.detect_ood_angle(batch, angle)
                results[split][angle].append(result)
                angle_true_digits.append(batch['digit'])
            true_digits[split][angle] = torch.cat(angle_true_digits)
    
    # Analyze results
    analysis_results = {
        'trained': {},
        'ood': {}
    }
    
    failure_analyses = {
        'trained': {},
        'ood': {}
    }
    
    print("\nPerforming detailed analysis...")
    for split, angles in [('trained', trained_angles), ('ood', ood_angles)]:
        for angle in tqdm(angles):
            # Performance metrics
            analysis_results[split][angle] = analyze_model_performance(
                results[split][angle],
                true_digits[split][angle],
                split,
                angle
            )
            
            # Failure analysis
            failure_analyses[split][angle] = analyze_failure_cases(
                results[split][angle],
                true_digits[split][angle],
                analysis_results[split][angle]
            )
    
    # Save results
    print("\nSaving analysis results...")
    with open(save_dir / 'detailed_metrics.txt', 'w') as f:
        f.write("Comprehensive Model Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        for split, angles in [('Trained', trained_angles), ('OOD', ood_angles)]:
            f.write(f"\n{split} Angles Analysis:\n")
            f.write("-" * 30 + "\n")
            
            for angle in angles:
                metrics = analysis_results[split.lower()][angle]
                failures = failure_analyses[split.lower()][angle]
                
                f.write(f"\nAngle: {angle}°\n")
                f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}\n\n")
                
                # Per-class metrics
                f.write("Per-Class Metrics:\n")
                for digit, class_metrics in metrics['per_class_metrics'].items():
                    f.write(f"\nDigit {digit}:\n")
                    f.write(f"- Accuracy: {class_metrics['accuracy']:.3f}\n")
                    f.write(f"- Precision: {class_metrics['precision']:.3f}\n")
                    f.write(f"- Recall: {class_metrics['recall']:.3f}\n")
                    f.write(f"- F1: {class_metrics['f1']:.3f}\n")
                    f.write(f"- Mean Confidence: {class_metrics['confidence']:.3f}\n")
                    
                    if digit in failures['systematic_errors']:
                        f.write("- Common misclassifications:\n")
                        for wrong_digit, count in failures['systematic_errors'][digit]:
                            f.write(f"  * Predicted as {wrong_digit}: {count} times\n")
                
                # Concept metrics
                f.write("\nConcept Metrics:\n")
                f.write(f"- Mean Invariant Similarity: {metrics['concept_metrics']['invariant_mean']:.3f}\n")
                f.write(f"- Mean Variant Similarity: {metrics['concept_metrics']['variant_mean']:.3f}\n")
                
                # High confidence mistakes
                f.write("\nHigh Confidence Mistakes:\n")
                for mistake in failures['high_confidence_mistakes']:
                    f.write(f"- True: {mistake['true']}, Predicted: {mistake['predicted']}, "
                           f"Confidence: {mistake['confidence']:.3f}\n")
                
                f.write("\n" + "=" * 50 + "\n")
                
                # Save visualizations
                plot_confusion_matrix(
                    metrics['confusion_matrix'],
                    save_dir / f'{split.lower()}_angle_{angle}_confusion.png'
                )
                plot_concept_similarities(
                    metrics['per_class_concepts'],
                    save_dir / f'{split.lower()}_angle_{angle}_concepts.png'
                )
                plot_calibration_curve(
                    metrics['calibration_stats'],
                    save_dir / f'{split.lower()}_angle_{angle}_calibration.png'
                )
    
    # Print summary
    print("\nSummary Statistics:")
    for split, angles in [('Trained', trained_angles), ('OOD', ood_angles)]:
        print(f"\n{split} Angles:")
        for angle in angles:
            metrics = analysis_results[split.lower()][angle]
            print(f"\nAngle {angle}°:")
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")
            print("Per-class Accuracy Range: "
                  f"{min(m['accuracy'] for m in metrics['per_class_metrics'].values()):.3f} - "
                  f"{max(m['accuracy'] for m in metrics['per_class_metrics'].values()):.3f}")
            print("Mean Confidence: "
                  f"{np.mean([m['confidence'] for m in metrics['per_class_metrics'].values()]):.3f}")
            print("Mean Invariant Similarity: "
                  f"{metrics['concept_metrics']['invariant_mean']:.3f}")
            print("Mean Variant Similarity: "
                  f"{metrics['concept_metrics']['variant_mean']:.3f}")

if __name__ == "__main__":
    main()