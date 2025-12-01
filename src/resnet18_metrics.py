import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from PIL import Image
import os

class ModelEvaluator:
    def __init__(self, model_path, num_classes=2, class_names=None):
        """
        Initialize the evaluator with a trained model
        
        Args:
            model_path: Path to saved model weights
            num_classes: Number of classes
            class_names: List of class names (e.g., ['algae', 'microplastics'])
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f'class_{i}' for i in range(num_classes)]
        
        # Load model
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def count_parameters(self):
        """Count model parameters in millions"""
        return sum(p.numel() for p in self.model.parameters()) / 1e6
    
    def estimate_flops(self):
        """Estimate FLOPs for ResNet18 (approximate)"""
        # ResNet18 FLOPs for 224x224 input ≈ 1.814 GFLOPs
        return 1.814
    
    def measure_inference_time(self, num_iterations=100):
        """Measure average inference time per image"""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
        
        avg_time_ms = (time.time() - start_time) / num_iterations * 1000
        return avg_time_ms
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test dataset
        
        Args:
            test_loader: PyTorch DataLoader for test data
            
        Returns:
            Dictionary containing all metrics and results
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, 
                                       target_names=self.class_names,
                                       output_dict=True)
        
        # Extract per-class metrics
        precision_per_class = [report[name]['precision'] for name in self.class_names]
        recall_per_class = [report[name]['recall'] for name in self.class_names]
        f1_per_class = [report[name]['f1-score'] for name in self.class_names]
        support_per_class = [report[name]['support'] for name in self.class_names]
        
        # Efficiency metrics
        params_million = self.count_parameters()
        flops_giga = self.estimate_flops()
        inference_time_ms = self.measure_inference_time()
        
        # Compile results
        results = {
            "accuracy": accuracy,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
            "support_per_class": support_per_class,
            "classification_report": report,
            "efficiency": {
                "params_million": params_million,
                "flops_giga": flops_giga,
                "inference_time_ms_per_image": inference_time_ms
            },
            "confusion_matrix": cm.tolist()
        }
        
        return results, all_preds, all_labels, all_probs
    
    def visualize_results(self, results, save_dir='evaluation_results'):
        """Create confusion matrix visualization"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Confusion Matrix only
        plt.figure(figsize=(8, 6))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300)
        plt.close()
        
        print(f"Confusion matrix saved to {save_dir}/")
    
    def visualize_predictions(self, test_loader, save_dir='evaluation_results'):
        """Save individual prediction images to separate folders"""
        correct_dir = os.path.join(save_dir, 'correct_predictions')
        incorrect_dir = os.path.join(save_dir, 'incorrect_predictions')
        os.makedirs(correct_dir, exist_ok=True)
        os.makedirs(incorrect_dir, exist_ok=True)
        
        self.model.eval()
        correct_count = 0
        incorrect_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images_dev = images.to(self.device)
                outputs = self.model(images_dev)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                for i in range(len(images)):
                    pred = preds[i].item()
                    label = labels[i].item()
                    prob = probs[i][pred].item()
                    
                    # Denormalize image
                    img = images[i].permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    
                    # Create figure for individual image
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(img)
                    ax.axis('off')
                    
                    if pred == label:
                        # Correct prediction
                        title = f"Predicted: {self.class_names[pred]}\nTrue: {self.class_names[label]}"
                        ax.set_title(title, color='green', fontsize=12, fontweight='bold', pad=20)
                        
                        filename = f"correct_{correct_count:04d}_{self.class_names[label]}_conf{prob:.2f}.png"
                        filepath = os.path.join(correct_dir, filename)
                        correct_count += 1
                    else:
                        # Incorrect prediction
                        title = f"Predicted: {self.class_names[pred]}\nTrue: {self.class_names[label]}"
                        ax.set_title(title, color='red', fontsize=12, fontweight='bold', pad=20)
                        
                        filename = f"incorrect_{incorrect_count:04d}_pred{self.class_names[pred]}_true{self.class_names[label]}_conf{prob:.2f}.png"
                        filepath = os.path.join(incorrect_dir, filename)
                        incorrect_count += 1
                    
                    plt.tight_layout()
                    plt.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close()
        
        print(f"\nSaved {correct_count} correct predictions to {correct_dir}/")
        print(f"Saved {incorrect_count} incorrect predictions to {incorrect_dir}/")



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path='resnet18_best.pth',  # Change this to your model path
        num_classes=2,
        class_names=['algae', 'microplastics']
    )
    
    # Load test data
    test_dataset = ImageFolder(r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\data\test", transform=evaluator.transform)  # Change this path
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    results, preds, labels, probs = evaluator.evaluate(test_loader)
    
    # Save results as JSON
    with open('metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results to console
    print(json.dumps(results, indent=2))
    
    # Create visualizations
    evaluator.visualize_results(results)
    evaluator.visualize_predictions(test_loader)