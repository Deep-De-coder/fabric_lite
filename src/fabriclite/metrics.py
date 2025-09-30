"""Evaluation metrics and visualization utilities."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score."""
    return accuracy_score(y_true, y_pred)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate macro-averaged F1 score."""
    return f1_score(y_true, y_pred, average='macro')


def confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    out_png: str,
    figsize: tuple = (10, 8),
    normalize: bool = True
) -> None:
    """
    Create and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        out_png: Output path for PNG file
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {out_png}")


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str]
) -> dict:
    """
    Generate classification report as dictionary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
    
    Returns:
        Classification report as dictionary
    """
    report = classification_report(
        y_true, y_pred, 
        target_names=labels, 
        output_dict=True
    )
    
    return report


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str]
) -> dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy(y_true, y_pred),
        'macro_f1': macro_f1(y_true, y_pred),
        'classification_report': classification_report_dict(y_true, y_pred, labels)
    }
    
    return metrics


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    out_png: str,
    figsize: tuple = (12, 5)
) -> None:
    """
    Plot training history.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch
        val_accs: Validation accuracies per epoch
        out_png: Output path for PNG file
        figsize: Figure size
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {out_png}")


def plot_class_distribution(
    labels: List[str],
    counts: List[int],
    out_png: str,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot class distribution.
    
    Args:
        labels: Class labels
        counts: Class counts
        out_png: Output path for PNG file
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    bars = plt.bar(labels, counts)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.title('Class Distribution')
    plt.xlabel('Fabric Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Class distribution plot saved to: {out_png}")
