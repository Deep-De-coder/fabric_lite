"""Command-line interface for FabricLite."""

import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .classifier import FabricClassifier
from .constants import CSV_HEADER
from .data import FolderDataset, make_dataloaders
from .export import to_onnx, to_torchscript
from .formatting import (
    build_record,
    get_output_format,
    write_csv_row,
    write_json_array,
    write_jsonl,
)
from .metrics import compute_metrics, confusion_matrix_plot, plot_training_history
from .preprocess import preprocess
from .taxonomy import FABRIC_LABELS
from .utils import setup_logger, save_json, safe_create_dir

app = typer.Typer(help="FabricLite: A tiny fabric/material classifier for garments")
console = Console()
logger = setup_logger()


@app.command()
def infer(
    image: Path = typer.Argument(..., help="Path to image file"),
    topk: int = typer.Option(3, "--topk", "-k", help="Number of top predictions"),
    white_balance: bool = typer.Option(False, "--wb", help="Apply white balance correction"),
    weights: Optional[Path] = typer.Option(None, "--weights", "-w", help="Path to model weights"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
    csv_out: bool = typer.Option(False, "--csv", help="Output as CSV"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON (only with --json)")
):
    """Infer fabric type from a single image."""
    try:
        # Validate mutual exclusivity
        if json_out and csv_out:
            raise typer.BadParameter("Use either --json or --csv, not both.")
        
        if not image.exists():
            console.print(f"[red]Error: Image file {image} does not exist[/red]")
            raise typer.Exit(1)
        
        # Load classifier
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=json_out or csv_out  # Disable progress for structured output
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            
            if weights and weights.exists():
                # Load custom weights
                model = FabricClassifier.from_pretrained()
                model.model.load_state_dict(torch.load(weights, map_location="cpu"))
            else:
                model = FabricClassifier.from_pretrained()
            
            progress.update(task, description="Running inference...")
            
            # Get probabilities for structured output
            if json_out or csv_out:
                prob_dict = model.predict_proba(image, white_balance=white_balance)
                record = build_record(str(image), prob_dict, k=topk)
            else:
                # Use existing predict method for human-readable output
                result = model.predict(image, topk=topk, white_balance=white_balance)
        
        # Handle structured output
        if json_out:
            if output:
                with open(output, 'w') as f:
                    json.dump(record, f, indent=2 if pretty else None, separators=(',', ':') if not pretty else None)
            else:
                json.dump(record, sys.stdout, indent=2 if pretty else None, separators=(',', ':') if not pretty else None)
                sys.stdout.write('\n')
        
        elif csv_out:
            if output:
                with open(output, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADER)
                    write_csv_row(writer, record)
            else:
                writer = csv.writer(sys.stdout)
                writer.writerow(CSV_HEADER)
                write_csv_row(writer, record)
        
        else:
            # Default human-readable output
            table = Table(title="Fabric Classification Results")
            table.add_column("Fabric Type", style="cyan")
            table.add_column("Probability", style="magenta")
            
            for label, prob in result.items():
                table.add_row(label, f"{prob:.3f}")
            
            console.print(table)
            
            # Save to file if requested (legacy behavior)
            if output:
                save_json(result, output)
                console.print(f"[green]Results saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def batch(
    folder: Path = typer.Argument(..., help="Path to folder containing images"),
    topk: int = typer.Option(3, "--topk", "-k", help="Number of top predictions"),
    white_balance: bool = typer.Option(False, "--wb", help="Apply white balance correction"),
    weights: Optional[Path] = typer.Option(None, "--weights", "-w", help="Path to model weights"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
    csv_out: bool = typer.Option(False, "--csv", help="Output as CSV"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON (only with --json)")
):
    """Batch inference on multiple images."""
    try:
        # Validate mutual exclusivity
        if json_out and csv_out:
            raise typer.BadParameter("Use either --json or --csv, not both.")
        
        if not folder.exists() or not folder.is_dir():
            console.print(f"[red]Error: Folder {folder} does not exist or is not a directory[/red]")
            raise typer.Exit(1)
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            console.print(f"[red]Error: No image files found in {folder}[/red]")
            raise typer.Exit(1)
        
        # Load classifier
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=json_out or csv_out  # Disable progress for structured output
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            
            if weights and weights.exists():
                # Load custom weights
                model = FabricClassifier.from_pretrained()
                model.model.load_state_dict(torch.load(weights, map_location="cpu"))
            else:
                model = FabricClassifier.from_pretrained()
            
            progress.update(task, description="Running batch inference...")
            
            # Get structured results
            if json_out or csv_out:
                records = []
                for img_file in image_files:
                    try:
                        prob_dict = model.predict_proba(img_file, white_balance=white_balance)
                        record = build_record(str(img_file), prob_dict, k=topk)
                        records.append(record)
                    except Exception as e:
                        # Handle unreadable images
                        error_record = build_record(str(img_file), {}, k=topk)
                        error_record["error"] = str(e)
                        records.append(error_record)
            else:
                # Use existing predict_batch method for human-readable output
                results = model.predict_batch(image_files, topk=topk, white_balance=white_balance)
        
        # Handle structured output
        if json_out:
            if output:
                output_format = get_output_format(output)
                with open(output, 'w') as f:
                    if output_format == "json":
                        write_json_array(f, records, pretty=pretty)
                    else:  # jsonl
                        write_jsonl(f, records)
            else:
                # Default to JSONL on stdout
                write_jsonl(sys.stdout, records)
        
        elif csv_out:
            if output:
                with open(output, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADER)
                    for record in records:
                        write_csv_row(writer, record)
            else:
                writer = csv.writer(sys.stdout)
                writer.writerow(CSV_HEADER)
                for record in records:
                    write_csv_row(writer, record)
        
        else:
            # Default human-readable output
            console.print(f"[green]Processed {len(results)} images[/green]")
            
            # Save to file if requested (legacy behavior)
            if output:
                save_json(results, output)
                console.print(f"[green]Results saved to {output}[/green]")
            else:
                # Display first few results
                table = Table(title="Batch Results (first 5)")
                table.add_column("Image", style="cyan")
                table.add_column("Top Prediction", style="magenta")
                table.add_column("Confidence", style="green")
                
                for i, result in enumerate(results[:5]):
                    if result:
                        top_label = list(result.keys())[0]
                        top_prob = result[top_label]
                        table.add_row(
                            image_files[i].name,
                            top_label,
                            f"{top_prob:.3f}"
                        )
                
                console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    data_dir: Path = typer.Argument(..., help="Path to training data directory"),
    val_dir: Path = typer.Argument(..., help="Path to validation data directory"),
    epochs: int = typer.Option(15, "--epochs", "-e", help="Number of training epochs"),
    lr: float = typer.Option(3e-4, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size"),
    model_name: str = typer.Option("mobilenet_v3_small", "--model", help="Model architecture"),
    white_balance: bool = typer.Option(False, "--wb", help="Apply white balance during training"),
    output_dir: Path = typer.Option(Path("artifacts"), "--output", "-o", help="Output directory")
):
    """Train fabric classifier."""
    try:
        if not data_dir.exists():
            console.print(f"[red]Error: Training data directory {data_dir} does not exist[/red]")
            raise typer.Exit(1)
        
        if not val_dir.exists():
            console.print(f"[red]Error: Validation data directory {val_dir} does not exist[/red]")
            raise typer.Exit(1)
        
        # Create output directory
        safe_create_dir(output_dir)
        
        # Load data
        console.print("Loading datasets...")
        train_loader, val_loader = make_dataloaders(
            str(data_dir), str(val_dir), batch_size=batch_size
        )
        
        # Create model
        console.print("Creating model...")
        model = FabricClassifier.from_pretrained(model_name)
        
        # Training setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        console.print("Starting training...")
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            # Training
            model.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(model.device), target.to(model.device)
                
                optimizer.zero_grad()
                output = model.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            # Validation
            model.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(model.device), target.to(model.device)
                    output = model.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            
            # Calculate F1 score
            from sklearn.metrics import f1_score
            val_f1 = f1_score(all_targets, all_preds, average='macro')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            console.print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}"
            )
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                weights_path = output_dir / "weights.pt"
                model.save(weights_path)
                
                # Save metadata
                metadata = {
                    "model_name": model_name,
                    "epoch": epoch + 1,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "labels": FABRIC_LABELS,
                    "white_balance": white_balance
                }
                save_json(metadata, output_dir / "metadata.json")
            
            scheduler.step()
        
        # Save training history
        plot_training_history(
            train_losses, val_losses, train_accs, val_accs,
            output_dir / "training_history.png"
        )
        
        console.print(f"[green]Training complete! Best model saved to {output_dir}/weights.pt[/green]")
        console.print(f"Best validation F1: {best_val_f1:.4f}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def eval(
    data_dir: Path = typer.Argument(..., help="Path to test data directory"),
    weights: Path = typer.Argument(..., help="Path to model weights"),
    report: Path = typer.Option(Path("report.json"), "--report", help="Output report file"),
    confusion_matrix: Path = typer.Option(Path("confusion_matrix.png"), "--cm", help="Confusion matrix plot")
):
    """Evaluate model on test data."""
    try:
        if not data_dir.exists():
            console.print(f"[red]Error: Data directory {data_dir} does not exist[/red]")
            raise typer.Exit(1)
        
        if not weights.exists():
            console.print(f"[red]Error: Weights file {weights} does not exist[/red]")
            raise typer.Exit(1)
        
        # Load model
        console.print("Loading model...")
        model = FabricClassifier.from_pretrained()
        model.model.load_state_dict(torch.load(weights, map_location="cpu"))
        
        # Load test data
        console.print("Loading test data...")
        test_dataset = FolderDataset(str(data_dir))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Evaluate
        console.print("Running evaluation...")
        model.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(model.device), target.to(model.device)
                output = model.model(data)
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = compute_metrics(
            all_targets, all_preds, FABRIC_LABELS
        )
        
        # Save report
        report.parent.mkdir(parents=True, exist_ok=True)
        save_json(metrics, report)
        
        # Create confusion matrix
        confusion_matrix.parent.mkdir(parents=True, exist_ok=True)
        confusion_matrix_plot(
            all_targets, all_preds, FABRIC_LABELS, str(confusion_matrix)
        )
        
        # Display results
        console.print(f"[green]Evaluation complete![/green]")
        console.print(f"Accuracy: {metrics['accuracy']:.4f}")
        console.print(f"Macro F1: {metrics['macro_f1']:.4f}")
        console.print(f"Report saved to {report}")
        console.print(f"Confusion matrix saved to {confusion_matrix}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def calibrate(
    val_dir: Path = typer.Argument(..., help="Path to validation data directory"),
    weights: Path = typer.Argument(..., help="Path to model weights"),
    output: Path = typer.Option(Path("temperature.json"), "--output", "-o", help="Output calibration file")
):
    """Calibrate model using temperature scaling."""
    try:
        if not val_dir.exists():
            console.print(f"[red]Error: Validation data directory {val_dir} does not exist[/red]")
            raise typer.Exit(1)
        
        if not weights.exists():
            console.print(f"[red]Error: Weights file {weights} does not exist[/red]")
            raise typer.Exit(1)
        
        # Load model
        console.print("Loading model...")
        model = FabricClassifier.from_pretrained()
        model.model.load_state_dict(torch.load(weights, map_location="cpu"))
        
        # Load validation data
        console.print("Loading validation data...")
        val_dataset = FolderDataset(str(val_dir))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Collect logits and labels
        console.print("Collecting predictions...")
        all_logits = []
        all_labels = []
        
        model.model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(model.device), target.to(model.device)
                output = model.model(data)
                
                all_logits.append(output.cpu())
                all_labels.append(target.cpu())
        
        # Concatenate all data
        val_logits = torch.cat(all_logits, dim=0)
        val_labels = torch.cat(all_labels, dim=0)
        
        # Fit temperature
        console.print("Fitting temperature scaling...")
        optimal_temp = model.calibrate(val_logits, val_labels)
        
        # Save calibration
        output.parent.mkdir(parents=True, exist_ok=True)
        model.temperature_scaling.save(output)
        
        console.print(f"[green]Calibration complete![/green]")
        console.print(f"Optimal temperature: {optimal_temp:.4f}")
        console.print(f"Calibration saved to {output}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    weights: Path = typer.Argument(..., help="Path to model weights"),
    format: str = typer.Option("onnx", "--format", "-f", help="Export format (onnx, torchscript)"),
    output: Path = typer.Option(Path("model.onnx"), "--output", "-o", help="Output file path")
):
    """Export model to ONNX or TorchScript format."""
    try:
        if not weights.exists():
            console.print(f"[red]Error: Weights file {weights} does not exist[/red]")
            raise typer.Exit(1)
        
        # Load model
        console.print("Loading model...")
        model = FabricClassifier.from_pretrained()
        model.model.load_state_dict(torch.load(weights, map_location="cpu"))
        
        # Export
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "onnx":
            console.print("Exporting to ONNX...")
            to_onnx(model.model, str(output))
        elif format.lower() == "torchscript":
            console.print("Exporting to TorchScript...")
            to_torchscript(model.model, str(output))
        else:
            console.print(f"[red]Error: Unsupported format {format}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[green]Export complete! Model saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
