"""
FPGA-Based Neuromorphic Plant Disease Detection Using SNNs
Software prototype matching the presentation's 3-8-2 architecture.

Dataset layout expected:
dataset/
    train/
        Apple___healthy/
        Apple___Apple_scab/
        ...
    valid/
        Apple___healthy/
        Apple___Apple_scab/
        ...

All folders containing "healthy" (case-insensitive) map to class 0.
All other folders map to class 1 (diseased).

Important:
This is a compact proof-of-concept. It uses the mean RGB value of each
image as the three input features to match the 3-input-neuron architecture
in the slides. Mean RGB alone is a weak representation for plant disease
recognition and should not be treated as a production-accuracy model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate


IMAGE_SIZE = 128
TIME_STEPS = 25
BETA = 0.9
HIDDEN_NEURONS = 8
NUM_OUTPUTS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RGBMeanDataset(Dataset):
    """Loads images and returns normalized mean R, G, B plus binary label."""

    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.ImageFolder(root=root, transform=self.transform)

        if len(self.dataset.classes) < 2:
            raise ValueError(f"Expected multiple class folders in {root}")

        self.samples = self.dataset.samples
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, _original_label = self.samples[index]
        image = self.dataset.loader(image_path).convert("RGB")
        image = self.transform(image)

        # Three features: mean intensity of the R, G and B channels.
        rgb_features = image.mean(dim=(1, 2)).clamp(0.0, 1.0)

        folder_name = Path(image_path).parent.name.lower()
        label = 0 if "healthy" in folder_name else 1

        return rgb_features, torch.tensor(label, dtype=torch.long)


class PlantDiseaseSNN(nn.Module):
    """3 input features -> 8 LIF hidden neurons -> 2 LIF output neurons."""

    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(3, HIDDEN_NEURONS)
        self.lif1 = snn.Leaky(
            beta=BETA,
            spike_grad=spike_grad,
            reset_mechanism="subtract"
        )

        self.fc2 = nn.Linear(HIDDEN_NEURONS, NUM_OUTPUTS)
        self.lif2 = snn.Leaky(
            beta=BETA,
            spike_grad=spike_grad,
            reset_mechanism="subtract"
        )

    def forward(self, features):
        batch_size = features.size(0)

        # Rate coding: each RGB intensity controls the probability of a spike.
        # Inputs are in [0, 1], and a new binary spike sample is generated
        # at every simulation time step.
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        output_spike_count = torch.zeros(
            batch_size, NUM_OUTPUTS, device=features.device
        )

        for _ in range(TIME_STEPS):
            input_spikes = (
                torch.rand_like(features) < features
            ).float()

            current1 = self.fc1(input_spikes)
            hidden_spikes, mem1 = self.lif1(current1, mem1)

            current2 = self.fc2(hidden_spikes)
            output_spikes, mem2 = self.lif2(current2, mem2)

            output_spike_count += output_spikes

        return output_spike_count / TIME_STEPS


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = output.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        output = model(features)
        loss = criterion(output, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def to_q8_8_hex(value):
    """Convert a float to signed 16-bit Q8.8 hexadecimal representation."""
    scaled = int(np.round(float(value) * 256.0))
    scaled = max(-32768, min(32767, scaled))
    return f"{scaled & 0xFFFF:04X}"


def export_fpga_weights(model, output_dir):
    """
    Export all 50 trained parameters as signed Q8.8 16-bit hex words.
    File order:
      fc1 weights (row-major), fc1 biases,
      fc2 weights (row-major), fc2 biases.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    ordered = [
        ("fc1.weight", state["fc1.weight"]),
        ("fc1.bias", state["fc1.bias"]),
        ("fc2.weight", state["fc2.weight"]),
        ("fc2.bias", state["fc2.bias"]),
    ]

    mem_path = output_dir / "snn_weights.mem"
    with mem_path.open("w", encoding="utf-8") as file:
        file.write("// Signed 16-bit Q8.8 weights; one hex word per line\n")
        for name, tensor in ordered:
            file.write(f"// {name} shape={tuple(tensor.shape)}\n")
            for value in tensor.detach().cpu().reshape(-1):
                file.write(to_q8_8_hex(value.item()) + "\n")

    metadata = {
        "architecture": "3-8-2",
        "encoding": "signed 16-bit Q8.8",
        "fractional_bits": 8,
        "time_steps": TIME_STEPS,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "memory_file": "snn_weights.mem",
        "parameter_order": [name for name, _ in ordered],
    }
    (output_dir / "snn_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"FPGA weights exported to: {mem_path}")


@torch.no_grad()
def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    features = tensor.mean(dim=(1, 2)).unsqueeze(0).to(DEVICE)

    model.eval()
    scores = model(features)[0]
    predicted_class = int(scores.argmax().item())
    label = "Healthy" if predicted_class == 0 else "Diseased"

    print(f"Prediction: {label}")
    print(f"Output spike rates [healthy, diseased]: {scores.cpu().tolist()}")
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset",
                        help="Folder containing train/ and valid/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--predict", default=None,
                        help="Optional path to one leaf image to classify")
    args = parser.parse_args()

    train_root = Path(args.data) / "train"
    valid_root = Path(args.data) / "valid"

    if not train_root.exists() or not valid_root.exists():
        raise FileNotFoundError(
            "Dataset folders not found. Expected dataset/train and dataset/valid."
        )

    train_dataset = RGBMeanDataset(train_root)
    valid_dataset = RGBMeanDataset(valid_root)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    print("Device:", DEVICE)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(valid_dataset))
    print("Original dataset folders:", train_dataset.classes)
    print("Binary mapping: healthy=0, diseased=1")

    model = PlantDiseaseSNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        valid_loss, valid_acc = evaluate(
            model, valid_loader, criterion
        )

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%"
        )

        if valid_acc > best_accuracy:
            best_accuracy = valid_acc
            torch.save(model.state_dict(), output_dir / "best_snn.pth")

    model.load_state_dict(
        torch.load(output_dir / "best_snn.pth", map_location=DEVICE)
    )
    export_fpga_weights(model, output_dir)

    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print("Saved model:", output_dir / "best_snn.pth")

    if args.predict:
        predict_image(model, args.predict)


if __name__ == "__main__":
    main()
