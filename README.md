# Plant Disease Detection using SNN (3-8-2)

## 1. Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Arrange the dataset
Download/extract the New Plant Diseases Dataset and arrange it like this:

```text
dataset/
  train/
    Apple___healthy/
    Apple___Apple_scab/
    ...
  valid/
    Apple___healthy/
    Apple___Apple_scab/
    ...
```

Folder names containing `healthy` are labelled Healthy (0); all other folders
are labelled Diseased (1).

## 3. Train
```bash
python train_snn.py --data dataset --epochs 20 --batch-size 64
```

## 4. Classify one image
```bash
python train_snn.py --data dataset --epochs 20 --predict path/to/leaf.jpg
```
This command trains first, then predicts the supplied image.

## 5. Outputs
- `outputs/best_snn.pth`: best model checkpoint
- `outputs/snn_weights.mem`: 50 signed 16-bit Q8.8 parameter words
- `outputs/snn_metadata.json`: architecture and memory format metadata

## Important limitations
This is a software prototype, not a complete FPGA bitstream. It exports the
trained linear-layer weights, but FPGA deployment also requires RTL/HLS logic
for RGB feature extraction, spike generation, LIF membrane updates, threshold
and reset behavior, output decision, and a testbench.

The three input features are global mean R/G/B values to match the slide's
3-input-neuron design. This discards spatial lesion information, so accuracy
may be limited. For a stronger disease detector, use spatial RGB features or
a larger feature extractor and update the FPGA architecture accordingly.
