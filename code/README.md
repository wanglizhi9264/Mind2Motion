# Mind2Motion EEG Encoder

Minimal public release of the Mind2Motion EEG encoder and its training code.

## Files

- `model.py`: multi-scale temporal-spatial EEG encoder with gated Transformer context.
- `train.py`: joint EEG-to-CLIP contrastive learning and motion classification.

The default architecture expects 7-channel, 1,500-sample EEG trials and produces a
512-dimensional normalized embedding plus 8-class logits. These values are inferred
from the input dataset where possible.

## Data format

Training uses one HDF5 file with the following datasets:

| Key | Content |
| --- | --- |
| `X` | `float32`, shape `(N, channels, time)` |
| `y` | integer class labels, shape `(N,)` |
| `split` | `train`, `val`, or `test` for each sample |
| `text_id` | caption identity for each sample |
| `text` | caption string for each sample |

Data and pretrained weights are not included.

## Train

```bash
pip install -r requirements.txt
python train.py --h5 /path/to/data.h5 --output runs/seed37 --seed 37
```

The released protocol uses AdamW with learning rate `7e-4`, 30 epochs, early stopping
on validation classification accuracy, and the joint objective
`0.25 InfoNCE + 0.025 cosine + 2.0 cross-entropy`.

The current dataset has motion class and acquisition session confounded. Results on
this split should not be interpreted as cross-session or cross-subject generalization.
