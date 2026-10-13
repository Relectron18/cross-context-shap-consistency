# Leakage-Safe Splitting Strategy

## Dataset Structure

The ISCX VPN-nonVPN dataset is analyzed separately for two contexts:

- Non-VPN
- VPN

Within each context, the task is 7-class application classification:

- BROWSING
- CHAT
- FT
- MAIL
- P2P
- STREAMING
- VOIP

The provided train/test partitions were retained rather than randomly re-splitting the complete dataset.

## Exact Feature-Vector Audit

Exact feature-vector overlap between training and test partitions was checked after removing repeated rows within each partition.

Results:

| Context | Shared unique feature vectors |
|---|---:|
| Non-VPN | 1 |
| VPN | 13 |

The shared vectors were subsequently checked against their class labels. The overlapping feature vectors showed conflicting labels between the training and test partitions. Therefore, they were not treated as straightforward duplicated labeled samples and were not removed automatically.

## Near-Duplicate Audit

Nearest-neighbor distances were calculated using standardized features.

Cross-partition nearest-neighbor distances were compared with nearest-neighbor distances among samples within the same partition.

The resulting distributions were broadly comparable, with no clear evidence of systematic near-duplicate contamination across the train/test boundary.

## Distribution Control

The natural VPN and Non-VPN class distributions differ.

Therefore, a controlled sensitivity-analysis dataset was also created by selecting the same number of samples per class from both contexts:

- 190 training samples per class
- 48 test samples per class
- 1,330 training samples per context
- 336 test samples per context

The original datasets were not modified.

The controlled dataset will be used only as a sensitivity analysis and not as the primary training/test dataset.

## Remaining Limitation

The available processed files contain the 23 numerical features but do not provide recoverable capture/session identifiers.

Therefore, a capture-level or flow-group-level split could not be reconstructed from the available files.

Consequently, the analysis does not claim absolute proof that no related flows occur across the original train/test boundary.

Instead, leakage was assessed using:

1. Exact feature-vector overlap analysis.
2. Label consistency inspection.
3. Cross-partition near-neighbor analysis.
4. Comparison with within-partition similarity.
5. A class-balanced sensitivity analysis.

This limitation will be explicitly reported in the paper.