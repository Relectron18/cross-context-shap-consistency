# Task F - SHAP Consequence Test Methodology

## Objective

Determine whether differences in SHAP-derived feature importance between
Non-VPN and VPN contexts have measurable consequences for predictive
performance.

## Leakage-safe feature selection

Task C SHAP rankings are not used for feature selection because they were
computed on the held-out test sets.

Instead, feature selection is performed using only the original training
partition of each context.

For each context/model combination:

1. Split the original training data into an inner-training set and an
   inner-validation set using a stratified split.
2. Train the model on the inner-training set.
3. Compute TreeSHAP on the inner-validation set.
4. Compute mean absolute SHAP importance across classes and inner-validation
   samples using the same definition as Task C.
5. Select the Top-10 features.
6. Discard the temporary inner split.
7. Retrain the model using the selected features on the FULL original
   training set.
8. Evaluate on the original untouched test set.

## Models

- Random Forest
- XGBoost

The same model hyperparameters used in Tasks B and C are retained.

## Feature-set conditions

For every target context and model:

### Full-feature model
Train and evaluate using all 23 features.

### Context-specific SHAP subset
Use the Top-10 features selected from the target context's own training
partition.

### Transferred SHAP subset
Use the Top-10 features selected from the other context.

Both transfer directions are evaluated:

- Non-VPN-selected features -> VPN
- VPN-selected features -> Non-VPN

## Evaluation

All final evaluations use the original untouched test sets.

Primary metric:
- Macro-F1

Secondary metrics:
- Per-class recall

The full-feature model provides the reference performance for each target
context/model.

## Interpretation

The consequence test asks whether context-specific SHAP differences have
predictive consequences.

If a transferred SHAP subset performs substantially worse than the
context-specific subset while the full-feature model remains strong, this
supports the conclusion that context-dependent feature importance is
practically relevant rather than purely explanatory.

If transferred and context-specific subsets perform similarly, the SHAP
differences may have limited predictive consequence.

The final conclusion must be based on the measured test-set results rather
than assumed from SHAP rankings alone.
