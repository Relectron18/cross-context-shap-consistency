# Task E - SHAP Feature Families

Feature families are defined before group-level SHAP analysis and are
based on the semantic structure of the 23 flow-level features.

## 1. Flow duration
- duration

## 2. Directional IAT statistics
- total_fiat
- total_biat
- min_fiat
- min_biat
- max_fiat
- max_biat
- mean_fiat
- mean_biat

## 3. Flow throughput
- flowPktsPerSecond
- flowBytesPerSecond

## 4. Flow IAT statistics
- min_flowiat
- max_flowiat
- mean_flowiat
- std_flowiat

## 5. Active-period statistics
- min_active
- mean_active
- max_active
- std_active

## 6. Idle-period statistics
- min_idle
- mean_idle
- max_idle
- std_idle

### Aggregation rule

For each context and tree model, the saved Task C SHAP values are used.
For each feature family, group attribution is calculated as the sum of the
mean absolute SHAP contributions of the member features.

Normalized group contribution is:

group attribution / sum of all group attributions.

Group rankings are ordered by descending normalized group contribution.

The same feature families and aggregation rule are applied to both VPN and
Non-VPN contexts.
