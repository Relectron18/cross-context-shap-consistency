# Task E - Individual-Feature vs Group-Level Conclusion

The individual-feature SHAP rankings from Task C show context- and model-dependent
changes in the most important features.

Random Forest:
- Non-VPN: total_biat, min_flowiat, total_fiat, flowBytesPerSecond, duration
- VPN: flowBytesPerSecond, min_fiat, min_biat, min_flowiat, total_biat

XGBoost:
- Non-VPN: min_flowiat, flowBytesPerSecond, total_biat, min_fiat, duration
- VPN: flowBytesPerSecond, min_flowiat, min_biat, min_fiat, ...

At the group level, Directional IAT statistics are the largest attribution
family for both contexts and both tree models. Flow IAT statistics and Flow
throughput are consistently the next major groups.

Therefore, individual feature importance shows context/model-dependent
variation, while the higher-level feature-family structure is substantially
more consistent across VPN and Non-VPN contexts.

This supports reporting both levels: individual-feature rankings reveal
specific changes in model reliance, while group-level analysis identifies
broader traffic-behavior patterns that remain important across contexts.
