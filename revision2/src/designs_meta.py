"""Capture counts used by the calibrated designs (shared by designs.py and p8_v13_audit.py).

CAPTURE_FILES: number of capture-file prefixes per class in PREFIX_TO_TRAFFIC_ID of utils.py in the independent
PyTorch implementation of Deep Packet by munhouiani (https://github.com/munhouiani/Deep-Packet, paper ref. [56];
the method itself is Lotfollahi et al. [7]). That list names 109 Non-VPN and 30 VPN prefixes for CHAT, FT, MAIL,
STREAMING and VOIP, one VPN P2P prefix (vpn_bittorrent) and no Non-VPN P2P prefix. The Non-VPN P2P entry below is
set to 1 by assumption; with the minimum of four groups per class it has no effect on any design.
BROWSING has no dedicated prefixes (its flows come from captures listed under other classes).
Class codes: 0 BROWSING, 1 CHAT, 2 FT, 3 MAIL, 4 P2P, 5 STREAMING, 6 VOIP.
ARFF_COUNTS: flows per class in the Scenario A2 15 s ARFF files before deduplication (results/arff_class_counts.csv).
"""
CAPTURE_FILES = {"novpn": {1: 20, 2: 32, 3: 4, 4: 1, 5: 19, 6: 34}, "vpn": {1: 10, 2: 6, 3: 2, 4: 1, 5: 5, 6: 7}}
ARFF_COUNTS = {"novpn": {1: 890, 2: 1018, 3: 249, 4: 1000, 5: 482, 6: 2826, 0: 2500},
               "vpn": {1: 1196, 2: 1932, 3: 491, 4: 928, 5: 475, 6: 2271, 0: 2500}}
