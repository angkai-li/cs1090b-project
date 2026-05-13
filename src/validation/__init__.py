"""src.validation - Sec 10 Event-Anchored Validation.

Modules:
    events        EVENT_ANCHORS (5 phones), KNOWN_EVENTS (21), tolerance constants
    hits          Sec 10.1 single-anchor hit-rate + Poisson-Binomial p-value
    extended      Sec 10.4 Moto G2 diagnostic + extended tolerance + edge filter + N=5 bootstrap
    magnitude     Sec 10.5/Sec 10.6 Cohen's d + Hedges' g with bootstrap CI
    reverse       Sec 10.5/Sec 10.6 reverse-direction lookup + random-cp null
    multi_anchor  Sec 10.5/Sec 10.6 N=14 multi-anchor hits + 10K bootstrap
    fdr           Sec 10.6 BH-FDR + Bonferroni-Holm corrections
    plots         Sec 10 overlay, bootstrap curves, forest, reverse bar
"""
