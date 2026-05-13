"""src.task2 - Task 2 change-point detection.

Modules:
    records      data prep: panels, weighted sentiment, records + RARE bucket, shrinkage_loss
    pelt         Sec 9.4 Route 1: CROPS-elbow PELT + Kneedle + Donoho-Johnstone sigma
    clasp        ClaSP wrapper
    synth        AR(1) synthetic generators for Routes 2/3
    autocpd      Sec 9.5 Route 2: AutoCPD MLP + product embedding
    tst          Sec 9.6 Route 3: TST encoder + MVP pretraining + DeepAR product bias
    moment       Sec 9.7 Route 4: MOMENT-1-large + LoRA + 2-layer MLP side-channel
    hybrid       Sec 9.8 hybrid decoding: PELT candidates + neural-prior filter (gamma sweep)
    evaluate     Sec 9.9 F1 vs PELT + phase profiling + KMeans
    legacy       Sec 9.2 legacy rbf-PELT baseline (7-feature multivariate)
    weekly       Sec 9.W full weekly pipeline mirroring monthly Sec 9.3-Sec 9.9 + Sec 9.2
"""
