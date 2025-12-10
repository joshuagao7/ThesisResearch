Document Outline
│
├── Abstract
│   └── Overview: Athletic training data analysis, automated detection challenge, framework (threshold & derivative), Langevin optimization, 279 jumps/27 participants, Vault One deployment
│
├── 1. Introduction
│   ├── Content: Data-driven athletic training importance, MNIST analogy (humans easily identify jumps like handwritten digits, encoding into deterministic machine language), automated analysis challenge vs manual labeling, GRF data & 48-sensor pressure plate introduction, deterministic+ML optimization approach emphasized
│   └── Figures
│       └── raw_with_annotations.png: Raw sensor data with ground truth jump markers. Explains: (1) Each colored line = unique sensor from 48-sensor array, (2) Thick black line = pooled signal (proportional to force, not calibrated, possibly nonlinear), (3) Vertical dashed lines = manually annotated jump events we aim to detect automatically
│
├── 2. Methods
│   ├── 2.1 Algorithms
│   │   ├── 2.1.1 Threshold Algorithm
│   │   │   ├── Content: Naive threshold approach explanation and limitations (false positives)
│   │   │   └── Figures
│   │   │       └── threshold_naive.png: Naive threshold detection (θ=90) detecting 21 jumps, many false positives
│   │   │
│   │   ├── 2.1.2 Derivative Algorithm
│   │   │   ├── Content: Derivative-based detection (takeoff/landing pairs), physics constraints (t_min, t_max), in-air validation
│   │   │   └── Figures
│   │   │       ├── derivative_overlay.png: Raw data with derivative overlay showing negative spikes at takeoff, positive at landing
│   │   │       └── derivative_thresholds.png: Derivative detection (thresholds: +20/-15) detecting 13 jumps
│   │   │
│   │   └── 2.1.3 Other Approaches Considered
│   │       ├── Content: Correlation, Hybrid, Ensemble, Landing Derivative algorithms
│   │       └── Tables
│   │           └── Table 1: Algorithm Performance Comparison and Optimized Parameters
│   │
│   └── 2.2 Optimization
│       ├── 2.2.1 Measuring Success
│       │   └── Content: Loss function definition (FP + FN), manual ground truth tagging explanation
│       │
│       ├── 2.2.2 Parameterization
│       │   └── Content: Explanation of which parameters to optimize for each algorithm (threshold, derivative, correlation, hybrid, ensemble, landing derivative)
│       │
│       └── 2.2.3 Optimization Methodology
│           └── Content: Grid search method first (for 2-3 parameters), challenge of high-dimensional data (curse of dimensionality), then Langevin Monte Carlo sampling introduction and explanation
│
├── 3. Results
│   ├── 3.1 Algorithm Implementation on Dataset
│   │   ├── Content: Performance on 279 jumps/27 participants. Derivative best (28 errors, 90% accuracy). All algorithm results in Table 2. Loss landscape visualizations integrated here (grid search comparison showing derivative superiority). Optimized threshold values mentioned: θ_upper = 17.71, θ_lower = -15.19, with loss landscape showing broad, stable optimal region
│   │   └── Tables
│   │       ├── Table 2: Algorithm Performance After Optimization (all algorithms)
│   │       └── Table 3: Jump Detection Performance by Participant
│   │   └── Figures
│   │       └── comparison_stacked_xz_view.png: Grid search loss landscapes comparing algorithms (side view)
│   │
│   └── 3.2 Downstream Analysis
│       ├── Content: What we can do when jumps are detected well: precise takeoff/landing boundary detection using half-peak thresholding, storing data over time, longitudinal tracking, trend analysis, monitoring athlete progress, tracking evolution of metrics (jump height, contact time, force production patterns)
│       └── Figures
│           └── jump_snapshots.png: Jump snapshots with initial detection (blue) and refined precise boundaries (yellow)
│
├── 4. Discussion
│   ├── 4.1 Generalizability and Algorithm Selection
│   │   └── Content: Why Derivative chosen over Ensemble (overfitting analogy, 30+ parameters, complex decision boundary) and Hybrid (weaker interpretation). Derivative represents acceleration (interpretable), closely tethered to real physical indicator. May not be most optimized on this dataset but most generalizable. Loss landscape trough (upper/lower thresholds) is stable, insensitive to manufacturing changes (foam stiffness, person weight). This generalizability possible because we didn't use black-box ML. Beauty of deterministic+ML optimization approach (uncommon in literature)
│   │
│   ├── 4.2 Future Work
│   │   └── Content: Extend to other tasks (sprints), adaptive tuning, sensor fusion. Discussion of lower derivative threshold asymmetry (θ_lower = -15.19 smaller in magnitude than θ_upper = 17.71): partially due to jump biomechanics (landing more gradual than takeoff), also material hysteresis and relaxation times. Future work: integrate with research on selecting optimal dielectric materials, characterize various foam compositions to optimize sensor design at intersection of hardware and software 
│   │
│   └── 4.3 Real-World Deployment: Vault One
│       ├── Content: JavaScript implementation for mobile app, real-time feedback
│       └── Figures
│           └── screenshot2.png: Vault One mobile app interface showing real-time jump detection
│
└── 5. Conclusion
    └── Content: Lightweight interpretable framework summary, contributions (algorithms, optimization, deployment), impact on sports science workflows
