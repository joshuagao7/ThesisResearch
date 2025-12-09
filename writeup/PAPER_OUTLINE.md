# Paper Outline (File System Style)

```
article.tex
├── preamble.tex
├── abstract.tex
├── intro.tex
│   ├── Data-Driven Sports Science
│   ├── Event Detection in Temporal Movement Data
│   ├── Limitations of Existing Approaches
│   ├── Human-Intuitive Pattern Detection
│   └── Proposed Framework and Example Domain
├── methods.tex
│   ├── Data Acquisition and Preprocessing
│   ├── Jump Detection Algorithms
│   │   ├── Threshold Algorithm
│   │   ├── Derivative Algorithm
│   │   └── Correlation Algorithm
│   └── Parameter Optimization
├── results.tex
│   ├── Visual Pipeline Demonstration
│   │   ├── Threshold Algorithm Pipeline
│   │   ├── Derivative Algorithm Pipeline
│   │   └── Correlation Algorithm Pipeline
│   ├── Loss Landscape Analysis
│   │   ├── Grid Search Methodology
│   │   ├── Algorithm Comparison
│   │   ├── Selected Parameters
│   │   ├── Parameter Robustness
│   │   └── Final Detection Performance
│   ├── Validation with Labeled Jump Heights
│   └── Downstream Analysis Examples
│       ├── Precise Boundary Detection
│       ├── PCA Analysis of Jump Segments
│       └── SVM Classification for Jumper Identification
├── discussion.tex
│   ├── Limitations
│   ├── Future Work
│   ├── Real-World Deployment: Vault One
│   └── Reflections on AI-Assisted Development
└── bibliography.bib
```

## Detailed Structure

### Root: article.tex
- Main document file that includes all sections
- Contains title, author information, and bibliography

### preamble.tex
- LaTeX packages and formatting
- Theorem environments
- Document settings

### abstract.tex
- Summary of the paper
- Key contributions and results

### intro.tex (Section 1)
- **Data-Driven Sports Science**: Context and motivation
- **Event Detection in Temporal Movement Data**: Problem statement and GRF data description
- **Limitations of Existing Approaches**: Current methods and their shortcomings
- **Human-Intuitive Pattern Detection**: Core philosophy of the approach
- **Proposed Framework and Example Domain**: Overview of the three algorithms
- **Contributions and Paper Roadmap**: Summary of contributions and paper organization

### methods.tex (Section 2)
- **Data Acquisition and Preprocessing**: Sensor system, pooling operation
- **Jump Detection Algorithms**: Three complementary approaches
  - **Threshold Algorithm**: Force threshold with physics constraints
  - **Derivative Algorithm**: Rate-of-change detection
  - **Correlation Algorithm**: Template matching approach
- **Precise Boundary Detection**: Refinement of jump boundaries
- **Parameter Optimization**: Grid search and loss landscape methodology

### results.tex (Section 3)
- **Visual Pipeline Demonstration**: Step-by-step visualizations
  - **Threshold Algorithm Pipeline**: Four-stage processing visualization
  - **Derivative Algorithm Pipeline**: Derivative-based detection visualization
  - **Correlation Algorithm Pipeline**: Template matching visualization
  - **Precise Boundary Detection**: Boundary refinement examples
- **Loss Landscape Analysis**: Parameter optimization results
  - **Grid Search Methodology**: Loss function definition
  - **Threshold Algorithm Loss Landscape**: Threshold algorithm performance
  - **Derivative Algorithm Loss Landscape**: Derivative algorithm performance
  - **Algorithm Comparison**: Direct comparison of all three algorithms
- **Optimal Parameter Selection**: Parameter choice and validation
  - **Selected Parameters**: Optimal parameter values
  - **Parameter Robustness**: Sensitivity analysis
  - **Final Detection Performance**: 94% accuracy results
- **Validation with Labeled Jump Heights**: Ground truth comparison
- **Downstream Analysis Examples**: Applications of detected jumps
  - **PCA Analysis of Jump Segments**: Dimensionality reduction
  - **SVM Classification for Jumper Identification**: Athlete classification

### discussion.tex (Section 4)
- **Limitations**: Current constraints and edge cases
- **Future Work**: Extensions to other tasks and improvements
- **Real-World Deployment: Vault One**: Mobile app deployment
- **Reflections on AI-Assisted Development**: Development process insights
- **Conclusion**: Summary and implications

### bibliography.bib
- Reference database

