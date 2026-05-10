# Double-KANs: Enhancing KANs with Simplified NURBS for Accurate Approximation of Discontinuous Signals
![Double-KAN Architecture](images/highlight.png)


## Abstract

Kolmogorov–Arnold Networks (KANs) have shown strong potential in scientific computing, including approximating physical equations and solving partial differential equations. While B-splines provide KAN with local control and robustness to catastrophic forgetting, the Uniform Rational B-Splines (URBS) used in vanilla KAN struggle to approximate abrupt or discontinuous signals, limiting their effectiveness. To address this issue, we introduce Double-KANs, a novel structure that expresses Non-Uniform Rational B-Splines (NURBS) as a simplified quotient of two B-spline summations. Double-KANs retains the structural simplicity of vanilla KAN while substantially enhancing the approximation of discontinuities. Furthermore, it can be seamlessly applied to existing KAN-based models and extensions. Experimental evaluations across multiple tasks demonstrate that Double-KANs achieves superior accuracy and generalization compared to multilayer perceptrons(MLP), vanilla KAN, and other improved variants, highlighting its effectiveness and broad applicability.

## Highlight
- Introduce NURBS to improve KAN’s ability to approximate abrupt signals
- Maintain compatibility with KAN-based models for seamless enhancement
- Show better performance than MLPs, vanilla KANs, and their variants in multiple tasks

We propose Double-KANs, an extension of KANs that overcomes their limitations in approximating abrupt and discontinuous signals. By reformulating NURBS as a quotient of two B-spline summations, Double-KANs preserves the simplicity of vanilla KANs while significantly enhancing their approximation ability. Experiments on multiple tasks, including bladder cancer classification and staging, show that Double-KANs consistently outperforms MLPs, vanilla KANs, and other variants, achieving superior accuracy and generalization.

## Environment Requirements

### Python Version
- Python 3.9

### Main Dependencies
```bash
# Core scientific computing
numpy=1.26.4
torch=2.7.1+cu118
torchvision>=0.22.1

# Machine learning utilities
scikit-learn>=1.1.0
pandas=2.1.4

# Visualization (optional)
matplotlib>=3.5.0
```

### Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/Double-KANs.git
cd Double-KANs

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install python=3.9.23 numpy=1.26.4 torch=2.7.1+cu118 torchvision>=0.22.1 scikit-learn>=1.1.0 pandas=2.1.4 matplotlib>=3.5.0
```

## Quick Start

```bash
# Basic training with default parameters
python train_minist_double_kan.py

# Custom training with specific parameters
python train_minist_double_kan.py \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --n_hidden 32 \
    --model_name "my_double_kan" \
    --note "experiment_1"

# View all available parameters
python train_minist_double_kan.py --help
```
