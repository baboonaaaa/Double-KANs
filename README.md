# Double-KANs: Enhancing KANs with Simplified NURBS for Accurate Approximation of Discontinuous Signals
The code for the paper “Double-KANs: Enhancing KANs with Simplified NURBS for Accurate Approximation of Discontinuous Signals” will be released after the paper is published.

## Abstract

Kolmogorov–Arnold Networks (KANs) have shown strong potential in scientific computing, including approximating physical equations and solving partial differential equations. While B-splines provide KAN with local control and robustness to catastrophic forgetting, the Uniform Rational B-Splines (URBS) used in vanilla KAN struggle to approximate abrupt or discontinuous signals, limiting their effectiveness. To address this issue, we introduce Double-KANs, a novel structure that expresses Non-Uniform Rational B-Splines (NURBS) as a simplified quotient of two B-spline summations. Double-KANs retains the structural simplicity of vanilla KAN while substantially enhancing the approximation of discontinuities. Furthermore, it can be seamlessly applied to existing KAN-based models and extensions. Experimental evaluations across multiple tasks demonstrate that Double-KANs achieves superior accuracy and generalization compared to multilayer perceptrons(MLP), vanilla KAN, and other improved variants, highlighting its effectiveness and broad applicability.

## Highlight
- Introduce NURBS to improve KAN’s ability to approximate abrupt signals
- Maintain compatibility with KAN-based models for seamless enhancement
- Show better performance than MLPs, vanilla KANs, and their variants in multiple tasks

We propose Double-KANs, an extension of KANs that overcomes their limitations in approximating abrupt and discontinuous signals. By reformulating NURBS as a quotient of two B-spline summations, Double-KANs preserves the simplicity of vanilla KANs while significantly enhancing their approximation ability. Experiments on multiple tasks, including bladder cancer classification and staging, show that Double-KANs consistently outperforms MLPs, vanilla KANs, and other variants, achieving superior accuracy and generalization.
