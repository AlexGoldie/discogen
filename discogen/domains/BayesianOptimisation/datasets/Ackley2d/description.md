DESCRIPTION
The negated Ackley function in 2D extends the 1D variant to a two-dimensional search space. It maintains the characteristic nearly flat outer region with a central area containing many local peaks surrounding a single global maximum. This benchmark is widely used to evaluate Bayesian optimization algorithms' ability to balance exploration and exploitation.

SEARCH SPACE
Dimensionality: 2D
Domain: [x₁, x₂] ∈ [-32.768, 32.768]²
Global maximum: [0, 0]

CHARACTERISTICS
One global maximum at the origin
Many local maxima arranged in a regular pattern
Exponentially decaying oscillations from the center
Tests systematic exploration and escape from local optima
