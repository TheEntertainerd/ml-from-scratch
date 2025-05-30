**Front:** Under what mathematical condition on the objective function is gradient descent guaranteed to converge to the global minimum (assuming an appropriate learning rate and other mild conditions)?
**Back:** When the objective function is <b>convex</b>. For convex functions, any local minimum is also a global minimum. Gradient descent, with a suitable learning rate, will converge to this global minimum.
**Tags:** gradient_descent, advanced, convexity, convergence, global_minimum

---
**Front:** What is a common mathematical definition of a convex function \(f\)?
**Back:** A function \(f\) defined on a convex set is convex if for any two points \(x_1, x_2\) in its domain and any \(\lambda \in [0, 1]\), the following inequality holds:
\(f(\lambda x_1 + (1-\lambda)x_2) \le \lambda f(x_1) + (1-\lambda)f(x_2)\).
This means the line segment connecting any two points on the graph of the function lies on or above the graph. For differentiable functions, an equivalent condition is \(f(y) \ge f(x) + \nabla f(x)^T (y-x)\) for all \(x,y\). If twice differentiable, \( \nabla^2 f(x) \succeq 0 \) (Hessian is positive semi-definite).
**Tags:** gradient_descent, advanced, convexity, definition

---
**Front:** How does the "curse of dimensionality" relate to the prevalence of local minima versus saddle points in high-dimensional optimization landscapes encountered in deep learning?
**Back:** In high-dimensional spaces, most critical points (where the gradient is zero) are overwhelmingly more likely to be <b>saddle points</b> rather than local minima. True local minima that are significantly worse than the global minimum become exponentially rarer as dimensionality increases. This implies that for very high-dimensional problems, escaping saddle points is often a more significant challenge than getting stuck in poor local minima.
**Tags:** gradient_descent, advanced, optimization_landscape, dimensionality, saddle_points, deep_learning

---
**Front:** What is the mathematical condition for a point \(\mathbf{x}^*\) to be a critical point (or stationary point) of a differentiable function \(f(\mathbf{x})\)?
**Back:** The gradient of the function at that point must be the zero vector: \(\nabla f(\mathbf{x}^*) = \mathbf{0}\). Gradient descent algorithms aim to find such points, which can be local minima, local maxima, or saddle points.
**Tags:** gradient_descent, advanced, critical_points, optimization_theory
