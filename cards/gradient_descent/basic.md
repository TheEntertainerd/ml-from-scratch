**Front:** What is the fundamental goal of the gradient descent algorithm?
**Back:** To iteratively find the minimum of a function by taking steps in the direction opposite to the function's gradient (or derivative in 1D) at the current point.
**Tags:** gradient_descent, basic

---
**Front:** In gradient descent, what does the gradient of a function tell us?
**Back:** The gradient points in the direction of the steepest increase of the function. Gradient descent moves in the opposite direction of the gradient to find a minimum, as this is the direction of steepest decrease.
**Tags:** gradient_descent, basic

---
**Front:** What is the "learning rate" in gradient descent and why is it important?
**Back:** The <b>learning rate</b> (often denoted as alpha, \(\alpha\)) is a hyperparameter that controls the size of the steps taken during each iteration. A well-chosen learning rate helps the algorithm converge to a minimum efficiently.
**Tags:** gradient_descent, basic, learning_rate

---
**Front:** What can happen if the learning rate in gradient descent is too large or too small?
**Back:**
- Too large: The algorithm might overshoot the minimum and fail to converge, potentially oscillating or diverging (bouncing around erratically).
- Too small: The algorithm will converge very slowly, requiring many iterations to reach the minimum.
**Tags:** gradient_descent, basic, learning_rate

---
**Front:** What are 2 common stopping criteria for the gradient descent algorithm?
**Back:**
- Reaching a predefined maximum number of iterations.
- The magnitude of the gradient becomes very small (below a specified tolerance), indicating that we are near a flat region (hopefully a minimum).
**Tags:** gradient_descent, basic

---
**Front:** Does gradient descent always find the global minimum of a function? Why or why not?
**Back:** No. Gradient descent is a local optimization algorithm. Depending on the function, it can get stuck in a local minimum or a saddle point, depending on the starting point. It doesn't guarantee finding the absolute lowest point (global minimum).
**Tags:** gradient_descent, basic, limitations

---
**Front:** In the context of gradient descent, what is a local minimum?
**Back:** A point on the function's surface that is lower than all its immediate neighboring points, but not necessarily the lowest point on the entire function. Gradient descent can converge to such a point.
**Tags:** gradient_descent, basic, optimization_landscape

---
**Front:** What is learning rate decay (or scheduling) and why might it be used in gradient descent?
**Back:** Learning rate decay is a technique where the learning rate is gradually reduced over iterations. It's used because a larger learning rate might be beneficial for making quick progress initially, while a smaller learning rate is better for fine-tuning and avoiding overshooting as the algorithm approaches a minimum.
**Tags:** gradient_descent, basic, learning_rate

---
**Front:** What is the mathematical update rule for gradient descent in multiple dimensions for a function \(J(\mathbf{w})\) at iteration \(k\)?
**Back:** If \(\mathbf{w}_k\) is the current parameter vector, the next vector \(\mathbf{w}_{k+1}\) is found by:
\(\mathbf{w}_{k+1} = \mathbf{w}_k - \alpha \nabla J(\mathbf{w}_k)\)
where \(\alpha\) is the learning rate and \(\nabla J(\mathbf{w}_k)\) is the gradient of the cost function \(J\) with respect to \(\mathbf{w}\) evaluated at \(\mathbf{w}_k\).
**Tags:** gradient_descent, basic, formula, multivariate

---
**Front:** Besides local and global minima/maxima, what is an other type of critical points (where the gradient is zero) that gradient descent encounter?
**Back:**
<b>Saddle points</b> are points where the gradient is zero, but they are neither a local minimum nor a local maximum. The function increases in some directions and decreases in others. These become more common in higher-dimensional problems.
**Tags:** gradient_descent, basic, optimization_landscape
