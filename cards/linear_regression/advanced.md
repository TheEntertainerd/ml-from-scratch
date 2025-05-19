**Front:** In deriving the closed-form solution for linear regression, we start with the linear model and MSE loss. What are these formulations?
**Back:** We start with a linear model \(\hat{y} = \mathbf{w}^T\mathbf{x}\) and use Mean Squared Error as our loss function:

\(MSE_{train} = \frac{1}{m}||\hat{\mathbf{y}}^{(train)} - \mathbf{y}^{(train)}||^2_2\)

To find the optimal weights, we need to minimize this MSE.
**Tags:** linear_regression, advanced, proof

---
**Front:** After defining the MSE loss for linear regression, what is the first step in finding the optimal weights?
**Back:** To minimize MSE, we set its gradient with respect to \(\mathbf{w}\) to zero:

\(\nabla_{\mathbf{w}}MSE_{train} = 0\)

\(\frac{1}{m}\nabla_{\mathbf{w}}||\hat{\mathbf{y}}^{(train)} - \mathbf{y}^{(train)}||^2_2 = 0\)

\(\frac{1}{m}\nabla_{\mathbf{w}}||\mathbf{X}^{(train)}\mathbf{w} - \mathbf{y}^{(train)}||^2_2 = 0\)
**Tags:** linear_regression, advanced, proof

---
**Front:** In the linear regression derivation, after setting \(\frac{1}{m}\nabla_{\mathbf{w}}||\mathbf{X}^{(train)}\mathbf{w} - \mathbf{y}^{(train)}||^2_2 = 0\), what is the expanded form of this gradient?
**Back:** 
\(\nabla_{\mathbf{w}}(\mathbf{X}^{(train)}\mathbf{w} - \mathbf{y}^{(train)})^T(\mathbf{X}^{(train)}\mathbf{w} - \mathbf{y}^{(train)}) = 0\)

\(\nabla_{\mathbf{w}}[\mathbf{w}^T\mathbf{X}^{(train)T}\mathbf{X}^{(train)}\mathbf{w} - 2\mathbf{w}^T\mathbf{X}^{(train)T}\mathbf{y}^{(train)} + \mathbf{y}^{(train)T}\mathbf{y}^{(train)}] = 0\)

\(2\mathbf{X}^{(train)T}\mathbf{X}^{(train)}\mathbf{w} - 2\mathbf{X}^{(train)T}\mathbf{y}^{(train)} = 0\)
**Tags:** linear_regression, advanced, proof

---
**Front:** In linear regression, after expanding the gradient and getting \(2\mathbf{X}^{(train)T}\mathbf{X}^{(train)}\mathbf{w} - 2\mathbf{X}^{(train)T}\mathbf{y}^{(train)} = 0\), what are the normal equations and the resulting closed-form solution?
**Back:** The normal equations are:

\(\mathbf{X}^{(train)T}\mathbf{X}^{(train)}\mathbf{w} = \mathbf{X}^{(train)T}\mathbf{y}^{(train)}\)

Solving for \(\mathbf{w}\) gives us the closed-form solution:
\(\mathbf{w} = (\mathbf{X}^{(train)T}\mathbf{X}^{(train)})^{-1}\mathbf{X}^{(train)T}\mathbf{y}^{(train)}\)
**Tags:** linear_regression, advanced, proof

---
**Front:** What is the basic prediction equation in linear regression? (How is the output calculated from the input?)
**Back:** Linear regression predicts a scalar value \(y \in \mathbb{R}\) from an input vector \(\mathbf{x} \in \mathbb{R}^n\). 

The prediction is defined as:
\(\hat{y} = \mathbf{w}^T\mathbf{x}\)

where \(\mathbf{w} \in \mathbb{R}^n\) is a vector of parameters (weights).
**Tags:** linear_regression, advanced, prediction

---
**Front:** In linear regression, if a feature's weight \(w_i\) is positive, how does this affect predictions?
**Back:** If \(w_i &gt; 0\): increasing the feature value \(x_i\) increases the prediction \(\hat{y}\)
**Tags:** linear_regression, advanced, weights

---
**Front:** In linear regression, if a feature's weight \(w_i\) is negative, how does this affect predictions?
**Back:** If \(w_i &lt; 0\): increasing the feature value \(x_i\) decreases the prediction \(\hat{y}\)
**Tags:** linear_regression, advanced, weights

---
**Front:** In linear regression, what does it mean when a feature's weight \(w_i\) has a large magnitude (absolute value)?
**Back:** If \(|w_i|\) is large: feature \(x_i\) has a large effect on the prediction
**Tags:** linear_regression, advanced, weights

---
**Front:** In linear regression, what does a weight value of \(w_i = 0\) tell us about the corresponding feature?
**Back:** If \(w_i = 0\): feature \(x_i\) has no effect on the prediction
**Tags:** linear_regression, advanced, weights


---
**Front:** What matrix condition must be satisfied for the closed-form solution of linear regression to exist and be unique?
**Back:** The closed-form solution \(\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\) exists and is unique when the matrix \(\mathbf{X}^T\mathbf{X}\) is invertible (non-singular).
**Tags:** linear_regression, advanced

---
**Front:** In linear regression, what three conditions ensure that the matrix \(\mathbf{X}^T\mathbf{X}\) is invertible?
**Back:** For \(\mathbf{X}^T\mathbf{X}\) to be invertible, we need:
- The feature matrix \(\mathbf{X}\) has full column rank
- There are at least as many examples as features (\(m \geq n\))
- No exact linear dependencies exist among the features (no multicollinearity)
**Tags:** linear_regression, advanced

---
**Front:** When the standard closed-form solution for linear regression doesn't exist uniquely, what are three alternative approaches?
**Back:** When the closed-form solution doesn't exist uniquely, the alternatives are:
- Using regularization (like ridge regression)
- Solving using the pseudoinverse
- Finding the minimum norm solution using iterative methods such as gradient descent
**Tags:** linear_regression, advanced