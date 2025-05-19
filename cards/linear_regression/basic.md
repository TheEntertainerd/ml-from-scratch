**Front:** What is the basic idea of linear regression?
**Back:** To find linear patterns in data by fitting a straight line (in 2D) or a hyperplane (in higher dimensions) that best represents the relationship between input and output variables.
**Tags:** linear_regression, basic

---
**Front:** Why can you sometimes choose linear regression over more complex machine learning models?
**Back:**
- It can give great results on the right datasets.
- It is computationally less expensive than many complex models.
**Tags:** linear_regression, basic

---
**Front:** What is an outlier in the context of a dataset?
**Back:** A data point that significantly deviates from the general trend of the other data points. Outliers can negatively impact the fitted line.
**Tags:** linear_regression, data_preprocessing

---
**Front:** In linear regression, what are residuals?
**Back:** The differences between the actual values of the target variable (y) and the values predicted by the linear regression model for the corresponding input variables (X). They represent the "mistakes" of the model.
**Tags:** linear_regression, basic

---
**Front:** Why do we square the residuals when evaluating a linear regression model?
**Back:**
- To ensure that both positive and negative errors contribute to the overall error (otherwise they could cancel each other out).
- To penalize larger errors more heavily.
**Tags:** linear_regression, basic

---
**Front:** What is the equation of a line in a 2D space, and what do its components represent in the context of linear regression?
**Back:** The equation is \(y = ax + b\), where:
- \(y\) is the predicted target variable (e.g., weight).
- \(x\) is the input variable (e.g., height).
- \(a\) is the slope, representing the change in \(y\) for a unit change in \(x\).
- \(b\) is the intercept, the value of \(y\) when \(x\) is zero.
In the context of the model, \(a\) and \(b\) are the model's <b>weights</b> that are learned from the data.
**Tags:** linear_regression, basic, weights

---
**Front:** How can we evaluate the performance of a linear regression model?
**Back:** By calculating metrics such as the sum of squared residuals or the R-squared value.
**Tags:** linear_regression, basic

---
**Front:** What does the R-squared value tell us about a linear regression model?
**Back:** It evaluates how well the linear regression line fits the data compared to a simple model that always predicts the average value of the target variable. It usually ranges from 0 to 1, with higher values indicating a better fit.
**Tags:** linear_regression, basic, r_squared

---
**Front:** What is a key limitation of the closed-form solution for linear regression when dealing with very large datasets and many features?
**Back:** The matrix inversion step can become computationally very expensive (roughly \(O(n^3)\) where \(n\) is the number of features), which makes it impractical for high-dimensional data.
**Tags:** linear_regression, basic