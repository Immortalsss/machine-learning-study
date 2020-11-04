function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


% 线性回归函数h(x) = theta0 +　theta1 * x + theta2 * x^2
h = X * theta;

% 代价函数J(theta0, theta1, theta2) = (h(x)- y)' * (h(x) - y) / (2 * m)
J = (h - y)' * (h - y) / (2 * m);



% =========================================================================

end
