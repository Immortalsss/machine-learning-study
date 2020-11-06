function g = sigmoid(z)
%   SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% 对传入的矩阵中的每一个元素做sigmoid函数，返回值位于0-1
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% 对矩阵中的每一个元素做sigmoid运算
g = 1.0 ./ (1.0 + exp(-z));


% =============================================================

end
