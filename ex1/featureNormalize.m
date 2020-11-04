function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly

X_norm = X;

% mu和sigma是两个1 x 2的矩阵
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% 存储房子大小和卧室数量的平均数
mu = mean(X);

% 存储房子大小和卧室数量的标准差
sigma = std(X);

% 房子大小缩放
X_norm(:,1) = (X(:,1) - mean(X(:,1))) / std(X(:,1));

% 房子卧室数量缩放
X_norm(:,2) = (X(:,2) - mean(X(:,2))) / std(X(:,2));


% ============================================================

end
