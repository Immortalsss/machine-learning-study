%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% 初始化
clear ; 
close all;
 clc

%% 加载数据
%  前两列是考试分数，第三列是标签

data = load('ex2data1.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

%% ==================== Part 1: 绘制 ====================
%  开始编码之前，用绘图来帮助理解数据集合和了解问题

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);

% 添加一些标签
hold on;
% 标签和图例
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============ Part 2: 计算成本函数和梯度 ============

%  存储X的行列值
[m, n] = size(X);

% 添加x0列，全为1
X = [ones(m, 1) X];

% 初始化拟合参数theta，全为0
initial_theta = zeros(n + 1, 1);

% 计算并展示cost和gradient,此时theta为[0,0,0]
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
fprintf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

% 计算并展示cost和gradient,此时theta为[-24,0.2,0.2]
test_theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(test_theta, X, y);

fprintf('\nCost at test theta: %f\n', cost);
fprintf('Expected cost (approx): 0.218\n');
fprintf('Gradient at test theta: \n');
fprintf(' %f \n', grad);
fprintf('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============= Part 3: 用Octave函数fminunc进行优化  =============

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

% 利用fminunc获取最优的theta
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% 输出最优的theta
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('Expected cost (approx): 0.203\n');
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('Expected theta (approx):\n');
fprintf(' -25.161\n 0.206\n 0.201\n');

% 画出边界
plotDecisionBoundary(theta, X, y);

% 标签和图例 
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============== Part 4: 预测和计算准确性 ==============
%  预测数据，使用逻辑回归模型预测一个学生第一次考试45分，第二次考试85分是否被录取
%
%  计算模型的准确性


% 利用计算好的theta，代入sigmoid函数计算可能性
prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n'], prob);
fprintf('Expected value: 0.775 +/- 0.002\n\n');

% 计算我们训练集的准确性
p = predict(theta, X);

% 计算p和y中对应编号的值相等的个数
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (approx): 89.0\n');
fprintf('\n');


