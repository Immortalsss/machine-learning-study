function plotData(X, y)
%   PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; 
hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% pos表示经过两次考试录取的学生编号
pos = find(y==1);
% neg表示经过两次考试没有录取的学生编号
neg = find(y==0);

% X(pos,1)表示录取学生的第一次成绩， X(pos,2)表示录取学生的第二次成绩
plot(X(pos,1), X(pos,2),'k+','LineWidth',2,'MarkerSize',7); 
plot(X(neg,1), X(neg,2),'ko','MarkerFaceColor','y','MarkerSize',7);


% =========================================================================



hold off;

end
