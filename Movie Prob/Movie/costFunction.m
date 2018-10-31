function [J, grad] = costFunction(theta, X, sales)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(sales); % number of training examples
%fprintf('\n no of training examples %f \n', m);

% You need to return the following variables correctly 
J = 0;
%fprintf('sixe of theta %f \n', size(theta));
grad = zeros(size(theta));

%fprintf('\n gradient %f \n', grad);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%fprintf('\n working 1 \n');
%fprintf('\n X is %f \n', X );

h = sigmoid(X*theta);

%figure
%plot(h);
%xlabel('Sigmoid Function')

%fprintf('\n h is %f \n');
%fprintf('\n working h from sigmoid \n');
% J = (1/m)*sum(-y .* log(h) - (1 - sales) .* log(1-h));
J = (1/m)*(-sales'* log(h) - (1 - sales)'* log(1-h));
grad = (1/m)*X'*(h - sales);


% =============================================================

end
