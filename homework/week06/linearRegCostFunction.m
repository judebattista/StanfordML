function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% 
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
err = h-y;
sqError = sum((err).^2);
J = 1/(2*m) * sqError;

regTheta = theta(2:end);
regTerm = lambda/(2*m) * dot(regTheta', regTheta);

J += regTerm;

% =========================================================================

%grad1 = 1/m * sum(err .* X(:, 1));
%gradRest = sum(err .* X(:, 2:end)) + theta'(:, 2:end);
%gradRest /= m;
%grad = [grad1 gradRest]';
% Works on example, but doesn't work on submit. Suspect dimensionality issue

% We can simplify this:
% Construct the regularization vector
gradReg = lambda / m * theta;
% First term is zero
gradReg(1) = 0;
% Construct the base gradient:
grad = X' * err;
grad /= m;
% Add 'em
grad += gradReg;


end
