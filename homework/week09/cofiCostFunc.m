function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

Jmatrix = (X*Theta' - Y);
Jmsq = Jmatrix .^ 2;
%S = R + 1;
%J = accumarray(S(:), Jmatrix(:))(2) / 2;
% Note: The above works, but can be done more simply by using R as an index into Jmatrix in the sum.
% Found at the MathWorks documentaion site:
% https://www.mathworks.com/help/matlab/matlab_prog/find-array-elements-that-meet-a-condition.html
logicalR = R == 1;
J = sum(Jmsq(logicalR));

% Regularize the cost:
costReg = lambda * sum(Theta(:) .^ 2) + lambda * sum(X(:) .^ 2);
costReg;
J += costReg;
J /= 2;

% Calulate the gradients:
%X_grad = ((X*Theta' - Y) .* R) * Theta;
%Theta_grad = (Jmatrix .* R)' * X; % Transposing inline does not work
% So, let's create a temporary variable
goodDelta = Jmatrix .* R;
X_grad = goodDelta * Theta + lambda * X;
% ... and transpose the temporary variable.
Theta_grad = goodDelta' * X + lambda * Theta;
% ... and that works. Same code, different lines! Go figure

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
