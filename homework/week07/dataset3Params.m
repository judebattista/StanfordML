function [c, sigma] = dataset3Params(x, y, xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
c = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
minErr = 1000;
cset = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sset = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
fooMax = size(cset, 2);
barMax = size(sset, 2);
for foo = 1:fooMax
    for bar = 1:barMax
        testc = cset(foo);
        tests = sset(bar);
        % Train the model on the training data using this comination of c and σ
        mod = svmTrain(x, y, testc, @(x1, x2) gaussianKernel(x1, x2, tests));
        % Use the model to make predictions on the xval set
        pred = svmPredict(mod, xval);
        % calculate the error
        err = mean(double(pred ~= yval))
        % if the error is better than our current error, keep c and σ
        if err < minErr
            minErr = err;
            c = testc;
            sigma = tests;
        end
    end
end

% =========================================================================

end
