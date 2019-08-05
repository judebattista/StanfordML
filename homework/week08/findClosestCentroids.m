function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% get the number of examples
m = size(X, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for foo = 1:m
    minDist = norm(X(foo, :) - centroids(1, :), 2);
    minNdx = 1;
    if K > 1
        for bar = 2:K
            dis = norm(X(foo, :) - centroids(bar, :), 2);
            if dis < minDist
                minDist = dis;
                minNdx = bar;
            end
        end
    end
    idx(foo) = minNdx;
end

% =============================================================

end

