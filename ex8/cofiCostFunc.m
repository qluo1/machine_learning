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
assert(size(X) == [num_movies,num_features]);
assert(size(Theta) == [num_users, num_features]);

%for i = 1:num_movies
%    xi = X(i,:);
%    for j = 1:num_users
%        thetaj = Theta(j,:);
%        if (R(i,j) == 1)
%            J = J + (thetaj * xi' - Y(i,j)) ^ 2;
%        end
%    end
%end
%
%
%J = 1/ 2 * J;

% right version is R == 1 
% J = sum(((Theta * X')' - Y)(R==1) .^ 2)/2;

%J = sum(sum(((Theta * X')' - Y) .^2 .* R))/2;

% regulized
%J = sum(sum(((Theta * X')' - Y) .^2 .* R))/2 + lambda/2 * sum(sum(Theta .^ 2)) + lambda/2 * sum(sum(X.^2));

% 1st
%J = sum(sum(((Theta * X')' - Y) .^2 .* R))/2 + lambda/2 * sum(sum(Theta .^ 2)) + lambda/2 * sum(sum(X.^2));

% 2nd option
J = sum(sum(((Theta * X')' - Y)(R==1) .^2 ))/2 + lambda/2 * sum(sum(Theta .^ 2)) + lambda/2 * sum(sum(X.^2));

for i = 1:num_movies
    xi = X(i,:);
    
    %X_grad(i,:) = ((Theta * xi' - Y(i,:)').* (R(i,:)'))' * Theta;
    
    % regualized version
    X_grad(i,:) = ((Theta * xi' - Y(i,:)').* (R(i,:)'))' * Theta + lambda * xi;
%    for j = 1:num_users
%        thetaj = Theta(j,:);
%        if (R(i,j) == 1)
%            X_grad(i,:) = X_grad(i,:) + (thetaj * xi' - Y(i,j)) * Theta(j,:);
%        end
%    end
end



for j = 1:num_users
    betaj = Theta(j,:);
    
    %Theta_grad(j,:) =  ((X * betaj' - Y(:,j)) .* R(:,j))' * X;
    
    % regualized version
    Theta_grad(j,:) =  ((X * betaj' - Y(:,j)) .* R(:,j))' * X + lambda * betaj;
%    for i = 1: num_movies
%        xi = X(i,:);
%        if (R(i,j) == 1)
%            Theta_grad(j,:) = Theta_grad(j,:) + (betaj * xi' - Y(i,j)) * X(i,:);
%        end
%    end
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
