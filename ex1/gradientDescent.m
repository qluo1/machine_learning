function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

theta_new = theta;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	
	for i = 1:length(theta)
		theta_new(i,1) = theta(i,1) - alpha * sum( (X * theta - y) .* X(:,i)) / m;
		fprintf("theta %f \t", theta_new(i,1));

	end
	
	theta = theta_new;
	
    % ============================================================
	cost = computeCost(X, y, theta);
	fprintf("iter: %d, cost: %f \n", iter, cost);
    % Save the cost J in every iteration    
    J_history(iter) = cost;

end
