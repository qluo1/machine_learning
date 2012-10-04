function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% add bias unit 1
a1 = [ones(size(X,1),1),X];

% assert 5000x401
%size(a1);
% assert(size(a1) == [5000 401])

% input layer
% assert(size(Theta1) == [25 401])
z2 = Theta1 * a1';
a2 = sigmoid(z2);

% hidden layer
% assert(size(a2) == [25 5000])

a2 = [ones(size(a2',1),1) a2'];
%assert(size(a2) == [5000 26])
% assert(size(Theta2) == [10 26])
z3 = Theta2 * a2';
a3 = sigmoid(z3);
% output layer
% assert(size(a3) ==[10 5000])
a3 = a3';

% assert(size(a3,2) == 10)

for i = 1:size(a3,2)
    yk = y == i;
    %J = J + sum(-yk .* log(a3(:,i)) - (1-yk) .* log(1 - a3(:,i)))/m;
    J = J + sum(-yk .* log(a3(:,i)) - (1-yk) .* log(1 - a3(:,i)))/m;
end
% assert(size(Theta1(:,2:end)) == [25 400])
tmp1 = sum(sum(Theta1(:,2:end).^2));
% assert(size(Theta2(:,2:end)) == [10 25])
tmp2 = sum(sum(Theta2(:,2:end).^2));
J = J + lambda * (tmp1 + tmp2) / (2 * m);

% -------------------------------------------------------------
% --- backprop for each X item to calculate Theta1_grad, Theta2_grad

for i =1:m
    % step 1
    a_1 = X(i,:);
    a_1 = [1;a_1'];
    %size(a_1)
    %assert(size(a_1) == [401,1]);
    % 25 * 401 x 401x1 = 25 * 1
    z_2 = Theta1 * a_1;
    a_2 = sigmoid(z_2);
    %assert(size(a_2) == [25,1]);
    a_2 = [1;a_2];
    %assert(size(a_2) ==[26,1]);
    % 10x26 * 26x1
    a_3 = sigmoid(Theta2 * a_2) ;
    %assert(size(a_3) == [10,1])
    y_1 = [1:num_labels] == y(i);
    % step 2
    delta_3 = a_3 - y_1';
    %assert(size(delta_3) == [10,1])
    % step 3
    % 26x10 * 10x1 = 25 x1
    delta_2 = (Theta2' * delta_3)(2:end).* sigmoidGradient(z_2);
    % 25x1

    % step 4 - accumulate delta
    %assert(size(delta_3) == [10 1])
    % 10x1 * 1x26
    %assert(size(a_2') == [1,26])
    Theta2_grad = Theta2_grad + delta_3 * a_2';

    %assert(size(a_1) == [401,1])
    Theta1_grad = Theta1_grad + delta_2 * a_1';
end

%assert(size(Theta1) == [25 401])
tmp_1 = Theta1(:,2:end);
tmp_1 = [zeros(size(tmp_1,1),1),tmp_1];
%assert(size(tmp_1) == [25 401])

%assert(size(Theta2) == [10,26])
tmp_2 = Theta2(:,2:end);
tmp_2 = [zeros(size(tmp_2,1),1),tmp_2];
%assert(size(tmp_2) == [10,26])

Theta1_grad = Theta1_grad / m + lambda * tmp_1 / m;
Theta2_grad = Theta2_grad / m + lambda * tmp_2 / m; 
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
