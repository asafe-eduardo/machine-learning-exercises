function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

   J_history(iter) = computeCost(X, y, theta);
         
   h = X * theta; %hypothesis function
   diff = h - y; %delta between hypothesis f unction and actual price
     
   %compute the new theta value by executing the derivation equation 
   theta_change = (diff * transpose(X) .* alpha) /m; 
   theta = theta - transpose(theta_change); 
     
   %make sure cost function J is always decreasing
   %fprintf('%f %f %f \n', J_history(iter), theta(1), theta(2)); 

   % ============================================================

   % Save the cost J in every iteration    
   % J_history(iter) = computeCost(X, y, theta);

end

end
