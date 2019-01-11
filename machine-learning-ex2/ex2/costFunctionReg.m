function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
hypothesis=zeros(m,1);
hypothesis=sigmoid(X*theta);
J=(1/m)*(-transpose(y)*log(hypothesis)-transpose(1-y)*log(1-hypothesis)) + (lambda/(2*m))*sum(theta(2:length(theta),:).^2);

del_theta=(1/m)*transpose(hypothesis-y)*X;
for i=2:size(theta,1)
    del_theta(1,i)=del_theta(1,i)+(lambda/m)*theta(i);
end
grad=transpose(del_theta);



% =============================================================

end
