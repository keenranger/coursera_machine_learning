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
X=[ones(m,1) X];
for i=1:m
    z_superscript_2=Theta1*transpose(X(i,:));%(25x401)*(1x401)T=25x1
    a_superscript_2=[1; sigmoid(z_superscript_2)];%activation,(25x1)->(26x1)
    
    z_superscript_3=Theta2*a_superscript_2;%10x26*(26x1)T=10x1
    a_superscript_3=sigmoid(z_superscript_3);%activation
    
    y_check=zeros(num_labels,1);%one hot encoding
    y_check(y(i))=1;%one hot encoding
   
    J=J+(1/m)*(-transpose(y_check)*log(a_superscript_3)-transpose(1-y_check)*log(1-a_superscript_3));%unregularized term
    
    delta_superscript_3=a_superscript_3-y_check;%10x1
    delta_superscript_2=transpose(Theta2)*delta_superscript_3;%(26x10)*(10x1)=(26*1)
    delta_superscript_2=delta_superscript_2(2:end).*sigmoidGradient(z_superscript_2);%remove delta super2 sub0 and elementwise gprime(zsuper2)
    
    Theta2_grad=Theta2_grad+(1/m)*delta_superscript_3*transpose(a_superscript_2);%(10x1)*(1x26)=(10x26)%backprpo without reg
    Theta1_grad=Theta1_grad+(1/m)*delta_superscript_2*X(i,:);%(25x1)*(1x401)=(25x401)
    

end
J=J+(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) +sum(sum(Theta2(:,2:end).^2)));%regularized term
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m)*Theta2(:,2:end);
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m)*Theta1(:,2:end);

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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
