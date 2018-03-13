% Define MU and SIGMA
MU = [0 0];
SIGMA = [ 1 -.9; -.9 1];
% % Evaluate the multivariate Gaussian at point X
% X = [1 1];
% mvnpdf(X, MU, SIGMA)
% 
% % Plot multivariate Gaussian
x1 = linspace(-5,5);
x2 = x1;

[X1, X2] = meshgrid(x1, x2);
% assign values for the mvgauss
X = [X1(:) X2(:)]; % 10000 x 2 double where each X1 
% will correspond to every X2
f = mvnpdf( X, MU, SIGMA); % 10000 x 1 double

% reshape(X,M,N) returns the M-by-N matrix 
% whose elements are taken columnwise from X. 
% An error results 
% if X does not have M*N elements.
 
% contour(X,Y,Z) draws a contour plot 
% of Z using vertices from the
% mesh defined by X and Y. 
% X and Y can be vectors or matrices.
res = reshape(f, length(x1), length(x1));
figure(2); contour(X1, X2, res);
xlabel('X1'); ylabel('X2'); colorbar