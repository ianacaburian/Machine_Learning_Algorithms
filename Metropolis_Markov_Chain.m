% Assignment 2: Question 3
% COSC380
% Author: Christian Caburian
% Date 14/08/17

% Define and plot sampling distribution
mu = [0 0]; % mean
rho = 0.998;
sigma = [1 rho; rho 1]; % covariance matrix
P = @(X) mvnpdf(X, mu, sigma); % X is implicitly a vector in R2
x1 = linspace(-1, 1);
x2 = x1;
[x1, x2] = meshgrid(x1, x2);
Pcontour = reshape(P([x1(:), x2(:)]), 100, 100);
figure(1);  clf;
contour(x1, x2, Pcontour, [1.6 3], 'k'); axis square

% Metropolis algorithm

% Initial State xt
% (arbitrarily chosen to begin on the target's edge)
xt = [ -0.55 -0.5]; 
eps = .05; 
sigQ = [ eps 0 ; 0 eps ];

% Random walk params
L = sqrt(2^2+2^2); 
T = ceil( (L / eps)^2 );
X = zeros(T, 2);

count_reject = 0;
count_accept = 0;
for t = 1:T 

    % Sample generated proposal Q
    xp = mvnrnd(xt, sigQ);    

    A = P(xp) / P(xt);    
    if A >= 1
        xt = xp;
        X(t,:) = xt;        
        count_accept = count_accept + 1;
    else
         if rand <= A
             xt = xp;
            X(t,:) = xt;
            count_accept = count_accept + 1;
         else
             % reject the state
             count_reject = count_reject + 1;
         end
    end    
end

% Plot samples
figure(1); hold on; scatter(X(:,1),X(:,2));