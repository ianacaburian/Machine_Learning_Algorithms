% Define and plot sampling distribution
mu = [0 0]; % mean
rho = 0.998;
sigma = [1 rho; rho 1]; % covariance matrix

% X is implicitly a vector in R2
P = @(X) mvnpdf(X, mu, sigma); 
x1 = linspace(-1, 1);
x2 = x1;
[x1, x2] = meshgrid(x1, x2);
Pcontour = reshape(P([x1(:), x2(:)]), 100, 100);
figure(1); clf; contour(x1, x2, Pcontour, [1.6 3], 'k'); 
axis square;

% Define E and gradE
A = inv(sigma);
% energy E, given by negative logarithm of P
find_E = @(X) 0.5*X*A*X'; 
find_gradE = @(X) A*X'; % gradient of E

% Hamiltonian algorithm

% Initialize values
T = 3;
Tau = 20;
epsilon = 0.05;
X_stored_sub = zeros(Tau, 2);
X_stored = zeros(1, 2);
X_term = zeros(T+1, 2);
accepted = 0;
X = [-0.9 -0.7];
X_stored(1,:) = X;
X_term = X;
E = find_E(X);
gradE = find_gradE(X);

% Loop T times
for i = 1:T
        P = randn(size(X));
        H = P*P'/2 + E;
        X_new = X;
        gradE_new = gradE;
        % Take Tau "leapfrog" steps
        for j = 1:Tau
                P = P - epsilon*gradE_new'/2;
                X_new = X_new + epsilon*P;
                gradE_new = find_gradE(X_new);
                P = P - epsilon*gradE_new'/2;
                X_stored_sub(j,:) = X_new;
        end
        
        % Update H
        E_new = find_E(X_new);
        H_new = P*P'/2 + E_new;
        dH = H_new - H;
        
        % Decide whether to accept
        if dH < 0
                accept = 1;
        elseif rand() < exp(-dH)
            accept = 1;
        else
            accept = 0;
        end
        if accept
            X = X_new;
            E = E_new;
            gradE = gradE_new;
        end
        
        accepted = accepted + accept;
        X_stored = [X_stored; X_stored_sub];
        X_term(i+1,:) = X;
end

% Plot samples
figure(1); 
%hold on; plot(X_stored(:,1), X_stored(:,2),'-x');
hold on; plot(X_term(:,1), X_term(:,2),'ksq');
acceptance_rate = accepted/T;

