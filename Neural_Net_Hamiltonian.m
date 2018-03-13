% Load dataset
load toy.dat
X = [ones(10,1) toy(:,1:2)];
t = toy(:,3);

% Define posterior distribution for W
alpha = 0.01;
y = @(W) sigmf(W*X',[1 0]);
G = @(W) -(t'*log(y(W)') + (1-t')*log(1-y(W))');
WDR = @(W) alpha*sum(W.^2, 2)'/2;
M = @(W)  G(W) + WDR(W);
% P = @(W) exp(-M(W));

% Define E and gradE
% energy E, given by negative logarithm of P
find_E = @(W) M(W); 
% Partial derivative of G with respect to wi.
find_gradE = @(W) (-(t' - y(W))*X)';

% Hamiltonian algorithm
% Initialize values
lag = 2000;
burn_in = 10000;
T = burn_in + 30*lag;
W_stored = zeros(T, 3);
accepted = 0;
W = [0 0 0];
W_stored(1,:) = W;

Tau = 20;
epsilon = 0.05;
E = find_E(W);
gradE = find_gradE(W);

% Loop T times
for i = 1:T-1
        P = randn(size(W));
        H = P*P'/2 + E;
        W_new = W;
        gradE_new = gradE;
        % Take Tau "leapfrog" steps
        for j = 1:Tau
                P = P - epsilon*gradE_new'/2;
                W_new = W_new + epsilon*P;
                gradE_new = find_gradE(W_new);
                P = P - epsilon*gradE_new'/2;
        end
        
        % Update H
        E_new = find_E(W_new);
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
            W = W_new;
            E = E_new;
            gradE = gradE_new;
        end
        
        accepted = accepted + accept;       
        W_stored(i+1,:) = W;
end

acceptance_rate = accepted/(T-1);
% Sum sampled output functions to find average neuron output
W_indep = W_stored(burn_in+lag:lag:T,:);
learned_y = @(x) zeros(1, length(x));
for i = 1:length(W_indep)
W = W_indep(i,:);
learned_y = @(x) [learned_y(x); sigmf(W*x',[1 0])];
end
learned_y = @(x) sum(learned_y(x))/length(W_indep);

% Plots
figure(1); clf

% Sample autocorrelation
subplot(1,2,1)
acf(W_stored(:,2), lag);

% Predictive distribution
subplot(1,2,2)
plot(X(1:5,2),X(1:5,3),'ks'); hold on
plot(X(6:10,2),X(6:10,3),'k*')
xlim([0 10]); ylim([0 10]); axis square
title('Predictive Distribution'); xlabel('x1'); ylabel('x2')
hold on
x1 = linspace(0,10);
x2 = x1;
[x1, x2] = meshgrid(x1, x2);
learned_y_cont = reshape(learned_y([ones(10000,1) x1(:) x2(:)]), 100, 100);
contour(x1, x2, learned_y_cont, [0.12 0.27 0.73 0.88],'--k'); hold on
contour(x1, x2, learned_y_cont, [0.5 0.5], 'k')


% Efficiency Comparison to the Metropolis Method

% Compared to the results found in the lecture, where
% the acceptance rate for the Metropolis method was 0.494,
% the Hamiltonian's rate is 0.938 -- a 90% improvement.
% A rough visual comparison of both method's sample
% autocorrelation plots in reveal that the Hamiltonian method 
% displays a significantly greater degree of efficiency 
% over the Metropolis method.
% In the Hamiltonian's plot, degree of autocorrelation
% (approx < 0.1) that is reached in 200 samples (lag length), 
% takes the Metropolis method approx. 2000 samples.
% Given that the time taken to compute each sample 
% is equal, the Hamiltonian method is ten times as fast.
% The Hamiltonian method also takes approx. 500 samples
% to reach 0, where the Metropolis method does not
% reach 0 in 2000 samples.

