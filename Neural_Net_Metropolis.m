% Load dataset
load toy.dat
X = [ones(10,1) toy(:,1:2)];
t = toy(:,3);

% Define posterior distribution for W
alpha = 0.01;
y = @(W) sigmf(W*X',[1 0]);
G = @(W) -(t'*log(y(W)') + (1-t')*log(1-y(W))') + alpha*sum(W.^2, 2)'/2;
P = @(W) exp(-G(W));

% Metropolis algorithm
% Initialize values
lag = 2000;
burn_in = 10000;
T = burn_in + 30*lag;
proposal_size = 0.1;
W_stored = zeros(T, 3);
accepted = 0;
W = [0 0 0];
W_stored(1,:) = W;

% Define proposal distribution and acceptance ratio
Q_sample = @(W) mvnrnd(W, diag(proposal_size*ones(length(W),1)));
A = @(Wprime, W) P(Wprime)/P(W);

% Loop T-1 times
for i = 1:T-1
Wprime = Q_sample(W);
A_value = A(Wprime, W);

% Decide whether to accept
if A_value >= 1
    accept = 1;
    elseif A_value > rand()
    accept = 1;
    else
        accept = 0;
end
if accept
    W = Wprime;
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