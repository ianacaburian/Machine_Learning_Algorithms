% Load dataset
load toy.dat
X = [ones(10,1) toy(:,1:2)];
t = toy(:,3);
% Plot data
figure(1); clf
plot(X(1:5,2),X(1:5,3),'ks'); hold on
plot(X(6:10,2),X(6:10,3),'k*')
xlim([0 10]); ylim([0 10]); axis square
xlabel('x1'); ylabel('x2')
xn = length(X(:,1));
wn = xn+1;
tn = length(t);

% Initialize weights and bias
W = [-3 1 2];
% Loop T times
T = 50000;
eta = 0.01;
alpha = 0.01;
y = @(W) sigmf(W*X', [1 0]);
for i = 1:T
grad = -(t' - y(W))*X + alpha*W;
W = W - eta*grad;
end
% Plot learned function
figure(1); hold on
learned_y = @(X) sigmf(W*X',[1 0]);
x1 = linspace(0,10);
x2 = x1;
[x1 x2] = meshgrid(x1, x2);
learned_y_cont = reshape(learned_y([ones(10000,1) x1(:), x2(:)]), 100, 100);
contour(x1, x2, learned_y_cont, [0.27 0.73],'--k'); hold on
contour(x1, x2, learned_y_cont, [0.5 0.5], 'k')