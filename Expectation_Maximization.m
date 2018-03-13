% Assignment 2: Question 1
% COSC380
% Author: Christian Caburian
% Date 14/08/17

% Load and Initialize Data
data = dlmread('old_faithful.dat', '\t', 26, 0);
x1 = data(:,2); x2 = data(:,3);
range1 = max(x1)-min(x1); 
range2 = max(x2)-min(x2); 
X = data(:,2:3);
N = length(X);
K = 2; % K > 2 also seems to work.
D = 2;

% Contour Plot variables
g1 = linspace(min(x1),max(x1));
g2 = linspace(min(x2),max(x2));
[G1, G2] = meshgrid(g1, g2);
G = [G1(:) G2(:)]; 

% Initial MU, SIGMA and PI
MU = zeros(K,D);
SIGMA = zeros(D,D,K);
PI = zeros(K,1);
for j = 1:K    
    m1 = min(x1) + range1*rand;
    m2 = min(x2) + range2*rand;
    MU(j,:) = [ m1 m2 ];
  
    s1 = range1/range2;
    s2 = range2/range1;
    SIGMA(:,:,j) = [ s1 0 ; 0 s2 ].*rand;
    
    if j ~= K
        PI(j,:) = (1-sum(PI))*rand;
    else
        PI(j,:) = 1-sum(PI);
    end
end

% EM algorithm
old_MU = zeros(K, D);
t_count = 0;
fig = 1;
% Initialize responsibilities.
respnum = zeros(1, K); % numerator for r
resp = zeros(N,K);
while (sum(sum(abs(MU - old_MU))) > 1e-2)    

    % Optional: Plot the state of the algorithm
%     t_count = t_count + 1;  
%     if t_count == 1 || t_count == 2 ...
%             || t_count == 3 || t_count == 15 ...
%             || t_count == 30 || t_count == 60
%         % Scatter Plot
%         figure(fig); clf;
%         scatter(x1, x2); axis square; box on
%         xlabel('eruption duration (min)'); 
%         ylabel('time to next eruption (min)');
% 
%         % Contour Plot
%         for j = 1:K
%             g = mvnpdf( G, MU(j,:), SIGMA(:,:,j));   
%             res = reshape(g, length(g1), length(g2));
%             figure(fig); hold on
%             contour(G1, G2, res);
%         end
%         fig = fig + 1;
%     end
           
    % E step
    for n = 1:N
        for j = 1:K
            f = mvnpdf( X(n,:), MU(j,:), SIGMA(:,:,j));    
            respnum(j) = PI(j,:)*f;
        end
        respden = sum(respnum);
        for j = 1:K
            resp(n,j) = respnum(j)/respden;
        end        
    end
     
    % M step    
     old_MU = MU;
     old_SIGMA = SIGMA;
     old_PI = PI;
    for j = 1:K        
        Nk = sum(resp(:,j));
        
        % new MU
        MU(j,:) = resp(:,j)'*X./Nk;
        
        % new SIG
        V = zeros(2, N);
        for i = 1:N
            V(:,i) = X(i,:) - mean(X);
        end
        Vresp = V.*resp(:,j)';
        % 2x272 double "V" should have each element
        % multiplied by 1x272 double "resp(:,j)",
        % resulting in 2x272 double.
        
        VrespVT = Vresp*V'; 
        % 2x2 matrix results from cross product
        % of prev result and its transpose.
        
        SIGMA(:,:,j) = VrespVT./Nk;
        % 2x2 matrix is divided by Nk to give the cov
        % matrix for the current K-gaussian.
        
        % new PI
        PI(j,:) = Nk/N;
    end
end % while end

% Final Plot
% Scatter Plot
figure(fig); clf;
scatter(x1, x2); axis square; box on
xlabel('eruption duration (min)'); 
ylabel('time to next eruption (min)');

% Contour Plot
for j = 1:K
    g = mvnpdf( G, MU(j,:), SIGMA(:,:,j));   
    res = reshape(g, length(g1), length(g2));
    figure(fig); hold on
    contour(G1, G2, res);
end 