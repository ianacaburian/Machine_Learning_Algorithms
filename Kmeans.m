% Load and plot "old faithful" dataset
data = dlmread('old_faithful.dat', '\t', 26, 0);
x1 = data(:,2); x2 = data(:,3);
figure(1); clf
scatter(x1, x2); axis square; box on
xlabel('eruption duration (min)'); ylabel('time to next eruption (min)')

%%%% The K-means algorithm %%%%
% define distance function
dist = @(x,y) 1/2*sum((x - y).^2);
% Initialization
K = 3;
N = length(data);
D = 2;
% Create a matrix of means initialized by a random scalar based
% on the range of the data.
m1 = (max(x1)-min(x1))*rand(K,1) + min(x1);
m2 = (max(x2)-min(x2))*rand(K,1) + min(x2);
means = [m1 m2];

% This is the matrix used to compare means that determines
% the loop break.
old_means = zeros(K, D);

% Repeat until no change in means
% Sums the the vectors of the difference in matrices,
% Break if within 1e-2
while (sum(sum(abs(means - old_means))) > 1e-2)

	% Assignment step
    % Initializes matrix of responsibilities, i.e. 0 or 1 for each
    % data point N per K-mean.
	resp = zeros(N,K);
    old_means = means;
    
    
	for n = 1:N % for each data point
		distance = zeros(1,K); % distance vector for all K-means
		for j = 1:K % for each K-mean find the distance to the n data point
			distance(j) = dist(means(j,:), data(n, 2:3)); 
        end
        % tuple assignment, minimum distance value assigned to val,
        % its corresponding cell in distance vector is assigned to k.
		[val, k] = min(distance);
        % the corresponding k's indicator variable is assigned a 1
        % the other 2 k's in the current loop's row remain 0.
		resp(n, k) = 1; 
	end

	% Update step
	for k = 1:K
		for d = 1:D
            % sum only all the data points for the current K-mean
			means(k, d) = sum(resp(:,k).*data(:,d+1)); % d+1 to ignore col 1.
        end
        % divide the sums by the total number of data points
        % for the corresponding K-mean
		means(k,:) = means(k,:)/sum(resp(:,k));
	end
end
%%%% end of algorithm %%%%

% Plot final means
figure(1); hold on
plot(means(:,1), means(:,2), 'ksq' , 'markersize', 15)
