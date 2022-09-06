%% Compute a vlad descriptors from a set of local descriptors
% Usage v = mean_vlad(centroids, s)
% where
%   centroids is the dictionary of centroids 
%   s is the set of descriptors
%
% Both centroids and descriptors are stored per column

function fine_grain_v = mean_vlad(centroids, s)

n = size (s, 1);          % number of descriptors
d = size (s, 2);          % descriptor dimensionality
k = size (centroids, 2);  % number of centroids

alpha = 0.5;              % Power Law parameter

%% create an empty 0-by-0 cell array
%residual_all = {};

%% RootSIFT
% s - sift descriptor, root_s - RootSIFT descriptor

%data = s'; % 128x1000
%root_sift = sqrt(data/sum(data));

%s=root_sift';

%dat=s;
%sum_val = sum(dat);
%for r = 1:1000
%    dat(r, :) = dat(r, :)./sum_val;
%end
%root_s = single(sqrt(dat));



%% find the nearest neigbhors for each descriptor

%[idx, dis] = yael_nn (centroids, root_s');

%%%%%%%%%%%%% MODIFICATION HERE FROM ME in lieu of Yael_nn () function %%%%%%%%%%%%%%%

%% create an assignment matrix, which has the dimensions NumberOfClusters-by-NumberOfDescriptors, % which assigns each descriptor to a cluster.
% e.g. use kd-trees
% vl_kdtreebuild returns a structure 'kdtree' 
kdtree = vl_kdtreebuild(centroids); 

s_new = single(s'); %

% Create assignment matrix
idx = vl_kdtreequery(kdtree, centroids, s_new); % nn - matrix with assignments

%%[nn, distx] = vl_kdtreequery(kdtree, centroids, single(sift_descr{k}));

assignments = zeros(64, numel(idx), 'single');
assignments(sub2ind(size(assignments), idx, 1:numel(idx))) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

v = zeros (d, k);
fine_grain_v = zeros (d, k);
%% create an empty 0-by-0 cell array
%residual_all = {};

% Aggregating residuals
for i = 1:n
  % Pure VLAD  
  res = (s_new(:, i) - centroids (:, idx(i)));
  v (:, idx(i)) = v (:, idx(i)) + res;
  residual(:,i) = res;
    
  %Residual Normalization
  %v (:, idx(i)) = v (:, idx(i)) + (s_new(:, i) - centroids (:, idx(i)))./norm(s_new(:, i) - centroids (:, idx(i)));
end

%residual_all=residual;

%for i = 1:n
%    residual_all{:,idx(i)} = residual(:,i);
%end  
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a)  Intra-normalization for VLAD
% Take unnormalized VLAD v
%for i = 1:k
%  v (i,:) = v (i,:) ./ norm(v(i,:));
%end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Concatenating all k subvectors/LDVs
% Unnormalised VLAD (k*d x 1)
%v = reshape (v, k*d, 1); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sign square root normalized VLAD
%for i = 1:(k*d)
%  v(i) = sign(v(i)) * sqrt(norm(v(i)));
%end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Storing residuals of each centroid in cell arrays
S_residual = mat2cell(residual, 128, ones(n,1));
%S_new_residual = accumarray(idx(:),1:n,[64 1],@(x){residual(:,x)});
s=s';
% Obtain descriptors in each cluster 
S_new_descriptors = accumarray(idx(:),1:n,[64 1],@(x){s(:,x)}); 

%for f = 1:64 % if else 
%  fgv(:,f) = fgvlad(centroids, f, S_new_residual{f});
%end

%%%%%%%%%%%%%%%%%%%%%%%%%% FgVLAD %%%%%%%%%%%%%%%%%%%%%%%%%%

for f = 1:64 
  if isempty(S_new_descriptors{f})
  % a cell may be empty
  mean_pool(:,f) = zeros(128,1,'single');
  sum_pool(:,f) = zeros(128,1,'single');
  else
  % compute mean of each cluster
  mean_pool(:,f) = mean(S_new_descriptors{f},2);
  sum_pool(:,f) = sum(S_new_descriptors{f},2);
  end 
end

% Aggregating residuals
%for i = 1:n
  % FgVLAD  
  % res = ((s_new(:, i) - mean_pool (:, idx(i)))) + (centroids(:,idx(i)) - mean_pool(:, idx(i)));
  % v_f (:, idx(i)) = v_f (:, idx(i)) + res;
  %residual(:,i) = res;
% end

for f = 1:64 % if else 
  %fgv{f,:} = fgvlad(mean_pool(:,f), centroids(:,f), f, S_new_descriptors{f});
  fine_grain_v(:, f) = fgvlad(mean_pool(:,f), sum_pool(:,f), centroids(:,f), f, S_new_descriptors{f});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Concatenating all k subvectors/LDVs
% Unnormalised VLAD (k*d x 1)
 %v_adapt = reshape (v_adapt, k*d, 1); 
 fine_grain_v = reshape (fine_grain_v, k*d, 1); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a) Power normalization 
for i = 1:(k*d)
  fine_grain_v(i) = sign(fine_grain_v(i)) * (norm(fine_grain_v(i)))^alpha;
end

%a)  Intra-normalization for VLAD+ADAPT
% Take unnormalized VLAD v
%for i = 1:k
%  v_adapt (i,:) = v_adapt (i,:) ./ norm(v_adapt(i,:));
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% b) L2 normalization
if norm (fine_grain_v) == 0
   fine_grain_v = ones (d * k, 1);
else
   fine_grain_v =  fine_grain_v ./ norm(fine_grain_v);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

