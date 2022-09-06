% Compute the set of VLAD descriptors for a set of images
% Usage: V = compute_vlad (centroids, S)
%
% where
%   centroids is the dictionary of centroids 
%   S is a cell structure. Each cell is a set of descriptors for an image
%
% Both centroids and descriptors are stored per column
function V = compute_vlad (centroids, S)

%fprintf ('Inside compute_vlad Pt1 \r\n');

nimg = length (S);
k = size (centroids, 2);
d = size (centroids, 1);
%k = 2*k;
k = 66048; %33024; % 2064; % 8256; %8320; 
%V = zeros (k * d, nimg, 'single');
V = zeros (k, nimg, 'single');

for i = 1:nimg
  %V(:, i) = vlad (centroids, S{i}); % original code
  % V(:, i) = vlad_k (centroids, S{i}); % new code
  fprintf ('HOVLAD - image = %f\n', i);
  %V(:, i) = minx_pooled_vlad (centroids, S{i}); % minx-pooled-vlad
  %V(:, i) = sumx_pooled_vlad (centroids, S{i}); % sumx-pooled-vlad
  %V(:, i) = sumpooled_desc_vlad (centroids, S{i}); % sumpooled_desc_vlad
  % V(:, i) = mean_pooled_vlad (centroids, S{i}); % mean_pooled_vlad
  %V(:, i) =  mean_vlad_adapt (centroids, S{i}); % mean_vlad_adapt - CIPR
  %V(:, i) =  mean_vlad(centroids, S{i}); % mean_vlad_adapt - Journal FgVLAD
   V(:, i) =  fgvlad (centroids, S{i});
  %fprintf ('Inside compute_vlad Pt2 \r\n');
end
