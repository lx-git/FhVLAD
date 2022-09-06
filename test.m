%----------------------------------------------------------------------------
% This test program reproduces the results obtained for the method 
% presented in the paper:
% "Aggregating local descriptors into a compact image reprensentation"
%
% Authors: Herve Jegou and Matthijs Douze
% Contact: herve.jegou@inria.fr and matthijs.douze@inria.fr
% Copyright INRIA 2010
%
% Licence: 
% This software and its subprograms are governed by the CeCILL license 
% under French law and abiding by the rules of distribution of free software. 
% See http://www.cecill.info/licences.en.html

% Various directories for mandatory libraries:
%   Yael and its Matlab interface should be compiled (mexfiles)
%   Pqcodes requires to compile sumidxtab (mex sumidxtab.c)
%----------------------------------------------------------------------------

dir_pqcodes = './pqcodes/';
dir_yael = './yael/';
dir_sift = './siftgeo/';
dir_data = './data/';

addpath ([dir_yael '/matlab']);
addpath (dir_pqcodes);

f_centroids = [dir_data 'clust_k64.fvecs'];    % given in the package
f_vlad = [dir_data 'vlad_k64_holidays.fvecs']; % computed
f_pca_proj = [dir_data 'pca_proj_matrix_vladk64_flickr1Mstar.fvecs']; %proj matrix

do_compute_vlad = false;          % compute vlads or use the pre-compiled ones

% Parameters
shortlistsize = 1000;             % number of elements ranked by the system
dd = 128;                         % number of components kept by PCA transform

nsq = 16;                         % ADC: number of subquantizers
nsq_bits = 8;                     % ADC: number of bit/subquantizer


%----------------------------------------------------------------------------
% Retrieve the list of images and construct the groundtruth
[imlist, sift, gnd, qidx] = load_holidays (dir_sift);

% compute or load the VLAD descriptors
if do_compute_vlad                 % compute VLADs from SIFT descriptors
  centroids = fvecs_read (f_centroids);
  v = compute_vlad (centroids, sift); 
else                               % load them from disk
  v = fvecs_read (f_vlad);
end
d_vlad = size (v, 1);              % dimension of the vlad vectors
  

%----------------------------------------------------------------------------
% Full VLAD
% perform the queries (without product quantization nor PCA) and find 
% the rank of the tp. Keep only top results (i.e., keep shortlistsize results). 
% for exact mAP, replace following line by k = length (imlist)
vn = yael_fvecs_normalize (v);
[idx, dis] = yael_nn (vn, vn(:,qidx), shortlistsize + 1);
idx = idx (2:end,:);  % remove the query from the ranking

map_vlad = compute_results (idx, gnd);
fprintf ('full VLAD.                           mAP = %.3f\n', map_vlad);

%----------------------------------------------------------------------------
% VLAD with PCA projection
% perform the PCA projection, and keep dd components only
f = fopen (f_pca_proj);
mu = fvec_read (f);     % mean. Note that VLAD are already almost centered. 
pca_proj = fvec_read (f);
pca_proj = reshape (pca_proj, d_vlad, 1024)'; % only the 1024 eigenvectors are stored
fclose (f);
pca_proj = pca_proj (1:dd,:);

% project the descriptors and compute the results after PCA
vp = pca_proj * (v - repmat (mu, 1, size (v,2)));
vp = yael_fvecs_normalize (vp);

[idx, dis] = yael_nn (vp, vp(:,qidx), shortlistsize + 1);
idx = idx (2:end,:);  % remove the query from the ranking

map_vlad_pca = compute_results (idx, gnd);
fprintf ('PCA VLAD (D''=%d)                    mAP = %.3f\n', dd, map_vlad_pca);


%----------------------------------------------------------------------------
% VLAD with PCA projection and PQcodes (ADC variant)
% Here, the PQcodes are learned on the Holidays dataset (not in our paper) 
% However, we observe comparable results for both learning sets. 
% Note: due to the randomness, the mAP varies from an experiment to another

% whitening with a random orthogonal matrix
Q = randn (dd);
Q = single (Q);
Q = yael_fvecs_normalize (Q)';
Q = yael_fvecs_normalize (Q)';

vpq = Q * vp;

% learn the pqcodes and encode the vectors
pq = pq_new (nsq, nsq_bits, vpq);
vpqcoded = pq_assign (pq, vpq);

% Perform the search
[idx, dis] = pq_search (pq, vpqcoded, vpq(:, qidx), shortlistsize);
idx = idx';
idx = idx (2:end, :);

map_vlad_pca_adc = compute_results (idx, gnd);
fprintf ('full VLAD.                           mAP = %.3f\n', map_vlad);
fprintf ('PCA VLAD (D''=%d)                    mAP = %.3f\n', dd, map_vlad_pca);
fprintf ('PCA+ADC VLAD (D''=%d, %d bytes/img)  mAP = %.3f\n', dd, nsq*nsq_bits/8, map_vlad_pca_adc);
