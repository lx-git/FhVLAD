% Author: Alexy Bhowmick
%
% FhVLAD algorithm testing on Holidays dataset SIFT descriptors and frames
% extracted by AB.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clear your command window, clear variables, breakpoints, cached memory, close all figures.
tic;
clc;
clear all;
close all;

%% Set up VlFeat
setup;

%mex siftgeo_read_fast.c;
fprintf('Driver code - Large Scale Image Retrieval - G1, T1\n');

dir_sift = './siftgeo/';
dir_data = './data/';

%% import the visual vocabulary
%  load Centroidi.mat
f_centroids = [dir_data 'clust_flickr60_k1000.fvecs'];
do_compute_vlad = true;          % compute vlads or use the pre-compiled ones

% Parameters
shortlistsize = 1000;             % number of elements ranked by the system

%% Retrieve the list of images (Holidays dataset) and construct the groundtruth
[imlist, sift, gnd, qidx] = load_holidays (dir_sift);

%%%%%%%%%%%%%%%%%% Inserting 3 missing values in Holidays %%%%%%%%%%%%%%%%
load d_125800;
load d_125801;
load d_125802;

d1 = rand500_sift_desc_125800';
d2 = rand500_sift_desc_125801';
d3 = rand500_sift_desc_125802';

sift{697,1} =  d1;
sift{698,1} =  d2;
sift{699,1} =  d3;

save complete_sift_Holidays.mat sift -v7.3

%% compute or load the VLAD descriptors
if do_compute_vlad                 % compute VLADs from SIFT descriptors
  centroids = fvecs_read (f_centroids);
  centroids_512 = centroids(:,[01:512]); % 0.591 + Use clust_flickr60_k5000.fvecs
  tic;
  v = compute_vlad (centroids_512, sift); 
else                               % load them from disk
  v = fvecs_read (f_vlad);
end

d_vlad = size (v, 1);              % dimension of the vlad vectors

%----------------------------------------------------------------------------
% Full VLAD
% perform the queries (without product quantization nor PCA) and find 
% the rank of the tp. Keep only top results (i.e., keep shortlistsize results). 
% for exact mAP, replace following line by k = length (imlist)

%vn = yael_fvecs_normalize (v);
vn = v;

[idx, dis] = yael_nn (vn, vn(:,qidx), shortlistsize + 1);
idx = idx (2:end,:);  % remove the query from the ranking

map_vlad = compute_results (idx, gnd);
fprintf ('full VLAD.                           mAP = %.3f\n', map_vlad);;
toc;

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
