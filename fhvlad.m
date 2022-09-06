%% Compute a fgvlad descriptors from a set of local descriptors in a cell
% Usage fine_res_vlad = fgvlad (mean_cell, sum_cell, centroid_cell, f, S)
% Both centroids and descriptors are stored per column

function fine_res_vlad = fgvlad (mean_cell, sum_cell, centroid_cell, f, S)

dimensions = size (S, 1);              % descriptor dimensionality
numDescriptors = size (S, 2);          % number of descriptors 

numCenters = 16; % 64 ; %K = number of clusters 
numTrials = 20 ;
maxNumIterations = 20 ; % 20
distance = 'l2' ;

fine_res_vlad = zeros (128, 1);
%fgv = zeros (dimensions, 1);
%% Normalizing 
% Normalizing mean_cell
if norm (mean_cell) == 0
   mean_cell = zeros(128, 1);
else
   mean_cell =  mean_cell ./ norm(mean_cell);
end

% Normalizing sum_cell
if norm (sum_cell) == 0
   sum_cell = zeros(128, 1);
else
   sum_cell =  sum_cell ./ norm(sum_cell);
end

%% Fine-grained concept
if (numCenters < numDescriptors)
%% Run ANN k-means algorithm on the data
[C, A, E] = vl_kmeans(S, ...
                      numCenters, 'Verbose', ...
                      'Distance', distance, ...
                      'MaxNumIterations', maxNumIterations, ...
                      'Algorithm', 'ANN', 'MaxNumComparisons', ceil(numCenters / 50));    
                  
 for i = 1:numDescriptors
    
    fine_res(:,i) = (centroid_cell - S(:,i)) - (S(:,i) - C(:,A(i))); 
    fine_res_vlad(:,1) = fine_res_vlad(:,1) + fine_res(:,i);
end 

elseif isempty(S)
    fine_res(:,1) = zeros(128,1,'single');
    fine_res_vlad(:,1) = zeros(128,1,'single');
        
else       
    for i = 1:numDescriptors
    fine_res(:,i) = (centroid_cell - mean_cell); 
    fine_res_vlad(:,1) = fine_res_vlad(:,1) + fine_res(:,i);
    end
end 

end
