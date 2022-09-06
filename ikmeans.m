%% Clear your command window, clear variables, breakpoints, cached memory, close all figures.
clc;
clear all;
close all;

% SETUP  Add the required search paths to MATLAB
%if exist('vl_version') ~= 3, run('vlfeat/toolbox/vl_setup') ; end
run('D:\vlfeat-0.9.20-bin\vlfeat-0.9.20\toolbox\vl_setup.m');

K = 3 ;
data = uint8(rand(128,1000) * 255) ;
[C,A] = vl_ikmeans(data,K) ;

datat = uint8(rand(2,10000) * 255) ;
AT = vl_ikmeanspush(datat,C) ;

cl = get(gca,'ColorOrder') ;
ncl = size(cl,1) ;
for k=1:K
  sel  = find(A  == k) ;
  selt = find(AT == k) ;
  plot(data(1,sel),  data(2,sel),  '.',...
       'Color',cl(mod(k,ncl)+1,:)) ;
  hold on ;
  plot(datat(1,selt),datat(2,selt),'+',...
       'Color',cl(mod(k,ncl)+1,:)) ;
end