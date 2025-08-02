function hrv(data)
% Function to calculate HRV metrics
% Input: data, MxN struct where the row is the time sample and the column is
%   the animal number


% Adapted from:
% * Rosenberg, A. A. (2018) ‘Non-invasive in-vivo analysis of intrinsic clock-like
%   pacemaker mechanisms: Decoupling neural input using heart rate variability
%   measurements.’ MSc Thesis. Technion, Israel Institute of Technology.
% 
% * Behar J. A., Rosenberg A. A. et al. (2018) ‘PhysioZoo: a novel open access
%   platform for heart rate variability analysis of mammalian
%   electrocardiographic data.’ Frontiers in Physiology.


my_dir = pwd
addpath(genpath(my_dir))


hrv = hrv_analysis(data);
save('hrv.mat','hrv')

hrv_plot(hrv)


end