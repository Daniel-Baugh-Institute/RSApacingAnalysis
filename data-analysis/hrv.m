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


addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))

% varargin = 19;
hrv = hrv_analysis(data);%hrv_analysis(data,varargin);
save('hrv_30min_all_baseline.mat','hrv')
% load('hrv_30min_paced_baseline.mat','hrv')

% Compare HRV metrics for HF and control animals
hrv_plot(hrv) % set varargin to 1 in hrv_analysis to get suitable hrv input

% Plot HRV metrics as timeseries 
hrv_timeseries(hrv)
end