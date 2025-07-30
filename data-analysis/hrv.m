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
% CHECK num_subjects OVERRIDE
% hrv = hrv_analysis(data);%hrv_analysis(data,varargin);
% load('PIP3plus.mat','hrv')
% save('hrv_S33_offPaceMono.mat','hrv')
% save('hrv_window1.mat','hrv')
% load('hrv_frag_filtered.mat','hrv')
% save('hrv_paced_test.mat',"hrv")
% load('hrv_30min_paced_samples_baseline.mat','hrv') % only paced samples
% hrv_paced = hrv;
load('hrv_30min_all_baseline.mat','hrv') % only baseline samples
% hrv_baseline = hrv;
% base_idx = [1:10 13 15 17 19 21 23 25 27 29 31 34]; 
% paced_idx = [11 14 16 18 20 22 24 26 28 30 32 35]; % paced
% 
% 
% hrv_combined(:,base_idx) = hrv_baseline(:,base_idx);
% hrv_combined(:,paced_idx) = hrv_paced(:,paced_idx);
% load('hrv_combined.mat','hrv_combined')
% hrv = hrv_combined;
% hrv_combined(:,33) = hrv(:,2);
% hrv_combined(:,12) = hrv(:,1);
% hrv = hrv_combined;
% save('hrv_combined.mat','hrv_combined')

% Compare HRV metrics for HF and control animals
% CURRENT NUM_SLICES OVERRIDE
hrv_plot(hrv) % set varargin to 1 in hrv_analysis to get suitable hrv input

% Plot HRV metrics as timeseries 
%hrv_timeseries(hrv)

% plot PIP change from baseline to pacing
% [num_slices,num_subjects] = size(hrv)
% % hrv(1,1).hrv_frag(1,1)
% for i = 1:num_subjects
%     for j = 1:num_slices
% 
%         if ~isempty(hrv(j,i).hrv_frag)
%             a = hrv(j,i).hrv_frag(1,1);
%             try
%                 subject_pip(j) = table2array(a);
%             catch
%                 disp('data missing')
%                 a
%             end
%         else
%             disp('Missing data probably?')
%             i
%             j
% 
%             subject_pip(j) = NaN;
%         end
%     end
%     rmIdx = isnan(subject_pip);
%     subject_pip_clean = subject_pip;
%     subject_pip_clean(rmIdx) = [];
%     % subject_pip_clean;
%     pip_mean(i) = mean(subject_pip_clean,'omitmissing');
%     pip_std(i) = std(subject_pip_clean);
% end
% work_mean = zeros(1,length(pip_mean));
% work_std = zeros(1,length(pip_std));
% pip_mean(13)
% pip_mean(33)
% 
% healthy_mean = pip_mean(5:9)
% healthy_std = pip_std(5:9)
% 
% control_flag = 1;
% plot_paced_efficiency(work_mean, pip_mean, work_std, pip_std,healthy_mean, healthy_std,control_flag)

end