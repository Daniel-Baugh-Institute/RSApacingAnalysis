%% Code to reproduce figures in Ch. 6 of Modeling and analysis of autonomic circuits underlying closed-loop cardiovascular control to identify key contributors to individual variability
% Michelle Gee
% August 2, 2025
% NOTE: You will need the mhrv toolbox and you will need to add the path of
% the mhrv toolbox to hrv_analysis

% Set up paths
clear; close all; restoredefaultpath;
my_dir = pwd;
addpath(genpath(my_dir))
% path to mhrv package
% addpath(genpath("/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/mhrv-master/"))

% local file paths
% cd 'C:\Users\mmgee\OneDrive - University of Delaware - o365\Documents\Github\NZ-physiology-data'
% addpath(genpath('C:\Users\mmgee\AppData\Local\Temp\Mxt231\RemoteFiles'))
% addpath 'C:\Users\mmgee\OneDrive - University of Delaware - o365\Documents\Github\NZ-physiology-data'

%% Load and preprocess data to save to smaller file
% Note, you can skip this step and load the processed data

filesToRead = {'1909 baseline 3.mat', '1909 RSA day 14.mat', '1909 post RSA day 2.mat', ...
    '2037 HF baseline.mat', '2037 RSA day 12.mat',... % 
    '2048 day 5 baseline.mat', '2048 RSA day 14 no CoBF.mat', ...
    '2102 baseline day 15.mat', '2102 RSA day 14.mat', ...
    '2123 baseline day 4.mat', '2123 RSA day 11.mat', ...
    '2125 baseline day 4.mat', '2125 RSA day 11.mat',...
    '2232 baseline day5.mat','2232 mono day 11.mat', ...
    '2236 baseline day 6.mat','2236 mono day 12.mat',...
    '2229 baseline day 7.mat', '2229 mono day 7 CoBF poor signal.mat',...
    '2228 baseline day 7.mat', '2228 mono day 12.mat',...
    '1829 baseline day 10.mat', '1829 mono day 12.mat', '1829 post mono day 2.mat',...
    '1833 baseline day 11.mat', '1833 mono day 13.mat'};



num_subjects = length(filesToRead);

% Pull data out from each animal's file and combine
% Struct data has fields HR, CO, CoBF, BP, timeRaw for each animal
% (data(1), data(2),...)

% Run data extraction in parallel
saveFilePrefix = '/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/data-processing/combinedData_all_30m_48slices_042125';
saveFileName = [saveFilePrefix '.mat'];
% data = combineSamples(filesToRead,saveFilePrefix);
% save(saveFileName,'data','-v7.3')
    
% To directly load the data, run this line instead
load(saveFileName)



%% Analysis and code to reproduce figures

%% Plot RR intervals show heart rate fragmentation segments (Figure 2)
fields = fieldnames(data)

% Heart failure example
% Find acceleration, deceleration, and alternation segments
nni = data(1,11).RRint(100:150);
nn_times = data(1,11).RRtime(100:150);
save('./plots/frag_HF1.mat',"nn_times","nni")
[ hrv_frag, acceleration_segment_boundaries_3plus, alternation_segment_boundaries_4plus ] = mhrv.hrv.hrv_fragmentation( nni );
save('./plots/plot_RR_frag_HF1.mat','acceleration_segment_boundaries_3plus','alternation_segment_boundaries_4plus','nni','nn_times')

% Plot
filename = 'plot_RR_frag_HF1.png';
plotFrag(nni, nn_times, acceleration_segment_boundaries_3plus, alternation_segment_boundaries_4plus,filename)

% Control example
% Find acceleration, deceleration, and alternation segments
nni = data(1,9).RRint(100:150);
nn_times = data(1,9).RRtime(100:150);
save('./plots/frag_C9.mat',"nn_times","nni")
[ hrv_frag, acceleration_segment_boundaries_3plus, alternation_segment_boundaries_4plus ] = mhrv.hrv.hrv_fragmentation( nni );
save('./plots/plot_RR_frag_C9.mat','acceleration_segment_boundaries_3plus','alternation_segment_boundaries_4plus','nni','nn_times')

% Plot
filename = 'plot_RR_frag_C9.png';
plotFrag(nni, nn_times, acceleration_segment_boundaries_3plus, alternation_segment_boundaries_4plus,filename)



%% Calculate cardiac efficiency (Figure 3)
restoredefaultpath
my_dir = pwd
addpath(genpath(my_dir))
[work_per_beat, efficiency_per_beat, work_mean, efficiency_mean, work_std, efficiency_std] = calcEfficiency(data);
save('efficiency_30m_all_CO-work.mat','work_per_beat','efficiency_per_beat','work_mean','efficiency_mean','work_std','efficiency_std','-v7.3')

% Plot change in efficiency from unpaced to paced
control_flag = 0; % not plotting healthy animals
healthy_mean = [];
healthy_std = [];
plot_paced_efficiency(work_mean, efficiency_mean, work_std, efficiency_std,healthy_mean, healthy_std,control_flag)


%% HRV analysis (Figure 4)
hrv(data)

%% Segment efficiency (Figure 5)
my_dir = pwd
addpath(genpath(my_dir))
[work_segment_accel, work_segment_alt, efficiency_segment_accel, efficiency_segment_alt] = calcEfficiencyFrag(data(:,1:7));











