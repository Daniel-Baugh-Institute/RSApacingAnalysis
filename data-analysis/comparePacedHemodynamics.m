function [CoBF_mean, CO_mean, RR_mean, MAP_mean, CoBF_std, CO_std, RR_std, MAP_std] = comparePacedHemodynamics(data)
% Function to calculate mean values and standard deviations for hemodynamic
% data. These data are fed into plot_paced_efficiency.m for data
% visualization.
% Input: data, MxN struct where the row is the time sample and the column is
%   the animal number
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
rng default
[num_slices, num_subjects] = size(data)
num_slices = num_slices;


% Preallocate
CO_mean = zeros(1,num_subjects);
CO_std = zeros(1,num_subjects);
CO_window_mean = zeros(num_slices,num_subjects);

CoBF_mean = zeros(1,num_subjects);
CoBF_std = zeros(1,num_subjects);
CoBF_window_mean = zeros(num_slices,num_subjects);

RR_mean = zeros(1,num_subjects);
RR_std = zeros(1,num_subjects);
RR_window_mean = zeros(num_slices,num_subjects);

MAP_mean = zeros(1,num_subjects);
MAP_std = zeros(1,num_subjects);
MAP_window_mean = zeros(num_slices,num_subjects);


for jj = 1:num_subjects
    
    for i = 1:num_slices

        % CO
        CO_window_mean(i,jj) = mean(data(i,jj).CO,'omitmissing');

        % CoBF
        CoBF_window_mean(i,jj) = mean(data(i,jj).CoBF,'omitmissing');


        % RR
        RR_window_mean(i,jj) = mean(data(i,jj).RRint,'omitmissing');
        

        % MAP
        MAP_window_mean(i,jj) = mean(data(i,jj).MAP,'omitmissing');
        
    end

    % CO
    CO_mean(jj) = mean(CO_window_mean(:,jj),'omitmissing');
    CO_window_mean_idx = ~isnan(CO_window_mean(:,jj));
    CO_window_mean_clean = CO_window_mean(CO_window_mean_idx,jj);
    CO_std(jj) = std(CO_window_mean_clean);

    % CoBF
    CoBF_mean(jj) = mean(CoBF_window_mean(:,jj),'omitmissing');
    CoBF_window_mean_idx = ~isnan(CoBF_window_mean(:,jj));
    CoBF_window_mean_clean = CoBF_window_mean(CoBF_window_mean_idx,jj);
    CoBF_std(jj) = std(CoBF_window_mean_clean);

    % RR
    RR_mean(jj) = mean(RR_window_mean(:,jj),'omitmissing');
    RR_window_mean_idx = ~isnan(RR_window_mean(:,jj));
    RR_window_mean_clean = RR_window_mean(RR_window_mean_idx,jj);
    RR_std(jj) = std(RR_window_mean_clean);

    % MAP
    MAP_mean(jj) = mean(MAP_window_mean(:,jj),'omitmissing');
    MAP_window_mean_idx = ~isnan(MAP_window_mean(:,jj));
    MAP_window_mean_clean = MAP_window_mean(MAP_window_mean_idx,jj);
    MAP_std(jj) = std(MAP_window_mean_clean);
end

save('hemodynamic_avg_30m_all.mat','CO_mean',"CO_std","CoBF_mean","CoBF_std","RR_mean","RR_std","MAP_mean","MAP_std")

% Statistical comparisons done in plot_paced_effciency.m

end