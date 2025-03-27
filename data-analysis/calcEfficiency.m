function [work_per_beat, efficiency_per_beat, work_mean, efficiency_mean, work_std, efficiency_std] = calcEfficiency(data)
% Function to calculate beat to beat cardiac work from cardiac output and
% arterial pressure
% Input: data, MxN struct where the row is the time sample and the column is
%   the animal number
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
rng default
[num_slices, num_subjects] = size(data)
num_slices = num_slices;


% SV = integral of CO * RR interval
% E = integral of BP * SV

% Preallocate
CO = struct([]);
BP = struct([]);
CoBF = struct([]);
SV = struct([]);

nb = 10; % number of heart beats for calculation
for jj = 1:num_subjects % issue with subject 6?
    for i = 1:num_slices

        % randomly sample RR interval
        sampleIdx = randi((length(data(i,jj).RRint) - 1),[nb,1]);
        RRsample = data(i,jj).RRint(sampleIdx); % s

        % Extract out RR, CO, BP for randomly sampled beats
        RR_start = data(i,jj).RRtime(sampleIdx); %s
        RR_stop = data(i,jj).RRtime(sampleIdx+1); % s

        s = RR_start; % sec
        e = RR_stop; % sec

        times_BP = data(i,jj).timeHRs_BP*3600; % convert to s
        times_CO = data(i,jj).timeHRs_CO*3600; % convert to s

        % TODO: because negative CO values are removed from the data during
        % data cleaning, the number of points in CO and BP are not the same
        % (~ 20 points less in CO than BP). For now, I'm just going to
        % truncate the last 20 points of BP
        for k = 1:nb
            % Extract BP and CO values for the beat
            BP_vals = data(i,jj).BP(times_BP > s(k) & times_BP < e(k)); % mmHg
            CO_vals = data(i,jj).CO(times_CO > s(k) & times_CO < e(k))/60/1000; % Convert CO to mL/s
            time_CO = times_CO(times_CO > s(k) & times_CO < e(k));

            % Calculate cumulative volume using cumulative integral
            if ~isempty(CO_vals) && numel(CO_vals) > 1
                vol_vals = cumtrapz(time_CO, CO_vals); % mL
            else
                vol_vals = NaN;
                disp('Warning: CO vector was empty or only has one data point');
            end

            % Compute Work per Beat using Pressure-Volume Loop
            sizediff = numel(BP_vals) - numel(CO_vals);
            % size(times_BP)
            % size(CO_vals)
            % size(time_CO)
            % size(vol_vals)
            if ~isempty(BP_vals) && numel(BP_vals) > 1 && ~any(isnan(vol_vals))
                work_per_beat_sample(k) = trapz(vol_vals, BP_vals(1:end-sizediff)); % mL * mmHg
            else
                work_per_beat_sample(k) = NaN;
                disp('Warning: Unable to calculate Work per Beat');
            end

            % Calculate Efficiency (Work/CoBF)
            % NOTE: Is this the best measure of efficiency?
            CoBF_vals = data(i,jj).CoBF(times_CO > s(k) & times_CO < e(k))/60; % Convert to mL/s
            if ~isempty(CoBF_vals) && numel(CoBF_vals) > 1
                efficiency_per_beat_sample(k) = work_per_beat_sample(k) / trapz(time_CO, CoBF_vals);
            else
                efficiency_per_beat_sample(k) = NaN;
                disp('Warning: Unable to calculate Efficiency per Beat');
            end
        end

    
        work_per_beat(i,jj) = mean(work_per_beat_sample,'all','omitmissing');
        efficiency_per_beat(i,jj)= mean(efficiency_per_beat_sample,'all','omitmissing');
    end

    
    
    % Average and std work and efficiency for each animal
    % TODO: this only samples the last i, k
    work_mean(jj) = mean(work_per_beat(:,jj),'all','omitmissing');
    efficiency_mean(jj) = mean(efficiency_per_beat(:,jj),'all','omitmissing');
    work_std(jj) = std(work_per_beat(:,jj));
    efficiency_std(jj) = std(efficiency_per_beat(:,jj));

end




% Compare work and efficiency for HF and control animals
HFidx = 1:5;
ctrlIdx = 6:9;

disp('h = 0 means no evidence that there are differences between groups')




% plot histogram and do t-test comparing distributions for HF and control
plotHF = work_mean(HFidx);%work_per_beat(:,HFidx);
plotHF = plotHF(:);
plotCtrl = work_mean(ctrlIdx);%work_per_beat(:,ctrlIdx);
plotCtrl = plotCtrl(:);

disp('Work')
[h,p,ci,stats] = ttest2(plotHF,plotCtrl)

figure;
h1 = histogram(plotHF,'FaceColor','r','FaceAlpha',0.4);
hold on
h2 = histogram(plotCtrl,'FaceColor','b','FaceAlpha',0.4);
h1.BinWidth = 0.001;
h2.BinWidth = 0.001;
xlabel('Work per beat (mL*mm Hg)')
ylabel('Counts')
legend('Heart failure', 'Control')
saveas(gcf,'work_per_beat.png')

% efficiency
size(efficiency_per_beat(:,HFidx))
plotHF = efficiency_mean(HFidx);%efficiency_per_beat(:,HFidx);
plotHF = plotHF(:);
plotCtrl = efficiency_mean(ctrlIdx);%efficiency_per_beat(:,ctrlIdx);
plotCtrl = plotCtrl(:);

% t-test
disp('Efficiency')
[h,p,ci,stats] = ttest2(plotHF,plotCtrl)

figure;
h1 = histogram(plotHF,'FaceColor','r','FaceAlpha',0.5);
hold on
h2 = histogram(plotCtrl,'FaceColor','b','FaceAlpha',0.5);
h1.BinWidth = 0.002;
h2.BinWidth = 0.002;
xlabel('Efficiency (mm Hg)') % mL*mmHg/mL
ylabel('Counts')
legend('Heart failure', 'Control')
saveas(gcf,'efficiency_per_beat.png')

end