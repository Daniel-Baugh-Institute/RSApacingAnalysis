function [work, efficiency, CoBF_per_beat, vol_max, bp_max] = effCalc(data,subject,slice,start,stop,num_beats_check)
% start: segment start beat index
% stop: segment end beat index
i = slice;
jj = subject;

s = start; % sec
e = stop; % sec



times_BP = data(i,jj).timeHRs_BP*3600; % convert to s
times_CO = data(i,jj).timeHRs_CO*3600; % convert to s

% TODO: because negative CO values are removed from the data during
% data cleaning, the number of points in CO and BP are not the same
% (~ 20 points less in CO than BP). For now, I'm just going to
% truncate the last 20 points of BP. Because we use times to
% extract the points, this should be fine.

    % Extract BP and CO values for the beat
    try
        BP_vals = data(i,jj).BP(times_BP > s & times_BP < e); % mmHg
        CO_vals = data(i,jj).CO(times_CO > s & times_CO < e)/60/1000; % Convert CO to mL/s
        time_CO = times_CO(times_CO > s & times_CO < e);

        % Count how many heart beats to calculate per beat average
        RR_times_idx = data(i,jj).RRtime >= s & data(i,jj).RRtime <= e;
        RR_times = data(i,jj).RRtime(RR_times_idx);

        num_beats = numel(RR_times) - 1;
        if num_beats ~= num_beats_check
            num_beats = numel(RR_times)
            num_beats_check
        elseif num_beats == 0
            disp('num beats = 0')
            num_beats = num_beats_check;
        end



        % figure;
        % histogram(BP_vals)
        % xlabel('BP')
        % saveas(gcf,['hist_BP_sample' num2str(jj) '.png'])
    catch
        sprintf('Missing data, sheep %d, window %d',jj,i)
        BP_vals = NaN;
        CO_vals = NaN;
        time_CO = NaN;
    end

    % Calculate cumulative volume using cumulative integral
    if ~isempty(CO_vals) && numel(CO_vals) > 1
        vol_vals = cumtrapz(time_CO, CO_vals); % mL
        % figure;
        % histogram(vol_vals)
        % xlabel('Vol')
        % saveas(gcf,['hist_vol_sample' num2str(jj) '.png'])
        % 
        % figure;
        % plot(time_CO,CO_vals)
        % xlabel('time')
        % ylabel('CO')
        % saveas(gcf,['plot_CO_sample' num2str(jj) '.png'])
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
    if ~isempty(BP_vals) && numel(BP_vals) > 1 && ~any(isnan(vol_vals)) && ~any(isnan(BP_vals))
        work_per_beat_sample = trapz(vol_vals, BP_vals(1:end-sizediff))/num_beats; % mL * mmHg
    else
        work_per_beat_sample = NaN;
        disp('Warning: Unable to calculate Work per Beat');
    end

    % Calculate Efficiency (Work/CoBF)
    % NOTE: Is this the best measure of efficiency?
    try
        CoBF_vals = data(i,jj).CoBF(times_CO > s & times_CO < e)/60; % Convert to mL/s
    catch
        CoBF_vals = NaN;
        disp('Warning: unable to calculate CoBF')
    end
    if ~isempty(CoBF_vals) && numel(CoBF_vals) > 1 && ~any(isnan(CoBF_vals))
        efficiency_per_beat_sample = work_per_beat_sample / trapz(time_CO, CoBF_vals); % units of trapz(time_CO, CoBF_vals) = mL
        CoBF_per_beat = trapz(time_CO, CoBF_vals) / num_beats;
    else
        efficiency_per_beat_sample = NaN;
        CoBF_per_beat = NaN;
        disp('Warning: Unable to calculate Efficiency per Beat');
    end



work = work_per_beat_sample;
efficiency = efficiency_per_beat_sample;
vol_max = max(vol_vals);
bp_max = max(BP_vals);



end
