function [work, efficiency, CoBF_per_beat, vol_max, bp_max] = effCalc(data,subject,slice,start,stop,num_beats_check)
% Calculate work and efficiency for a small time window (tens of beats)
%  Input: data, MxN struct where the row is the time sample and the column is
%   the animal number
% subject: which subject in data (column) to analyze slice: which time
% window in data (row) to analyze start: segment start beat index stop:
% segment end beat index num_beats_check: check that number of beats in
% RR_interval for that time window is the same as the number of beats
% calculated from the function
i = slice;
jj = subject;

s = start; % sec
e = stop; % sec



times_BP = data(i,jj).timeHRs_BP*3600; % convert to s
times_CO = data(i,jj).timeHRs_CO*3600; % convert to s


    % Extract BP and CO values for the beat
    try
        BP_vals = data(i,jj).BP(times_BP > s & times_BP < e); % mmHg
        CO_vals = data(i,jj).CO(times_CO > s & times_CO < e)*1000/60; % Convert CO to mL/s
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

    catch
        sprintf('Missing data, sheep %d, window %d',jj,i)
        BP_vals = NaN;
        CO_vals = NaN;
        time_CO = NaN;
    end

    % Calculate cumulative volume using cumulative integral
    if ~isempty(CO_vals) && numel(CO_vals) > 1
        vol_vals = cumtrapz(time_CO, CO_vals); % mL
    else
        vol_vals = NaN;
        disp('Warning: CO vector was empty or only has one data point');
    end

    % Compute Work per Beat using Pressure-Volume Loop
    sizediff = numel(BP_vals) - numel(CO_vals);
    if ~isempty(BP_vals) && numel(BP_vals) > 1 && ~any(isnan(vol_vals)) && ~any(isnan(BP_vals))
        work_per_beat_sample = trapz(vol_vals, BP_vals(1:end-sizediff))/num_beats; % mL * mmHg
    else
        work_per_beat_sample = NaN;
        disp('Warning: Unable to calculate Work per Beat');
    end

    % Calculate Efficiency (Work/CoBF)
    try
        CoBF_vals = data(i,jj).CoBF(times_CO > s & times_CO < e)/60; % Convert to mL/s
    catch
        CoBF_vals = NaN;
        disp('Warning: unable to calculate CoBF')
    end
    if ~isempty(CoBF_vals) && numel(CoBF_vals) > 1 && ~any(isnan(CoBF_vals)) % for model ~isempty(CO_vals)
        % efficiency_per_beat_sample = work_per_beat_sample / trapz(time_CO, CoBF_vals); % units of trapz(time_CO, CoBF_vals) = mL
        % efficiency_per_beat_sample = vol_vals(end) / trapz(time_CO, CoBF_vals); % 7/31/25: no grouping of data. Change so that efficiency = CO/CoBF
        efficiency_per_beat_sample = vol_vals(end) / work_per_beat_sample; % 7/31/25: higher efficiency for alt: CO/work
        CoBF_per_beat = trapz(time_CO, CoBF_vals) / num_beats; % for model: NaN
    else
        efficiency_per_beat_sample = NaN;
        CoBF_per_beat = NaN;
        disp('Warning: Unable to calculate Efficiency per Beat');
    end



work = work_per_beat_sample;
efficiency = efficiency_per_beat_sample;
vol_max = max(vol_vals); % This is per beat CO
bp_max = max(BP_vals);



end
