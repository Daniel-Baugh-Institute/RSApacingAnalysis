clear; close all; restoredefaultpath;
addpath(genpath('C:\Users\mmgee\AppData\Local\Temp\Mxt231\RemoteFiles'))
addpath(genpath('C:\Users\mmgee\MATLAB\PhysioNet-Cardiovascular-Signal-Toolbox-eec46e75e0b95c379ecb68cb0ebee0c4c9f54605'))
fileToRead1 = '1828 day 11 baseline.mat';
readRamchandraData(fileToRead1)

channelNum = [1 6];%[1 3 4 6];
prefix = 'Baseline_11_Ch';
plotName = ['plotRawData_' prefix '.png'];
% plotRamchandraData(channelNum,prefix,plotName)
n = length(channelNum); % 4 data channels
tiledlayout(n,1)

for i = 1:n
    structName = [prefix num2str(channelNum(i))];
    try
        start(i) = eval([structName '.start']);
    catch
        start(i) = 0;
    end
    try
        interval(i) = eval([structName '.interval']);
    catch
        interval(i) = eval([structName '.resolution']);
    end
    stop(i) = eval([structName '.length'])*interval(i);
    timeRaw = start(i):interval(i):stop(i); % seconds

    % HR data needs to be evaluated differently
    if strcmp(eval([structName '.title']), 'HR')
        RRintervals = diff(eval([structName '.times']));
        timeRR = eval([structName '.times'])/60./60; % hours
        %%
        nexttile(n)
        stairs(timeRR(1:end-1),RRintervals)
        % xlim([5 5.05])
        xlabel('Time (hrs)')
        ylabel('RR interval (s)')

        %%
    else
        if length(timeRaw) ~= length(eval([structName '.values']))
            time = timeRaw(1:end-1)./60./60; % hours
        else
            time = timeRaw./60./60;
        end

        nexttile(i)
        plot(time,eval([structName '.values']))
        % xlim([5 5.05])


        % ylabel
        if strcmp(eval([structName '.title']), 'BP')
            ylab = 'BP (mm Hg)';
        elseif strcmp(eval([structName '.title']), 'CoBF')
            ylab = 'Coronary blood flow (mL/min)';
        elseif strcmp(eval([structName '.title']), 'CO')
            ylab = 'Cardiac output (L/min)';
        elseif strcmp(eval([structName '.title']), 'HR')
            ylab = 'Heart rate (bpm)';
        else
            disp('ylabel not any of the ones used previously')
            ylab = '';
        end

        ylabel(ylab)
    end
end

saveas(gcf,plotName)

%% reconstruct baroreflex curve from data
% sequence method to get heart rate increases and decrecrease? If I recall,
% this didn't work (see slides from 2/23/24 one on one)
subjectID = '1828_day_11';
% valid_indices = 30000:1:40000;
BPstruct = [prefix num2str(channelNum(1))];
OnsetTimes = run_wabp(eval([BPstruct '.values']));
FsBP = 1000; % Hz

r = abpfeature(eval([BPstruct '.values']),OnsetTimes, FsBP);
%  1:  Time of systole   [samples]
%  2:  Systolic BP       [mmHg]
%  3:  Time of diastole  [samples]
%  4:  Diastolic BP      [mmHg]
%  5:  Pulse pressure    [mmHg]
%  6:  Mean pressure     [mmHg]
%  7:  Beat Period       [samples]
%  8:  mean_dyneg
%  9:  End of systole time  0.3*sqrt(RR)  method
% 10:  Area under systole   0.3*sqrt(RR)  method
% 11:  End of systole time  1st min-slope method
% 12:  Area under systole   1st min-slope method
% 13:  Pulse             [samples]
% NOTE: this is not good at identifying SBP peaks. Use BP_annotate.m
% instead

%% try BP_annotate.m
addpath('C:\Users\mmgee\OneDrive - University of Delaware - o365\Documents\Github\NZ-physiology-data')
inFs = 1/Baseline_11_Ch1.interval;
verbose = 1; % 1 if figures wanted

[ footIndex, systolicIndex, notchIndex, dicroticIndex, time, bpwaveform ] = BP_annotate( Baseline_11_Ch1.values, inFs, verbose );
% need to align systolic index to time (time(systolicIndex))
%% sequence method
systolicTime = time(systolicIndex);
indices = matchSampleTimes(systolicTime, Baseline_11_Ch6.times);
% then get RR intervals at those times:
RR = RRintervals(indices(1:end-2));
SBP = bpwaveform(systolicIndex);

% RR = diff(r(:,1))./1000; % convert to s from ms
X = [RR,SBP(1:end-2)']; % Matrix with two columns representing RR intervals and
%   systolic blood pressure values
% valid_indices = 1:1:10000;
% % % bar plot of binned MAP values and average RR values
% [valid_indices_linear,slope] = sequenceMethod(valid_indices,X,subjectID);
% 
% 
% % plot mean up slopes binned and back calculate baroreflex curve
% plotPositiveSlopes(valid_indices_linear, slope, X);


%% moving average method
windowPoints = 50;
[SBPmovingAvg, RRmovingAvg] = movingAverageSBPRR(X,windowPoints);
filename = ['plot_movingAvg_1828_day_11_' num2str(windowPoints) 'beatWindow.png'];
plotMovingAvg(SBPmovingAvg, RRmovingAvg, filename)



% based on the method used in:
%   https://doi.org/10.1161/01.HYP.29.6.1284. This method still relies on
%   phenylephrine injections
% [interpolated_RR, RR_times] = interpolate_to_1000Hz(RRintervals(1:5762), timeRR(1:5762).*60.*60); % HR is sampled at 1000 Hz, but RR is converted
% BPidx = 1;
% BP_times = start(BPidx):interval(BPidx):3600;%stop(BPidx);
% blood_pressure = eval([prefix num2str(BPidx) '.values']);
% [baroreflex_RR, baroreflex_MAP] = baroreflex_curve_from_continuous(interpolated_RR, RR_times, blood_pressure(1:3600/1e-3), BP_times,subjectID);

%% Function for rrinterval interpolation
function [interpolated_data, interpolated_times] = interpolate_to_1000Hz(data, times)
% INTERPOLATE_TO_1000HZ interpolates data with inconsistent sample times to 1000 Hz.
%   [interpolated_data, interpolated_times] = INTERPOLATE_TO_1000HZ(data, times)
%   interpolates the input data with its corresponding sample times to a consistent
%   sampling rate of 1000 Hz using linear interpolation.
%   Written by ChatGPT 2/19/24
%
%   Input:
%   - data: Input data vector.
%   - times: Corresponding time vector for the input data.
%
%   Output:
%   - interpolated_data: Interpolated data vector at 1000 Hz.
%   - interpolated_times: Time vector corresponding to the interpolated data.

% Determine the new time vector with a sampling rate of 1000 Hz
new_times = linspace(times(1), times(end), round((times(end) - times(1)) * 1000) + 1);

% Interpolate the data to the new time vector
interpolated_data = interp1(times, data, new_times, 'linear');

% Exclude the last data point if the new time vector has one extra point
if numel(new_times) > numel(interpolated_data)
    new_times = new_times(1:end-1);
end

% Assign the interpolated times
interpolated_times = new_times;
end

function indices = matchSampleTimes(timeA, timeB)
% matchSampleTimes finds the indices in timeB that best match each time in timeA
%
% Inputs:
%   timeA - vector of sample times for signal A
%   timeB - vector of sample times for signal B
%
% Outputs:
%   indices - vector of indices in timeB that most closely match the times in timeA

% Initialize the output indices vector
indices = zeros(size(timeA));

% Loop through each time point in timeA
for i = 1:length(timeA)
    % Find the index in timeB with the minimum absolute time difference to timeA(i)
    [~, indices(i)] = min(abs(timeB - timeA(i)));
end
end

function [SBPmovingAvg, RRmovingAvg] = movingAverageSBPRR(X,windowPoints)
% movingAverageSBPRR finds the moving average for SBP and corresponding RR interval
%
% Inputs:
%   X - array where first column is RR intervals and second column is
%   corresponding MAP values
%   window - sliding window of length k to calculate moving average
%   across neighboring elements of BPsignal and RR signal
%
% Outputs:
%   MAPmovingAvg - MAP moving average
%   RRmovingAvg - corresponding RR moving average
RRsignal = X(:,1);
BPsignal = X(:,2);
SBPmovingAvg = movmean(BPsignal,windowPoints);
RRmovingAvg = movmean(RRsignal,windowPoints);

end

function plotMovingAvg(SBPmovingAvg, RRmovingAvg, filename)
figure;
tiledlayout(2, 1)

% First panel: Original moving average plot
nexttile(1)
plot(SBPmovingAvg, RRmovingAvg,'o')
xl1 = xlim;
xlabel('Avg. systolic blood pressure (mm Hg)')
ylabel('Avg. RR interval (s)')

% Second panel: Binned bar plot with error bars
% Define SBP bins
binWidth = 5; % Define bin width for SBP
SBP_min = floor(min(SBPmovingAvg) / binWidth) * binWidth;
SBP_max = ceil(max(SBPmovingAvg) / binWidth) * binWidth;
SBP_bins = SBP_min:binWidth:SBP_max;

% Initialize arrays for binned averages and standard deviations
avg_SBP = zeros(1, length(SBP_bins) - 1);
avg_RR = zeros(1, length(SBP_bins) - 1);
std_RR = zeros(1, length(SBP_bins) - 1);

% Calculate mean and std deviation for each bin
for j = 1:length(SBP_bins) - 1
    binIndices = SBPmovingAvg >= SBP_bins(j) & SBPmovingAvg < SBP_bins(j + 1);
    if any(binIndices)
        avg_SBP(j) = mean(SBPmovingAvg(binIndices));
        avg_RR(j) = mean(RRmovingAvg(binIndices));
        std_RR(j) = std(RRmovingAvg(binIndices)); % Standard deviation for error bars
    else
        avg_SBP(j) = NaN;
        avg_RR(j) = NaN;
        std_RR(j) = NaN;
    end
end

% Remove NaN values
valid_bins = ~isnan(avg_SBP) & ~isnan(avg_RR);
avg_SBP = avg_SBP(valid_bins);
avg_RR = avg_RR(valid_bins);
std_RR = std_RR(valid_bins);

% Plot the bar plot with error bars
nexttile(2)
bar(avg_SBP, avg_RR, 'FaceColor', [0.2, 0.6, 0.8]);
hold on
errorbar(avg_SBP, avg_RR, std_RR, 'k', 'LineStyle', 'none', 'LineWidth', 1.2);
xl2 = xlim;
xlabel('Binned systolic blood pressure (mm Hg)')
ylabel('Binned avg. RR interval (s)')
title('Binned Avg. RR Interval with Standard Deviation')

% Find leftmost xLeft
xLeft = min([xl1(1), xl2(1)]);
% Find rightmost xRight
xRight = max([xl1(2), xl2(2)]);

nexttile(1)
xlim([xLeft,xRight])
nexttile(2)
xlim([xLeft,xRight])

% Save the figure
saveas(gcf, filename)
end

function plotPositiveSlopes(valid_indices_linear, slope, X)
    % Filter for positive slopes
    positiveSlopeIndices = slope > 0;
    positiveIndices = valid_indices_linear(positiveSlopeIndices);
    positiveSlopes = slope(positiveSlopeIndices);

    % Extract systolic blood pressure and RR interval values for positive slopes
    SBP_values = X(positiveIndices, 2);
    RR_values = X(positiveIndices, 1);

    % Define bins for systolic blood pressure
    binWidth = 10; % Adjust bin width as desired
    SBP_min = floor(min(SBP_values) / binWidth) * binWidth;
    SBP_max = ceil(max(SBP_values) / binWidth) * binWidth;
    SBP_bins = SBP_min:binWidth:SBP_max;

    % Initialize arrays for binned slopes and standard deviation
    avg_SBP = zeros(1, length(SBP_bins) - 1);
    avg_slope = zeros(1, length(SBP_bins) - 1);
    std_slope = zeros(1, length(SBP_bins) - 1);
    avg_RR = zeros(1, length(SBP_bins) - 1);

    % Calculate average slope and standard deviation for each bin
    for j = 1:length(SBP_bins) - 1
        binIndices = SBP_values >= SBP_bins(j) & SBP_values < SBP_bins(j + 1);
        if any(binIndices)
            avg_SBP(j) = mean(SBP_values(binIndices));
            avg_slope(j) = mean(positiveSlopes(binIndices));
            std_slope(j) = std(positiveSlopes(binIndices)); % Standard deviation for error bars
            avg_RR(j) = mean(RR_values(binIndices));
        else
            avg_SBP(j) = NaN;
            avg_slope(j) = NaN;
            std_slope(j) = NaN;
            avg_RR(j) = NaN;
        end
    end

    % Remove NaN values from bins
    valid_bins = ~isnan(avg_SBP) & ~isnan(avg_slope);
    avg_SBP = avg_SBP(valid_bins);
    avg_slope = avg_slope(valid_bins);
    std_slope = std_slope(valid_bins);

    % Plot the bar chart with error bars for each bin
    figure;
    bar(avg_SBP, avg_slope, 'FaceColor', [0.4, 0.7, 0.2]);
    hold on
    errorbar(avg_SBP, avg_slope, std_slope, 'k', 'LineStyle', 'none', 'LineWidth', 1.2);
    xlabel('Systolic Blood Pressure (mm Hg)')
    ylabel('Average Slope')
    title('Average Positive Slopes for SBP Bins')
    grid on

    % Save figure
    saveas(gcf, 'PositiveSlopeBins.png');

    figure;
    % reconstruct baroreflex curve by connecting slopes of bins together
    for i = 1:length(avg_SBP)-2 % might change the -2 depending on whether pressure data covers upper range
        % find the equation of the line for this segment
        x = avg_SBP(i);
        y = avg_RR(i);
        y_intercept = y - avg_slope(i)*x;

        % solve for the end points of the line at the bin edges
        point1 = avg_slope(i)*SBP_bins(i) + y_intercept;
        point2 = avg_slope(i)*SBP_bins(i+1) +y_intercept;

        % plot the segment
        plot([SBP_bins(i) SBP_bins(i+1)],[point1 point2],'-','LineWidth',2)
        hold on
    end

    xlabel('Systolic Blood Pressure (mm Hg)')
    ylabel('RR interval (s)')
    title('Reconstructed baroreflex curve')
    grid on

    % Save figure
    saveas(gcf, 'plot_reconstructedBaroreflexCurve.png');
end