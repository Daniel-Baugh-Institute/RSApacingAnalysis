%% CLUSTER COPYcalcTE
% TODO: fix plotting of CO and CoBF for 4, 6, 9,

clear; close all; restoredefaultpath;
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))

% local file paths
% cd 'C:\Users\mmgee\OneDrive - University of Delaware - o365\Documents\Github\NZ-physiology-data'
% addpath(genpath('C:\Users\mmgee\AppData\Local\Temp\Mxt231\RemoteFiles'))
% addpath 'C:\Users\mmgee\OneDrive - University of Delaware - o365\Documents\Github\NZ-physiology-data'

%% Set up parallel pool
% myCluster = parcluster('local');
% myCluster.NumWorkers = str2double(getenv('SLURM_CPUS_ON_NODE')) / str2double(getenv('SLURM_CPUS_PER_TASK'));
% myCluster.JobStorageLocation = getenv('TMPDIR');
% myPool = parpool(myCluster, myCluster.NumWorkers);
%% Load and preprocess data to save to smaller file

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

% {'2019 HF baseline.mat', '2031 HF baseline.mat', '2035 HF baseline.mat','2037 HF baseline.mat',...
%     '1828 day 11 baseline.mat', '2445 baseline.mat','2446 baseline.mat',...
%     '2454 baseline.mat','2478 baseline.mat','2453 baseline no CoBF.mat'};

num_subjects = length(filesToRead);

% Pull data out from each animal's file and combine
% Struct data has fields HR, CO, CoBF, BP, timeRaw for each animal
% (data(1), data(2),...)

% Run data extraction in parallel

    saveFilePrefix = '/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/data-processing/combinedData_all_30m_48slices_042125';
    % combinedData_all_30m_48slices only has second data set with 26
    % samples

    % data = combineSamples(filesToRead,saveFilePrefix);


% Repackage data into single file
% for i = 1:num_subjects
%     fileName = ['combinedData_RRfromRaw' num2str(i) '.mat'];
%     load(fileName)
%     combinedData(i).time = data(1).time';
%     combinedData(i).BP = data(1).BP;
%     combinedData(i).CoBF = data(1).CoBF;
%     combinedData(i).CO = data(1).CO;
%     combinedData(i).RRtime = data(1).RRtime;
%     combinedData(i).RRint = data(1).RRint;
%     combinedData(i).MAP = data(1).MAP;
% end
saveFileName = [saveFilePrefix '.mat'];
% save(saveFileName,'data','-v7.3')
% load(saveFileName)

% save sample of data for local testing
% f=fieldnames(data)
% 
% for i = 1:length(f)
%     try
%         sample_data(1,1).(f{i})(1:2000)=data(2,6).(f{i})(1:2000); 
%     catch
%         disp('Field not a timeseries?')
%         f{i}
%         sample_data(1,1).(f{i})=data(2,6).(f{i})(1:end);
%     end
% end
% 
% save('sample_data_S6_W2.mat','sample_data')


% disp('data size')
% size(data)
% data = data(1:2,1);
% save('data_test.mat',"data")
% [work_segment_accel, work_segment_alt, efficiency_segment_accel, efficiency_segment_alt] = calcEfficiencyFrag(data)

% 
% disp('1 hr')
% load('combinedData_classificationTest_1hr.mat')
% size(data)

% combinedAnnotatedData has first 2 HF samples
% load("combinedAnnotatedData_HF.mat","data","annotated_data")

%% Analysis
% Test stationarity of mean
% T_SBP = test_mean_stationarity(data)
% RR, BP, SBP, CO, CoBF were all stationary for all subjects with p = 0.001
% tested using the augmented dickey-fuller test

% HRV analysis
% data = [];%data(:,[12 33]);
% hrv(data)

% Calculate cardiac efficiency
% [work_per_beat, efficiency_per_beat, work_mean, efficiency_mean, work_std, efficiency_std] = calcEfficiency(data);
% save('efficiency_30m_all_CO-work.mat','work_per_beat','efficiency_per_beat','work_mean','efficiency_mean','work_std','efficiency_std','-v7.3')
load 'efficiency_30m_all_CO-work.mat'
% size(work_mean)


% plot comparison of paced and unpaced efficiency and work
control_flag = 0;
healthy_mean = [];
healthy_std = [];
plot_paced_efficiency(work_mean, efficiency_mean, work_std, efficiency_std,healthy_mean, healthy_std,control_flag)

% Plot RR intervals show heart rate fragmentation segments
% fields = fieldnames(data)

% nni = data(1,11).RRint(100:150);
% nn_times = data(1,11).RRtime(100:150);
% save('frag_HF1.mat',"nn_times","nni")
% [ hrv_frag, acceleration_segment_boundaries_3plus, alternation_segment_boundaries_4plus ] = mhrv.hrv.hrv_fragmentation( nni );
% save('plot_RR_frag_HF1.mat','acceleration_segment_boundaries_3plus','alternation_segment_boundaries_4plus','nni','nn_times')
% % load 'test_plot_RR_frag_HF.mat'
% filename = 'plot_RR_frag_HF1.png';
% plotFrag(nni, nn_times, acceleration_segment_boundaries_3plus, alternation_segment_boundaries_4plus,filename)

% Test of plotting CO vs RR
% plotRR_CO(nni, nn_times, data(1,35))

% Plot points of HR and SBP pairs and find PDF. Plot potential energy surface to visualize attractors
% disp('HR and BP')
% var1 = 'SBP';
% var2 = 'RR';
% verbose = 0; % 1 for histogram, 0 for no histogram
% OVERRIDE NUM_SLICES
% [PDF, PES] = plotPES2(data,var1,var2); % returns struct with F_grid for each individual
% save('PDF_RR_MAP_30m_paced.mat','PDF','PES','-v7.3')
% disp('Saved RR, MAP .mat')
% load 'PDF_RR_MAP_30m_paced.mat'
% size(PES)
% 
% PDF_paced = PDF;
% PES_paced = PES;
% 
% load('PDF_RR_MAP_30m_48slices.mat') % 10 original samples
% % Remove duplicate sample
% PDF(:,4) = [];
% size(PDF)
% % combine data
% PDF_combined = [PDF, PDF_paced];
% size(PDF_combined)
% 
% % Remove duplicate sample
% PES(:,4) = [];
% size(PES)
% % combine data
% PES_combined = [PES, PES_paced];
% size(PES_combined)

% transition state analysis
% calculate transition energy
% calcTE(PES_combined)

% Plot potential energy surface for CO and CoBF
% disp('CO and CoBF')
% var1 = 'CO';
% var2 = 'CoBF';
% verbose = 0;
% % OVERRIDE NUM_SLICES
% [PDF, PES] = plotPES2(data,var1,var2);
% save('PDF_30min_CO_CoBF_paced.mat','PDF','PES','-v7.3')
% % disp('Saved mat file')
% load('PDF_30min_CO_CoBF_paced.mat')% 16 paced samples
% PDF_paced = PDF;
% PES_paced = PES;
% load('PDF_30m_48slices_CO_CoBF.mat') % 10 original samples

% Remove duplicate sample
% PDF(:,4) = [];
% size(PDF)
% % combine data
% PDF_combined = [PDF, PDF_paced];
% size(PDF_combined)
% 
% % Remove duplicate sample
% PES(:,4) = [];
% size(PES)
% % combine data
% PES_combined = [PES, PES_paced];
% size(PES_combined)
% 
% calcTE(PES_combined)


% Calculate Bhattacharyya distance between potential energy surfaces
% load '/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/PDF_30min_CO_CoBF_paced.mat'
% PES_paced = PES;
% PDF_paced = PDF;
% load '/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/PDF_30m_48slices_CO_CoBF.mat'
% PES_combined = [PES PES_paced];
% PDF_combined = [PDF PDF_paced];
% size(PES_combined)
% D_B = bhattacharyya_distance_2d(PDF_combined)

% Classifier
% size(PES)
% load 'PDF_RR_MAP_1hr_24slices.mat'
% classify_pes(PES_combined)

% PCA
% filename = 'PCA_test.png';
% [coeff, score, latent] = pca_timeseries(data,filename);


%% HRV
% mhrv_init()
% % time domain analysis
% [ hrv_td, plot_data_timeDomain ] = mhrv.hrv.hrv_time( RRintervals );
%
% % frequency domain analysis
% [ hrv_fd, pxx, f_axis, plot_data_freqDomain ] = mhrv.hrv.hrv_freq( RRintervals );



%% Baroreflex curve reconstruction
% % plot mean up slopes binned and back calculate baroreflex curve
% [A1,A2,A3,A4] = plotPositiveSlopes(valid_indices_linear, slope, X);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % TODO change this function to remove outliers based on IQR
%
%
% % bar plot of binned MAP values and average RR values
% [valid_indices_linear,slope] = sequenceMethod(valid_indices,X,subjectID);
%
%
% % sequence method
% systolicTime = time(systolicIndex);
% indices = matchSampleTimes(systolicTime, Baseline_11_Ch6.times);
% % then get RR intervals at those times:
% RR = RRintervals(indices(1:end-2));
%
%
% % RR = diff(r(:,1))./1000; % convert to s from ms
% X = [RRfromBP',SBP(2:end)'];%[RR,SBP(1:end-2)']; % Matrix with two columns representing RR intervals and
% %   systolic blood pressure values
% % valid_indices_slice = 1:1:length(RRfromBP)/2;
%
% % Remove indices from valid_indices if RR < 0.01 or RR > 2
% % valid_indices_filtered = find(RRfromBP > 0.01 & RRfromBP < 2);
%
% valid_indices = IdUpSequences(X);
% % valid_indices = intersect(valid_indices,valid_indices_slice);


% moving average method
% windowPoints = 50;
% [SBPmovingAvg, RRmovingAvg] = movingAverageSBPRR(X,windowPoints);
% filename = ['plot_movingAvg_1828_day_11_' num2str(windowPoints) 'beatWindow.png'];
% plotMovingAvg(SBPmovingAvg, RRmovingAvg, filename)


% Baroreflex curve from single pressure spike
% based on the method used in:
%   https://doi.org/10.1161/01.HYP.29.6.1284. This method still relies on
%   phenylephrine injections
% [interpolated_RR, RR_times] = interpolate_to_1000Hz(RRintervals(1:5762), timeRR(1:5762).*60.*60); % HR is sampled at 1000 Hz, but RR is converted
% BPidx = 1;
% BP_times = start(BPidx):interval(BPidx):3600;%stop(BPidx);
% blood_pressure = eval([prefix num2str(BPidx) '.values']);
% [baroreflex_RR, baroreflex_MAP] = baroreflex_curve_from_continuous(interpolated_RR, RR_times, blood_pressure(1:3600/1e-3), BP_times,subjectID);
%% NOTES

%%%%% WHAT ABOUT DENSITY PLOT OF COBF AND HR. AS HEART GETS MORE EFFICIENT,
%%%%% SHOULD BE ABLE TO SEE THIS

% local energy minimum location and depth
% Disconnectivity of local minima
% Relationship between basin size and mean duration
% Correlate transition energies to coronary blood flow, infarct size or
% other anatomical disease marker? Are these data that we have?
% Ref: https://www.nature.com/articles/ncomms5765

%% Exit code
% delete(myPool);
% exit
%% Functions from baroreflex curve reconstruction
% Function for rrinterval interpolation
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

function  [A1,A2,A3,A4] = plotPositiveSlopes(valid_indices_linear, slope, X)
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
IQR =  zeros(1,length(SBP_bins) - 1);

% Calculate average slope and standard deviation for each bin
x_swarm = [];
y_swarm = [];
for j = 1:length(SBP_bins) -1
    binIndices = SBP_values >= SBP_bins(j) & SBP_values < SBP_bins(j + 1);
    if any(binIndices)
        avg_SBP(j) = mean(SBP_values(binIndices));
        avg_slope(j) = mean(positiveSlopes(binIndices));
        std_slope(j) = std(positiveSlopes(binIndices)); % Standard deviation for error bars
        avg_RR(j) = mean(RR_values(binIndices));
        IQR(j) = iqr(positiveSlopes(binIndices),'all');
        bin_slopes = positiveSlopes(binIndices);

        % Remove outliers (1.5IQR)
        filtered_slopes = bin_slopes(bin_slopes <= 0.5);
        avg_slope_filtered(j) = mean(filtered_slopes);

        % construct x value vector for swarm plot
        num_in_bin(j) = length(filtered_slopes);%length(positiveSlopes(binIndices));
        bin_midpoint = (SBP_bins(j) + SBP_bins(j+1))/2;
        x_swarm = [x_swarm; repmat(bin_midpoint, length(filtered_slopes), 1)];

        % construct y value vector for swarm plot
        % y_swarm_temp = positiveSlopes(binIndices(filtered_slopes))';
        y_swarm = [y_swarm; filtered_slopes'];
    else
        avg_SBP(j) = NaN;
        avg_slope(j) = NaN;
        std_slope(j) = NaN;
        avg_RR(j) = NaN;
        IQR(j) = NaN;
    end
end

% Remove NaN values from bins
valid_bins = ~isnan(avg_SBP) & ~isnan(avg_slope);
avg_SBP = avg_SBP(valid_bins);
% avg_slope = avg_slope(valid_bins);
std_slope = std_slope(valid_bins);

% Plot the bar chart with error bars for each bin
figure;
% x = bin_num;
% y = slope;
swarmchart(x_swarm,y_swarm,'HandleVisibility','off')
hold on
% plot mean line for bin
for j = 1:length(avg_slope)
    plot([SBP_bins(j) SBP_bins(j+1)],[avg_slope_filtered(j) avg_slope_filtered(j)],'k-')
end
% bar(avg_SBP, avg_slope, 'FaceColor', [0.4, 0.7, 0.2]);
% hold on
% errorbar(avg_SBP, avg_slope, std_slope, 'k', 'LineStyle', 'none', 'LineWidth', 1.2);
legend('Bin mean')
xlabel('Systolic Blood Pressure (mm Hg)')
ylabel('Average Slope')
% ylim([0 0.2])
title('Average Positive Slopes for SBP Bins')
grid on

% Save figure
saveas(gcf, 'PositiveSlopeBins.png');

figure;
% reconstruct baroreflex curve by connecting slopes of bins together
RR = [];
BP = [];
for i = 1:length(avg_SBP) % might change the -2 depending on whether pressure data covers upper range
    % find the equation of the line for this segment
    x = avg_SBP(i);
    y = avg_RR(i);
    y_intercept = y - avg_slope(i)*x;

    % solve for the end points of the line at the bin edges
    point1(i) = avg_slope(i)*SBP_bins(i) + y_intercept;
    point2(i) = avg_slope(i)*SBP_bins(i+1) +y_intercept;

    % plot the segment
    plot([SBP_bins(i) SBP_bins(i+1)],[point1(i) point2(i)],'-','LineWidth',2,'HandleVisibility','off')
    hold on
    RR = [RR point1(i) y point2(i)];
    BP = [BP SBP_bins(i) SBP_bins(i) + 5 SBP_bins(i+1)];
end

xlabel('Systolic Blood Pressure (mm Hg)')
ylabel('RR interval (s)')
title('Reconstructed baroreflex curve')
% xlim([60 130])
grid on

% Save figure
saveas(gcf, 'plot_reconstructedBaroreflexCurve.png');

% Fit points of reconstructed curve to sigmoid
% Use end points and mid point

rmNan = ~isnan(RR);
BP = BP(rmNan);
RR = RR(rmNan);
[A1,A2,A3,A4] = sigmoidRegression(BP',RR');
legend('fit')



end

function plotBinnedRRvsSBP(X)
% Input:
%   X - a 2-column array where:
%       Column 1 is RR intervals (in seconds)
%       Column 2 is corresponding SBP values (in mm Hg)

% Extract RR intervals and SBP values from X
RR_intervals = X(:, 1);
SBP_values = X(:, 2);

% Define bin edges for SBP (5 mm Hg intervals)
binWidth = 10;
SBP_min = floor(min(SBP_values) / binWidth) * binWidth;
SBP_max = ceil(max(SBP_values) / binWidth) * binWidth;
binEdges = SBP_min:binWidth:SBP_max;

% Initialize arrays for storing mean and standard deviation of RR intervals
avg_RR = zeros(1, length(binEdges) - 1);
std_RR = zeros(1, length(binEdges) - 1);
avg_SBP = zeros(1, length(binEdges) - 1);  % For the x-axis (bin centers)

% Calculate the mean and standard deviation of RR intervals for each SBP bin
for i = 1:length(binEdges) - 1
    % Get indices of SBP values that fall within the current bin
    binIndices = SBP_values >= binEdges(i) & SBP_values < binEdges(i + 1);

    if any(binIndices)
        avg_RR(i) = mean(RR_intervals(binIndices));  % Mean RR interval for this bin
        std_RR(i) = std(RR_intervals(binIndices));   % Standard deviation of RR interval
        avg_SBP(i) = (binEdges(i) + binEdges(i + 1)) / 2;  % Midpoint of the bin
    else
        avg_RR(i) = NaN;
        std_RR(i) = NaN;
        avg_SBP(i) = (binEdges(i) + binEdges(i + 1)) / 2;
    end
end

% Remove bins without data (NaNs)
valid_bins = ~isnan(avg_RR);
avg_RR = avg_RR(valid_bins);
std_RR = std_RR(valid_bins);
avg_SBP = avg_SBP(valid_bins);

% Plot the bar chart with error bars
figure;
bar(avg_SBP, avg_RR, 'FaceColor', [0.2, 0.6, 0.8]); % Bar chart for average RR intervals
hold on;
errorbar(avg_SBP, avg_RR, std_RR, 'k', 'LineStyle', 'none', 'LineWidth', 1.2); % Error bars
xlabel('Systolic Blood Pressure (mm Hg)');
ylabel('Average RR Interval (s)');
title('Binned RR Interval vs SBP');
grid on;
hold off;
end


