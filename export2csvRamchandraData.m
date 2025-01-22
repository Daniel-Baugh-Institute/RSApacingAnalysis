% slice data for use in tiny time mixers
% 11/5/24 (election day) 
% Michelle Gee

clear; close all; restoredefaultpath;
addpath(genpath('C:\Users\mmgee\AppData\Local\Temp\Mxt231\RemoteFiles'))
addpath(genpath('C:\Users\mmgee\MATLAB\PhysioNet-Cardiovascular-Signal-Toolbox-eec46e75e0b95c379ecb68cb0ebee0c4c9f54605'))
addpath(genpath('C:\Users\mmgee\MATLAB\mhrv-master'))
addpath 'C:\Users\mmgee\OneDrive - University of Delaware - o365\Documents\Github\NZ-physiology-data'
fileToRead1 = '1828 day 11 baseline.mat';
readRamchandraData(fileToRead1)

%% Raw data
% numSamplePoints = 60000;
% channelNum = [1 3 4 6];
% prefix = 'Baseline_11_Ch';
% n = length(channelNum); % 4 data channels
% 
% Idx = 1:1:numSamplePoints+1;
% T = table(Idx');
% colName = {'BP','CoBF','CO','HR'};
% 
% for i = 1:n - 1
%     structName = [prefix num2str(channelNum(i))];
%     try
%         start(i) = eval([structName '.start']);
%     catch
%         start(i) = 0;
%     end
%     try
%         interval(i) = eval([structName '.interval']);
%     catch
%         interval(i) = eval([structName '.resolution']);
%     end
%     stop(i) = eval([structName '.length'])*interval(i);
%     timeRaw = start(i):interval(i):stop(i); % seconds
% 
%     % HR data needs to be evaluated differently
%     if strcmp(eval([structName '.title']), 'HR')
%         RRintervals = diff(eval([structName '.times']));
%         timeRR = eval([structName '.times'])/60./60; % hours
% 
%     else
%         if length(timeRaw) ~= length(eval([structName '.values']))
%             time = timeRaw(1:end-1)./60./60; % hours
%         else
%             time = timeRaw./60./60;
%         end
% 
% 
%         % reformat time data
%         if i == 1
%             % Define the starting reference date and time
%             start_date = datetime('01/01/2024 00:00:00', 'InputFormat', 'MM/dd/yyyy HH:mm:ss');
% 
%             % Convert time_data_seconds to a duration array
%             time_durations = seconds(time*1000000);
% 
%             % Add the duration to the starting reference date
%             time_data_datetime = start_date + time_durations;
% 
%             % Set the display format to MM/dd/yyyy HH:mm:ss.SSS for milliseconds
%             time_data_datetime.Format = 'MM/dd/yyyy HH:mm:ss.SSS';
% 
%             % Display result
%             T.('Time') = time_data_datetime(1000:1000+numSamplePoints)';
%         end
% 
%         newTabCol = eval([structName '.values(1000:1000+numSamplePoints)']);
%         T.(colName{i}) = newTabCol;
%         % TODO: align times properly for different columns
%         % TODO: add HR into data table. with aligned sample times?
%     end
% 
% 
% end
% 
% writetable(T,'ramchandra_small.csv')

%% RR intervals and SBP
numSamplePoints = 15000;
channelNum = [1 3 4 6];
prefix = 'Baseline_11_Ch';
n = length(channelNum); % 4 data channels

Idx = 1:1:numSamplePoints+1;
T = table(Idx');
colName = {'BP','CoBF','CO','HR'};

for i = 1:n - 1
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

    else
        if length(timeRaw) ~= length(eval([structName '.values']))
            time = timeRaw(1:end-1)./60./60; % hours
        else
            time = timeRaw./60./60;
        end


        % reformat time data
        if i == 1
            % Define the starting reference date and time
            start_date = datetime('01/01/2024 00:00:00', 'InputFormat', 'MM/dd/yyyy HH:mm:ss');

            % Convert time_data_seconds to a duration array
            time_durations = seconds(time*1000000);

            % Add the duration to the starting reference date
            time_data_datetime = start_date + time_durations;

            % Set the display format to MM/dd/yyyy HH:mm:ss.SSS for milliseconds
            time_data_datetime.Format = 'MM/dd/yyyy HH:mm:ss.SSS';

            % Display result
            T.('Time') = time_data_datetime(1000:1000+numSamplePoints)';
        end
        
    end
    

end

%% PeakID index
addpath('C:\Users\mmgee\OneDrive - University of Delaware - o365\Documents\Github\NZ-physiology-data')
inFs = 1/Baseline_11_Ch1.interval;
verbose = 1; % 1 if figures wanted

% ID SBP peaks and RR intervals
[ footIndex, systolicIndex, notchIndex, dicroticIndex, time, bpwaveform ] = BP_annotate( Baseline_11_Ch1.values, inFs, verbose );

%% Calculate average CO during RR beat interval
% SBPtimes = time(systolicIndex);
% meanCO = zeros(length(SBPtimes),1);
% timevec = Baseline_11_Ch4.start:Baseline_11_Ch4.interval:Baseline_11_Ch4.interval*Baseline_11_Ch4.length;
% for i = 1:length(SBPtimes) - 1
%     % ID time slice of a beat
%     start = SBPtimes(i);
%     stop = SBPtimes(i+1);
%     COidx = find(timevec > start & timevec < stop);
%     COslice = Baseline_11_Ch4.values(COidx);
% 
%     % average over time slice
%     COsliceMean = mean(COslice);
%     meanCO(i) = COsliceMean;
% end

SBPtimes = time(systolicIndex);
num_times = 15100;%length(SBPtimes) - 1;
meanCO = zeros(num_times, 1);
meanCoBF = zeros(num_times,1);
timevec = Baseline_11_Ch4.start:Baseline_11_Ch4.interval:Baseline_11_Ch4.interval*Baseline_11_Ch4.length;

% Vectorized search for start and stop times
for i = 1:num_times
    % Get start and stop times for the beat interval
    start = SBPtimes(i);
    stop = SBPtimes(i + 1);

    % Logical indexing to find values within the time interval
    % COslice = Baseline_11_Ch4.values(timevec > start & timevec < stop);
    CoBFslice = Baseline_11_Ch3.values(timevec > start & timevec < stop);

    % Calculate mean directly for the interval
    % meanCO(i) = mean(COslice);
    meanCoBF(i) = mean(CoBFslice);
end

%% Clean data
load 'meanCO.mat'
SBP = bpwaveform(systolicIndex);
RRfromBP = diff(time(systolicIndex));
% remove RR and SBP pairs where RR > 2
idx2rm = find(RRfromBP > 2);
RRclean = RRfromBP;
RRclean(idx2rm) = [];
SBPclean = SBP;
SBPclean(idx2rm) = [];
COclean = meanCO;
COclean(idx2rm) = [];
CoBFclean = meanCoBF;
CoBFclean(idx2rm) = [];
%% Make table
Idx = 1:1:numSamplePoints+1;
T = table(Idx');

% add time to table as beat #
% Define the starting reference date and time
            start_date = datetime('01/01/2024 00:00:00', 'InputFormat', 'MM/dd/yyyy HH:mm:ss');

            % Convert time_data_seconds to a duration array
            time_table = 1:1:numSamplePoints+1;
            time_durations = seconds(time_table);

            % Add the duration to the starting reference date
            time_data_datetime = start_date + time_durations;

            % Set the display format to MM/dd/yyyy HH:mm:ss.SSS for milliseconds
            time_data_datetime.Format = 'MM/dd/yyyy HH:mm:ss.SSS';

            % Display result
            T.('Time') = time_data_datetime';

% add sbp and rr to table            
colName = {'SBP','RR','CO','CoBF'};

for i = 1:length(colName)
    if i == 1
        newTabCol = SBPclean(101:101+numSamplePoints)';
    elseif i == 2
        newTabCol = RRclean(100:100+numSamplePoints)';
    elseif i == 3
        newTabCol = COclean(100:100+numSamplePoints);
    elseif i == 4
        newTabCol = CoBFclean(100:100+numSamplePoints);
    else
        disp('Check length colName')
    end
        T.(colName{i}) = newTabCol;
end
%%
writetable(T,'ramchandra_small.csv')