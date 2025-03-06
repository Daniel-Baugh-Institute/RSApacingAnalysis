function sampleSection(filesToRead, sampleLength, filename)
% Combine samples from individuals into struct data that has fields HR, CO,
% BP, CoBF for each animal
% sampleLength: minutes of sample to output

num_subjects = length(filesToRead);
meanRR = zeros(1,num_subjects);
inFs = 1/1e-3;
num_points = sampleLength*60*inFs;
sampleLength_hrs = sampleLength/60;
verbose = 1; % 1 if figures wanted
k = 50000; % number of points for moving average to detect signal loss
kfkb = [150000 150000]; % 5 min window size. At edges, ignores missing values and averages over available window
rng default


for i = 1:num_subjects
    vars = readRamchandraData(filesToRead{i});

    % Handle missing CoBF in one sample
    if strcmp(filesToRead{i}, '2453 baseline no CoBF.mat')
        num_channels = 3;
    else
        num_channels = 4;
    end

    % extract time vector for CO, CoBF, BP (HR has different sampling rate)
    structName = vars{1};
    struct = eval(structName);
    try
        start(i) = struct.start;
    catch
        start(i) = 0;
    end
    try
        interval(i) = struct.interval;
    catch
        interval(i) = struct.resolution;
        disp('Time vector is for HR')
    end
    stop(i) = struct.length*interval(i);
    timeRaw = start(i):interval(i):stop(i); % seconds
    data(i).time = timeRaw;
    timeHRs = data(i).time./60./60;

    % extract data from each channel
    for j = 1:num_channels
        structName = vars{j};
        struct = eval(structName);

        % debugging
        try
            disp(struct.title)
        catch
            disp('Animal number: ')
            disp(i)
            disp('Channel number: ')
            disp(j)
        end

        if strcmp('BP',struct.title)
            data(i).BP = struct.values;

            % Remove BP signals where probe has failed
            rm_idx = find(data(i).BP < 0);
            data(i).BP(rm_idx) = [];
            num_points_removed = length(rm_idx);
            time_removed = num_points_removed*0.001;
            sprintf('No signal was detected in the BP data in animal %d.',i)
            sprintf('%d points were removed (%0.3f seconds)',num_points_removed,time_removed)

            % Adjust time vector to account for missing signal
            timeHRs_BP = timeHRs;
            timeHRs_BP(rm_idx) = [];
            

        elseif strcmp('CoBF',struct.title)
            data(i).CoBF = struct.values;
            timeHRs_CoBF = timeHRs;

        elseif strcmp('CO',struct.title)
            data(i).CO = struct.values;
            struct.interval
            disp('Most intervals are 1e-3')
            timeHRs_CO = timeHRs;


        elseif strcmp('HR',struct.title) || strcmp('HR(Peak)',struct.title)
            data(i).HR = struct.times;
        else
            disp('Title is not BP, CoBF, CO, or HR.')
            disp('Animal number: ')
            disp(i)
            disp('Channel number: ')
            disp(j)
        end


    end

    % CoBF missing for sample 10
    if i ~=10
        % Remove noise with negative signals for CO and CoBF
        % Check that CoBF and CO data are aligned
        if length(data(i).CO) ~= length(data(i).CoBF)
            disp('Length of CO and CoBF raw data is not the same')
            length(data(i).CO)
            length(data(i).CoBF)
        end
        % CO
        CO_idx_negsignal = find(data(i).CO < 0);
        sprintf('%d points removed because of negative signal for animal %d CO',length(CO_idx_negsignal),i)
        % CoBF
        CoBF_idx_negsignal = find(data(i).CoBF < 0);
        sprintf('%d points removed because of negative signal for animal %d CoBF',length(CoBF_idx_negsignal),i)
        idx_negsignal = [CoBF_idx_negsignal CoBF_idx_negsignal]; % combine indices for CO and CoBF so they stay aligned for potential energy surface
        timeHRs_CoBF(idx_negsignal) = [];
        data(i).CoBF(idx_negsignal) = [];
        timeHRs_CO(idx_negsignal) = [];
        data(i).CO(idx_negsignal) = [];

        % Remove sections with no signal
        % CO
        M = movmean(data(i).CO,[0 k],"Endpoints",'discard');
        if i == 3 % remove lost signal in animal 3
            thresh = 3;
        else
            thresh = -5;
        end
        CO_idx_nosignal = find(M < thresh);
        %CoBF
        M = movmean(data(i).CoBF,[0 k],"Endpoints",'discard');
        CoBF_idx_nosignal = find(M < -5);
        idx_nosignal = [ CoBF_idx_nosignal CO_idx_nosignal];

        if ~isempty(idx_nosignal)
            idx2rm_start = min(idx_nosignal);
            idx2rm = idx2rm_start:1:length(data(i).CoBF);
            data(i).CoBF(idx2rm) = [];
            data(i).CO(idx2rm) = [];
            sprintf('No signal was detected in CoBF and/or CO data in animal %d. %d points were removed (%0.3f seconds)',i,length(idx2rm),0.001*length(idx2rm))
            % Adjust time vector to account for missing signal
            timeHRs_CoBF(idx2rm) = [];
            timeHRs_CO(idx2rm) = [];
        end


    end



    % Annotate BP waveform
    [ ~, systolicIndex, ~, ~, time, bpwaveform ] = BP_annotate( data(i).BP, inFs, verbose );
    annotated_data(i).systolicIndex = systolicIndex;
    annotated_data(i).time = time;
    annotated_data(i).bpwaveform = bpwaveform;

    systolicTime = annotated_data(i).time(annotated_data(i).systolicIndex);
    RRfromBP = diff(systolicTime);
    meanRR(i) = mean(RRfromBP);
    SBP = annotated_data(i).bpwaveform(annotated_data(i).systolicIndex);
    X = [RRfromBP',SBP(2:end)',systolicTime(2:end)'];

    % Remove double counted peaks
    lowRRidx = find(X(:,1)<0.1);
    X_clean = X;
    X_clean(lowRRidx,:) = [];
    annotated_data(i).X = X_clean;
    timeHRs_RR = annotated_data(i).X(:,3)./60./60;

    % Find sample indices
    tstart = randi([10 500],1,1)/60 % hours
    tend = (tstart+sampleLength_hrs) % hours
    COsampleIdx = find(timeHRs_CO > tstart & timeHRs_CO < tend);
    CoBFsampleIdx = find(timeHRs_CoBF > tstart & timeHRs_CoBF < tend);
    BPsampleIdx = find(timeHRs_BP > tstart & timeHRs_BP < tend);
    max(timeHRs_RR)
    min(timeHRs_RR)
    RRsampleIdx = find(timeHRs_RR > tstart & timeHRs_RR < tend)
    

    % Extract samples and time vectors
    data(i).CO = data(i).CO(COsampleIdx);
    timeHRs_CO = timeHRs_CO(COsampleIdx);
    data(i).CoBF = data(i).CoBF(CoBFsampleIdx);
    timeHRs_CoBF = timeHRs_CoBF(CoBFsampleIdx);
    data(i).BP = data(i).BP(BPsampleIdx);
    timeHRs_BP = timeHRs_BP(BPsampleIdx);
    RRsample = annotated_data(i).X(RRsampleIdx,1);
    length(RRsample)
    timeHRs_RR = timeHRs_RR(RRsampleIdx);

    % Set up plot
    tiledlayout(num_channels,1)
    plotName = ['plot_sample_data_' num2str(i) '.png'];

    disp('debug')
    size(timeHRs_RR)
    size(RRsample)
    any(isnan(RRsample))

    
    nexttile;
    stairs(timeHRs_RR,RRsample,'HandleVisibility','off')
    ylabel('RR interval (s)')
    hold off

    nexttile;
    if length(timeHRs_BP) == length(data(i).BP)
        plot(timeHRs_BP,data(i).BP)
    elseif (length(timeHRs_BP) - 1) == length(data(i).BP)
        timeHRs_BP = timeHRs_BP(1:end-1);
        plot(timeHRs_BP,data(i).BP)
        disp('time vector was one data point longer than BP')
    elseif (length(timeHRs_BP) + 1) == length(data(i).BP)
        disp('time vector was one data point shorter than BP')
        timeHRs_BP = [timeHRs_BP, timeHRs_BP(end) + 0.001];
        plot(timeHRs_BP,data(i).BP)
    else
        length(timeHRs_BP)
        length(data(i).BP)
        sprintf('BP and time vectors not the same length for animal %d',i)
    end
    
    ylabel({'Blood pressure'; '(mm Hg)'})
    hold off

    nexttile;
    if length(timeHRs_CO) == length(data(i).CO)
        plot(timeHRs_CO,data(i).CO)
    elseif (length(timeHRs_CO) - 1) == length(data(i).CO)
        timeHRs_CO = timeHRs_CO(1:end-1);
        plot(timeHRs_CO,data(i).CO)
        disp('time vector was one data point longer than CO')
    elseif (length(timeHRs_CO) + 1) == length(data(i).CO)
        disp('time vector was one data point shorter than CO')
        timeHRs_CO = [timeHRs_CO, timeHRs_CO(end) + 0.001];
        plot(timeHRs_CO,data(i).CO)
    else
        sprintf('CO and time vectors not the same length for animal %d',i)
        length(timeHRs_CO)
        length(data(i).CO)
    end
    
    ylabel({'Cardiac output'; '(L/min)'})
    hold off

    if num_channels > 3
        nexttile;
        if length(timeHRs_CoBF) == length(data(i).CoBF)
            plot(timeHRs_CoBF,data(i).CoBF)
        elseif (length(timeHRs_CoBF) - 1) == length(data(i).CoBF)
            timeHRs_CoBF = timeHRs_CoBF(1:end-1);
            plot(timeHRs_CoBF,data(i).CoBF)
            disp('time vector was one data point longer than CoBF')
        elseif (length(timeHRs_CoBF) + 1) == length(data(i).CoBF)
            disp('time vector was one data point shorter than CoBF')
            timeHRs_CoBF = [timeHRs_CoBF, timeHRs_CoBF(end) + 0.001];
            plot(timeHRs_CoBF,data(i).CoBF)
        else
            sprintf('CoBF and time vectors not the same length for animal %d',i)
            length(timeHRs_CoBF)
            length(data(i).CoBF)
        end
        ylim([-5 inf])
        ylabel({'Coronary blood'; 'flow (mL/min)'})

    end
    
    xlabel('Time (hrs)')

    saveas(gcf,plotName)
    
    data(i).timeHRs_CO = timeHRs_CO;
    data(i).timeHRs_CoBF = timeHRs_CoBF;
    data(i).timeHRs_BP = timeHRs_BP;
    data(i).timeHRs_RR = timeHRs_RR;
    data(i).RR = RRsample;

    data_sample = data;
   

% save data
save(filename,'data_sample','-v7.3')


end