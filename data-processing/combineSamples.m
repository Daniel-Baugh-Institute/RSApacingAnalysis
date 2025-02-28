function combineSamples(filesToRead)
% Combine samples from individuals into struct data that has fields HR, CO,
% BP, CoBF for each animal
% TODO?: subsets random sample of CoBF and CO for
% testing. Make size of this sample a function input
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
num_subjects = length(filesToRead);
meanRR = zeros(1,num_subjects);
meanBP = zeros(1,num_subjects);
meanCoBF = zeros(1,num_subjects);
meanCO = zeros(1,num_subjects);
inFs = 1/1e-3;
verbose = 1; % 1 if figures wanted
k = 50000; % number of points for moving average to detect signal loss
kfkb = [150000 150000]; % 5 min window size. At edges, ignores missing values and averages over available window


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

            % Calculate moving average mean
            M_BP = movmean(data(i).BP,kfkb);

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

        M_CoBF = movmean(data(i).CoBF,kfkb);

    end

    % Calculate moving average mean
    M_CO = movmean(data(i).CO,kfkb);

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


    % stats on data
    meanRR(i) = mean(annotated_data(i).X(:,1));
    meanSBP(i) = mean(annotated_data(i).X(:,2));
    meanBP(i) = mean(data(i).BP);
    meanCO(i) = mean(data(i).CO);
    meanCoBF(i) = mean(data(i).CoBF);



    % Set up plot
    tiledlayout(num_channels,1)
    plotName = ['plot_raw_data_' num2str(i) '.png'];
    timeHRs_RR = annotated_data(i).X(:,3)./60./60;
    nexttile;
    stairs(timeHRs_RR,annotated_data(i).X(:,1),'HandleVisibility','off')
    hold on
    plot(timeHRs_RR,movmean(annotated_data(i).X(:,1),kfkb),'r')
    legend('5 min moving avg')
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
    hold on
    plot(timeHRs_BP,M_BP,'r-')
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
    hold on
    length(timeHRs_CO)
    length(M_CO)
    plot(timeHRs_CO,M_CO,'r-')
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

        hold on
        plot(timeHRs_CoBF,M_CoBF,'r-')
    end
    
    xlabel('Time (hrs)')

    saveas(gcf,plotName)

    % Test seasonality of data
    quarter_length = round(length(data(i).CO)/4);
    q1_idx = 1:1:quarter_length;
    q2_idx = quarter_length:1:2*quarter_length;
    q3_idx = 2*quarter_length:1:3*quarter_length;
    q4_idx = 3*quarter_length:1:length(data(i).CO);
    rng default
    sample = randi([1 quarter_length],30,1); % randomly sample 30 values in each quarter%

    % CoBF
    if i ~= 10 % missing CoBF data in sample 10
        sample_mat_CoBF = [data(i).CoBF(q1_idx(sample)),data(i).CoBF(q2_idx(sample)),data(i).CoBF(q3_idx(sample)),data(i).CoBF(q4_idx(sample))];
    end
    % CO
    sample_mat_CO = [data(i).CO(q1_idx(sample)),data(i).CO(q2_idx(sample)),data(i).CO(q3_idx(sample)),data(i).CO(q4_idx(sample))];
    disp('nan test')
    any(isnan(sample_mat_CO))
    any(isnan(q1_idx(sample)))


    % BP (different length than CO and CoBF)
    quarter_length_BP = round(length(data(i).BP)/4);
    q1_idx_BP = 1:1:quarter_length_BP;
    q2_idx_BP = quarter_length_BP:1:2*quarter_length_BP;
    q3_idx_BP = 2*quarter_length_BP:1:3*quarter_length_BP;
    q4_idx_BP = 3*quarter_length_BP:1:length(data(i).BP);
    sample = randi([1 quarter_length_BP],30,1); 
    disp('ll')
    size(q1_idx_BP(sample))
    size(data(i).BP(q1_idx_BP(sample)))
    sample_mat_BP = [data(i).BP(q1_idx_BP(sample)),data(i).BP(q2_idx_BP(sample)),data(i).BP(q3_idx_BP(sample)),data(i).BP(q4_idx_BP(sample))];
    disp('mat size')
    size(sample_mat_BP)
    
    for jj = 1:3
        for kk = jj+1:4
            disp('variances')
            disp(var(sample_mat_CoBF(:,jj)))
            disp(var(sample_mat_CoBF(:,kk)))
            disp('size')
            disp(size(sample_mat_CoBF(:,jj)))
            disp(size(sample_mat_CoBF(:,kk)))
            disp('inf')
            any(isinf(sample_mat_CO))
            any(isinf(q1_idx_BP(sample)))
            if i ~= 10 % missing CoBF data in sample 10
                sprintf('t-test for CoBF quarter %d and %d',jj,kk)
                [h,p] = ttest2(sample_mat_CoBF(:,jj),sample_mat_CoBF(:,kk))
            end
            sprintf('t-test for CO quarter %d and %d',jj,kk)
            [h,p] = ttest2(sample_mat_CO(:,jj),sample_mat_CO(:,kk))
            sprintf('t-test for BP quarter %d and %d',jj,kk)
            [h,p] = ttest2(sample_mat_BP(:,jj),sample_mat_BP(:,kk))
        end
    end

end

% save data
save('combinedAnnotatedData_HF.mat','annotated_data','data','-v7.3')

% Test differences between HF and normal hemodynamics
disp('HF meanRR')
mean(meanRR(1:4))
std(meanRR(1:4))
disp('Control meanRR')
mean(meanRR(5:10))
std(meanRR(5:10))
[h,p,ci,stats] = ttest2(meanRR(1:4),meanRR(5:10))

meanBP
mean(meanBP(1:4))
std(meanBP(1:4))
mean(meanBP(5:10))
std(meanBP(5:10))
[h,p,ci,stats] = ttest2(meanBP(1:4),meanBP(5:10))

meanCoBF
mean(meanCoBF(1:4))
std(meanCoBF(1:4))
mean(meanCoBF(5:10))
std(meanCoBF(5:10))
[h,p,ci,stats] = ttest2(meanCoBF(1:4),meanCoBF(5:10))

meanCO
mean(meanCO(1:4))
std(meanCO(1:4))
mean(meanCO(5:10))
std(meanCO(5:10))
[h,p,ci,stats] = ttest2(meanCO(1:4),meanCO(5:10))


end