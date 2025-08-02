function data = combineSamples(filesToRead,saveFileName)
% Combine samples from individuals into struct data that has fields HR, CO,
% BP, CoBF for each animal

addpath(genpath('../'))
num_subjects = length(filesToRead)
meanRR = zeros(1,num_subjects);
meanMAP = zeros(1,num_subjects);
meanBP = zeros(1,num_subjects);
meanCoBF = zeros(1,num_subjects);
meanCO = zeros(1,num_subjects);

k_avg = 50000; % number of points for moving average to detect signal loss
kfkb = [150000 150000]; % 5 min window size. At edges, ignores missing values and averages over available window

% Time length for slice
t_hours = 0.5; % Total hours * num_slices shouldn't exceed recording time ~24 hrs
t_seconds = t_hours*60*60; % seconds
num_points = round(t_seconds * 1000);
num_slices = 60;%25;%


for i = 19:19%length(filesToRead)
    vars = readRamchandraData(filesToRead{i});

    % Handle missing CoBF in one sample
    if strcmp(filesToRead{i}, '2453 baseline no CoBF.mat') 
        num_channels = 3;
    else
        num_channels = 4;
    end
    for k = 1:num_slices
        tic
        structName = vars{1};
        struct = eval(structName);
        try
            start(k) = 600000+num_points*(k-1);%struct.start;
        catch
            start(k) = 1;
        end
        try
            interval(k) = struct.interval;
        catch
            interval(k) = struct.resolution;
            disp('Time vector is for HR')
        end
        stop(k) = 600000+num_points*k;
        timeRaw = start(k)*interval(k):interval(k):stop(k)*interval(k); % seconds
        data(k,i).time = timeRaw;
        timeHRs = data(k,i).time./60./60;


        % extract data from each channel
        num_channels_temp = num_channels;
        tiledlayout(4,1)
        for j = 1:num_channels_temp
            structName = vars{j};
            if strcmp(structName, 'file')
                num_channels_temp = num_channels_temp + 1;
                continue
            end
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

            if strcmp('BP',struct.title) || strcmp('BP: 1',struct.title) || strcmp('1.BP',struct.title)
                try
                    data(k,i).BP = struct.values(start(k):stop(k));
                catch
                    sprintf('Data missing for sheep %d, time window %d, BP', i, k)
                    data(k,i).BP = NaN;
                end

                % Remove BP signals where probe has failed
                % Adjust time vector to account for missing signal
                if isnan(data(k,i).BP)
                    timeHRs_BP = NaN;
                else
                    timeHRs_BP = timeHRs;
                end

                data(k,i).timeHRs_BP = timeHRs_BP;
                nexttile;
                plot(timeHRs_BP,data(k,i).BP)
                xlabel('Time (hrs)')
                ylabel({'Blood pressure'; '(mm Hg)'})

                % Calculate moving average mean
                M_BP = movmean(data(k,i).BP,kfkb);

            elseif strcmp('CoBF',struct.title) || strcmp('1.CoBF',struct.title)
                try
                    data(k,i).CoBF = struct.values(start(k):stop(k));
                    timeHRs_CoBF = timeHRs;
                catch
                    sprintf('Data missing for sheep %d, time window %d, CoBF', i, k)
                    data(k,i).CoBF = NaN;
                    timeHRs_CoBF = NaN;
                end

                nexttile;
                plot(timeHRs_CoBF,data(k,i).CoBF)
                xlabel('Time (hrs)')
                ylabel({'Coronary blood'; 'flow (mL/min)'})

            elseif strcmp('CO',struct.title) || strcmp('1.CO',struct.title)
                try
                    data(k,i).CO = struct.values(start(k):stop(k));
                    timeHRs_CO = timeHRs;
                catch
                    sprintf('Data missing for sheep %d, time window %d, CO', i, k)
                    data(k,i).CO = NaN;
                    timeHRs_CO = NaN;
                end

                nexttile;
                plot(timeHRs_CO,data(k,i).CO)
                xlabel('Time (hrs)')
                ylabel({'Cardiac output'; '(L/min)'})



            elseif strcmp('HR',struct.title) || strcmp('HR(Peak)',struct.title) || strcmp('1.HR',struct.title)
                tempRRtime = struct.times; % s

                try
                    RRidx_slice = find(tempRRtime > start(k)/1000 & tempRRtime < stop(k)/1000);
                    data(k,i).RRtime = tempRRtime(RRidx_slice);%struct.times; % seconds (time heart beat occurs)
                    data(k,i).RRint = diff(data(k,i).RRtime); % seconds (heart beat length)
                    timeHRs_RR = data(k,i).RRtime/60./60;
                    nexttile;
                stairs(timeHRs_RR(2:end),data(k,i).RRint)
                xlabel('Time (hrs)')
                ylabel('RR interval (s)')

                    if isempty(data(k,i).RRint)
                        sprintf('Sheep %d, slice %d is empty',i,k)
                    end
                catch
                    sprintf('Data missing for sheep %d, time window %d, RR', i, k)
                    data(k,i).RRtime = NaN;
                    data(k,i).RRint = NaN;
                    timeHRs_RR = NaN;
                    nexttile;
                stairs(timeHRs_RR,data(k,i).RRint)
                xlabel('Time (hrs)')
                ylabel('RR interval (s)')
                end

                
                
                plotNameBySheep = ['./raw-data-plots/plot_raw_data_paced_' num2str(i) '_S' num2str(k) '.png'];
                saveas(gcf,plotNameBySheep)

                plotNameBySheep = ['./raw-data-plots/plot_window_paced_' num2str(i) '_S' num2str(k) '.png'];
                saveas(gcf,plotNameBySheep)

            else
                disp('Title is not BP, CoBF, CO, or HR.')
                disp('Animal number: ')
                disp(i)
                disp('Slice number: ')
                disp(k)
                disp('Channel number: ')
                disp(j)
            end

        end

        % Skip to next animal if data for all channels is missing
        if sum(isnan(data(k,i).RRint)) == 1 && ...
                sum(isnan(data(k,i).CO)) == 1 && ...
                sum(isnan(data(k,i).BP)) == 1 && ...
                sum(isnan(data(k,i).CoBF)) == 1
            continue
        end

        % CoBF missing for sample 10
        if isfield(data,'CoBF') == 1
            % Remove noise with negative signals for CO and CoBF
            % Check that CoBF and CO data are aligned
            if length(data(k,i).CO) ~= length(data(k,i).CoBF)
                disp('Length of CO and CoBF raw data is not the same')
                length(data(k,i).CO)
                length(data(k,i).CoBF)
            end
            % CO
            CO_idx_negsignal = find(data(k,i).CO < -3);
            sprintf('%d points removed because of negative signal for animal %d CO',length(CO_idx_negsignal),i)
            % CoBF
            CoBF_idx_negsignal = find(data(k,i).CoBF < -5);
            sprintf('%d points removed because of negative signal for animal %d CoBF',length(CoBF_idx_negsignal),i)
            idx_negsignal = [CoBF_idx_negsignal CoBF_idx_negsignal]; % combine indices for CO and CoBF so they stay aligned for potential energy surface
            timeHRs_CoBF(idx_negsignal) = [];
            if ~strcmp(filesToRead{i}, '2453 baseline no CoBF.mat') ...
                && ~strcmp(filesToRead{i}, '2229 mono day 7 CoBF poor signal.mat')...
                && ~strcmp(filesToRead{i},'2048 RSA day 14 no CoBF.mat')
                try
                    data(k,i).CoBF(idx_negsignal) = [];
                    timeHRs_CO(idx_negsignal) = [];
                    data(k,i).CO(idx_negsignal) = [];
                catch
                    disp('CoBF or CO empty')
                end
            end
            

            % Remove sections with no signal
            % CO
            M = movmean(data(k,i).CO,[0 k_avg],"Endpoints",'discard');

                thresh = -10;
            % end
            CO_idx_nosignal = find(M < thresh);


            %CoBF
            if ~strcmp(filesToRead{i}, '2453 baseline no CoBF.mat') ...
                && ~strcmp(filesToRead{i}, '2229 mono day 7 CoBF poor signal.mat')...
                && ~strcmp(filesToRead{i},'2048 RSA day 14 no CoBF.mat')
                
                M = movmean(data(k,i).CoBF,[0 k_avg],"Endpoints",'discard');
                CoBF_idx_nosignal = find(M < -5);
                idx_nosignal = [ CoBF_idx_nosignal CO_idx_nosignal];
            else
                idx_nosignal = CO_idx_nosignal;
            end

            if ~isempty(idx_nosignal)
                idx2rm_start = min(idx_nosignal);
                idx2rm = idx2rm_start:1:length(data(k,i).CO);
                if ~strcmp(filesToRead{i}, '2453 baseline no CoBF.mat') ...
                && ~strcmp(filesToRead{i}, '2229 mono day 7 CoBF poor signal.mat')...
                && ~strcmp(filesToRead{i},'2048 RSA day 14 no CoBF.mat')
                    data(k,i).CoBF(idx2rm) = [];
                end
                data(k,i).CO(idx2rm) = [];
                sprintf('No signal was detected in CoBF and/or CO data in animal %d. %d points were removed (%0.3f seconds)',i,length(idx2rm),0.001*length(idx2rm))
                % Adjust time vector to account for missing signal
                timeHRs_CoBF(idx2rm) = [];
                timeHRs_CO(idx2rm) = [];
                data(k,i).timeHRs_CO = timeHRs_CO;
            end

            if ~strcmp(filesToRead{i}, '2453 baseline no CoBF.mat') ...
                && ~strcmp(filesToRead{i}, '2229 mono day 7 CoBF poor signal.mat')...
                && ~strcmp(filesToRead{i},'2048 RSA day 14 no CoBF.mat')
                    M_CoBF = movmean(data(k,i).CoBF,kfkb);
                    data(k,i).timeHRs_CO = timeHRs_CO;
            end

        end
        
        % Calculate moving average mean
        M_CO = movmean(data(k,i).CO,kfkb);

        % Calculate MAP from time periods for RR
        N = round(length(data(k,i).RRtime)) - 1 % 5000 for reasonable viewing size
        RR_start = data(k,i).RRtime(1:N); %s
        RR_stop = data(k,i).RRtime(2:N+1); % s
        length(timeHRs_BP)
        length(data(k,i).BP)

        % Vectorized computation using arrayfun
        if ~strcmp(filesToRead{i}, '2453 baseline no CoBF.mat') ...
                && ~strcmp(filesToRead{i}, '2229 mono day 7 CoBF poor signal.mat')...
                && ~strcmp(filesToRead{i},'2048 RSA day 14 no CoBF.mat')
            try
                data(k,i).MAP = arrayfun(@(s, e) mean(data(k,i).BP(timeHRs_BP(1:end-1) > s & timeHRs_BP(1:end-1) < e)), RR_start/60/60, RR_stop/60/60);
            catch
                data(k,i).MAP = NaN;
            end
            try
                data(k,i).CO_mean = arrayfun(@(s, e) mean(data(k,i).CO(timeHRs_CO(1:end-1) > s & timeHRs_CO(1:end-1) < e)), RR_start/60/60, RR_stop/60/60);
            catch
                data(k,i).CO_mean = NaN;
            end

            try
                data(k,i).CoBF_mean = arrayfun(@(s, e) mean(data(k,i).CoBF(timeHRs_CoBF(1:end-1) > s & timeHRs_CoBF(1:end-1) < e)), RR_start/60/60, RR_stop/60/60);
            catch
                data(k,i).CoBF_mean = NaN;
            end
        end
    end



    % stats on data
    try
        meanRR(i) = mean(data(i).RRint);
        meanMAP(i) = mean(data(i).MAP);
        meanBP(i) = mean(data(i).BP);
        meanCO(i) = mean(data(i).CO);
        if ~strcmp(filesToRead{i}, '2453 baseline no CoBF.mat') ...
                && ~strcmp(filesToRead{i}, '2229 mono day 7 CoBF poor signal.mat')...
                && ~strcmp(filesToRead{i},'2048 RSA day 14 no CoBF.mat')
            meanCoBF(i) = mean(data(i).CoBF); %
        else
            meanCoBF(i) = NaN;
        end
    catch
        meanCO(i) = NaN;
    end


        tiledlayout(num_channels,1)
        plotName = ['./raw-data-plots/plot_filtered_data_paced' num2str(i) '.png'];
        timeHRs_RR = data(k,i).RRtime/60./60;
        nexttile;
        stairs(timeHRs_RR(2:end),data(k,i).RRint,'HandleVisibility','off')
        hold on
        plot(timeHRs_RR(2:end),movmean(data(k,i).RRint,kfkb),'r')
        legend('5 min moving avg')
        ylabel('RR interval (s)')
        hold off

        nexttile;
        if length(timeHRs_BP) == length(data(k,i).BP)
            plot(timeHRs_BP,data(k,i).BP)
        elseif (length(timeHRs_BP) - 1) == length(data(k,i).BP)
            timeHRs_BP = timeHRs_BP(1:end-1);
            plot(timeHRs_BP,data(k,i).BP)
            disp('time vector was one data point longer than BP')
        elseif (length(timeHRs_BP) + 1) == length(data(k,i).BP)
            disp('time vector was one data point shorter than BP')
            timeHRs_BP = [timeHRs_BP, timeHRs_BP(end) + 0.001];
            plot(timeHRs_BP,data(k,i).BP)
        else
            length(timeHRs_BP)
            length(data(k,i).BP)
            sprintf('BP and time vectors not the same length for animal %d',i)
        end
        hold on
        plot(timeHRs_BP,M_BP,'r-')
        ylabel({'Blood pressure'; '(mm Hg)'})
        hold off

        nexttile;
        if length(timeHRs_CO) == length(data(k,i).CO)
            plot(timeHRs_CO,data(k,i).CO)
        elseif (length(timeHRs_CO) - 1) == length(data(k,i).CO)
            timeHRs_CO = timeHRs_CO(1:end-1);
            plot(timeHRs_CO,data(k,i).CO)
            disp('time vector was one data point longer than CO')
        elseif (length(timeHRs_CO) + 1) == length(data(k,i).CO)
            disp('time vector was one data point shorter than CO')
            timeHRs_CO = [timeHRs_CO, timeHRs_CO(end) + 0.001];
            plot(timeHRs_CO,data(k,i).CO)
        else
            sprintf('CO and time vectors not the same length for animal %d',i)
            length(timeHRs_CO)
            length(data(k,i).CO)
        end
        hold on
        length(timeHRs_CO)
        length(M_CO)
        plot(timeHRs_CO,M_CO,'r-')
        ylabel({'Cardiac output'; '(L/min)'})
        hold off

        if ~strcmp(filesToRead{i}, '2453 baseline no CoBF.mat') ...
                && ~strcmp(filesToRead{i}, '2229 mono day 7 CoBF poor signal.mat')...
                && ~strcmp(filesToRead{i},'2048 RSA day 14 no CoBF.mat')
            nexttile;
            if length(timeHRs_CoBF) == length(data(k,i).CoBF)
                plot(timeHRs_CoBF,data(k,i).CoBF)
            elseif (length(timeHRs_CoBF) - 1) == length(data(k,i).CoBF)
                timeHRs_CoBF = timeHRs_CoBF(1:end-1);
                plot(timeHRs_CoBF,data(k,i).CoBF)
                disp('time vector was one data point longer than CoBF')
            elseif (length(timeHRs_CoBF) + 1) == length(data(k,i).CoBF)
                disp('time vector was one data point shorter than CoBF')
                timeHRs_CoBF = [timeHRs_CoBF, timeHRs_CoBF(end) + 0.001];
                plot(timeHRs_CoBF,data(k,i).CoBF)
            else
                sprintf('CoBF and time vectors not the same length for animal %d',i)
                length(timeHRs_CoBF)
                length(data(k,i).CoBF)
            end
            ylim([-5 inf])
            ylabel({'Coronary blood'; 'flow (mL/min)'})

            hold on
            plot(timeHRs_CoBF,M_CoBF,'r-')
        end

        xlabel('Time (hrs)')

    
    % save after each animal
    saveFileNameBySheep = [saveFileName '_' num2str(i) '.mat'];
    dataBySheep = data(:,i);
    save(saveFileNameBySheep,'dataBySheep','-v7.3')
     
    % Check if time exeeded and processing stuck
    toc
    disp('Animal')
    disp(num2str(i))

end
saveFileNameMat = [saveFileName '.mat'];
save(saveFileNameMat,'data','-v7.3')
disp('Saved .mat file')

% stats on data
meanRR
meanMAP
meanBP
meanCO


end