%% CLUSTER COPYcalcTE
% TODO: fix plotting of CO and CoBF for 4, 6, 9,

clear; close all; restoredefaultpath;
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))

% local file paths
% cd 'C:\Users\mmgee\OneDrive - University of Delaware - o365\Documents\Github\NZ-physiology-data'
% addpath(genpath('C:\Users\mmgee\AppData\Local\Temp\Mxt231\RemoteFiles'))
% addpath 'C:\Users\mmgee\OneDrive - University of Delaware - o365\Documents\Github\NZ-physiology-data'

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



% Pull data out from each animal's file and combine
% Struct data has fields HR, CO, CoBF, BP, timeRaw for each animal
% (data(1), data(2),...)

% Run data extraction in parallel

saveFilePrefix = 'combinedData_paced_30m_48slices';

data = combineBySheepFunc(filesToRead,saveFilePrefix);


saveFileName = [saveFilePrefix '.mat'];
save(saveFileName,'data','-v7.3')

function data = combineBySheepFunc(filesToRead,saveFilePrefix)
    num_subjects = length(filesToRead);
    num_slices = 60;
    data(num_slices,num_subjects).time = [];

    for j = 1:num_subjects
        
        fileName = [saveFilePrefix '_' num2str(j) '.mat'];
        disp(['Checking existence: ', fileName])
        exist(fileName, 'file')
        
        try
            load(fileName,'dataBySheep')
            fprintf('Loaded file %s with %d time slices\n', fileName, length(dataBySheep))
        catch
            warning('Could not load dataBySheep from %s', fileName)
            continue
        end

        sprintf('Processing sheep %d',j)
        for i = 1:num_slices % number of time slices
            
            data(i,j).time = dataBySheep(i).time;
            data(i,j).BP = dataBySheep(i).BP;
            data(i,j).timeHRs_BP = dataBySheep(i).timeHRs_BP; 
            data(i,j).CO = dataBySheep(i).CO;
            data(i,j).RRtime = dataBySheep(i).RRtime;
            data(i,j).RRint = dataBySheep(i).RRint;
            try
                data(i,j).timeHRs_CO = dataBySheep(i).timeHRs_CO;
            catch
                sprintf('Missing CO for sheep %d, window %d', j,i)
                data(i,j).timeHRs_CO = NaN;
            end

            try
                data(i,j).MAP = dataBySheep(i).MAP;
            catch
                data(i,j).MAP = NaN;
                sprintf('Missing MAP for sheep %d, window %d', j,i)
            end

            try
                data(i,j).CO_mean = dataBySheep(i).CO_mean;
            catch
                data(i,j).CO_mean = NaN;
                sprintf('Missing CO for sheep %d, window %d', j,i)
            end
            
            try
                data(i,j).CoBF = dataBySheep(i).CoBF;
            catch
                data(i,j).CoBF = NaN;
                sprintf('Missing CoBF for sheep %d, window %d', j,i)
            end

            try
                data(i,j).CoBF_mean = dataBySheep(i).CoBF_mean;
            catch
                data(i,j).CoBF_mean = NaN;
                sprintf('Missing CoBF_mean for sheep %d, window %d', j,i)
            end
        end
        size(data)
    end
end





