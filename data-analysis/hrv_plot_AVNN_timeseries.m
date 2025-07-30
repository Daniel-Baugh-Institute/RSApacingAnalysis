function hrv_plot_AVNN_timeseries(data)
% Function to calculate plot AVNN by animal at different time windows
% Input: data, MxN struct where the row is the time sample and the column is
%   the animal number


addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
[num_slices, num_subjects] = size(data);


% select particular animals for analysis
AIDs = 1:10;%[11 12 14 16 18 20 22 24 26 28 30 32 33 35];
num_subjects = length(AIDs);
% AIDs = [1:10 13 15 17 19 21 23 25 27 29 31 34]; 
% num_subjects = length(AIDs);


% Prepare environment for mhrv package
folderPath = '/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/mhrv-master/bin/wfdb'; % Provide the full or relative path to the folder

% Delete the folder and its contents
status = rmdir(folderPath, 's');

if status
    disp('Folder deleted successfully.');
else
    disp('Failed to delete folder.');
end

mhrv_init()

stdRR = zeros(num_slices,num_subjects);
for jj = 1:num_subjects
    g = [];
    x = [];
    for i = 1:num_slices
        
        RRint = data(i,AIDs(jj)).RRint;

        %Filter out RR interval > 1500 ms
        idx2rm = find(RRint > 1.5);
        RRint(idx2rm) = [];
        if ~isempty(idx2rm)
            sprintf('RR intervals > 1.5 s removed for animal %d, slice %d',jj,i)
        end
        
        if ~isempty(RRint)
            window_time = i * 30 / 60; % Convert num_slices to hrs
            g_new = repmat({num2str(window_time)},length(RRint),1);
            g = [g; g_new];
            x = [x; RRint(:)];
            stdRR(i,jj) = std(RRint);
        else
            sprintf('Missing data for sheep %d, time window %d',jj,i)        
        end

    end
    std_by_animal(jj) = mean(stdRR(:,jj));
    figure;
    hold on
    boxplot(x,g,'PlotStyle','compact')
    ylim([0.1 1.5])
    xlabel('Time (hrs)')
    ylabel('RR interval (s)')
    title(['Sheep ' num2str(jj)])
    set(gca,'FontSize',16)
    set(gcf,'Position',[0 0 1800,500])
    filename = ['RR_box_timeseries_sheep_' num2str(jj) '.png'];
    saveas(gcf,filename)
    

    sprintf('Finished sheep %d',jj)
end


% compare variability in HF vs control animals
HFidx = [1:4 10];% 13 15 17 19 21 23 25 27 29 31 34]; 
ctrlIdx = 5:9; 
    for n = 1:length(HFidx)
        for p = 1:num_slices
            try
                HFval(p,n) = std_by_animal(HFidx(n));
            catch
                HFval(p,n) = NaN;
            end
        end
    end

    for n = 1:length(ctrlIdx)
        for p = 1:num_slices
            try
                ctrlVal(p,n) = std_by_animal(ctrlIdx(n));
            catch
                ctrlVal(p,n) = NaN;
            end
        end
    end
    
    [h,p,ci,stats] = ttest2(HFval(:),ctrlVal(:))

end