function hrv(data)
% Function to calculate HRV metrics
% Input: data, MxN struct where the row is the time sample and the column is
%   the animal number


% Adapted from:
% * Rosenberg, A. A. (2018) ‘Non-invasive in-vivo analysis of intrinsic clock-like
%   pacemaker mechanisms: Decoupling neural input using heart rate variability
%   measurements.’ MSc Thesis. Technion, Israel Institute of Technology.
% 
% * Behar J. A., Rosenberg A. A. et al. (2018) ‘PhysioZoo: a novel open access
%   platform for heart rate variability analysis of mammalian
%   electrocardiographic data.’ Frontiers in Physiology.


addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
[num_slices, num_subjects] = size(data);

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

% HRV analysis
i = 1;
for jj = 1:num_subjects 

        RRint = data(i,jj).RRint;

        %Filter out RR interval > 1500 ms
        idx2rm = find(RRint > 1.5);
        RRint(idx2rm) = [];

        % time domain metrics
        filename = ['hrv_time' num2str(jj) '.png'];
        [ hrv_td, plot_data ] = mhrv.hrv.hrv_time( RRint, filename );
        hrv(i,jj).hrv_td = table2array(hrv_td)

        % frequency domain metrics
        filename = {['hrv_fd_spectrum' num2str(jj) '.png'],['hrv_fd_beta' num2str(jj) '.png']};
        [ hrv_fd, pxx, f_axis, plot_data ] = mhrv.hrv.hrv_freq( RRint, filename );
        hrv(i,jj).hrv_fd = table2array(hrv_fd);

        % nonlinear analysis
        filename = ['hrv_nonlinear' num2str(jj) '.png'];
        [ hrv_nl, plot_data ] = mhrv.hrv.hrv_nonlinear( RRint, filename );
        hrv(i,jj).hrv_nl = table2array(hrv_nl);

        % fragmentation analysis
        [ hrv_frag ] = mhrv.hrv.hrv_fragmentation( RRint );
        hrv(i,jj).hrv_frag = table2array(hrv_frag);
       
        
end


% Compare HRV metrics for HF and control animals
HFidx = 1:5;
ctrlIdx = 6:9;

% Get metric names from table
headers_td = hrv_td.Properties.VariableNames;
headers_fd = hrv_fd.Properties.VariableNames;
headers_nl = hrv_nl.Properties.VariableNames;
headers_frag = hrv_frag.Properties.VariableNames;


disp('h = 0 means no evidence that there are differences between groups')


disp('Frequency domain metrics')
metrics = [headers_fd]
labels = metrics;
labels = strrep(labels, '_', ' ');
HFval = zeros(1,length(HFidx));
ctrlVal = zeros(1,length(ctrlIdx));
% binWidth = [1 2 1 1 1 1 1 5 2 3 4];
for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        HFval(n) = hrv(i,HFidx(n)).hrv_fd(1,m);
    end

    for n = 1:length(ctrlIdx)
        ctrlVal(n) = hrv(i,ctrlIdx(n)).hrv_fd(1,m);
    end
    disp(['Comparing ' metrics{m} ' between HF and control groups'])
    [h,p,ci,stats] = ttest2(HFval,ctrlVal)

    % plot histogram of metrics ratio 
    figure;
    h1 = histogram(HFval,'FaceAlpha',0.4,'FaceColor','r');
    hold on
    h2 = histogram(ctrlVal,'FaceAlpha',0.4,'FaceColor','b');
    binWidth = h1.BinWidth;% = binWidth(m);
    h2.BinWidth = binWidth;
    xlabel(labels{m})
    ylabel('Counts')
    legend('Heart failure', 'Control')
    saveas(gcf,[metrics{m} '_hist.png'])

end

disp('Time domain metrics')
metrics = [headers_td];
labels = metrics;
labels = strrep(labels, '_', ' ');
for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        HFval(n) = hrv(i,HFidx(n)).hrv_td(1,m);
    end

    for n = 1:length(ctrlIdx)
        ctrlVal(n) = hrv(i,ctrlIdx(n)).hrv_td(1,m);
    end
    disp(['Comparing ' metrics{m} ' between HF and control groups'])
    [h,p,ci,stats] = ttest2(HFval,ctrlVal)

    % plot histogram of metrics ratio 
    figure;
    h1 = histogram(HFval,'FaceAlpha',0.4,'FaceColor','r');
    hold on
    h2 = histogram(ctrlVal,'FaceAlpha',0.4,'FaceColor','b');
    binWidth = h1.BinWidth;% = binWidth(m);
    h2.BinWidth = binWidth;
    xlabel(labels{m})
    ylabel('Counts')
    legend('Heart failure', 'Control')
    saveas(gcf,[metrics{m} '_hist.png'])
end


disp('Nonlinear analysis metrics')
metrics = [headers_nl];
labels = metrics;
labels = strrep(labels, '_', ' ');
for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        HFval(n) = hrv(i,HFidx(n)).hrv_nl(1,m);
    end

    for n = 1:length(ctrlIdx)
        ctrlVal(n) = hrv(i,ctrlIdx(n)).hrv_nl(1,m);
    end
    disp(['Comparing ' metrics{m} ' between HF and control groups'])
    [h,p,ci,stats] = ttest2(HFval,ctrlVal)

    % plot histogram of metrics ratio 
    figure;
    h1 = histogram(HFval,'FaceAlpha',0.4,'FaceColor','r');
    hold on
    h2 = histogram(ctrlVal,'FaceAlpha',0.4,'FaceColor','b');
    binWidth = h1.BinWidth;% = binWidth(m);
    h2.BinWidth = binWidth;
    xlabel(labels{m})
    ylabel('Counts')
    legend('Heart failure', 'Control')
    saveas(gcf,[metrics{m} '_hist.png'])
end

disp('Fragmentation analysis metrics')
metrics = [headers_frag];
labels = metrics;
labels = strrep(labels, '_', ' ');
for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        HFval(n) = hrv(i,HFidx(n)).hrv_frag(1,m);
    end

    for n = 1:length(ctrlIdx)
        ctrlVal(n) = hrv(i,ctrlIdx(n)).hrv_frag(1,m);
    end
    disp(['Comparing ' metrics{m} ' between HF and control groups'])
    [h,p,ci,stats] = ttest2(HFval,ctrlVal)

    % plot histogram of metrics ratio 
    figure;
    h1 = histogram(HFval,'FaceAlpha',0.4,'FaceColor','r');
    hold on
    h2 = histogram(ctrlVal,'FaceAlpha',0.4,'FaceColor','b');
    binWidth = h1.BinWidth;% = binWidth(m);
    h2.BinWidth = binWidth;
    xlabel(labels{m})
    ylabel('Counts')
    legend('Heart failure', 'Control')
    saveas(gcf,[metrics{m} '_hist.png'])
end
end