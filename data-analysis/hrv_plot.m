function hrv_plot(hrv)
% Function to plot HRV metrics
% Input: hrv, MxN struct where the row is the time sample and the column is
%   the animal number

addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
[num_slices, num_subjects] = size(hrv);
num_slices = 1;

hrv_td = hrv.hrv_td;
hrv_fd = hrv.hrv_fd;
hrv_nl = hrv.hrv_nl;
hrv_frag = hrv.hrv_frag;

%% CHANGE HERE FOR COMPARING HF VS CONTROL
% Compare HRV metrics for HF and control animals
% HFidx = 1:5;
% ctrlIdx = 6:10;

% Compare HRV metrics for expanded dataset with extra HF samples
HFidx = [1:4 10 13 15 17 19 21 23 25 27 29 31 34];
ctrlIdx = 5:9;
%%

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
HFval = zeros(num_slices,length(HFidx));
ctrlVal = zeros(num_slices,length(ctrlIdx));

for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        for p = 1:num_slices
            try
                HFval(p,n) = table2array(hrv(p,HFidx(n)).hrv_fd(1,m));
            catch
                HFval(p,n) = NaN;
            end
        end
    end

    for n = 1:length(ctrlIdx)
        for p = 1:num_slices
            try
                ctrlVal(p,n) = table2array(hrv(p,ctrlIdx(n)).hrv_fd(1,m));
            catch
                ctrlVal(p,n) = NaN;
            end
        end
    end
    disp(['Comparing ' metrics{m} ' between HF and control groups'])
    [h,p,ci,stats] = ttest2(HFval(:),ctrlVal(:))

    % plot histogram of metrics ratio 
    figure;
    h1 = histogram(HFval(:),'FaceAlpha',0.4,'FaceColor','r');
    hold on
    h2 = histogram(ctrlVal(:),'FaceAlpha',0.4,'FaceColor','b');
    binWidth = h1.BinWidth;% = binWidth(m);
    h2.BinWidth = binWidth;
    xlabel(labels{m})
    ylabel('Counts')
    legend('Heart failure', 'Control')
    saveas(gcf,[metrics{m} '_hist.png'])

    % Plot
    ydata = [ctrlVal(:); HFval(:)];
    xgroupdata = [categorical(repmat({'Control'}, 1, length(ctrlVal(:)))), ....
    categorical(repmat({'HF'}, 1, length(HFval(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata,'MarkerStyle','none');
    hold on;

    % Plot raw data points directly above their corresponding box plots
    a = -0.1;
    b = 0.1;
    n = numel(HFval);
    jitterHF = a + (b-a).*rand(n,1);
    n = numel(ctrlVal);
    jitterCtrl = a + (b-a).*rand(n,1);
    xHF = 1.25*ones(size(HFval(:))) + jitterHF;
    xCtrl = 0.75 * ones(size(ctrlVal(:))) + jitterCtrl;
    scatter(xHF, HFval(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);
    scatter(xCtrl, ctrlVal(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);

    ax = gca;
    set(ax,'xticklabel',[])
    ylabel(metrics{m});
    legend({'Control','HF'}, 'Location', 'best');
    hold off;
    set(gca,'FontSize',16)
    saveas(gcf,[metrics{m} '_box.png'])


end

disp('Time domain metrics')
metrics = [headers_td];
labels = metrics;
labels = strrep(labels, '_', ' ');
for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        for p = 1:num_slices
            try
                HFval(p,n) = table2array(hrv(p,HFidx(n)).hrv_td(1,m));
            catch
                HFval(p,n) = NaN;
            end
        end
    end

    for n = 1:length(ctrlIdx)
        for p = 1:num_slices
            try
                ctrlVal(p,n) = table2array(hrv(p,ctrlIdx(n)).hrv_td(1,m));
            catch
                ctrlVal(p,n) = NaN;
            end
        end
    end
    disp(['Comparing ' metrics{m} ' between HF and control groups'])
    [h,p,ci,stats] = ttest2(HFval(:),ctrlVal(:))

    % plot histogram of metrics ratio 
    figure;
    h1 = histogram(HFval(:),'FaceAlpha',0.4,'FaceColor','r');
    hold on
    h2 = histogram(ctrlVal(:),'FaceAlpha',0.4,'FaceColor','b');
    binWidth = h1.BinWidth;% = binWidth(m);
    h2.BinWidth = binWidth;
    xlabel(labels{m})
    ylabel('Counts')
    legend('Heart failure', 'Control')
    saveas(gcf,[metrics{m} 'paced_hist.png'])

    % Plot
    ydata = [ctrlVal(:); HFval(:)];
    xgroupdata = [categorical(repmat({'Control'}, 1, length(ctrlVal(:)))), ....
    categorical(repmat({'HF'}, 1, length(HFval(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata,'MarkerStyle','none');
    hold on;

    % Plot raw data points directly above their corresponding box plots
    a = -0.1;
    b = 0.1;
    n = numel(HFval);
    jitterHF = a + (b-a).*rand(n,1);
    n = numel(ctrlVal);
    jitterCtrl = a + (b-a).*rand(n,1);
    xHF = 1.25*ones(size(HFval(:))) + jitterHF;
    xCtrl = 0.75 * ones(size(ctrlVal(:))) + jitterCtrl;
    scatter(xHF, HFval(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);
    scatter(xCtrl, ctrlVal(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);

    ax = gca;
    set(ax,'xticklabel',[])
    ylabel(metrics{m});
    legend({'Control','HF'}, 'Location', 'best');
    hold off;
    set(gca,'FontSize',16)
    saveas(gcf,[metrics{m} 'paced_box.png'])

    % Combined boxplot per slice
    figure;
    hold on;
    for p = 1:num_slices
        hfSlice = HFval(p, :);
        ctrlSlice = ctrlVal(p, :);
        ydata = [ctrlSlice(:); hfSlice(:)];
        xgroup = [repmat(p - 0.2, numel(ctrlSlice), 1); repmat(p + 0.2, numel(hfSlice), 1)];
        groups = [categorical(repmat({'Control'}, numel(ctrlSlice), 1));
                  categorical(repmat({'HF'}, numel(hfSlice), 1))];

        % Box for control and HF
        boxchart((repmat(p - 0.2, numel(ctrlSlice), 1))/2, ctrlSlice(:), 'BoxFaceColor', 'b', 'MarkerStyle', 'none','BoxWidth',0.1);
        boxchart((repmat(p + 0.2, numel(hfSlice), 1))/2, hfSlice(:), 'BoxFaceColor', 'r', 'MarkerStyle', 'none','BoxWidth',0.1);

        % Overlay raw data
        scatter((randn(size(ctrlSlice(:)))*0.05 + (p - 0.2))/2, ctrlSlice(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);
        scatter((randn(size(hfSlice(:)))*0.05 + (p + 0.2))/2, hfSlice(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);
    end
    xlabel('Time (h)')
    ylabel(labels{m});
    xlim([0 30])
    legend({'Control Box','HF Box','Control Data','HF Data'}, 'Location', 'best');
    set(gcf,'Position',[0 0 2400 500])
    set(gca, 'FontSize', 16)
    % title(['Per-Time-Slice Boxplot for ' labels{m}])
    saveas(gcf, sprintf('%s_box_per_slice_paced.png', metrics{m}))


end


disp('Nonlinear analysis metrics')
metrics = [headers_nl]
labels = metrics;
labels = strrep(labels, '_', ' ');
for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        for p = 1:num_slices
            try
                HFval(p,n) = table2array(hrv(p,HFidx(n)).hrv_nl(1,m));
            catch
                HFval(p,n) = NaN;
            end
        end
    end

    for n = 1:length(ctrlIdx)
        for p = 1:num_slices
            try
                ctrlVal(p,n) = table2array(hrv(p,ctrlIdx(n)).hrv_nl(1,m));
            catch
                ctrlVal(p,n) = NaN;
            end
        end
    end
    disp(['Comparing ' metrics{m} ' between HF and control groups'])
    [h,p,ci,stats] = ttest2(HFval(:),ctrlVal(:))

    % plot histogram of metrics ratio 
    figure;
    h1 = histogram(HFval(:),'FaceAlpha',0.4,'FaceColor','r');
    hold on
    h2 = histogram(ctrlVal(:),'FaceAlpha',0.4,'FaceColor','b');
    binWidth = h1.BinWidth;% = binWidth(m);
    h2.BinWidth = binWidth;
    xlabel(labels{m})
    ylabel('Counts')
    legend('Heart failure', 'Control')
    saveas(gcf,[metrics{m} '_hist.png'])

    % Plot
    ydata = [ctrlVal(:); HFval(:)];
    xgroupdata = [categorical(repmat({'Control'}, 1, length(ctrlVal(:)))), ....
    categorical(repmat({'HF'}, 1, length(HFval(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata,'MarkerStyle','none');
    hold on;

    % Plot raw data points directly above their corresponding box plots
    a = -0.1;
    b = 0.1;
    n = numel(HFval);
    jitterHF = a + (b-a).*rand(n,1);
    n = numel(ctrlVal);
    jitterCtrl = a + (b-a).*rand(n,1);
    xHF = 1.25*ones(size(HFval(:))) + jitterHF;
    xCtrl = 0.75 * ones(size(ctrlVal(:))) + jitterCtrl;
    scatter(xHF, HFval(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);
    scatter(xCtrl, ctrlVal(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);

    ax = gca;
    set(ax,'xticklabel',[])
    ylabel(metrics{m});
    legend({'Control','HF'}, 'Location', 'best');
    hold off;
    set(gca,'FontSize',16)
    saveas(gcf,[metrics{m} '_box.png'])

    if m == 3 % alpha 1
        alpha1HF = HFval;
        alpha1Ctrl = ctrlVal;
    elseif m == 4 % alpha2
        alpha2HF = HFval;
        alpha2Ctrl = ctrlVal;
    end

end

% Alpha 1 and alpha 2 phase plot
figure;
h1 = scatter(alpha1Ctrl(1), alpha2Ctrl(1), 'bo', 'filled', ...
    'MarkerFaceAlpha', 0.5, 'DisplayName', 'Control');
hold on;

% Plot remaining Control points without legend entry
scatter(alpha1Ctrl(2:end), alpha2Ctrl(2:end), 'bo', 'filled', ...
    'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');

% Plot first HF point (for legend)
h2 = scatter(alpha1HF(1), alpha2HF(1), 'r^', 'filled', ...
    'MarkerFaceAlpha', 0.5, 'DisplayName', 'HF');

% Plot remaining HF points without legend entry
scatter(alpha1HF(2:end), alpha2HF(2:end), 'r^', 'filled', ...
    'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
legend([h1 h2], 'Location', 'best');
xlabel('Alpha 1')
ylabel('Alpha 2')
hold off;
set(gca,'FontSize',16)
saveas(gcf,'alpha_phase_plot.png')






disp('Fragmentation analysis metrics')
metrics = [headers_frag];
labels = metrics;
labels = strrep(labels, '_', ' ');
for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        for p = 1:num_slices
            try
                HFval(p,n) = table2array(hrv(p,HFidx(n)).hrv_frag(1,m));
            catch
                HFval(p,n) = NaN;
            end
        end
    end

    for n = 1:length(ctrlIdx)
        for p = 1:num_slices
            try
                ctrlVal(p,n) = table2array(hrv(p,ctrlIdx(n)).hrv_frag(1,m));
            catch
                ctrlVal(p,n) = NaN;
            end
        end
    end
    disp(['Comparing ' metrics{m} ' between HF and control groups'])
    [h,p,ci,stats] = ttest2(HFval(:),ctrlVal(:))

    % plot histogram of metrics ratio 
    figure;
    h1 = histogram(HFval(:),'FaceAlpha',0.4,'FaceColor','r');
    hold on
    h2 = histogram(ctrlVal(:),'FaceAlpha',0.4,'FaceColor','b');
    binWidth = h1.BinWidth;% = binWidth(m);
    h2.BinWidth = binWidth;
    xlabel(labels{m})
    ylabel('Counts')
    legend('Heart failure', 'Control')
    saveas(gcf,[metrics{m} '_hist.png'])

    % Plot
    ydata = [ctrlVal(:); HFval(:)];
    xgroupdata = [categorical(repmat({'Control'}, 1, length(ctrlVal(:)))), ....
    categorical(repmat({'HF'}, 1, length(HFval(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata,'MarkerStyle','none');
    hold on;

    % Plot raw data points directly above their corresponding box plots
    a = -0.1;
    b = 0.1;
    n = numel(HFval);
    jitterHF = a + (b-a).*rand(n,1);
    n = numel(ctrlVal);
    jitterCtrl = a + (b-a).*rand(n,1);
    xHF = 1.25*ones(size(HFval(:))) + jitterHF;
    xCtrl = 0.75 * ones(size(ctrlVal(:))) + jitterCtrl;
    scatter(xHF, HFval(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);
    scatter(xCtrl, ctrlVal(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);

    ax = gca;
    set(ax,'xticklabel',[])
    ylabel(metrics{m});
    legend({'Control','HF'}, 'Location', 'best');
    hold off;
    set(gca,'FontSize',16)
    saveas(gcf,[metrics{m} '_box.png'])


end
end