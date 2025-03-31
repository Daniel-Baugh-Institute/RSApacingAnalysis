function hrv_plot(hrv)
% Function to plot HRV metrics
% Input: hrv, MxN struct where the row is the time sample and the column is
%   the animal number

addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))

hrv_td = hrv.hrv_td;
hrv_fd = hrv.hrv_fd;
hrv_nl = hrv.hrv_nl;
hrv_frag = hrv.hrv_frag;

% Compare HRV metrics for HF and control animals
HFidx = 1:5;
ctrlIdx = 6:9;

% Get metric names from table
save('hrv_test.mat','hrv')
headers_td = hrv_td(1,1).Properties.VariableNames;
headers_fd = hrv_fd(1,1).Properties.VariableNames;
headers_nl = hrv_nl(1,1).Properties.VariableNames;
headers_frag = hrv_frag(1,1).Properties.VariableNames;


disp('h = 0 means no evidence that there are differences between groups')


disp('Frequency domain metrics')
metrics = [headers_fd]
labels = metrics;
labels = strrep(labels, '_', ' ');
HFval = zeros(1,length(HFidx));
ctrlVal = zeros(1,length(ctrlIdx));
i = 1;
for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        HFval(n) = table2array(hrv(HFidx(n)).hrv_fd(1,m));
    end

    for n = 1:length(ctrlIdx)
        ctrlVal(n) = table2array(hrv(ctrlIdx(n)).hrv_fd(1,m));
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

    % Plot
    ydata = [ctrlVal, HFval];
    xgroupdata = [categorical(repmat({'Control'}, 1, length(ctrlVal(:)))), ....
    categorical(repmat({'HF'}, 1, length(HFval(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata);
    hold on;

    % Plot raw data points directly above their corresponding box plots
    xHF = 1.25*ones(size(HFval(:)));
    xCtrl = 0.75 * ones(size(ctrlVal(:)));
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
        HFval(n) = table2array(hrv(HFidx(n)).hrv_td(1,m));
    end

    for n = 1:length(ctrlIdx)
        ctrlVal(n) = table2array(hrv(ctrlIdx(n)).hrv_td(1,m));
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

    % Plot
    ydata = [ctrlVal, HFval];
    xgroupdata = [categorical(repmat({'Control'}, 1, length(ctrlVal(:)))), ....
    categorical(repmat({'HF'}, 1, length(HFval(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata);
    hold on;

    % Plot raw data points directly above their corresponding box plots
    xHF = 1.25*ones(size(HFval(:)));
    xCtrl = 0.75 * ones(size(ctrlVal(:)));
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


disp('Nonlinear analysis metrics')
metrics = [headers_nl];
labels = metrics;
labels = strrep(labels, '_', ' ');
for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        HFval(n) = table2array(hrv(HFidx(n)).hrv_nl(1,m));
    end

    for n = 1:length(ctrlIdx)
        ctrlVal(n) = table2array(hrv(ctrlIdx(n)).hrv_nl(1,m));
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

    % Plot
    ydata = [ctrlVal, HFval];
    xgroupdata = [categorical(repmat({'Control'}, 1, length(ctrlVal(:)))), ....
    categorical(repmat({'HF'}, 1, length(HFval(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata);
    hold on;

    % Plot raw data points directly above their corresponding box plots
    xHF = 1.25*ones(size(HFval(:)));
    xCtrl = 0.75 * ones(size(ctrlVal(:)));
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

disp('Fragmentation analysis metrics')
metrics = [headers_frag];
labels = metrics;
labels = strrep(labels, '_', ' ');
for m = 1:length(metrics)
    for n = HFidx(1):HFidx(end)
        HFval(n) = table2array(hrv(HFidx(n)).hrv_frag(1,m));
    end

    for n = 1:length(ctrlIdx)
        ctrlVal(n) = table2array(hrv(ctrlIdx(n)).hrv_frag(1,m));
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

    % Plot
    ydata = [ctrlVal, HFval];
    xgroupdata = [categorical(repmat({'Control'}, 1, length(ctrlVal(:)))), ....
    categorical(repmat({'HF'}, 1, length(HFval(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata);
    hold on;

    % Plot raw data points directly above their corresponding box plots
    xHF = 1.25*ones(size(HFval(:)));
    xCtrl = 0.75 * ones(size(ctrlVal(:)));
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