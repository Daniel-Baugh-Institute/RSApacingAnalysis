function hrv_timeseries(hrv)
% Function to calculate HRV metrics
% Input: data, MxN struct where the row is the time sample and the column is
%   the animal number

[num_slices, num_subjects] = size(hrv);

addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))

%% CHANGE THIS SECTION FOR DIFFERENT DATA INPUTS
% Compare HRV metrics for HF and control animals
HFidx = 1:5;
ctrlIdx = 6:10;
timevec = 0:0.5:30;%0:12:12*(num_slices-1);


% Compare HRV metrics for expanded dataset with extra HF samples
HFidx = [1:4 10 13 15 17 19 21 23 25 27 29 31 34];
ctrlIdx = 5:9;
AIDs = [1:10 13 15 17 19 21 23 25 27 29 31 34]; 

num_subjects = sum(numel(HFidx), numel(ctrlIdx));
%%


hrv_td = hrv.hrv_td;
hrv_fd = hrv.hrv_fd;
hrv_nl = hrv.hrv_nl;
hrv_frag = hrv.hrv_frag;



% Get metric names from table
headers_td = hrv_td.Properties.VariableNames;
headers_fd = hrv_fd.Properties.VariableNames;
headers_nl = hrv_nl.Properties.VariableNames;
headers_frag = hrv_frag.Properties.VariableNames;


% Plot timeseries progression of hrv metrics for each animal
disp_labels = {'Frequency domain metrics','Time domain metrics','Non-linear metrics','Fragmentation analysis metrics'};
header_names = {'headers_fd','headers_td','headers_nl','headers_frag'};
struct_names = {'hrv_fd','hrv_td','hrv_nl','hrv_frag'};
for p = 1:4
disp(disp_labels{p})
metrics = eval(header_names{p});
labels = metrics;
labels = strrep(labels, '_', ' ');
% HFval = zeros(1,length(HFidx));
% ctrlVal = zeros(1,length(ctrlIdx));


for m = 1:length(metrics)
    figure;
    for j = 1:num_subjects
        for k = 1:num_slices
            name = ['hrv(k,AIDs(j)).' struct_names{p} '(1,m)'];
            try
                timeseries_metric(k) = table2array(eval(name));
            catch
                timeseries_metric(k) = NaN;
                sprintf('NaN for sample %d, time window %d',AIDs(j),k)
                disp('metric: ')
                metrics{m}
            end
        end
        % plot metric
        if AIDs(j) < 6 || AIDs(j) > 10 % HF
            color = 'r';
            label = 'HF';
        else
            color = 'b';
            label = 'Control';
        end

    
        % Show legend for one HF and one control animal
        if (j == 1 && color == 'r') || (j == 6 && color == 'b')
            h = plot(timevec(2:end),timeseries_metric,'o-','Color',color,'DisplayName',label);
        else
            h = plot(timevec(2:end),timeseries_metric,'o-','Color',color,'HandleVisibility','off');
        end

        hold on
        
    end
    ylabel(metrics{m});
    xlabel('Time (hrs)')
    xlim([0 30])
    legend({'HF','Control'}, 'Location', 'best');
    hold off;
    set(gca,'FontSize',16)
    saveas(gcf,[metrics{m} '_30m_paced_timeseries.png'])


end
end

end