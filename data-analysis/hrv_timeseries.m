function hrv_timeseries(hrv)
% Function to calculate HRV metrics
% Input: data, MxN struct where the row is the time sample and the column is
%   the animal number

[num_slices, num_subjects] = size(hrv);

addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
hrv = hrv_analysis(data,varargin);
hrv_td = hrv.hrv_td;
hrv_fd = hrv.hrv_fd;
hrv_nl = hrv.hrv_nl;
hrv_frag = hrv.hrv_frag;

% Compare HRV metrics for HF and control animals
HFidx = 1:5;
ctrlIdx = 6:10;

% Get metric names from table
headers_td = hrv_td(1,1).Properties.VariableNames;
headers_fd = hrv_fd(1,1).Properties.VariableNames;
headers_nl = hrv_nl(1,1).Properties.VariableNames;
headers_frag = hrv_frag(1,1).Properties.VariableNames;


% Plot timeseries progression of hrv metrics for each animal


disp('Frequency domain metrics')
metrics = [headers_fd]
labels = metrics;
labels = strrep(labels, '_', ' ');
HFval = zeros(1,length(HFidx));
ctrlVal = zeros(1,length(ctrlIdx));
timevec = 0:12:12*num_slices;

for m = 1:length(metrics)
    figure;
    for j = 1:num_subjects
        
        % plot metric
        if j < 6 % HF
            color = 'r';
        else
            color = 'b';
        end
        timeseries_metric = table2array(hrv(:,j).hrv_fd(1,m));
        h = plot(timevec,timeseries_metric,'o-','Color',color,'HandleVisibility','off');

        % Show legend for one HF and one control animal
        if j == 1 || j == 6
            h.Visible = 'on';
        end

        hold on
        
    end
    ylabel(metrics{m});
    xlabel('Time (min)')
    legend({'Control','HF'}, 'Location', 'best');
    hold off;
    set(gca,'FontSize',16)
    saveas(gcf,[metrics{m} '_timeseries.png'])


end

end