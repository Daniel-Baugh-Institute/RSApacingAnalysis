function CO_work_curve(struct_CO_work, filename)
% Function to plot CO vs work for multiple animals
[~,num_subjects] = size(struct_CO_work);
num_subjects = 12;


% sample IDS for RSA and mono paced
pacedRSAidx = [11 14 16 18 20 22];  % paced
baseRSAidx = [10 13 15 17 19 21];  % unpaced
offPaceRSAidx = 12;
ctrlIdx = 5:9;

% Compare unpaced HF and mono paced HF
baseMonoIdx = [23 25 27 29 31 34];  % unpaced
pacedMonoIdx = [24 26 28 30 32 35];  % paced
offPaceMono = 33;

baselineIdx = [baseRSAidx baseMonoIdx];

legend_entries = [];  % To hold legend handles
legend_labels = {};   % To hold legend labels

for i = 1:num_subjects
    sampleID = baselineIdx(i)
    if ismember(sampleID,baseMonoIdx)
        color = 'r';
        label = 'HF';
        color2 = 'm';
        label2 = 'Mono paced';
    elseif ismember(sampleID,pacedRSAidx)
        color = 'c';
        label = 'RSA paced';
    elseif ismember(sampleID,baseRSAidx)
        color = 'r';
        label = 'HF';
        color2 = 'c';
        label2 = 'RSA paced';
    elseif ismember(sampleID,pacedMonoIdx)
        color = 'm';
        label = 'Mono paced';
    elseif ismember(sampleID,ctrlIdx)
        color = 'b';
        label = 'Control';
    elseif ismember(sampleID,offPaceMono) || ismember(i,offPaceRSAidx)
        color = 'g';
        label = 'Off-pace';
    else
        disp('Sample index out of range')
    end 

    CO = struct_CO_work(sampleID).CO;
    work = struct_CO_work(sampleID).work;
    CO_paced = struct_CO_work(sampleID+ 1).CO;
    work_paced = struct_CO_work(sampleID+1).work;
    figure;
    hold on
        % Plot data points
    if ~any(strcmp(legend_labels, label))
        % Plot with visible legend only once per group
        h = plot(CO, work, 'o', 'DisplayName', label,'MarkerEdgeColor',color);
        
        plot(CO_paced,work_paced, 'o', 'DisplayName', label2,'MarkerEdgeColor',color2)
        legend_entries(end+1) = h;
        legend_labels{end+1} = label;
    else
        % Suppress legend entry for other subjects
        plot(CO, work, 'o', 'Color', color, 'HandleVisibility', 'off');
        plot(CO_paced, work_paced, 'o', 'Color', color2, 'HandleVisibility', 'off');
    end

    % Add linear fit
    if length(CO) >= 2  % At least two points required
        coeffs = polyfit(CO, work, 1);
        xfit = linspace(min(CO), max(CO), 100);
        yfit = polyval(coeffs, xfit);
        plot(xfit, yfit, '--', 'Color', color, 'HandleVisibility', 'off');
    end

    if length(CO_paced) >= 2  % At least two points required
        coeffs = polyfit(CO_paced, work_paced, 1);
        xfit = linspace(min(CO_paced), max(CO_paced), 100);
        yfit = polyval(coeffs, xfit);
        plot(xfit, yfit, '--', 'Color', color2, 'HandleVisibility', 'off');
    end

    xlabel('Cardiac output (mL)')
ylabel('Work per beat (mm Hg * mL)')
title(['Animal ' num2str(sampleID)])
set(gca, 'FontSize', 16)
legend(legend_entries, legend_labels, 'Location', 'best')
filenameFull = [filename '_S' num2str(sampleID) '.png'];
saveas(gcf, filenameFull)
end


end