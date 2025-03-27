function [coeff, score, latent] = pca_timeseries(data,filename)
% PCA of RR interval, MAP, CoBF, CO for each heart beat
% data: mxn struct with fields for RRint, MAP, CoBF, CO. n is the number of
% animals. m is the number of time slices from an animal
% Have to remove animal 10 since data is missing for CoBF

addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))

[num_slices, num_subjects] = size(data);
% num_slices = 1; % for now, just using one slice

% Define colors
colors = [  % First 4 subjects (warm colors)
    1, 0, 0;   % Red
    1, 0.5, 0; % Orange
    1, 0, 1;   % Magenta
    1, 0.6, 0.6; % Pink
    % Last 5 subjects (cool colors)
    0, 0, 1;   % Blue
    0, 1, 0;   % Green
    0.5, 0, 0.5; % Purple
    0, 0.7, 1; % Light Blue
    0, 0.5, 0.5 % Teal
];

% Data formatting and normalization
RRint = [];
MAP = [];
CO = [];
CoBF = [];
num_points = zeros(num_subjects,1);
subject_indices = [];

for jj = 1:num_subjects
    for i = 1:num_slices
        num_pts = length(data(i,jj).RRint);
        RRint = [RRint; data(i,jj).RRint];
        MAP = [MAP; data(i,jj).MAP];
        CO = [CO; data(i,jj).CO_mean];
        CoBF = [CoBF; data(i,jj).CoBF_mean];
        num_points(jj) = num_pts;
        subject_indices = [subject_indices; jj * ones(num_pts, 1)];
    end
end

X(:,1) = RRint(:);
X(:,2) = MAP(:);
X(:,3) = CO(:);
X(:,4) = CoBF(:);

X = normalize(X,1); % z-score normalization

% PCA
[coeff, score, latent] = pca(X,'Centered','on');
save('pca.mat','coeff','score','subject_indices')

% Plotting
figure;

% for jj = 1:num_subjects
biplot(coeff(:,1:2),'scores',score(:,1:2),'varlabels',{'RR','MAP','CO','CoBF'},'HandleVisibility','off');
% end
saveas(gcf,'PCA_compare.png')

score_plot = max(max(coeff(:,1:2)))*score/max(max(abs(score(:,1:2))));
figure;
hold on 
biplot(coeff(:,1:2),'scores',score(:,1:2),'varlabels',{'RR','MAP','CO','CoBF'},'HandleVisibility','off');
for jj = 1:num_subjects
    scatter(score_plot(subject_indices == jj, 1), score_plot(subject_indices == jj, 2), 36, colors(jj,:), 'filled');
end

xlim([-1 1])
ylim([-1 1])
xlabel('Component 1')
ylabel('Component 2')
legend(arrayfun(@(x) sprintf('Subject %d', x), 1:num_subjects, 'UniformOutput', false))
set(gcf,'Position',[0,0,600,500])
hold off
saveas(gcf, filename);


end