function [PDF, PES] = plotPES2(data,var1,var2)
% Function to plot potential energy surface
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))

[num_slices, num_subjects] = size(data)

% num_slices = 5;%num_slices - 1;

% myCluster = parcluster('local');
% myCluster.NumWorkers = str2double(getenv('SLURM_CPUS_ON_NODE')) / str2double(getenv('SLURM_CPUS_PER_TASK'));
% myCluster.JobStorageLocation = getenv('TMPDIR');
% myPool = parpool(myCluster, myCluster.NumWorkers);

fig = figure;
fig.Position = [100, 100, 1600, 800];
PES = [];
PDF = [];

missing_data_flag = 0;
PES(num_slices, num_subjects).F_grid = [];  % preallocate struct array
PDF(num_slices, num_subjects).F_grid = [];

t = tiledlayout(fig,num_subjects,num_slices,'TileSpacing','compact','Padding','compact');
parfor jj = 1:num_subjects
    for i = 1:num_slices
        X = [];
        if strcmp(var1, 'CO') && strcmp(var2, 'CoBF')
            xx = 0:1:50;
            yy = 0:10:400;
            F_grid = NaN(length(yy), length(xx));
            plot_type = 1;

            try
                X(:,1) = data(i,jj).CO(:,1);
                X(:,2) = data(i,jj).CoBF(:,1);

                % probability density plot
                [Xpts, Ypts] = meshgrid(xx,yy);
                pts1 = Xpts(:);
                pts2 = Ypts(:);
                pts = [pts1 pts2];
                [f, xi] = ksdensity(X,pts,'PlotFcn', 'surf');%,'Support','positive');
                F_grid = reshape(f, size(Xpts));
            catch
                F_grid = NaN;
                disp('Missing data detected')
            end




        elseif strcmp(var1, 'SBP') && strcmp(var2,'RR')
            xx = 0.18:0.02:1.4;
            yy = 70:2:200;
            F_grid = NaN(length(yy), length(xx));
            plot_type = 2;
            try
                l_RR = length(data(i,jj).RRint);
                l_MAP = length(data(i,jj).MAP);
            catch
                sprintf('Missing data for sheep %d window %d', jj, i)
                F_grid = NaN;
                continue
            end
            if l_RR ~= l_MAP
                min_length = min (l_RR,l_MAP);
                data(i,jj).RRint = data(i,jj).RRint(1:min_length);
                data(i,jj).MAP = data(i,jj).MAP(1:min_length);
            end
            X =[data(i,jj).RRint, data(i,jj).MAP];

            % probability density plot
            [Xpts, Ypts] = meshgrid(xx,yy);
            pts1 = Xpts(:);
            pts2 = Ypts(:);
            pts = [pts1 pts2];
            try
                [f, xi] = ksdensity(X,pts,'PlotFcn', 'surf');%,'Support','positive');
                F_grid = reshape(f, size(Xpts));
            catch
                F_grid = NaN;
            end



        else
            F_grid = NaN;
        end

        PES(i,jj).F_grid = -log(F_grid);
        PDF(i,jj).F_grid = F_grid;
    end
end
disp('PES size')
size(PES)
%% Exit code
% delete(myPool);
% exit
end