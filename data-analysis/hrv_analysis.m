function hrv = hrv_analysis(data,varargin)
% Function to calculate HRV metrics
% Input: data, MxN struct where the row is the time sample and the column is
%   the animal number
% varargin: integer number of time windows to use that will override the
% total number of time windows.


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
% Allow override of number of slices
varargin
if nargin > 1
    num_slices = override;
    if override > num_slices
    disp('Error: Override number of time windows entered is greater than the number of time windows in the data set')
    return
    end
end


for jj = 1:num_subjects
    for i = 1:num_slices
        
        RRint = data(i,jj).RRint;

        %Filter out RR interval > 1500 ms
        idx2rm = find(RRint > 1.5);
        RRint(idx2rm) = [];
        
        if ~isempty(RRint)
        % time domain metrics
        filename = ['hrv_time' num2str(jj) '.png'];
        size(RRint)
        any(isnan(RRint))
        [ hrv_td, plot_data ] = mhrv.hrv.hrv_time( RRint, filename );
        hrv(i,jj).hrv_td = hrv_td;

        % frequency domain metrics
        filename = {['hrv_fd_spectrum' num2str(jj) '.png'],['hrv_fd_beta' num2str(jj) '.png']};
        [ hrv_fd, pxx, f_axis, plot_data ] = mhrv.hrv.hrv_freq( RRint, filename );
        hrv(i,jj).hrv_fd = hrv_fd;

        % nonlinear analysis
        filename = ['hrv_nonlinear' num2str(jj) '.png'];
        [ hrv_nl, plot_data ] = mhrv.hrv.hrv_nonlinear( RRint, filename );
        hrv(i,jj).hrv_nl = hrv_nl;

        % fragmentation analysis
        [ hrv_frag ] = mhrv.hrv.hrv_fragmentation( RRint );
        hrv(i,jj).hrv_frag = hrv_frag;
        else
            sprintf('Sheep %d, time window %d is empty',jj,i)
            hrv(i,jj).hrv_td = NaN;
            hrv(i,jj).hrv_fd = NaN;
            hrv(i,jj).hrv_nl = NaN;
            hrv(i,jj).hrv_frag = NaN;
        end

    end
end


end