%%
RR = diff(V20_31_baseline_4_Ch8.times);
RR_intervals = RR;  % RR intervals
sample_end = 10000;
RR_times = V20_31_baseline_4_Ch8.times(1:sample_end-1) + RR_intervals(1:sample_end-1) / 2;  % Midpoint between R-peaks

% 2. Define resampled time grid (1000 Hz = 1 ms step)
t_resampled = V20_31_baseline_4_Ch8.times(1):0.001:V20_31_baseline_4_Ch8.times(sample_end);  % in seconds

% 3. Interpolate RR intervals onto the new time grid
resampledRR = interp1(RR_times, RR_intervals(1:sample_end-1), t_resampled, 'linear', 'extrap');

% Optional: Plot to visualize
figure;
plot(RR_times, RR_intervals(1:sample_end-1), 'ko-', 'DisplayName', 'Original RR');
hold on;
plot(t_resampled, resampledRR, 'r', 'DisplayName', 'Resampled @ 1000 Hz');
xlabel('Time (s)');
ylabel('RR Interval (s)');
legend;
title('Resampled RR Interval at 1000 Hz');

A = [RR_intervals(1:50000), V20_31_baseline_4_Ch1.values(1:50000), V20_31_baseline_4_Ch4.values(1:50000), V20_31_baseline_4_Ch3.values(1:50000)];

plot_time = 0:0.001:5000/1000;
figure;
plot(plot_time(1:end-1),V20_31_baseline_4_Ch3.values(1:5000))
xlabel('Time (s)');
ylabel('CoBF (mL/min)');
saveas(gcf,'plot-raw-CoBF.png')
% writematrix(A,'ramchandra_raw.csv')