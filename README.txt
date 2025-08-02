# Code to reproduce figures in Chapter 6 of the dissertation Modeling and analysis of autonomic circuits underlying closed-loop cardiovascular control to identify key contributors to individual variability by Michelle Gee
Summer 2025

# Purpose
The purpose of this code is to test the hypothesis that reinstating heart rate variability in a sheep model of heart failure increases cardiac efficiency
We test this hypothesis using RR interval, arterial pressure, cardiac output, and coronary blood flow data from healthy and heart failure sheep treated with monotonic pacing and pacing mimicking respiratory sinus arrhythmia from Shanks et al. 2022
We use these data to calculate pressure-volume work

# Note
This code uses a modified version of the mhrv toolbox (https://github.com/physiozoo/mhrv.git)
The files in /mhrv/+mhrv/+hrv were modified
Additional arguments were added to allow for plotting options
The outputs of hrv_fragmentation.m were modified to include the indices of the start and end of fragmentation windows
An additional function, hrv_fragmentation_filtered.m was added to test the effect of noise in the RR interval data on the results

# Reproducing the figures
Run mainRamchandra.m to reproduce the figures. Please note that the data are not available here but are available upon request from Shanks et al.

If you have questions or find a bug, please contact mmgee@udel.edu

# References
Behar J. A., Rosenberg A. A. et al. (2018) ‘PhysioZoo: a novel open access platform for heart rate variability analysis of mammalian electrocardiographic data.’ Frontiers in Physiology.

Rosenberg, A. A. (2018) ‘Non-invasive in-vivo analysis of intrinsic clock-like pacemaker mechanisms: Decoupling neural input using heart rate variability measurements.’ MSc Thesis. Technion, Israel Institute of Technology.

Shanks J, Abukar Y, Lever NA, Pachen M, LeGrice IJ, Crossman DJ, Nogaret A, Paton JFR, Ramchandra R. Reverse re-modelling chronic heart failure by reinstating heart rate variability. Basic Res Cardiol. 2022 Feb 1;117(1):4. doi: 10.1007/s00395-022-00911-0. PMID: 35103864; PMCID: PMC8807455.