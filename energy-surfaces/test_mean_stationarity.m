function T = test_mean_stationarity(data)
num_subjects = length(data);
hStore = zeros(num_subjects,1);
pStore = zeros(num_subjects,1);
for i = 1:num_subjects-1
    y = data(i).CoBF;%X(:,1); % second column is SBP, first column of X is RR interval
    [h,pValue,~,~] = adftest(y);
    hStore(i) = h;
    pStore(i) = pValue;
end
T = table(hStore,pStore);

end