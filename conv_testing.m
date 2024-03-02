%% setup data
%training data -> do we want to shuffle?
trainData = readmatrix('data/train.csv', 'Range', 'C2:ADF27456');
trainLabel = readmatrix('data/train.csv', 'Range', 'B2:B27456');
%testing data
testData = readmatrix('data/test.csv', 'Range', 'B2:ADE7173');
%get id labels (to use for output file)
testID = readmatrix('data/test.csv', 'Range', 'A2:A7173');

%% setup network


%% train network


%% validate network performance


%% test network


%% produce figs for doc ?


%% export resutls
testResults = []; %classification (w forward) of test data

%create table and export
columnNames = {'id', 'label'};
fileName = 'convResults.csv';
outputTable = array2table([testID, testResults], 'VariableNames', columnNames);
writetable(outputTable, fileName);