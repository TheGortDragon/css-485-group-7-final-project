%% setup data
%training data
trainData = readmatrix('data/train.csv', 'Range', 'C2:ADF27456');
trainLabel = readmatrix('data/train.csv', 'Range', 'B2:B27456');
%testing data
testData = readmatrix('data/test.csv', 'Range', 'B2:ADE7173');
%get id labels
testID = readmatrix('data/test.csv', 'Range', 'A2:A7173');

%% setup network


%% train network


%% validate network performance


%% test network


%% produce figs for doc


%% export resutls
testClass = []; %classification (w forward) of test data
% Define the column names and the file name
columnNames = {'id', 'label'};
fileName = 'backpropClassification.csv';
% Create a table with the output data and column names
outputTable = array2table([testID, testClass], 'VariableNames', columnNames);
% Write the table to a CSV file
writetable(outputTable, fileName);