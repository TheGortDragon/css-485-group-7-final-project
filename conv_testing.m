%% get data
%training data -> do we want to shuffle?
trainData = readmatrix('data/train.csv', 'Range', 'C2:ADF27456');
trainLabel = readmatrix('data/train.csv', 'Range', 'B2:B27456');
%testing data
testData = readmatrix('data/test.csv', 'Range', 'B2:ADE7173');
%get id labels (to use for output file)
testID = readmatrix('data/test.csv', 'Range', 'A2:A7173');

%% normalize data?
% scale to 0-1?
% zero out values below a threshold (~0.3) ?

%% reformat data (784x1 -> 28x28)
% training data
numImages = size(trainData, 1);
imgSize = [28, 28];
train3D = zeros([imgSize, numImages]);
for i = 1:numImages
    train3D(:, :, i) = reshape(trainData(i, :), image_size);
end

% test data
numImages = size(testData, 1);
test3D = zeros([imgSize, numImages]);
for i = 1:numImages
    test3D(:, :, i) = reshape(testData(i, :), image_size);
end

%% setup network


%% train network


%% validate network performance


%% test network


%% produce figs for doc ?


%% export resutls
testResults = []; %classification (w forward) of test data

% create table and export
columnNames = {'id', 'label'};
fileName = 'convResults.csv';
outputTable = array2table([testID, testResults], 'VariableNames', columnNames);
writetable(outputTable, fileName);