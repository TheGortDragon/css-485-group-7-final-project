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
    if trainLabel(i, 1) == 9 || trainLabel(i, 1) == 25
        train3D(:, :, i) = [];
        continue
    end
    train3D(:, :, i) = reshape(trainData(i, :), imgSize);
end

% convert labels to binary
label2D = zeros(24, numImages);
for i = 1:size(trainLabel, 1)
    num = trainLabel(i, 1);
    if num == 9 || num == 25
        label2D(:, i) = [];
        continue
    end  
    if num > 9
        num = num - 1;
    end
    label2D(num + 1, i) = 1;
end

% test data
numImages = size(testData, 1);
test3D = zeros([imgSize, numImages]);
for i = 1:numImages
    test3D(:, :, i) = reshape(testData(i, :), imgSize);
end

train3D = train3D / 255;
test3D = test3D / 255;

%% setup network
cnn = CNN();

%% train network
cnn.train(train3D(:,:,:), label2D(:,:), 3, 12, 24, 3, .01);

%% validate network performance


%% test network

testResults = zeros(size(testID)); %classification (w forward) of test data

for i = 1:size(testID, 1)
    output = cnn.predict(test3D(:, :, i));
    num = find(output == max(output)) - 1;
    if num >= 9
        num = num + 1;
    end
    testResults(i, 1) = num;
end


%% produce figs for doc ?



%% create table and export
columnNames = {'id', 'label'};
fileName = 'convResults.csv';
outputTable = array2table([testID, testResults], 'VariableNames', columnNames);
writetable(outputTable, fileName);