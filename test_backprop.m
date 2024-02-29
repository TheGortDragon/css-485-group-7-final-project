import BackPropLayer.*

%% setup data
%training data
trainData = (readmatrix('data/train.csv', 'Range', 'C2:ADF27456'))';
trainData = trainData / 255;
trainLabel = (readmatrix('data/train.csv', 'Range', 'B2:B27456'))';
%testing data
testData = (readmatrix('data/test.csv', 'Range', 'B2:ADE7173'))';
testData = testData / 255;
%get id labels
testID = readmatrix('data/test.csv', 'Range', 'A2:A7173');


%% setup network
network = BackPropLayer(size(trainData, 1), 200, 1, 0.001);
network.outputLayer.transferFunc = "purelin";
network.hiddenLayer.transferFunc = "purelin";


%% do the training
epoch = 20;
for rounds = 1:epoch
    for i = 1:size(trainData, 2)
        % Get the ith input pattern and target patterns
        inputPattern = trainData(:, i);
        targetPattern = trainLabel(:, i);
        %disp(targetPattern);

        % Train the network with the current input and target pattern
        network = network.train(targetPattern', inputPattern, 1);
    end
end


%% classify test data
testClass = zeros(size(testID));
for i = 1:size(testData, 2)
    input = testData(:, i);
    output = network.compute(input);
    testClass(i) = round(output);
end


%% output
% Define the column names and the file name
columnNames = {'id', 'label'};
fileName = 'backpropClassification.csv';
% Create a table with the output data and column names
outputTable = array2table([testID, testClass], 'VariableNames', columnNames);
% Write the table to a CSV file
writetable(outputTable, fileName);

