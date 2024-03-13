load('cnn13.mat');
load('cnn16.mat');
load('cnn112.mat');
load('cnn33.mat');
load('cnn36.mat');
load('cnn312.mat');
load('cnn53.mat');
load('cnn56.mat');
load('cnn512.mat');

load('validationData.mat');
load('validationLabels.mat');

cnns = [cnn13 cnn16 cnn112 cnn33 cnn36 cnn312 cnn53 cnn56 cnn512];

score = zeros(9);

for i = 1:length(cnns)
    for j = 1:size(validationLabels, 2)
        input = validationData(:, :, j);
        inLabel = validationLabels(:, j);
        output = cnns(i).predict(input);
        inNum = find(inLabel == max(inLabel));
        outNum = find(output == max(output));
        if (inNum == outNum)
            score(i) = score(i) + 1;
        end
    end
    disp(score(i) / 2455);
end

