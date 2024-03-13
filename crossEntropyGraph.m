x = [1 2 3 4 5 6 7];
cnnY13 = [3.1839 3.1782 3.1782 3.1782 3.1782 3.1782 3.1782];
cnnY16 = [3.182 3.1782 3.1782 3.1782 3.1782 3.1782 3.1782];
cnnY112 = [3.185 3.1782 3.1782 3.1782 3.1782 3.1782 3.1782];
cnnY33 = [1.8748 .7156 .47479 .34068 .2656 .20952 .18753];
cnnY36 = [3.2021 3.1783 3.1783 3.1782 3.1782 3.1782 3.1782];
cnnY312 = [2.0883 .86294 .64633 .54192 .43228 .29698 .1962];
cnnY53 = [2.6322 .87629 .62552 .52008 .49239 .48156 .45179];
cnnY56 = [NaN .64317 .32765 .20831 .18904 .13772 .1159];
cnnY512 = [NaN .97273 .64956 .37988 .2319 .16321 .11563];

y = [cnnY13; cnnY16; cnnY112; cnnY33; cnnY36; cnnY312; cnnY53; cnnY56; cnnY512];

yName = ["cnn13" 'cnn16' 'cnn112' 'cnn33' 'cnn36' 'cnn312' 'cnn53' 'cnn56' 'cnn512'];
disp(size(yName));
hold on
for i=1:size(y, 1)
    plot(x, y(i, :), 'DisplayName', yName(i));
    if mod(i, 3) == 0
        legend;
        title("Cross Entropy Loss For Each Neural Network");
        xlabel("Number of Iterations");
        ylabel("Cross Entropy Loss");
        figure;
        hold on
    end
end