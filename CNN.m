classdef CNN < handle

    %%Basically a layer manager; gets called by Main
    %% Layers to include/layer order is TBD at the moment
    %% as for activation functions (cross entropy, mse), matlab has them
    %% already implemented.
    properties
        conLayer
        reluLayer
        poolingLayer
        flattenLayer
        fcLayer
    end

    methods
        function output = predict(this, input)
            output = this.conLayer.forward(input);
            output = this.reluLayer.forward(output);
            output = this.poolingLayer.forward(output);
            output = this.flattenLayer.forward(output);
            output = this.fcLayer.forward(output);            
        end

        function train(this, data, labels, kernelSize, depth, outputSize, epoch, learningRate)
            this.conLayer = ConvolutionLayer(size(data, 1), kernelSize, depth);
            this.reluLayer = ReLULayer();
            this.poolingLayer = PoolingLayer(2, 2, 3, 'max');
            this.flattenLayer = FlattenLayer();
            this.fcLayer = FCLayer((size(data, 1) - kernelSize + 1) * (size(data, 2) - kernelSize + 1) * depth / 4, outputSize);
            % sequentially
            % conv -> ReLU -> pooling -> fully connected
            % cross entropy loss -> backprop -> update weights
            for i = 1:epoch
                for j = 1:size(data, 3)
                    result = this.predict(data(:, :, j));
                    loss = labels(:, j) - result;
                    currGradient = this.fcLayer.backward(loss, learningRate);
                    currGradient = this.flattenLayer.backward(currGradient);
                    currGradient = this.poolingLayer.backward(currGradient);
                    currGradient = this.reluLayer.backward(currGradient);
                    currGradient = this.conLayer.backward(currGradient, learningRate);
                end
            end
        end
    end
end