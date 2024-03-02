classdef ConvLayer < handle
    properties 
        inputSize
        kernelSize
        outputSize
        numKernels
        kernels % weights
        biases
    end

    methods
        % constructor
        function this = ConvLayer(inputShape, outputShape, kernelShape, depth)
            this.inputSize = inputShape;
            this.outputSize = outputShape;
            this.kernelSize = kernelShape;
            this.numKernels = depth;
            this.kernels = rand(kernelSize, kernelSize);
            this.biases = rand(outputSize, 1);
        end

        % forward function // iffy on the execution but math logic seems ok
        function output = forward(this, input)
            for i = 1:this.outputSize % idk what size exaclty this would be
                sum = 0;
                for j = 1:this.inputSize
                    %maybe use conv2 instead?
                    sum = sum + xcorr(input(j), this.kernels(i, j)); 
                end
                output = this.biases(i) + sum;
            end
        end

        % training function
        function this = train(this, inputPattern, targetPattern, epochs, learningRate)
            for e = 1:epochs
                
            end
        end

    end


    methods(Static)
        % compute cross entropy loss
        function val = xEnLoss(target, actual) %should be vectors
            val = -sum((target * log(actual)), 2);
        end

        %find the error
        function val = error(target, actual)
            val = target - actual;
        end

        % find the mean squared error
        function val = mse(target, actual)
            val = sum(error(target, actual), 2); % sum, if they are vectors
            val = power(val, 2); % square
            val = (1 / size(target, 1)) * val; % mean
        end

    end

end