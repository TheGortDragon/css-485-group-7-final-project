classdef ConvLayer < handle
    properties 
        inputSize
        kernelSize
        outputSize
        kernels
        biases
    end

    methods
        % constructor
        function this = ConvLayer(inputShape, outputShape, kernelShape, depth)
            this.inputSize = inputShape;
            this.outputSize = outputShape;
            this.kernelSize = kernelShape;
            this.kernels = rand(kernelSize, kernelSize);
            this.biases = rand(outputSize, 1;)
        end

        % forward function
        function output = forward(input)
            output = 0; %temp
        end

    end


    methods(Static)

    end

end