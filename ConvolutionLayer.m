classdef ConvolutionLayer < Layer

    properties
        input
        output
        kernelW % weights for the kernel
        bias
    end

    methods
        function this = ConvolutionLayer(inputSize, kernelSize) %kernel size is the same as output size
            this.input.size = inputSize;
            this.kernelW = rand(kernelSize);
            this.output.size = inputHeight - kernelSize + 1;
            this.bias = rand(this.output.size);
        end

        function forward(this)
            
        end

        function backward(this)
            
        end
    end
end