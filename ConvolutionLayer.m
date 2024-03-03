classdef ConvolutionLayer < Layer % for a (depth) 1 input network

    properties
        input % has sub variables !
            % size, lastResults
        output % has sub variables !
            % size, results
        kernelW % weights for the kernel
        bias
        depth % number of filters
    end

    methods
        %constructor
        function this = ConvolutionLayer(inputSize, kernelSize, depth) %kernel size is the same as output size
            this.depth = depth;
            this.input.size = inputSize;
            this.kernelW = rand(kernelSize, kernelSize, depth);
            this.output.size = inputHeight - kernelSize + 1;
            this.bias = rand(this.output.size);
        end

        % forward function (does not include any activation)
        function output = forward(this, input)
            this.input.lastResult = input; % to use in backward
            this.output.result = zeros(this.output.size);
            for i = 1:this.depth
                convResult = conv2(input, this.kernelW(:, :, i), 'valid');
                this.output.result(i, :, :) = convResult + this.bias(i);
            end
            output = this.output.result;
        end

        % backward function
        function input_gradient = backward(this, output_gradient, learningRate)
            kernel_gradient = zeros(size(this.kernelW));
            input_gradient = zeros(this.input.size);


            % update values
            this.kernelW = this.kernelW - (learningRate * kernel_gradient);
            this.bias = this.bias - (learningRate * output_gradient);
        end
    end
end