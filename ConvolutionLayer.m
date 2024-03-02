classdef ConvolutionLayer < Layer % for a depth 0 input network

    properties
        input % has sub variables !
            % size, lastResults
        output % has sub variables !
            % size, results
        kernelW % weights for the kernel
        bias
    end

    methods
        %constructor
        function this = ConvolutionLayer(inputSize, kernelSize) %kernel size is the same as output size
            this.input.size = inputSize;
            this.kernelW = rand(kernelSize);
            this.output.size = inputHeight - kernelSize + 1;
            this.bias = rand(this.output.size);
        end

        % forward function (does not include any activation)
        function output = forward(this, input)
            this.input.lastResult = input; % to use in backward
            
            output = conv2(input, this.kernelW, 'valid');
            output = output + this.bias;
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