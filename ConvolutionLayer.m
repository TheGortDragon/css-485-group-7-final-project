classdef ConvolutionLayer < Layer % for a (depth) 1 input network

    properties
        input % has sub variables !
            % size, lastResults
        output % has sub variables !
            % size, result
        kernelW % weights for the kernel, 3D, tensor
        bias
        depth % number of filters
    end

    methods
        %constructor
        function this = ConvolutionLayer(inputSize, kernelSize, depth) %kernel size is the same as output size
            this.depth = depth;
            this.input.size = inputSize;
            this.kernelW = rand(kernelSize, kernelSize, depth);
            this.output.size = inputSize - kernelSize + 1;
            this.bias = rand(this.output.size);
        end

        % forward function (does not include any activation)
        function output = forward(this, input)
            this.input.lastResult = input; % to use in backward
            this.output.result = zeros(this.output.size); %pre-allocate

            for i = 1:this.depth % each filter type
                convResult = conv2(input, this.kernelW(:, :, i), 'valid');
                this.output.result(i, :, :) = convResult + this.bias(i);
            end

            output = this.output.result;
        end

        % backward function
        function input_gradient = backward(this, output_gradient, learningRate)
            kernel_gradient = zeros(size(this.kernelW));
            input_gradient = zeros(this.input.size);

            for i = 1:this.depth
                kernel_gradient(:, :, i) = conv2(this.input.lastResult, output_gradient(i, :, :), 'valid');
                input_gradient = input_gradient + conv2(output_gradient(i, :, :), this.kernelW(:, :, i), 'full');
            end

            % update values
            this.kernelW = this.kernelW - (learningRate * kernel_gradient);
            this.bias = this.bias - ( learningRate * sum(output_gradient, 1) );
        end
    end
end