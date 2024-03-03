classdef DenseLayer < Layer
    %DENSELAYER Summary of this class goes here
    %   Detailed explanation goes here

    properties
        input % has sub variables
            % size, lastInput
        output % has sub variables
            % size
        weights
        bias
    end

    methods
        %constructor
        function this = DenseLayer(inputSize,outputSize)
            this.input.size = inputSize; %square matrices
            this.output.size = outputSize; % ^
            this.weights = rand(this.output.size, this.input.size);
            this.bias = rand(this.output.size, 1);
        end

        %forward function
        function output = forward(this, input)
            this.input.lastInput = input;
            output = (this.weights * input) + this.bias;
        end

        %backward function
        function input_gradient = backward(this, output_gradient, learningRate)
            weight_gradient = output_gradient * this.input.lastInput';
            input_gradient = this.weights' * output_gradient;

            %update values
            this.weights = this.weights - (learningRate * weight_gradient);
            this.bias = this.bias - (learningRate * output_gradient);
        end

    end
end