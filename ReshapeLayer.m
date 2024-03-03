classdef ReshapeLayer < Layer

    properties % both have sub variable of 'size'
        input
        output
    end

    methods
        %constructor
        function this = ReshapeLayer(inputSize, outputSize)
            this.input.size = inputSize;
            this.output.size = outputSize;
        end

        %forward
        function output = forward(this, input)
            output = reshape(input, this.output.size);
        end

        %backward
        function input_gradient = backward(this, output_gradient, learningRate) % learning rate from classDef but is not used
            input_gradient = reshape(output_gradient, this.input.size);
        end
    end
end