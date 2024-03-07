classdef FlattenLayer < Layer

    properties % both have sub variable of 'size'
        input
    end

    methods
        %forward
        function output = forward(this, input)
            this.input = input;
            output = reshape(input, size(input, 1) * size(input, 2) * size(input, 3), 1);
        end

        %backward
        function input_gradient = backward(this, output_gradient) % learning rate from classDef but is not used
            input_gradient = reshape(output_gradient, [],size(this.input, 2), size(this.input, 3));
        end
    end
end