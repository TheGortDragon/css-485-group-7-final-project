classdef ReLULayer < Layer
    %RELULAYER Summary of this class goes here
    %   Detailed explanation goes here

    properties
        input
        output 
    end

    methods
        %forward function
        function output = forward(this, input)
            this.input = input;
            output = input;
            output(output <= 0) = 0;
        end

        %backward function
        function input_gradient = backward(this, output_gradient)
            input_gradient = max(0, this.input);
            input_gradient(input_gradient > 0) = output_gradient(input_gradient > 0);
        end
    end
end