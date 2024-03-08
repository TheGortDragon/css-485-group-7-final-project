classdef SoftMaxLayer < Layer
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
            output = exp(input - max(input)) / sum(exp(input - max(input)));
        end

        %backward function
        function input_gradient = backward(this, output_gradient)

        end
    end
end