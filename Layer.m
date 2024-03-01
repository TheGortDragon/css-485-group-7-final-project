classdef Layer < handle
    %% abstract class

    properties
        input
        output
    end

    methods (Abstract)
        forward(this)
        backward(this)
    end
end