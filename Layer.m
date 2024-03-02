classdef Layer < handle
    %% abstract class

    properties %idk if we need input and output honestly, but can keep it
        input
        output
    end

    methods (Abstract)
        forward(this)
        backward(this)
    end
end