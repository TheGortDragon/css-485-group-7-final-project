classdef Layer < handle
    %% abstract class

    methods (Abstract)
        forward(this)
        backward(this)
    end
end