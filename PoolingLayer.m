classdef PoolingLayer < Layer

    properties
        input % has sub var, size
        output % has sub var, shape
        stride
        poolSize
        mode
    end

    methods
        % constructor
        function this = PoolingLayer(poolSize, strideSize, type) % square
            if ~(strcmp(mode, 'min')) % add more options as we see fit
                error('Invalid pooling type. Supported types are "min".');
            end

            this.poolSize = poolSize;
            this.stride = strideSize;
            this.mode = type;
        end

        %forward
        function output = forward(this, input)
            this.input.size = size(input);
            sqrSize = this.input.size(1);

            % Compute output shape
            this.output.shape = [ceil(sqrSize / this.stride), ceil( sqrSize / this.stride)];
            output = zeros(this.output.shape);
            
            % Perform pooling operation
            for j = 1:this.output.shape(1)
                for k = 1:this.output.shape(2)
                    start_row = (j - 1) * this.stride + 1;
                    end_row = min(start_row + this.poolSize - 1, sqrSize);
                    start_col = (k - 1) * this.stride + 1;
                    end_col = min(start_col + this.poolSize - 1, sqrSize);
                    
                    if strcmp(obj.mode, 'max')
                        output(j, k) = min(min(input(start_row:end_row, start_col:end_col)));
                    end
                end
            end
        end

        %backward -> dunno
        function backward(this)
            
        end
    end
end