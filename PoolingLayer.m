classdef PoolingLayer < Layer

    properties
        input % has sub var, size
        output % has sub var, shape
        stride
        poolSize
        depth
        mode
    end

    methods
        % constructor
        function this = PoolingLayer(poolSize, strideSize, depth, type) % square
            if ~(strcmp(type, 'max')) % add more options as we see fit
                error('Invalid pooling type. Supported types are "max".');
            end

            this.poolSize = poolSize;
            this.stride = strideSize;
            this.mode = type;
            this.depth = depth;
        end

        %forward
        function output = forward(this, input)
            this.input.size = size(input);
            this.input.lastInput = input;
            sqrSize = this.input.size(1);

            % Compute output shape
            this.output.shape = [ceil(sqrSize / this.stride), ceil( sqrSize / this.stride), this.depth];
            output = zeros(this.output.shape);
            
            % Perform pooling operation
            for i = 1:this.depth
                for j = 1:this.output.shape(1)
                    for k = 1:this.output.shape(2)
                        start_row = (j - 1) * this.stride + 1;
                        end_row = min(start_row + this.poolSize - 1, sqrSize);
                        start_col = (k - 1) * this.stride + 1;
                        end_col = min(start_col + this.poolSize - 1, sqrSize);
                        
                        if strcmp(this.mode, 'max')
                            output(j, k, i) = max(max(input(start_row:end_row, start_col:end_col, i)));
                        end
                    end
                end
                this.output.lastOutput = output;
            end
        end

        %backward
        function input_gradient = backward(this, output_gradient)
            input_gradient = zeros(this.input.size);
          %feed backward the sensitivity since everything else is 1
             for i = 1:this.depth
                for j = 1:this.input.size(1)
                    for k = 1:this.input.size(2)
                       currMax = this.output.lastOutput(int32(j / this.poolSize), int32(k / this.poolSize), i);
                       if currMax == this.input.lastInput(j, k, i)
                            input_gradient(j, k, i) = output_gradient(int32(j / this.poolSize), int32(k / this.poolSize), i);
                       end
                    end
                end
            end
        end
    end
end