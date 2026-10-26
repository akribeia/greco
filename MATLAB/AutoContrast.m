function [TS] = AutoContrast(TS)

    for i = 1:size(TS,3)
        TS(:,:,i) = imadjust(TS(:,:,i));
    end