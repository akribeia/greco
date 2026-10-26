function [TS, TSA, TSO, shift2D] = Align2D(mrc, rotation, scale)

    TS = single(ReadMRC(mrc));
    TSO = TS; % guardar el TS original
    %TS = permute(TS, [2,1,3]);%
    TS = Resize(TS,scale);

    %Denoise
    TS = Denoise(TS);

    %Contrast adjustment
    TS = AutoContrast(TS);

    %alinear tilts
    [TSA shift2D] = SeriesAlignment(TS, rotation);
    