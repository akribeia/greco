function [TS] = Resize(TS, resize)
    
    TS = imresize(TS,resize);