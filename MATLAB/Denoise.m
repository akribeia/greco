function [TS] = Denoise(TS)

    net = denoisingNetwork('DnCNN');
    for i = 1:size(TS,3)
        TS(:,:,i) = denoiseImage(TS(:,:,i),net);
    end