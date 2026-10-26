function [mask3d] = cylinderMask(TS, pad)

    rec = Radon(imcomplement(TS(:,:,median(1:size(TS,3)))), 0);

    % mask
    m2d = zeros(size(rec, 1:3));
    mask = ellipseMatrix(ceil(size(m2d,1)/2), ceil(size(m2d,1)/2), ceil(size(m2d,1)/2)-pad, ceil(size(m2d,1)/2)-pad, 0, zeros(size(m2d,1), size(m2d,1)),1,1,0);
    mask3d = repmat(mask, [1 1 size(rec, 2)]);
    mask3d = permute(mask3d, [1 3 2]);
    mask3d(:,1:pad,:) = zeros;
    mask3d(:,end-pad:end,:) = zeros;
