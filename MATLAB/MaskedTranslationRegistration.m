function [transform,maxC,C,numberOfOverlapMaskedPixels] = MaskedTranslationRegistration(fixedImage,movingImage,fixedMask,movingMask,overlapRatio)

% [transform,maxC,C,numberOfOverlapMaskedPixels] =
% MaskedTranslationRegistration(fixedImage,movingImage,fixedMask,movingMask,overlapRatio) 
%   Masked FFT normalized cross-correlation registration of movingImage and
%   fixedImage under masks movingMask and fixedMask.
%   movingMask and fixedMask should consist of only 1s and 0s, where 1
%   indicates locations of useful information in the corresponding image,
%   and 0 indicates locations that should be masked (ignored).
%   fixedImage and movingImage need not be the same size, but fixedMask
%   must be the same size as fixedImage, and movingMask must be the same
%   size as movingImage.
%   If a mask is not needed for either the fixedImage or the movingImage,
%   the fixedMask and/or movingMask can be set to an image of all ones of
%   the same size as the corresponding fixedImage and/or movingImage.
%   The optional overlapRatio specifies the number of pixels needed in the
%   overlap region for meaningful results.  It is specified as a ratio of the
%   maximum number of overlap pixels.  Regions in the resulting correlation
%   image that have fewer than this number of pixels will be set to 0.   
%
%   If you use this code in your work, please reference the following: 
%   D. Padfield. "Masked Object Registration in the Fourier Domain".
%   Transactions on Image Processing. 
%   D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
%   Pattern Recognition, 2010. 
%
%   Author: Dirk Padfield, GE Global Research, padfield@research.ge.com

if( nargin < 5 )
    overlapRatio = 3/10;
end

[C,numberOfOverlapMaskedPixels] = normxcorr2_masked(fixedImage,movingImage,fixedMask,movingMask);

imageSize = size(movingImage);

% Mask the borders;
numberOfPixelsThreshold = overlapRatio * max(numberOfOverlapMaskedPixels(:));
C(numberOfOverlapMaskedPixels < numberOfPixelsThreshold) = 0;

search_mask = ellipseMatrix(ceil(size(C,1)/2), ceil(size(C,2)/2), 20, 20, 0, zeros(size(C,1:2)),1,1,0); %ESTO LO HE PUESTO YO PARA METERLE UNA MASK DE 20 px
C = C.*search_mask;

[maxC, imax] = max(C(:));

[ypeak, xpeak] = ind2sub(size(C),imax(1));
transform = [(xpeak-imageSize(2)) (ypeak-imageSize(1))];

% Take the negative of the transform so that it has the correct sign.
transform = -transform;
