function [outImage] = padmask(inImage,maskImage,paddingMethod,rampDistance)

% [outImage] = padmask(inImage,maskImage,paddingMethod,rampDistance)
%   Given an input image and a mask image, this function fills all of the 
%   pixels with the mask value using the specified method.
%   paddingMethod is 'ramp' (ramp) or 'symmetric'
%
%   If you use this code in your work, please reference the following: 
%   D. Padfield. "Masked Object Registration in the Fourier Domain".
%   Transactions on Image Processing. 
%   D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
%   Pattern Recognition, 2010. 
%
%   Author: Dirk Padfield, GE Global Research, padfield@research.ge.com

if( nargin < 3 )
    paddingMethod = 'ramp';
end
if( nargin < 4 )
    % This parameter is particular to the "ramp" method.
    % The ramp distance should be as small as possible while still removing
    % the artifacts of the strong edges resulting from the masks.
    rampDistance = 60;
end
    
inImage = double(inImage);

if( strcmp(paddingMethod,'ramp') )    
    % Include a border around the image.
    backgroundMask = true(size(inImage,1)+2,size(inImage,2)+2);
    backgroundMask(2:end-1,2:end-1) = ~maskImage;
    distImage = bwdist(backgroundMask);
    % eps is needed to avoid problems with dividing by zero.
    distImage(distImage > rampDistance) = rampDistance+eps;
    distImage = distImage / max(distImage(:));
    distImage = distImage(2:end-1,2:end-1);    
    outImage = inImage .* distImage;
        
elseif( strcmp(paddingMethod,'hanning') )
    % Include a border around the image.
    backgroundMask = true(size(inImage,1)+2,size(inImage,2)+2);
    backgroundMask(2:end-1,2:end-1) = ~maskImage;
    distImage = bwdist(backgroundMask);
    distImage(distImage > rampDistance) = rampDistance;
    
    % Hann window
    % w = 0.5 * (1 - cos(2*pi*(0:m-1)'/(n-1)));
    a0 = 0.5;
    a1 = 0.5;
    a2 = 0;
    a3 = 0;
    a4 = 0;
    x = 0.5 * unique(distImage) / max(distImage(:));
    w = a0 - a1*cos(2*pi*x) + a2*cos(4*pi*x) - a3*cos(6*pi*x) + a4*cos(8*pi*x);

    % Use the squared distance image because it has whole integers so that
    % it can be used for indexing.
    distImageSquared = round(distImage.^2);
    mapper(unique(distImageSquared)+1) = w;
    mappedImage = mapper(distImageSquared+1);
    mappedImage = mappedImage(2:end-1,2:end-1);
    outImage = inImage .* mappedImage;

elseif( strcmp(paddingMethod,'symmetric') )
    outImage = inImage;
    imageSize = size(inImage);
    maskValue = 0;

    while( sum(maskImage(:)) < numel(maskImage) )
        [distImage,distIndex] = bwdist(maskImage,'cityblock');

        [backgroundIndicesY,backgroundIndicesX] = find(maskImage==0);
        [closestIndicesY,closestIndicesX] = ind2sub(imageSize,distIndex(maskImage==0));

        directionVectors = [closestIndicesX-backgroundIndicesX, closestIndicesY-backgroundIndicesY];

        replacementIndicesX = backgroundIndicesX + 2*directionVectors(:,1);
        replacementIndicesY = backgroundIndicesY + 2*directionVectors(:,2);

        replacementIndicesX(replacementIndicesX<1) = 1;
        replacementIndicesX(replacementIndicesX>imageSize(2)) = imageSize(2);
        replacementIndicesY(replacementIndicesY<1) = 1;
        replacementIndicesY(replacementIndicesY>imageSize(1)) = imageSize(1);

        replacementIndicesX = round(replacementIndicesX);
        replacementIndicesY = round(replacementIndicesY);
        replacementIndices = sub2ind(imageSize,replacementIndicesY,replacementIndicesX);
        outImage(maskImage==0) = outImage(replacementIndices);

        % Not all the image pixels will be filled because some may land outside
        % of the image.  Therefore, we have to iterate.
        % Recalculate the image mask.
        maskImage = outImage ~= maskValue;
    end
end
