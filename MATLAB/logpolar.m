function [outImage,b,thetaStep,rhoStep,radius] = logpolar(inImage,thetaRange,rhoRange,interpolationMethod)

% [outImage,b,thetaStep,rhoStep,radius] = logpolar(inImage,thetaRange,rhoRange,interpolationMethod)
%   This function converts an image from Cartesian to log-polar coordinates.
%
%   If you use this code in your work, please reference the following: 
%   D. Padfield. "Masked Object Registration in the Fourier Domain".
%   Transactions on Image Processing. 
%   D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
%   Pattern Recognition, 2010. 
%
%   Author: Dirk Padfield, GE Global Research, padfield@research.ge.com

[rows,cols,depth] = size(inImage); % This will be the size of the output.

if( nargin < 2 || isempty(thetaRange) )
    thetaStep = 2*pi/360;
    thetaRange = 0:thetaStep:2*pi-thetaStep;
end
if( nargin < 3 )
    radius = min([(cols-1)/2 (rows-1)/2]);
    rhoStep = 0.5;
    rhoRange = 0:rhoStep:radius;
end
if( nargin < 4 )
    interpolationMethod = 'nearest';
end

radius = rhoRange(end);
thetaStep = thetaRange(2)-thetaRange(1);
rhoStep = rhoRange(2) - rhoRange(1);

origin = [(cols+1)/2,(rows+1)/2];

% Find the base of the conversion to log coordinates.  According to the
% paper, the base is 1.044, which is found as log_1.044(256) = 128.  256 is
% the size of the input image, and 128 is the number of rows in the polar
% representation.  Symbolically, this can be represented as log_b(x) = y,
% where we know x and y and want to find b.  Using the relation that
% log_b(x) = log_a(x)/log_a(b) = ln(x)/ln(b), we find that ln(x)/y = ln(b)
% and b = exp(ln(x)/y).
b = exp(log(radius) / rhoRange(end));       % base for the log-polar conversion

% Initialize xArray and yArray.
xArray = zeros(length(thetaRange),length(rhoRange));
yArray = xArray;
for i = 1:length(thetaRange)                    % rows
    for j = 1:length(rhoRange)                  % columns
        rho = b ^ (rhoRange(j));                % the log-polar
        theta = thetaRange(i);
        xArray(i,j) = rho * cos(theta) + origin(1);
        yArray(i,j) = rho * sin(theta) + origin(2);
    end
end

% Take care of pixels outside of the image.
onesImage = ones([size(inImage,1) size(inImage,2)]);
xxBinary = xArray;
yyBinary = yArray;
xxBinary(xArray < 1) = 0;
xxBinary(xArray > cols) = 0;
yyBinary(yArray < 1) = 0;
yyBinary(yArray > rows) = 0;

outImageBinary = interp2(onesImage,xxBinary,yyBinary,'nearest');

% Check for values out of bounds due to round off error.
xArray(xArray > cols) = cols;
xArray(xArray < 1) = 1;
yArray(yArray > rows) = rows;
yArray(yArray < 1) = 1;

for i = 1:size(inImage,3)
    transformedImage = interp2(inImage(:,:,i),xArray,yArray,interpolationMethod);
    transformedImage(isnan(outImageBinary)) = 0;
    outImage(:,:,i) = transformedImage;
end
