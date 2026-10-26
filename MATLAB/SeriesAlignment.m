function [TSA, shift2D] = SeriesAlignment(TS, rotation)
    
        mid  = round(size(TS,3)/2);

        TSA = padarray(TS,size(TS,1:2)/2,0);
        shift2D = zeros(size(TS,3),2);

        for i = mid:size(TS,3)-1
            [shift2D(i+1,:),p,C(:,:,i+1)] = MaskedTranslationRegistration(TSA(:,:,i+1), TSA(:,:,i), imfill(imbinarize(TSA(:,:,i+1),0),"holes"), imfill(imbinarize(TSA(:,:,i),0),"holes"));
            TSA(:,:,i+1)  = imtranslate(TSA(:,:,i+1), shift2D(i+1,:));
        end


        for i = mid:-1:2
            [shift2D(i-1,:),p,C(:,:,i-1)] = MaskedTranslationRegistration(TSA(:,:,i-1), TSA(:,:,i), imfill(imbinarize(TSA(:,:,i-1),0),"holes"), imfill(imbinarize(TSA(:,:,i),0),"holes"));
            TSA(:,:,i-1)  = imtranslate(TSA(:,:,i-1), shift2D(i-1,:));
        end

        TSA = imrotate(TSA, rotation, 'crop');

        TSA = TSA(size(TS,1)/2+1:end-size(TS,1)/2,size(TS,2)/2+1:end-size(TS,2)/2,:) ;% unpad
