    %Boxes
        SE = strel('sphere',convsize);
    SEmask = strel('sphere',box/2);
    %3d cylinder
    [mask3d] = cylinderMask(TSS, mask_pad);


    %find mid and initial refine range

    mid = median(1:size(TS,3));
    refine = [mid-2 mid-1 mid mid+1 mid+2];

    if mid ~= ceil(mid)
    refine = [floor(mid-1) floor(mid) ceil(mid) ceil(mid)+1];
    end
    tiles_temp = [];
    mask_temp = [];    
    maxim = [];

    drift = zeros(size(shift2D));
    %padarray and apply shifts
while length(refine) <= size(TSB,3) || k < 25

    TSC = padarray(TSS,[size(TSS,1) size(TSS,2)],'replicate');

    for i = 1:size(TS,3)
        temp = imrotate(imtranslate(TSC(:,:,i), flip(shift2D(i,:)) - drift(i,:)), -rotation, 'crop');
        win = centerCropWindow2d(size(TSC, 1:2) ,size(TSS,1:2));
        TSD(:,:,i) = temp(win.YLimits(1):win.YLimits(2),win.XLimits(1):win.XLimits(2),:);
    end

    maximas = [];
    tiles_temp = [];
    mask_temp = [];    
spmd (ngpu)
gpuDevice(labindex);
TSE = gpuArray(TSD); 
 for t = refine   
 if ceil(length(refine(1):t)/(length(refine)/ngpu)) == labindex
 rango = refine;
    if length(refine) ~= size(TSD,3)
        %take the KAZE points for the given tilt,
        % use previous KAZE points the first time a tilt is being
        % refined
        if t < refine(2)
        cloud_for_t = round(cloud(cloud(:,3)==t+1,:));
        elseif t > refine(end-1)
        cloud_for_t = round(cloud(cloud(:,3)==t-1,:));
        else
        cloud_for_t = round(cloud(cloud(:,3)==t,:));
        end
    else
    cloud_for_t = round(cloud(cloud(:,3)==t,:)); %case refinement is finished

    end
    cloud_for_t(:,1) = cloud_for_t(:,1) + drift(t,1);
    cloud_for_t(:,2) = cloud_for_t(:,2) + drift(t,2);
    %clean the cloud for outliers after apply drift
    cloud_for_t = CleanCloud(TS, cloud_for_t, 0, 20);

    %make columns from KAZE points
    columns = zeros(size(mask3d));
    for i = 1:length(cloud_for_t)
    columns(cloud_for_t(i,2) - box/2 : cloud_for_t(i,2) + box/2, cloud_for_t(i,1) - box/2 : cloud_for_t(i,1) + box/2,: ) = 1*i;
    end
    %rotate the columns according to angles(t)
    R = rotx(angles(t));
    tform = rigid3d((R),[0 0 0]);
    transformed = imwarp(columns,tform,'nearest');
    win = centerCropWindow3d(size(transformed, 1:3) ,size(mask3d,1:3));
    transformed = gpuArray(transformed(win.YLimits(1):win.YLimits(2),win.XLimits(1):win.XLimits(2),win.ZLimits(1):win.ZLimits(2)));
    %remove the tilt is being refined, reconstruct and convolve
    tilt = angles(t);
    rango(rango==t) = [];
    rec = gpuArray(Radon(imcomplement(TSE(:,:,rango)), angles(rango)));
    rec = convn(rec, SE.Neighborhood, 'same'); %different ways to filter the volume
    %rec = imopen(gather(rec), SE.Neighborhood);
    %rec = imclose(gather(rec), SE.Neighborhood);
    
    %project the volume
    recB = gpuArray(rec.*mask3d.*imbinarize(gather(transformed))); %%%%%%%%%%%%%
    [recS,h] = sort(reshape(recB,[size(rec,1)*size(rec,2)*size(rec,3), 1]),'descend');
    %find the maximas
    [r,s] = intersect(transformed(h),[1:length(cloud_for_t)],'stable');
    list = h(s);
    %make the list of maximas, make the mask of spheres to project the maximas
    spheres_mask = gpuArray(logical(zeros(size(mask3d))));
        for i = 1:length(cloud_for_t)
        [ind1, ind2, ind3] = ind2sub(size(recB),list(i));
        spheres_mask(ind1, ind2, ind3) = 1;
        maximas = [maximas ; [ind1 ind2 ind3 t]];
        end



%project reconstruction (masked) and project the mask
    spheres_mask = convn(spheres_mask, SEmask.Neighborhood, 'same'); %convolve the spheres_mask to convert points into spheres
    tiles_temp = cat(3, tiles_temp, imcomplement(ProjectionST(rec.*spheres_mask, tilt)));
    mask_temp = cat(3, mask_temp, ProjectionST(spheres_mask, tilt));
 
 
 end
 end
end

    maxim = cat(1,gather(maximas{:}));
    tiles = cat(3,tiles_temp{:});
    mask = cat(3,mask_temp{:});

    %cosine stretching to make it more robust in higher angle tilts
for t = 1:length(refine)
    cosStretch = ceil((size(tiles,1) - (size(tiles,1)*cosd(angles(refine(t)))))/2);
    tiles(1:cosStretch,:,t) = 0;
    tiles(end-cosStretch:end,:,t) = 0;
    mask(1:cosStretch,:,t) = 0;
    mask(end-cosStretch:end,:,t) = 0;
 end


    %make the search mask, do the roseman correlation using the projection
    %of the reconstruction, the projection of the mask and the apropriate
    %tilt, for the correlation we also consider the previous tilt to make
    %it more robust
    drift_temp = zeros(size(shift2D));
    for t = refine
         [drift_temp(t,:), peak, corr(:,:,t), D] = MaskedTranslationRegistration(tiles(:,:,length(refine(1):t)), TSD(:,:,t), imbinarize(mask(:,:,length(refine(1):t))), ones(size(TSD(:,:,t))),0.95);
    end
    drift = drift + drift_temp;


       if refine(end) < size(TSB,3)
         %estimate the drift in X direction, here I use a more simple model
             x = [refine(1):mid]-mid;
             y = drift(refine(1):mid);
             myfit = fit(x',y','m*x^2','StartPoint',[0])

                    drift(refine(1)-1) = fix(myfit.m*((refine(1)-1)-mid)^2);

             x = [refine(refine==mid):refine(end)]-mid;
             y = drift(refine(refine==mid):refine(end));
             myfit = fit(x',y','m*x^2','StartPoint',[0])

                    drift(refine(end)+1) = fix(myfit.m*((refine(end)+1)-mid)^2);
     end



     if refine(end) < size(TSB,3)
         %estimate where to place the following tilt, we need to estimate
         %the value of d and h.
             x = angles(refine);
             y = drift(refine,2)';
             myfit = fit(x',y','(d-cosd(x)*d)+(h-cosd(x)*h)*sign(x)','StartPoint',[0,0])

                    drift(refine(1)-1,2) = fix((myfit.d-cosd(angles(refine(1)-1))*myfit.d)+(myfit.h-cosd(angles(refine(1)-1))*myfit.h)*sign(angles(refine(1)-1)));

                    drift(refine(end)+1,2) = fix((myfit.d-cosd(angles(refine(end)+1))*myfit.d)+(myfit.h-cosd(angles(refine(end)+1))*myfit.h)*sign(angles(refine(end)+1)));
     end

         if length(refine) ~= size(TSB,3)
        refine = [refine(1)-1 refine refine(end)+1];
    end

refine
drift'
end