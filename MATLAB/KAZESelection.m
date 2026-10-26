function [cloud_idx, points, count] = KAZESelection(image, max_points, box ,shift2D)

box = box*2;

method = ["detectKAZEFeatures(image(:,:,i),'NumOctaves' ,1)" ]
    for i = 1:size(image,3)
        Ipoints = eval(method);
        points{i} = Ipoints;
        count(i) = points{i}.Count;
    end
    
        %Extract xy of points
    cloud_idx = [0 0 0]
    for i = 1:size(image,3)
    mask = ones(size(image,1:2));

        for t = 1:size(points{i}.Location,1)
        coord = [points{i}.selectStrongest(t).Location(:,2) + shift2D(i,2)  points{i}.selectStrongest(t).Location(:,1) + shift2D(i,1)];
        coord = coord(end,:);
        coord = coord(coord(:,1) < size(image,1) , :) ; %%%% crop based on TSA size
        coord = coord(coord(:,2) < size(image,2) , :) ;
        idx = all(coord>=1,2); %% discard coordinates smaller than 0.
        coord = coord(idx,:); %%%
            if ~isempty(coord) 
            if  mask(round(coord(1)), round(coord(2))) > 0 && round(coord(1)) < size(image,1) - box && round(coord(2)) < size(image,1) - box &&  round(coord(1)) > box  && round(coord(2)) > box  
                mask(round(coord(1)) - box/2 : round(coord(1)) + box/2, round(coord(2)) - box/2 : round(coord(2)) + box/2,: ) = 0;
                cloud_idx = [cloud_idx ; [ coord   (i)*ones(size(points{i}.selectStrongest(1)))]];
            end
            end
        if size(cloud_idx(cloud_idx(:,3)==i,:),1) == max_points
            break
        end
        end
    end

    cloud_idx(1,:) = [];
    cloud_idx = [cloud_idx [1:length(cloud_idx)]']
    
    %trim cloud
    idx = all(cloud_idx>=1,2); %% discard coordinates smaller than 0.
    cloud_idx = cloud_idx(idx,:); %%%

    cloud_idx = cloud_idx(cloud_idx(:,1) < size(image,1) , :) ; %%%% crop based on TSA size
    cloud_idx = cloud_idx(cloud_idx(:,2) < size(image,2) , :) ;