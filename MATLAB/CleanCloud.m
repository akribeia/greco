function [cloud] = CleanCloud(TS, cloud_idx, rotation, box_size)
    d = [0];
    angles = [-60:3:60];
    cloud_idx(:,1)= cloud_idx(:,1)-ceil(size(TS,1)/2);
    cloud_idx(:,2)= cloud_idx(:,2)-ceil(size(TS,2)/2);

    % Create rotation matrix
    theta = rotation; % to rotate counterclockwise
    R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];

    cloud_idx(:,1:2) = (R*cloud_idx(:,1:2)')';
    cloud_idx(:,1)= cloud_idx(:,1)+ceil(size(TS,1)/2);
    cloud_idx(:,2)= cloud_idx(:,2)+ceil(size(TS,2)/2);

    cloud = cloud_idx;
    cloud = round(cloud);
    cloud(:,2) = cloud(:,2) + round(d-cosd(angles(cloud(:,3)))*d)';

    cloud = cloud(cloud(:,1) < size(TS,1)-box_size , :);
    cloud = cloud(cloud(:,2) < size(TS,2)-box_size , :);
    cloud = cloud(cloud(:,1) > box_size , :);
    cloud = cloud(cloud(:,2) > box_size , :);