function [tiles] = ProjectionST(rec, tilt)
tic
%project
%rec = permute(rec,[2 1 3]);
tilt = (-1 * tilt ) + 90;

%first permute
for t = 1:length(tilt)
    R{t} = [];
    for i = 1:size(rec,2)
        R{t} = [R{t} [radon(reshape(rec(:,i,:),size(rec,1), size(rec,3)),tilt(t))]];
    end
end
tiles =[];
for i = 1:length(R)
    tiles = cat(3,tiles, R{i});
end

tiles = tiles(ceil((size(tiles,1)-size(rec,1))/2):size(tiles,1)-ceil((size(tiles,1)-size(rec,1))/2),:,:);
tiles = gather(tiles);
tiles = transpose(imrotate(tiles,270));
toc
end