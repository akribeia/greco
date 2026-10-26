scale= 0.1
rotation = 5.47
box = 20
convsize = 2
ngpu = 8
mask_pad = 20
angles = [-60:3:60]

[TS, TSA, TSO, shift2D] = Align2D('ts01.mrc', -rotation, scale);

[cloud_idx, points, count] = KAZESelection(TS,250, box, shift2D);

[cloud] = CleanCloud(TS, cloud_idx, -rotation, box);

%ROTATE
TSB = permute(TSA, [2,1,3]);
TSS = permute(TS, [2,1,3]);
cloudB = cloud(:,[2 1 3 4])

%


