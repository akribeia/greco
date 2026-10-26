function [vol] = Radon(TS, tilts)
tic
vol = zeros(size(TS,1), size(TS,2),size(TS,1),'single');
        for s=1:size(TS,2)
             projections = single(squeeze(TS(:,s,:)));
              sliceXZ = iradon(projections,tilts,'linear','Ram-Lak',size(TS,1));
             vol(:,s,:) = sliceXZ';
        end
toc
end


