"""Template sampling and browser diagnostics, with explicit zyx/xyz conversion."""
import io
import numpy as np
import mrcfile

MAX_INPUT_VOXELS=512*1024**2


def _density_range(data,mask=False):
    """Validate mmap data in bounded slabs, without whole-volume temporaries."""
    low=float('inf');high=-float('inf')
    for start in range(0,data.shape[0],4):
        block=data[start:start+4]
        if not np.isfinite(block).all():
            raise ValueError('Mask must contain finite weights.' if mask else 'Template must contain finite, nonconstant density.')
        low=min(low,float(block.min()));high=max(high,float(block.max()))
    return low,high


def inspect_template(path):
    with mrcfile.mmap(path,mode='r',permissive=False) as handle:
        shape=tuple(handle.data.shape)
        if len(shape)!=3 or len(set(shape))!=1 or shape[0]<4:
            raise ValueError('Template must be a cubic 3D MRC with at least 4 voxels per side.')
        if handle.data.size>MAX_INPUT_VOXELS:raise ValueError('Template is too large (512 Mi voxels maximum).')
        low,high=_density_range(handle.data)
        if high<=low:
            raise ValueError('Template must contain finite, nonconstant density.')
        if (int(handle.header.mapc),int(handle.header.mapr),int(handle.header.maps))!=(1,2,3):
            raise ValueError('Template MRC must use standard XYZ axis mapping.')
        voxel=np.array([float(handle.voxel_size.x),float(handle.voxel_size.y),float(handle.voxel_size.z)])
        known=bool(np.isfinite(voxel).all() and np.all(voxel>0))
        if known and not np.allclose(voxel,voxel[0],rtol=1e-4):
            raise ValueError('Template has anisotropic voxels; resample it to isotropic voxels first.')
    return dict(shape=list(shape),pixel_size_A=float(voxel[0]) if known else None)


def inspect_mask(path):
    with mrcfile.mmap(path,mode='r',permissive=False) as handle:
        shape=tuple(handle.data.shape)
        if len(shape)!=3 or len(set(shape))!=1 or shape[0]<4:raise ValueError('Mask must be a cubic 3D MRC.')
        if handle.data.size>MAX_INPUT_VOXELS:raise ValueError('Mask is too large (512 Mi voxels maximum).')
        # Resampled masks can ring slightly beyond 0/1 (the existing Greco picker
        # mask spans -4e-5..1.00034). Accept only small interpolation overshoot.
        low,high=_density_range(handle.data,mask=True)
        if low<-.001 or high>1.001 or high<=0:
            raise ValueError('Mask must contain finite weights approximately between 0 and 1 with nonzero support.')
        if (int(handle.header.mapc),int(handle.header.mapr),int(handle.header.maps))!=(1,2,3):raise ValueError('Mask must use standard XYZ MRC axes.')
        voxel=np.array([float(handle.voxel_size.x),float(handle.voxel_size.y),float(handle.voxel_size.z)])
        known=bool(np.isfinite(voxel).all() and np.all(voxel>0))
        if known and not np.allclose(voxel,voxel[0],rtol=1e-4):raise ValueError('Mask has anisotropic voxels.')
    return dict(shape=list(shape),pixel_size_A=float(voxel[0]) if known else None)


def prepare_template(path, pixel_A, target_A, flip_z=False, invert_contrast=False,*,mask_path):
    """Resample density and explicit mask about their shared physical centre."""
    from scipy.ndimage import affine_transform,gaussian_filter
    with mrcfile.open(path,permissive=False) as handle:source=np.array(handle.data,np.float32)
    if flip_z:source=source[::-1].copy()
    if invert_contrast:source=-source
    source-=source.mean()
    scale=float(target_A)/float(pixel_A)
    size=max(4,int(round(source.shape[0]/scale)))
    if size>128:raise ValueError('Resampled template exceeds 128 voxels per side. Use coarser reconstruction or a smaller template.')
    if scale>1:source=gaussian_filter(source,.5*np.sqrt(scale*scale-1),mode='constant')
    offset=(np.array(source.shape)-1)/2-scale*(size-1)/2
    template=affine_transform(source,np.eye(3)*scale,offset=offset,output_shape=(size,)*3,
        order=3,mode='constant',cval=0).astype(np.float32)
    metadata=inspect_mask(mask_path)
    if tuple(metadata['shape'])!=source.shape:raise ValueError('Mask must have the same box size as the template.')
    with mrcfile.open(mask_path) as handle:mask_source=np.array(handle.data,np.float32).clip(0,1)
    if flip_z:mask_source=mask_source[::-1].copy()
    mask=affine_transform(mask_source,np.eye(3)*scale,offset=offset,output_shape=(size,)*3,
        order=1,mode='constant',cval=0).clip(0,1).astype(np.float32)
    support=template[mask>0]
    if support.size<2 or np.std(support)<1e-8:raise ValueError('Template has no usable contrast inside the resampled mask.')
    return template,mask


def particle_examples(volume_path, xyz, box_size, count=24):
    """Read small native XY patches from MRC; no whole-volume materialisation."""
    from PIL import Image,ImageDraw
    count=min(count,len(xyz));tile=128;columns=6;rows=max(1,int(np.ceil(count/columns)))
    image=Image.new('RGB',(columns*tile,rows*(tile+24)),(20,28,40));draw=ImageDraw.Draw(image)
    if not count:draw.text((20,20),'No particles above the detection threshold.',fill='white')
    with mrcfile.mmap(volume_path,mode='r') as handle:
        data=handle.data
        for i,(x,y,z) in enumerate(xyz[:count]):
            x,y,z=np.rint([x,y,z]).astype(int);half=box_size//2
            z=np.clip(z,0,data.shape[0]-1)
            patch=np.array(data[z,max(0,y-half):min(data.shape[1],y+half+1),max(0,x-half):min(data.shape[2],x+half+1)],np.float32)
            if not patch.size:continue
            lo,hi=np.percentile(patch,[1,99]);patch=np.uint8(255*np.clip((patch-lo)/max(float(hi-lo),1e-12),0,1))
            thumb=Image.fromarray(patch).resize((tile,tile),Image.Resampling.BILINEAR).convert('RGB')
            left=(i%columns)*tile;top=(i//columns)*(tile+24)
            image.paste(thumb,(left,top));draw.text((left+5,top+tile+5),f'#{i+1}  z={z}',fill='white')
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()
