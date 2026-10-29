"use strict";

function tiltDisplayParams(index) {
  const params=new URLSearchParams({project_id:state.project.id,series_id:state.selected,index,
    display_version:3,
    ctf_revision:state.preprocessStatus?.ctf_revision ?? '',
    rotation_deg:Number(state.orientationPreview ?? state.project.orientation?.rotation_deg ?? 0),
    flip_y:state.orientationFlipY?'1':'0'});
  if(state.stage==='reconstruction' && state.reconstructionCTFPreview)params.set('preview_geometry','1');
  if([-1,1].includes(state.orientationHandPreview))params.set('df_hand',state.orientationHandPreview);
  if(Number.isFinite(state.orientationCTFAxisPreview))params.set('axis_deg',state.orientationCTFAxisPreview);
  return params;
}
function showTiltPlane() {
  const slider=$('tilt-slider'), job=state.preprocessStatus;
  if(!slider || !job?.angles?.length) return;
  const index=Number(slider.value);
  $('tilt-angle').textContent=`${index+1} / ${job.angles.length} · ${angle(job.angles[index])}`;
  const image=$('tilt-plane-image');
  if(image) {
    image.onload=()=>matchTiltBackground(image);
    image.onerror=()=>{
      if(state.ctfView==='landscape') {
        state.ctfView=null;showTiltPlane();
        notify('Defocus landscape could not be loaded; showing the tilt image. Restart the server if it is still using the previous viewer.');
      }
    };
    const params=tiltDisplayParams(index);
    if(state.ctfView==='landscape')params.set('landscape','1');
    const src='/api/tilt-plane?'+params;
    if(image.getAttribute('src')!==src) image.src=src;
    const raster=tiltRaster();
    if(image.parentElement!==raster)raster.append(image);
    image.style.transform='none';
    raster.style.transform='none';
  }
  const area=$('tilt-image-area');
  area.classList.toggle('show-axis',!!state.axisVisible);
  showCTFView(index);
}
function tiltRaster() {
  let raster=$('tilt-raster');
  if(!raster){raster=document.createElement('div');raster.id='tilt-raster';$('tilt-image-area').prepend(raster);}
  return raster;
}
function showCTFView(index) {
  const area=$('tilt-image-area');
  let layer=$('ctf-view-image');
  if(!layer){layer=document.createElement('img');layer.id='ctf-view-image';area.append(layer);}
  layer.hidden=!state.ctfView || state.ctfView==='landscape';
  const source=$('tilt-plane-image');
  if(source)source.style.visibility=['rings','comparison'].includes(state.ctfView)?'hidden':'visible';
  const note=$('ctf-view-note');
  if(!state.ctfView){if(note)note.remove();return;}
  layer.alt=state.ctfView==='comparison'?'CTF model on left and measured spectrum on right':state.ctfView==='rings'?'Measured cleaned CTF spectrum with Thon rings':'Modelled spatial defocus landscape';
  layer.className=state.ctfView==='landscape'?'ctf-landscape':'ctf-rings';
  // Landscape is already composited with its tilt in one backend response.
  // Spectra stay in the estimator's raw Fourier coordinates.
  const parent=area;
  if(layer.parentElement!==parent)parent.append(layer);
  layer.style.transform='none';
  if(state.ctfView!=='landscape') {
    const params=new URLSearchParams({project_id:state.project.id,series_id:state.selected,index,mode:state.ctfView});
    const src='/api/ctf-plane?'+params;
    if(layer.getAttribute('src')!==src)layer.src=src;
  }
  const caption=note || document.createElement('div');caption.id='ctf-view-note';
  caption.textContent=state.ctfView==='comparison'?'Left: fitted |CTF| model · Right: measured amplitude spectrum · grey: outside fit interval · raw Fourier orientation; independent display contrast':state.ctfView==='rings'?'Measured CTF spectrum · same tilt':'Defocus gradient relative to tilt centre · pink: lower / blue: higher · intensity scaled to maximum gradient in this series';
  if(state.ctfView==='landscape') {
    const fit=state.preprocessStatus?.ctf_local_geometry;
    const savedHand=fit?.model==='image_defocus_plane_v1'?null:(fit?.handedness ?? state.preprocessStatus?.options?.df_hand);
    caption.textContent+=fit?.model==='image_defocus_plane_v1'?' · measured image defocus plane':fit?' · saved tilt CTF model with fitted offsets':' · saved nominal CTF model (no local fit)';
    if(fit?.model==='image_defocus_plane_v1')caption.textContent+=' · archived free image plane; Redo CTF uses the tilt model';
    if([-1,1].includes(savedHand))caption.textContent+=` · fitted hand ${savedHand>0?'+1':'−1'}`;
    if(state.orientationHandUnsaved)caption.textContent+=' · selected sign applies after Redo CTF';
  }
  if(state.stage==='reconstruction' && state.reconstructionCTFPreview && state.ctfView==='landscape')caption.textContent='Defocus preview · selected handedness and axis · CTF will be re-estimated with the final geometry';
  if(!note)area.append(caption);
  layer.onerror=()=>{layer.hidden=true;if(source)source.style.visibility='visible';caption.textContent='CTF view unavailable. Restart the web server if it is outdated; check that this tilt has saved CTF spectra.';};
}
function matchTiltBackground(image) {
  // 2D alignment fills missing pixels with the plane mean. That constant
  // padding is the most frequent border intensity in the displayed PNG.
  const canvas=document.createElement('canvas');
  canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
  const context=canvas.getContext('2d',{willReadFrequently:true});
  if(!context || !canvas.width || !canvas.height) return;
  context.drawImage(image,0,0);
  const pixels=context.getImageData(0,0,canvas.width,canvas.height).data;
  const histogram=new Uint32Array(256);
  let sum=0;
  for(let y=0;y<canvas.height;y++)for(let x=0;x<canvas.width;x++){
    const value=pixels[4*(y*canvas.width+x)];sum+=value;
    if(x===0||y===0||x===canvas.width-1||y===canvas.height-1)histogram[value]++;
  }
  const mode=histogram.reduce((best,count,i)=>count>histogram[best]?i:best,0);
  const count=canvas.width*canvas.height;
  const borderCount=count-Math.max(0,canvas.width-2)*Math.max(0,canvas.height-2);
  const background=histogram[mode]>=borderCount*.1?mode:Math.round(sum/count);
  $('tilt-image-area').style.backgroundColor=`rgb(${background}, ${background}, ${background})`;
}

function selectTiltDiagnostic(mode,controls){
  if(mode==='axis')state.axisVisible=!state.axisVisible;
  else{
    state.ctfView=state.ctfView===mode?null:mode;
    if(mode==='comparison')state.axisVisible=false;
  }
  controls.querySelectorAll('[data-tilt-diagnostic]').forEach(button=>{
    const kind=button.dataset.tiltDiagnostic;
    button.setAttribute('aria-pressed',String(kind==='axis'?!!state.axisVisible:state.ctfView===kind));
  });
}
