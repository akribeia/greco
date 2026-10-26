"use strict";

const reconstructionViewCache=new Map();
function reconstructionViewKey(projectId,seriesId,job) {
  return JSON.stringify([projectId,seriesId,job?.work,job?.job_id,job?.state,job?.volumes?.original,job?.volumes?.tinycare,job?.landmarks?.path,job?.landmarks?.revision]);
}
function rememberReconstructionView() {
  const image=$('volume-image'),key=image?.dataset.reconstructionViewKey;
  image?.grecoZoomCleanup?.();
  if(image)delete image.grecoZoomCleanup;
  if(!key || !image.complete || !image.naturalWidth || image.grecoImageQueue?.active)return;
  const pixels=image.naturalWidth*image.naturalHeight;
  if(pixels>8*1024*1024)return;
  const view={image,job:state.reconstructionStatus,index:state.volumeIndex,method:state.volumeMethod,chosen:!!state.volumeMethodChosen,axis:$('volume-axis').value,pixels};
  image.remove();reconstructionViewCache.delete(key);reconstructionViewCache.set(key,view);
  let total=Array.from(reconstructionViewCache.values()).reduce((sum,item)=>sum+item.pixels,0);
  while(reconstructionViewCache.size>4 || total>8*1024*1024){
    const oldest=reconstructionViewCache.keys().next().value;
    total-=reconstructionViewCache.get(oldest).pixels;reconstructionViewCache.delete(oldest);
  }
}
function restoreReconstructionView(row) {
  const key=reconstructionViewKey(state.project.id,row.id,row.reconstruction);
  const cached=reconstructionViewCache.get(key);
  let image=$('volume-image');
  if(cached){
    reconstructionViewCache.delete(key);
    image.replaceWith(cached.image);image=cached.image;
    state.volumeMethod=cached.method;state.volumeMethodChosen=cached.chosen;state.volumeIndex=cached.index;state.reconstructionStatus=cached.job;
    $('volume-axis').value=cached.axis;
  }else if(['completed','partial'].includes(row.reconstruction?.state) && row.reconstruction?.shape){
    state.reconstructionStatus={...row.reconstruction,volumes:{...row.reconstruction.volumes}};
  }
  image.dataset.reconstructionViewKey=key;
  if(state.reconstructionStatus){
    $('recon-status').textContent=reconstructionStateLabel(state.reconstructionStatus);
    $('recon-status').className='badge '+(state.reconstructionStatus.state==='completed'?'':'pending');
    showVolumePlane();
  }
}

function reconstructionPanel(row) {
  const selected=state.project.reconstruction_denoisers || [];
  const alignment=state.project.reconstruction_alignment_options || {smoothness_strength:0,rescue_tilts:true};
  return `<section class="reconstruct-step"><div id="reconstruct-step-header" class="reconstruct-step-header"><span class="reconstruct-step-chevron" aria-hidden="true">▾</span><strong>Reconstruct 001</strong></div><div class="preprocessing-layout"><div class="reconstruction-viewer-column"><section class="panel tilt-viewer"><div class="panel-header"><span id="volume-title">${esc(row.name)} · Tomogram</span><span id="recon-status" class="badge pending">Checking output…</span></div><div class="tilt-image-area tomogram-area"><div id="volume-empty" class="viewer-empty">The tomogram will appear here after reconstruction.</div><img id="volume-image" hidden alt="Reconstructed tomogram slice"><div class="volume-variants volume-landmarks"><button type="button" class="button secondary" data-volume-method="landmarks" aria-pressed="false" title="Landmark projections from the latest available alignment pass" hidden>Landmarks</button></div><div class="volume-variants"><button type="button" class="button secondary" data-volume-method="tinycare" aria-pressed="false" title="Click again to return to the original volume" disabled>TinyCARE</button></div></div><div class="tilt-scroll"><select id="volume-axis" aria-label="Slice axis"><option value="z">Z</option><option value="y">Y</option><option value="x">X</option></select><input id="volume-slider" type="range" min="0" max="0" value="0" disabled aria-label="Tomogram slice"><span id="volume-slice">No tomogram</span></div><div id="recon-job-info" class="hint viewer-info"></div></section></div><form id="reconstruction-form" class="panel processing-parameters"><div class="panel-header">Greco Optimizer</div><div class="reconstruction-details"><label class="metadata-field"><span>Tomogram thickness (%)</span><input id="recon-thickness" type="number" min="30" max="100" step="1" required value="${esc(state.project.reconstruction_thickness_percent ?? 50)}"></label><label class="metadata-field"><span>Final reconstruction binning</span><select id="recon-final-binning">${[1,2,4,8,16].map(bin=>`<option value="${bin}" ${bin===(state.project.reconstruction_final_binning ?? row.preprocessing?.output_binning ?? row.preprocessing?.options?.output_binning ?? 8)?'selected':''}>Bin ${bin}</option>`).join('')}</select></label><label class="metadata-field"><span>Smoothness</span><input id="recon-smoothness" type="number" min="0" max="5" step="0.05" required value="${esc(alignment.smoothness_strength)}"></label><label class="check-line"><span>Rescue tilts (optional)</span><input id="recon-rescue" type="checkbox" ${alignment.rescue_tilts?'checked':''}></label><p class="hint">Greco Optimizer aligns the tilt series.<br>Smoothness penalizes abrupt tilt-to-tilt shifts.<br>Rescue retries difficult tilts with a wider search.</p></div><div class="panel-header">Final tomogram denoising</div><div class="parameter-section"><div class="metadata-grid">${[['tinycare','TinyCARE']].map(([key,label])=>`<label class="check-line"><span>${label}</span><input type="checkbox" data-recon-denoiser="${key}" ${selected.includes(key)?'checked':''}></label>`).join('')}</div></div><p id="recon-error" class="form-error viewer-info" role="alert"></p><div class="submit-actions"><button type="submit" class="button primary" data-submit-recon="one">Submit</button><button type="submit" class="button secondary" data-submit-recon="all">Submit all</button></div></form></div></section>${processingStepPanels(row)}`;
}
async function pollReconstruction(projectId,seriesId) {
  try {
    const job=await api('/api/reconstruction-status?'+new URLSearchParams({project_id:projectId,series_id:seriesId}));
    if(state.project?.id!==projectId||state.selected!==seriesId||state.stage!=='reconstruction'||!$('volume-image'))return;
    job.volumes=Object.fromEntries(Object.entries(job.volumes || {}).filter(([key])=>['original','tinycare'].includes(key)));
    state.reconstructionStatus=job;selectedSeries().reconstruction=job;renderStages();
    if($('recon-status')){
      $('recon-status').textContent=reconstructionStateLabel(job);
      $('recon-status').className='badge '+(job.state==='completed'?'':'pending');
      bindPreprocessingActions(job,'reconstruction','recon-status');
    }
    $('recon-job-info').textContent=[job.job_id?'Slurm job '+job.job_id:'','Greco Optimizer',job.alignment_options?`Smoothness ${job.alignment_options.smoothness_strength} · Rescue ${job.alignment_options.rescue_tilts?'±16 px':'off'}`:'',job.active_count?`${job.active_count}/${job.total_tilts} tilts`:'',job.final_binning?`Final bin ${job.final_binning}`:'',job.work || '',job.error || job.status_error || job.warning || '',...Object.entries(job.denoising_errors || {}).map(([m,e])=>m+': '+e)].filter(Boolean).join(' · ');
    showVolumePlane();
    if(state.activeGeometryStep)showGeometryVolume();
    if(['submitted','running','unknown'].includes(job.state))state.reconstructionTimer=setTimeout(()=>pollReconstruction(projectId,seriesId),8000);
  } catch(error) {if($('recon-error'))$('recon-error').textContent=error.message;}
}
function applyVolumeContrast() {
  const contrast=state.volumeContrast ?? 1;
  if($('volume-image'))$('volume-image').style.filter='none';
  if($('volume-autocontrast'))$('volume-autocontrast').setAttribute('aria-pressed',String(!!state.volumeAutoContrast));
  if($('volume-autocontrast'))$('volume-autocontrast').classList.toggle('active',!!state.volumeAutoContrast);
  if($('volume-brightness-value'))$('volume-brightness-value').textContent=Math.round((state.volumeBrightness ?? 0)*100)+'%';
  if($('volume-contrast-value'))$('volume-contrast-value').textContent=contrast.toFixed(2)+'×';
}
function toggleReconstructionMethod(method){
  const fallback=method==='landmarks' && state.reconstructionStatus?.volumes?.tinycare?'tinycare':'original';
  const next=state.volumeMethod===method?fallback:method;
  state.volumeMethodChosen=true;
  if((next==='landmarks')!==(state.volumeMethod==='landmarks'))state.volumeIndex=null;
  state.volumeMethod=next;
  showVolumePlane();
}
function showVolumePlane() {
  const job=state.reconstructionStatus;
  if(!job?.shape || !$('volume-slider'))return;
  const available=method=>method==='landmarks'?Boolean(job.landmarks):Boolean(job.volumes?.[method]);
  if(!state.volumeMethodChosen && available('tinycare'))state.volumeMethod='tinycare';
  if(!available(state.volumeMethod))state.volumeMethod='original';
  const landmarks=state.volumeMethod==='landmarks',preview=job.landmarks;
  const axisControl=$('volume-axis');axisControl.hidden=landmarks;axisControl.disabled=landmarks;
  const axis=landmarks?'z':axisControl.value;
  const dimension={z:0,y:1,x:2}[axis];
  const size=landmarks?preview.angles.length:(job.volume_shapes?.[state.volumeMethod] || job.shape)[dimension];
  const middle=landmarks?preview.angles.reduce((best,a,i)=>Math.abs(a)<Math.abs(preview.angles[best])?i:best,0):Math.floor(size/2);
  const index=state.volumeIndex==null?middle:Math.max(0,Math.min(size-1,state.volumeIndex));
  const slider=$('volume-slider');
  state.volumeIndex=index;slider.disabled=false;slider.max=size-1;slider.value=index;
  slider.setAttribute('aria-label',landmarks?'Landmark tilt':'Tomogram slice');
  $('volume-slice').textContent=landmarks?`${index+1} / ${size} · ${angle(preview.angles[index])}`:`${axis.toUpperCase()} · ${index+1} / ${size}`;
  $('volume-slice').title=landmarks?`Tilt ${preview.tilt_numbers[index]} · ${preview.stage}`:'';
  const availableVolumes=Object.values(job.volumes || {}).filter(Boolean).length;
  document.querySelectorAll('[data-volume-method]').forEach(b=>{
    const method=b.dataset.volumeMethod;b.disabled=!available(method);
    b.hidden=b.disabled || (method!=='landmarks' && availableVolumes<2);
    b.classList.toggle('active',method===state.volumeMethod);b.setAttribute('aria-pressed',String(method===state.volumeMethod));
  });
  if($('volume-title'))$('volume-title').textContent=selectedSeries().name+' · '+(landmarks?'Landmarks':'Tomogram');
  $('volume-empty').hidden=true;const image=$('volume-image');image.hidden=false;
  image.alt=landmarks?'Landmark projection at the selected tilt':'Reconstructed tomogram slice';
  image.dataset.reconstructionViewKey=reconstructionViewKey(state.project.id,state.selected,job);
  applyVolumeContrast();
  requestViewerImage(image,'/api/volume-plane?'+new URLSearchParams({project_id:state.project.id,series_id:state.selected,index,axis,method:state.volumeMethod,revision:JSON.stringify([job.work,job.job_id,landmarks?[preview.path,preview.revision]:job.volumes?.[state.volumeMethod]]),autocontrast:state.volumeAutoContrast?'1':'0',contrast:state.volumeContrast ?? 1,brightness:state.volumeBrightness ?? 0}));
  image.onerror=()=>{const error=$('recon-error') || $('recon-job-info');if(error)error.textContent=landmarks?'Could not load this landmark tilt.':'Could not load this tomogram slice.';};
}

async function submitReconstruction(event) {
  event.preventDefault();
  const all=event.submitter.dataset.submitRecon==='all';
  const ids=all?submittableSeries('reconstruction'):[state.selected];
  if(all&&!confirm(`Submit Greco Optimizer reconstruction for ${ids.length} tilt series? Preprocessing must be completed or submitted to Slurm.`))return;
  const denoisers=Array.from(document.querySelectorAll('[data-recon-denoiser]:checked')).map(input=>input.dataset.reconDenoiser);
  const buttons=Array.from(event.currentTarget.querySelectorAll('button'));buttons.forEach(b=>b.disabled=true);$('recon-error').textContent='';
  try {
    const optimizer='greco_optimizer';
    const smoothness=Number($('recon-smoothness').value),rescue=$('recon-rescue').checked;
    const entrySigma=Number(state.project.reconstruction_entry_sigma_px ?? 8);
    const thickness=Number($('recon-thickness').value);
    if(!Number.isFinite(thickness) || thickness<30 || thickness>100)throw new Error('Tomogram thickness must be between 30% and 100%.');
    {
      let capabilities;
      try {capabilities=await api('/api/reconstruction-capabilities');}
      catch(error) {throw new Error('Restart the Greco web server and reload this page to enable the Greco Optimizer. No job was submitted.');}
      if(!capabilities.final_binning || !capabilities.integrated_smoothness || capabilities.optimizer!==optimizer || capabilities.optimizer_version!==1)
        throw new Error('The server does not support the Greco Optimizer. Restart it and reload this page. No job was submitted.');
    }
    const result=await api('/api/submit-reconstruction',{project_id:state.project.id,series_ids:ids,denoisers,optimizer,entry_search_sigma_px:entrySigma,tomogram_thickness_percent:thickness,final_binning:Number($('recon-final-binning').value),smoothness_strength:smoothness,rescue_tilts:rescue,submit_all:all});
    saveProject(result);notify(result.submission_errors?.length?result.submission_errors.join(' · '):`${'Submitted'} ${ids.length} Greco Optimizer job(s).`);
  } catch(error){if($('recon-error'))$('recon-error').textContent=error.message;}
  finally{buttons.forEach(b=>b.disabled=false);}
}
