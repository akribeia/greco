"use strict";

function geometryRefinementPanel(row) {
  const step=processingSteps(row).find(s=>s.id===state.activeGeometryStep);
  const prior=step?precedingGeometry(row,step):null;
  const options=(step && processingStepJob(row,step)?.options) || (prior?.kind==='geometry_refinement' && processingStepJob(row,prior)?.options) || state.project.geometry_refinement_options || {binning:8,patch_size:33,max_passes:10,optimizer_iterations:1000,search_support_A:16};
  const ready=['completed','partial','submitted','running'].includes(row.reconstruction?.state);
  return `<div class="preprocessing-layout geometry-refinement-layout"><div class="reconstruction-viewer-column">${geometryVolumePanel(row)}</div><form id="geometry-refinement-form" class="panel processing-parameters"><div class="panel-header"><span>Joint landmark geometry</span><span id="angle-status" class="badge pending">${ready?'Ready':'Pending'}</span></div><div class="parameter-section"><p>Polishing-style joint fit: three rigid rotations and two translations per tilt, together with static landmark XYZ.</p><div class="metadata-grid"><label><span>Joint geometry binning</span><select id="geometry-binning">${[8,4,2,1].map(bin=>`<option value="${bin}" ${options.binning===bin?'selected':''}>Bin ${bin}</option>`).join('')}</select></label><label><span>Landmark patch (px)</span><input id="geometry-patch-size" type="number" value="${geometryPatchSize(options.binning)}" disabled><small>Fixed at 33 px in bin8; scales with binning.</small></label><label><span>Maximum passes</span><input id="geometry-max-passes" type="number" min="1" max="100" value="${options.max_passes}" required></label></div><details><summary>Advanced</summary><div class="metadata-grid"><label><span>Optimizer iterations per pass</span><input id="geometry-iterations" type="number" min="1" max="10000" value="${options.optimizer_iterations}" required></label><label><span>Correlation support (Å)</span><input id="geometry-support" type="number" min="0.1" max="500" step="0.1" value="${options.search_support_A}" required></label></div><p class="hint">Correlation support is limited by the landmark patch size. It does not bound the rotation angles.</p></details></div><div class="reconstruction-details"><p>${prior?.kind==='manual_curation'?'Rebuilds landmarks from the corrected manual geometry before fitting.':'Inherits landmarks from the source geometry.'}</p><p>Uses the rigid GPU optimizer from polishing. The first acquired tilt is anchored. Steps are accepted, reduced or rejected using the likelihood of the full coupled geometry.</p><p>Local landmark references exclude the fitted tilt and are rebuilt each pass. Bin 1/2/4 replays saved Greco motion correction and CTF corrections; at least eight supported landmarks per tilt are required.</p><p>Saves full camera matrices, landmark XYZ, a bin8 projection stack for review and a tomogram at the selected final binning.</p></div>${finalReconstructionOptions('geometry',options)}<div id="angle-job-info" class="hint viewer-info"></div><p id="angle-error" class="form-error viewer-info" role="alert"></p><div class="submit-actions"><button type="submit" class="button primary" data-submit-angle="one" ${ready?'':'disabled'}>Submit</button><button type="submit" class="button secondary" data-submit-angle="all">Submit all</button></div></form></div>`;
}
function geometryPatchSize(binning){
  return 1+256/Number(binning);
}
function mountTinycareReuse(denoiser,reuse){
  const update=()=>{reuse.disabled=!denoiser.checked;if(reuse.disabled)reuse.checked=false;};
  denoiser.addEventListener('change',update);update();
}
function mountGeometryPatchScaling(){
  mountTinycareReuse(document.querySelector('[data-geometry-denoiser="tinycare"]'),$('geometry-reuse-tinycare'));
  const binning=$('geometry-binning'),patch=$('geometry-patch-size');
  const update=()=>{patch.value=geometryPatchSize(binning.value);};
  binning.addEventListener('change',update);update();
}
function geometryVolumePanel(row){
  return `<section class="panel tilt-viewer geometry-volume-viewer"><div class="panel-header"><span>${esc(row.name)} · Refined geometry</span></div><div class="tilt-image-area tomogram-area"><div id="geometry-volume-empty" class="viewer-empty">The final tomogram will appear here after geometry refinement.</div><img id="geometry-volume-image" hidden alt="Geometry-refined tomogram slice"><div class="volume-variants"><button type="button" class="button secondary" id="geometry-view-tinycare" aria-pressed="false">TinyCARE</button><span id="geometry-volume-sources"></span></div></div><div class="tilt-scroll"><select id="geometry-volume-axis" aria-label="Geometry slice axis"><option value="z">Z</option><option value="y">Y</option><option value="x">X</option></select><input id="geometry-volume-slider" type="range" min="0" max="0" value="0" disabled aria-label="Geometry tomogram slice"><span id="geometry-volume-slice">No tomogram</span></div></section>`;
}
function mountReconstructionZoom(image){
  const area=image.parentElement,view={zoom:1,x:0,y:0};
  area.insertAdjacentHTML('beforeend','<div class="picker-view-tools"><button type="button" class="button secondary" data-reconstruction-zoom="in" aria-label="Zoom in" title="Zoom in · drag to move">🔍 +</button><button type="button" class="button secondary" data-reconstruction-zoom="out" aria-label="Zoom out" title="Zoom out">🔍 −</button></div>');
  const apply=()=>{
    view.zoom=Math.max(1,Math.min(8,view.zoom));
    const maxX=area.clientWidth*(view.zoom-1)/2,maxY=area.clientHeight*(view.zoom-1)/2;
    view.x=Math.max(-maxX,Math.min(maxX,view.x));view.y=Math.max(-maxY,Math.min(maxY,view.y));
    image.style.transform=`translate(${view.x}px, ${view.y}px) scale(${view.zoom})`;
    area.classList.toggle('reconstruction-pan-enabled',view.zoom>1);
    area.querySelector('[data-reconstruction-zoom="out"]').disabled=view.zoom===1;
    area.querySelector('[data-reconstruction-zoom="in"]').disabled=view.zoom===8;
  };
  area.querySelectorAll('[data-reconstruction-zoom]').forEach(button=>button.addEventListener('click',()=>{
    view.zoom*=button.dataset.reconstructionZoom==='in'?1.5:1/1.5;apply();
  }));
  let drag=null;image.draggable=false;
  area.addEventListener('pointerdown',event=>{
    if(view.zoom<=1 || event.button!==0 || event.target!==image)return;
    event.preventDefault();drag={x:event.clientX,y:event.clientY,px:view.x,py:view.y};area.setPointerCapture(event.pointerId);
  });
  area.addEventListener('pointermove',event=>{if(!drag)return;view.x=drag.px+event.clientX-drag.x;view.y=drag.py+event.clientY-drag.y;apply();});
  for(const name of ['pointerup','pointercancel','lostpointercapture'])area.addEventListener(name,()=>{drag=null;});
  image.grecoZoomCleanup?.();
  image.grecoZoomCleanup=()=>image.removeEventListener('load',apply);
  image.addEventListener('load',apply);
  apply();
}
function mountGeometryContrast(){
  clearTimeout(state.geometryContrastTimer);
  $('geometry-volume-image').insertAdjacentHTML('beforebegin',`<details class="volume-contrast picker-contrast-panel"><summary>Contrast</summary><div class="picker-contrast-controls"><div class="contrast-actions"><button id="geometry-autocontrast" type="button" class="button secondary">Auto contrast</button><button id="geometry-contrast-reset" type="button" class="button secondary">Reset contrast</button></div><label>Contrast <input id="geometry-contrast-slider" type="range" min="0.25" max="4" step="0.05" value="${state.geometryContrast ?? 1}" aria-label="Geometry tomogram contrast"><output id="geometry-contrast-value"></output></label><label>Brightness <input id="geometry-brightness-slider" type="range" min="-1" max="1" step="0.02" value="${state.geometryBrightness ?? 0}" aria-label="Geometry tomogram brightness"><output id="geometry-brightness-value"></output></label></div></details>`);
  const update=()=>{
    $('geometry-contrast-value').textContent=(state.geometryContrast ?? 1).toFixed(2)+'×';
    $('geometry-brightness-value').textContent=Math.round((state.geometryBrightness ?? 0)*100)+'%';
    $('geometry-autocontrast').classList.toggle('active',!!state.geometryAutoContrast);
    $('geometry-autocontrast').setAttribute('aria-pressed',String(!!state.geometryAutoContrast));
  };
  const refresh=()=>{update();clearTimeout(state.geometryContrastTimer);state.geometryContrastTimer=setTimeout(showGeometryVolume,120);};
  $('geometry-contrast-slider').addEventListener('input',()=>{state.geometryContrast=Number($('geometry-contrast-slider').value);refresh();});
  $('geometry-brightness-slider').addEventListener('input',()=>{state.geometryBrightness=Number($('geometry-brightness-slider').value);refresh();});
  const reset=automatic=>{clearTimeout(state.geometryContrastTimer);state.geometryAutoContrast=automatic;state.geometryContrast=1;state.geometryBrightness=0;$('geometry-contrast-slider').value=1;$('geometry-brightness-slider').value=0;update();showGeometryVolume();};
  $('geometry-autocontrast').title='Compute a robust intensity window from the current MRC slice';
  $('geometry-autocontrast').addEventListener('click',()=>reset(true));
  $('geometry-contrast-reset').addEventListener('click',()=>reset(false));
  update();
}
async function loadGeometryVolumeSources(projectId,seriesId,stepId){
  const steps=processingSteps(selectedSeries());
  const results=await Promise.allSettled(steps.filter(step=>processingStepJob(selectedSeries(),step)?.work).map(async step=>{
    const job=await api('/api/geometry-refinement-status?'+new URLSearchParams({project_id:projectId,series_id:seriesId,step_id:step.id}));
    return {id:step.id,number:step.number,job};
  }));
  if(state.project?.id!==projectId || state.selected!==seriesId || state.activeGeometryStep!==stepId)return;
  state.geometryVolumeSources=results.filter(result=>result.status==='fulfilled').map(result=>result.value);
  showGeometryVolume();
}
function toggleGeometryTinycare(){
  state.geometryViewTinycare=!state.geometryViewTinycare;
  state.geometryTinycareChoices[state.geometryTinycareChoiceKey]=state.geometryViewTinycare;
  state.geometryVolumeIndex=null;showGeometryVolume();
}
function showGeometryVolume(){
  if(!$('geometry-volume-image'))return;
  const row=selectedSeries();
  const sources=[{id:'reconstruction',number:1,job:state.reconstructionStatus || row.reconstruction},
    ...processingSteps(row).map((step,index)=>({id:step.id,number:step.number ?? index+2,
      job:step.id===state.activeGeometryStep?state.geometryStatus:
        state.geometryVolumeSources?.find(source=>source.id===step.id)?.job || processingStepJob(row,step)}))]
    .filter(source=>source.job?.state==='completed' && (source.job.preview_shape || source.job.shape) &&
      (source.job.volumes?.original || source.job.tomogram))
    .sort((a,b)=>a.number-b.number);
  if(!sources.some(source=>source.id===state.geometryVolumeSource))state.geometryVolumeSource=sources.at(-1)?.id;
  const source=sources.find(source=>source.id===state.geometryVolumeSource);
  state.geometryTinycareChoices ??= {};
  state.geometryTinycareChoiceKey=JSON.stringify([state.project.id,state.selected,source?.id,source?.job.work,source?.job.job_id,source?.job.volumes?.original]);
  const available=Boolean(source?.job.volumes?.tinycare);
  state.geometryViewTinycare=available && (state.geometryTinycareChoices[state.geometryTinycareChoiceKey] ?? true);
  const method=state.geometryViewTinycare?'tinycare':'original',tinycare=$('geometry-view-tinycare');
  tinycare.hidden=!available;tinycare.disabled=!available;
  tinycare.classList.toggle('active',state.geometryViewTinycare);
  tinycare.setAttribute('aria-pressed',String(state.geometryViewTinycare));
  $('geometry-volume-sources').innerHTML=sources.map(item=>`<button type="button" class="button secondary ${item.id===source?.id?'active':''}" data-geometry-volume-source="${esc(item.id)}">R ${String(item.number).padStart(3,'0')}</button>`).join('');
  document.querySelectorAll('[data-geometry-volume-source]').forEach(button=>button.addEventListener('click',()=>{
    state.geometryVolumeSource=button.dataset.geometryVolumeSource;state.geometryVolumeIndex=null;showGeometryVolume();
  }));
  const image=$('geometry-volume-image');
  if(!source){
    image.hidden=true;$('geometry-volume-empty').hidden=false;
    $('geometry-volume-empty').textContent=state.geometryViewTinycare?'No TinyCARE tomograms available.':'No reconstructed tomograms available.';
    $('geometry-volume-slider').disabled=true;$('geometry-volume-slice').textContent='No tomogram';return;
  }
  const job=source.job,axis=$('geometry-volume-axis').value,size=(job.preview_shape || job.shape)[{z:0,y:1,x:2}[axis]];
  const index=state.geometryVolumeIndex==null?Math.floor(size/2):Math.max(0,Math.min(size-1,state.geometryVolumeIndex));
  state.geometryVolumeIndex=index;const slider=$('geometry-volume-slider');slider.disabled=false;slider.max=size-1;slider.value=index;
  $('geometry-volume-slice').textContent=`${axis.toUpperCase()} · ${index+1} / ${size}`;
  $('geometry-volume-empty').hidden=true;image.hidden=false;
  const query={project_id:state.project.id,series_id:state.selected,index,axis,method,autocontrast:state.geometryAutoContrast?'1':'0',contrast:state.geometryContrast ?? 1,brightness:state.geometryBrightness ?? 0};
  if(source.id!=='reconstruction')query.step_id=source.id;
  requestViewerImage(image,'/api/volume-plane?'+new URLSearchParams(query));
  image.onerror=()=>{if($('angle-error'))$('angle-error').textContent='Could not load the geometry tomogram slice.';};
}
async function requireLinearSteps(){
  const capabilities=await api('/api/reconstruction-capabilities');
  if(!capabilities.linear_refinement_steps)throw new Error('Restart the Greco server to enable linear refinement steps.');
}
async function pollAngleRefinement(projectId,seriesId){
  const stepId=state.activeGeometryStep;
  if(!stepId)return;
  try{
    const job=await api('/api/geometry-refinement-status?'+new URLSearchParams({project_id:projectId,series_id:seriesId,step_id:stepId}));
    if(job.step_id!==stepId)throw new Error('Restart the Greco server to enable linear refinement steps.');
    if(state.project?.id!==projectId || state.selected!==seriesId || state.stage!=='reconstruction' || !$('angle-status') || state.activeGeometryStep!==stepId)return;
    const step=processingSteps(selectedSeries()).find(s=>s.id===stepId);
    if(stepId==='geometry')selectedSeries().geometry_refinement=job;else if(step)step.job=job;
    state.geometryStatus=job;showGeometryVolume();
    $('angle-status').textContent='Geometry · bin'+(job.binning ?? job.options?.binning ?? 8)+' · '+job.state;
    $('angle-status').className='badge '+(job.state==='completed'?'':'pending');
    $('angle-job-info').textContent=[job.state,job.job_id?'Slurm '+job.job_id:'',job.geometry_model?'3 rotations + XY per tilt':'',job.max_rotation_change_deg!=null?'Max rotation Δ '+Number(job.max_rotation_change_deg).toFixed(3)+'°':'',job.landmark_count?job.landmark_count+' landmarks':'',job.passes?job.passes+' passes':'',job.converged===true?'Converged':job.converged===false?'Not converged':'',job.stop_reason || '',job.tomogram || '',job.report || '',job.warning || ''].filter(Boolean).join(' · ');
    $('angle-error').textContent=job.error || job.status_error || '';
    if(job.state==='completed' && !state.reconstructionStatus?.volumes?.geometry)pollReconstruction(projectId,seriesId);
    if(['submitted','running','unknown'].includes(job.state))state.angleRefinementTimer=setTimeout(()=>pollAngleRefinement(projectId,seriesId),8000);
  }catch(error){if($('angle-error'))$('angle-error').textContent=error.message;}
}
async function submitAngleRefinement(event){
  event.preventDefault();
  const stepId=event.currentTarget.dataset.geometryStepId;
  if(!stepId){$('angle-error').textContent='Reopen the geometry refinement step before submitting.';return;}
  const all=event.submitter.dataset.submitAngle==='all';
  const ids=all?firstGeometrySeriesIds():[state.selected];
  if(!ids.length){notify('No new jobs to submit. Existing jobs are preserved.');return;}
  if(all&&!confirm(`Submit landmark geometry refinement for ${ids.length} tilt series?`))return;
  const options={binning:Number($('geometry-binning').value),patch_size:geometryPatchSize($('geometry-binning').value),max_passes:Number($('geometry-max-passes').value),optimizer_iterations:Number($('geometry-iterations').value),search_support_A:Number($('geometry-support').value)};
  options.denoisers=Array.from(event.currentTarget.querySelectorAll('[data-geometry-denoiser]:checked')).map(input=>input.dataset.geometryDenoiser);
  options.reuse_tinycare_model=$('geometry-reuse-tinycare').checked;
  options.final_binning=Number($('geometry-final-binning').value);
  const buttons=Array.from(event.currentTarget.querySelectorAll('button'));buttons.forEach(b=>b.disabled=true);
  $('angle-error').textContent='';
  try{await requireLinearSteps();const result=await api('/api/submit-geometry-refinement',{project_id:state.project.id,series_ids:ids,options,submit_all:all,step_id:stepId});saveProject(result);const panel=document.querySelector(`[data-step-id="${stepId}"]`);if(panel)panel.open=true;notify(result.submission_errors?.length?result.submission_errors.join(' · '):all?'Submitted landmark geometry refinement.':'Submitted landmark geometry refinement.');}
  catch(error){if($('angle-error'))$('angle-error').textContent=error.message;}
  finally{buttons.forEach(b=>b.disabled=false);}
}
