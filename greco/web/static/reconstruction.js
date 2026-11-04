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

const reconstructionAdvancedFields=[['refine_common_axis','Refine β · common tilt axis'],['refine_per_view_axis','Refine β · per view'],['refine_tilt_angles','Refine α · tilt angles'],['refine_gamma','Refine γ (experimental)']];
function reconstructionAdvancedDraft() {
  if(state.reconstructionAdvancedProject!==state.project.id+':'+(state.project.active_reconstruction_assembly || '')){
    state.reconstructionAdvancedProject=state.project.id+':'+(state.project.active_reconstruction_assembly || '');
    state.reconstructionAdvancedDraft={...(state.project.reconstruction_advanced_options || {})};
    const record=reconstructionAssembly();if(record)state.reconstructionAdvancedDraft.enabled=record.mode==='advanced';
  }
  return state.reconstructionAdvancedDraft;
}
function reconstructionAdvancedPanel() {
  const options=reconstructionAdvancedDraft();
  const hand=options.defocus_handedness ?? state.project.orientation?.df_hand ?? state.project.preprocessing_options?.df_hand ?? 1;
  return `<div id="recon-advanced-options" class="parameter-section" hidden><div class="metadata-grid">${reconstructionAdvancedFields.map(([key,label])=>`<label class="check-line"><span>${label}</span><input type="checkbox" data-recon-advanced="${key}" ${key==='refine_gamma'?'title="Final individual camera refinement · fixed landmarks · minimum total change"':''} ${options[key]?'checked':''}></label>`).join('')}<div class="metadata-field"><label for="recon-df-hand">Defocus handedness</label><select id="recon-df-hand">${[-1,1].map(value=>`<option value="${value}" ${Number(hand)===value?'selected':''}>${value}</option>`).join('')}</select></div><label class="check-line"><span>Flip tomogram Z</span><input id="recon-flip-z" type="checkbox" ${options.flip_tomogram_z?'checked':''}></label></div><p class="hint">CTF is re-estimated with the final geometry in Advanced mode.<br>Check defocus handedness using the TS defocus view before submitting all.<br>Flip tomogram Z rotates the tilt series by 180°.</p>${defocusHandHelp()}</div>`;
}
function reconstructionFineScalePanel(){
  const options=reconstructionAdvancedDraft();
  return `<div id="recon-fine-scale-options" class="parameter-section" hidden><label class="check-line"><span>Fine-scale refinement</span><input id="recon-fine-scale" type="checkbox" ${options.fine_scale_refinement?'checked':''}></label><div id="recon-fine-scale-fields" class="fine-scale-controls"><label class="fine-scale-field"><span>Binning</span><select id="recon-fine-binning">${[4,2,1].map(bin=>`<option value="${bin}" ${bin===(options.fine_scale_binning ?? 4)?'selected':''}>Bin ${bin}</option>`).join('')}</select></label><label class="fine-scale-field"><span>Passes</span><input id="recon-fine-rounds" type="number" min="1" max="20" step="1" value="${options.fine_scale_rounds ?? 5}"></label></div><small class="recon-fine-help">Same landmarks and refinement settings. CTF is re-estimated before and after.<br><strong>Warning:</strong> This can be slow and is not required to reconstruct at bin 4, bin 2 or bin 1.</small></div><div id="recon-leveling-options" class="parameter-section" hidden><label class="check-line"><span>Tomogram leveling</span><input id="recon-leveling" type="checkbox" ${options.tomogram_leveling?'checked':''}></label><small class="recon-fine-help">Fits a reference plane to the reconstructed landmark positions and applies a rigid 3D rotation to level the tomogram.</small></div>`;
}
function reconstructionAdvancedSelection() {
  return {enabled:!!reconstructionAdvancedDraft().enabled,...Object.fromEntries(reconstructionAdvancedFields.filter(([key])=>key!=='refine_gamma' || document.querySelector(`[data-recon-advanced="${key}"]`).checked).map(([key])=>[key,document.querySelector(`[data-recon-advanced="${key}"]`).checked])),defocus_handedness:Number($('recon-df-hand').value),flip_tomogram_z:$('recon-flip-z').checked,tomogram_leveling:!!$('recon-leveling')?.checked,fine_scale_refinement:!!$('recon-fine-scale')?.checked,fine_scale_binning:Number($('recon-fine-binning')?.value ?? 4),fine_scale_rounds:Number($('recon-fine-rounds')?.value ?? 5)};
}
function mountReconstructionAdvanced() {
  const smoothness=$('recon-smoothness');
  let previousSmoothness=smoothness.value,locked=false;
  function update(){
    const options=reconstructionAdvancedDraft(),active=!!options.enabled;
    $('recon-advanced-options').hidden=!active;
    $('recon-parameter-title').textContent=active?'Advanced reconstruction':'Greco Optimizer';
    const angular=active;
    if(angular && !locked){previousSmoothness=smoothness.value;smoothness.value=0;}
    if(!angular && locked)smoothness.value=previousSmoothness;
    locked=angular;smoothness.disabled=angular;
    smoothness.closest('label').hidden=active;
    $('recon-normal-help').hidden=active;
    $('recon-rescue').closest('label').hidden=active;
    const ts=$('recon-ts');
    if(ts){
      ts.hidden=!active;
      if(!active && ts.checked){ts.checked=false;ts.dispatchEvent(new Event('change'));}
    }
    for(const [key] of reconstructionAdvancedFields)options[key]=document.querySelector(`[data-recon-advanced="${key}"]`).checked;
    options.defocus_handedness=Number($('recon-df-hand').value);options.flip_tomogram_z=$('recon-flip-z').checked;
    const fine=$('recon-fine-scale');
    if(fine){
      $('recon-fine-scale-options').hidden=!active;
      if($('recon-leveling-options'))$('recon-leveling-options').hidden=!active;
      options.tomogram_leveling=active && !!$('recon-leveling')?.checked;
      options.fine_scale_refinement=active && fine.checked;
      options.fine_scale_binning=Number($('recon-fine-binning').value);
      options.fine_scale_rounds=Number($('recon-fine-rounds').value);
      $('recon-fine-binning').disabled=$('recon-fine-rounds').disabled=!fine.checked;
      $('recon-fine-scale-fields').classList.toggle('fine-scale-disabled',!fine.checked);
    }
    if($('recon-ts')?.checked)showReconstructionTS();
  }
  document.querySelectorAll('[data-recon-advanced],#recon-df-hand,#recon-flip-z,#recon-fine-scale,#recon-fine-binning,#recon-fine-rounds,#recon-leveling').forEach(input=>input.addEventListener('change',update));
  update();
}
function mountReconstructionTS(){
  state.reconstructionTSStatus=null;state.axisVisible=false;state.ctfView=null;
  const check=$('recon-ts');
  $('recon-ts-slider').addEventListener('input',()=>{ showReconstructionTS(); });
  const controls=$('recon-ts-controls');
  for(const [mode,label,title] of [['comparison','◐','Compare CTF model (left) and measured spectrum (right)'],['landscape','ΔZ','Show saved defocus landscape'],['axis','[|]','Show vertical tilt-axis guide']]){
    const button=document.createElement('button');button.type='button';button.className=mode==='axis'?'button secondary':'button secondary ctf-view-toggle';if(mode==='axis')button.id='axis-toggle';button.dataset.tiltDiagnostic=mode;button.textContent=label;button.title=title;button.setAttribute('aria-label',title);button.setAttribute('aria-pressed','false');
    button.addEventListener('click',()=>{
      selectTiltDiagnostic(mode,controls);
      if(mode==='axis')$('tilt-image-area').classList.toggle('show-axis',state.axisVisible);
      else showReconstructionTS();
    });controls.append(button);
  }
  check.checked=false;check.textContent=reconstructionViewsEditable()?'Exclude views':'Tomogram';
  check.addEventListener('click',()=>{check.checked=!check.checked;check.dispatchEvent(new Event('change'));});
  let tomogramView=null;
  check.addEventListener('change',()=>{
    if(check.checked){
      tomogramView={index:state.volumeIndex,axis:$('volume-axis').value,method:state.volumeMethod,chosen:state.volumeMethodChosen};
    }else if(tomogramView){
      state.volumeIndex=tomogramView.index;
      state.volumeMethod=tomogramView.method;
      state.volumeMethodChosen=tomogramView.chosen;
      $('volume-axis').value=tomogramView.axis;
    }
    check.textContent=reconstructionViewsEditable()?'Exclude views':check.checked?'Tilt series':'Tomogram';
    $('recon-view-grid').hidden=!check.checked;
    if(check.checked)$('volume-empty').hidden=true;
    check.classList.toggle('active',check.checked);check.setAttribute('aria-pressed',String(check.checked));
    $('recon-ts-viewer').hidden=!check.checked;
    $('recon-ts-scroll').hidden=!check.checked;
    $('volume-slider').closest('.tilt-scroll').hidden=check.checked;
    $('volume-image').hidden=check.checked;
    $('volume-slider').disabled=check.checked;
    $('volume-image').parentElement.querySelector('.volume-contrast').hidden=check.checked;
    if(!check.checked && state.volumeMethod==='landmarks'){state.volumeMethod='original';state.volumeMethodChosen=false;}
    document.querySelectorAll('[data-volume-method]').forEach(button=>button.hidden=button.dataset.volumeMethod==='landmarks'?!check.checked:check.checked);
    if(check.checked)showReconstructionTS();else{
      if(!state.reconstructionStatus?.shape){$('volume-empty').hidden=false;$('volume-image').hidden=true;$('volume-title').textContent=selectedSeries().name+' · Tomogram';}
      showVolumePlane();
    }
  });
  const projectId=state.project.id,seriesId=state.selected;
  api('/api/preprocessing-status?'+new URLSearchParams({project_id:projectId,series_id:seriesId})).then(job=>{
    if(state.project?.id!==projectId || state.selected!==seriesId || !check.isConnected)return;
    state.reconstructionTSStatus=job;
    renderReconstructionViewGrid();
    check.disabled=job.state!=='completed' || !job.angles?.length;
    if(check.checked)showReconstructionTS();
  }).catch(error=>{check.title=error.message;});
  bindSliceWheel($('recon-ts-viewer'),$('recon-ts-slider'));
}
function showReconstructionTS(){
  if(!$('recon-ts')?.checked)return;
  renderReconstructionViewGrid();
  const landmarkButton=document.querySelector('[data-volume-method=landmarks]');
  if(landmarkButton){landmarkButton.hidden=!state.reconstructionStatus?.landmarks;landmarkButton.disabled=!state.reconstructionStatus?.landmarks;landmarkButton.classList.toggle('active',state.volumeMethod==='landmarks');landmarkButton.setAttribute('aria-pressed',String(state.volumeMethod==='landmarks'));}
  const landmarks=state.volumeMethod==='landmarks';
  $('recon-ts-viewer').hidden=landmarks;$('recon-ts-scroll').hidden=landmarks;
  $('volume-slider').closest('.tilt-scroll').hidden=!landmarks;
  $('volume-image').hidden=!landmarks;
  if(landmarks){showVolumePlane();return;}
  const refined=state.reconstructionStatus?.tilt_series;
  const buttons=$('recon-ts-controls').querySelectorAll('button');
  buttons.forEach((button,i)=>{if(i<2){button.disabled=!!refined && !refined.final_ctf;button.title=refined?(i===0?'Compare final CTF model and measured spectrum':'Show final refined defocus landscape'):(i===0?'Compare CTF model and measured spectrum':'Show saved defocus landscape');}});
  if(refined?.angles?.length){
    if(!refined.final_ctf)state.ctfView=null;
    const slider=$('recon-ts-slider');slider.disabled=false;slider.max=refined.angles.length-1;
    if(slider.dataset.geometryRevision!==String(refined.revision)){slider.value=refined.angles.reduce((best,a,i)=>Math.abs(a)<Math.abs(refined.angles[best])?i:best,0);slider.dataset.geometryRevision=String(refined.revision);}
    const index=Math.max(0,Math.min(refined.angles.length-1,Number(slider.value)));
    highlightReconstructionView(refined.source_indices?.[index] ?? index);
    $('recon-ts-angle').textContent=`${index+1} / ${refined.angles.length} · ${angle(refined.angles[index])}`;
    const image=$('tilt-plane-image');image.style.visibility='visible';
    requestViewerImage(image,'/api/reconstruction-tilt-plane?'+new URLSearchParams({project_id:state.project.id,series_id:state.selected,index,revision:refined.revision,...(state.ctfView?{mode:state.ctfView}:{})}));
    const preview=state.reconstructionStatus?.landmarks;
    if(preview){
      const landmarkIndex=matchingReconstructionTilt(preview.tilt_numbers.map(n=>n-1),preview.angles,refined.source_indices,refined.angles,index);
      prefetchReconstructionImage(reconstructionLandmarkURL(landmarkIndex));
    }
    const layer=$('ctf-view-image');if(layer)layer.hidden=true;
    $('ctf-view-note')?.remove();
    if(state.ctfView){const note=document.createElement('div');note.id='ctf-view-note';note.textContent=state.ctfView==='landscape'?'Final refined CTF · pink: lower defocus · blue: higher defocus · common scale across tilts':'Final refined CTF model (left) · measured spectrum (right) · raw Fourier coordinates';$('tilt-image-area').append(note);}
    $('tilt-image-area').classList.toggle('show-axis',!!state.axisVisible);
    $('volume-title').textContent=selectedSeries().name+' · Refined tilt series';$('volume-empty').hidden=true;
    return;
  }
  const job=state.reconstructionTSStatus;if(!job?.angles?.length)return;
  state.preprocessStatus=job;
  const options=reconstructionAdvancedSelection();
  let rotation=Number(state.project.orientation?.rotation_deg ?? 0);
  if(options.enabled && options.flip_tomogram_z)rotation=(rotation+360)%360-180;
  state.orientationPreview=rotation;state.orientationFlipY=!!state.project.orientation?.flip_y;
  state.orientationHandPreview=options.enabled?options.defocus_handedness:Number(state.project.orientation?.df_hand ?? 1);
  state.orientationCTFAxisPreview=state.orientationFlipY?-rotation:rotation;
  state.reconstructionCTFPreview=options.enabled;
  state.orientationHandUnsaved=(job.ctf_local_geometry?.handedness ?? job.options?.df_hand)!==state.orientationHandPreview;
  const slider=$('recon-ts-slider');slider.disabled=false;slider.max=job.angles.length-1;
  if(!slider.dataset.initialized){slider.value=job.angles.reduce((best,a,i)=>Math.abs(a)<Math.abs(job.angles[best])?i:best,0);slider.dataset.initialized='1';}
  $('tilt-slider').max=slider.max;$('tilt-slider').value=slider.value;showTiltPlane();
  $('recon-ts-angle').textContent=$('tilt-angle').textContent;
  highlightReconstructionView(Number(slider.value));
  $('volume-title').textContent=selectedSeries().name+' · Preprocessed tilt series';
  $('volume-empty').hidden=true;
}

function reconstructionPanel(row) {
  const selected=state.project.reconstruction_denoisers ?? (reconstructionAssembly()?.mode==='advanced'?['tinycare']:[]);
  const alignment=state.project.reconstruction_alignment_options || {smoothness_strength:0,rescue_tilts:true};
  return `<section class="reconstruct-step"><div class="reconstruction-root-actions"></div><div class="preprocessing-layout"><div class="reconstruction-viewer-column"><section class="panel tilt-viewer"><div class="panel-header"><span id="volume-title">${esc(row.name)} · Tomogram</span><span id="recon-status" class="badge pending">Checking output…</span></div><div class="tilt-image-area tomogram-area"><div id="volume-empty" class="viewer-empty">The tomogram will appear here after reconstruction.</div><img id="volume-image" hidden alt="Reconstructed tomogram slice"><div id="recon-ts-viewer" hidden><div id="tilt-image-area" class="tilt-image-area"><img id="tilt-plane-image" alt="Preprocessed tilt series"><div id="recon-ts-controls" class="orientation-controls"></div></div><input id="tilt-slider" type="range" hidden><span id="tilt-angle" hidden></span></div><div class="volume-variants volume-landmarks"><button id="recon-ts" type="button" class="button secondary" aria-pressed="false" disabled hidden>Tomogram</button><button type="button" class="button secondary" data-volume-method="landmarks" aria-pressed="false" title="Landmark projections from the latest available alignment pass" hidden>Landmarks</button></div><div class="volume-variants"><button type="button" class="button secondary" data-volume-method="tinycare" aria-pressed="false" title="Click again to return to the original volume" disabled>TinyCARE</button></div></div><div id="recon-ts-scroll" class="tilt-scroll" hidden><input id="recon-ts-slider" type="range" min="0" max="0" value="0" disabled aria-label="Tilt angle"><span id="recon-ts-angle"></span></div><div id="recon-view-grid" class="curation-strip-scroll" hidden><div class="curation-tilts" role="group" aria-label="Included reconstruction views"></div></div><div class="tilt-scroll"><select id="volume-axis" aria-label="Slice axis"><option value="z">Z</option><option value="y">Y</option><option value="x">X</option></select><input id="volume-slider" type="range" min="0" max="0" value="0" disabled aria-label="Tomogram slice"><span id="volume-slice">No tomogram</span></div><div id="recon-job-info" class="hint viewer-info"></div></section>${reconstructionAssembly()?.mode==='advanced'?'<details id="recon-angle-table" class="panel"><summary class="panel-header">Refined angles</summary><p class="hint">Available after geometry refinement.</p></details>'+landmarkCloudPanel():''}</div><form id="reconstruction-form" class="panel processing-parameters"><div id="recon-parameter-title" class="panel-header">Greco Optimizer</div><div class="reconstruction-details"><label class="metadata-field"><span>Tomogram thickness (%)</span><input id="recon-thickness" type="number" min="30" max="100" step="1" required value="${esc(state.project.reconstruction_thickness_percent ?? 50)}"></label><label class="metadata-field"><span>Final reconstruction binning</span><select id="recon-final-binning">${[1,2,4,8,16].map(bin=>`<option value="${bin}" ${bin===(state.project.reconstruction_final_binning ?? row.preprocessing?.output_binning ?? row.preprocessing?.options?.output_binning ?? 8)?'selected':''}>Bin ${bin}</option>`).join('')}</select></label><label class="metadata-field"><span>Smoothness</span><input id="recon-smoothness" type="number" min="0" max="5" step="0.05" required value="${esc(alignment.smoothness_strength)}"></label><label class="check-line"><span>Rescue tilts (optional)</span><input id="recon-rescue" type="checkbox" ${alignment.rescue_tilts?'checked':''}></label><p id="recon-normal-help" class="hint">Greco Optimizer aligns the tilt series.<br>Smoothness penalizes abrupt tilt-to-tilt shifts.<br>Rescue retries difficult tilts with a wider search.</p></div>${reconstructionAdvancedPanel()}${reconstructionFineScalePanel()}<div class="panel-header">Final tomogram denoising</div><div class="parameter-section"><div class="metadata-grid">${[['tinycare','TinyCARE']].map(([key,label])=>`<label class="check-line"><span>${label}</span><input type="checkbox" data-recon-denoiser="${key}" ${selected.includes(key)?'checked':''}></label>`).join('')}</div></div><p id="recon-error" class="form-error viewer-info" role="alert"></p><div class="submit-actions"><button type="submit" class="button primary" data-submit-recon="one">Submit</button><button type="submit" class="button secondary" data-submit-recon="all">Submit all</button></div></form></div>${processingStepPanels(row)}</section>`;
}
async function pollReconstruction(projectId,seriesId) {
  const reconstructionId=state.project?.active_reconstruction_assembly;
  try {
    const job=await api('/api/reconstruction-status?'+new URLSearchParams({project_id:projectId,series_id:seriesId}));
    if(state.project?.id!==projectId||state.selected!==seriesId||state.stage!=='reconstruction'||state.project?.active_reconstruction_assembly!==reconstructionId||!$('volume-image'))return;
    job.volumes=Object.fromEntries(Object.entries(job.volumes || {}).filter(([key])=>['original','tinycare'].includes(key)));
    state.reconstructionStatus=job;selectedSeries().reconstruction=job;renderStages();
    renderReconstructionViewGrid();
    if($('recon-status')){
      $('recon-status').textContent=reconstructionStateLabel(job);
      $('recon-status').className='badge '+(job.state==='completed'?'':'pending');
      bindPreprocessingActions(job,'reconstruction','recon-status');
    }
    $('recon-job-info').textContent=[job.job_id?'Slurm job '+job.job_id:'','Greco Optimizer',job.alignment_options?`Smoothness ${job.alignment_options.smoothness_strength} · Rescue ${job.alignment_options.rescue_tilts?'±16 px':'off'}`:'',job.active_count?`${job.active_count}/${job.total_tilts} tilts`:'',job.final_binning?`Final bin ${job.final_binning}`:'',job.work || '',job.error || job.status_error || '',...Object.entries(job.denoising_errors || {}).map(([m,e])=>m+': '+e)].filter(Boolean).join(' · ');
    if(job.warning){
      const warning=document.createElement('div');warning.className='reconstruction-warning';
      const label=document.createElement('strong');label.textContent='Warning: ';
      warning.append(label,document.createTextNode(job.warning));$('recon-job-info').append(warning);
    }
    renderReconstructionAngles(job);
    loadReconstructionLandmarks(job);
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
  if(method==='landmarks' && $('recon-ts')?.checked){
    const preview=state.reconstructionStatus?.landmarks;
    const refined=state.reconstructionStatus?.tilt_series;
    const tsAngles=refined?.angles || state.reconstructionTSStatus?.angles || [];
    const tsSources=refined?.source_indices || tsAngles.map((_,i)=>i);
    const landmarkSources=(preview?.tilt_numbers || []).map(n=>n-1);
    const slider=$('recon-ts-slider');
    if(state.volumeMethod==='landmarks'){
      slider.value=matchingReconstructionTilt(tsSources,tsAngles,landmarkSources,preview?.angles || [],Number(state.volumeIndex ?? 0));
      state.volumeMethod='original';
    }else{
      state.volumeIndex=matchingReconstructionTilt(landmarkSources,preview?.angles || [],tsSources,tsAngles,Number(slider.value));
      state.volumeMethod='landmarks';
    }
    state.volumeMethodChosen=true;showReconstructionTS();return;
  }
  const fallback=method==='landmarks' && state.reconstructionStatus?.volumes?.tinycare?'tinycare':'original';
  const next=state.volumeMethod===method?fallback:method;
  state.volumeMethodChosen=true;
  if((next==='landmarks')!==(state.volumeMethod==='landmarks'))state.volumeIndex=null;
  state.volumeMethod=next;
  showVolumePlane();
}
function showVolumePlane() {
  if($('recon-ts')?.checked && state.volumeMethod!=='landmarks'){showReconstructionTS();return;}
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
  renderReconstructionViewGrid();
  if(landmarks)highlightReconstructionView(preview.tilt_numbers[index]-1);
  slider.setAttribute('aria-label',landmarks?'Landmark tilt':'Tomogram slice');
  $('volume-slice').textContent=landmarks?`${index+1} / ${size} · ${angle(preview.angles[index])}`:`${axis.toUpperCase()} · ${index+1} / ${size}`;
  $('volume-slice').title=landmarks?`Tilt ${preview.tilt_numbers[index]} · ${preview.stage}`:'';
  const availableVolumes=Object.values(job.volumes || {}).filter(Boolean).length;
  document.querySelectorAll('[data-volume-method]').forEach(b=>{
    const method=b.dataset.volumeMethod;b.disabled=!available(method);
    b.hidden=b.disabled || (method==='landmarks'?!$('recon-ts')?.checked:!!$('recon-ts')?.checked || availableVolumes<2);
    b.classList.toggle('active',method===state.volumeMethod);b.setAttribute('aria-pressed',String(method===state.volumeMethod));
  });
  if($('volume-title'))$('volume-title').textContent=selectedSeries().name+' · '+(landmarks?'Landmarks':'Tomogram');
  $('volume-empty').hidden=true;const image=$('volume-image');image.hidden=false;
  image.alt=landmarks?'Landmark projection at the selected tilt':'Reconstructed tomogram slice';
  image.dataset.reconstructionViewKey=reconstructionViewKey(state.project.id,state.selected,job);
  applyVolumeContrast();
  requestViewerImage(image,'/api/volume-plane?'+new URLSearchParams({project_id:state.project.id,series_id:state.selected,index,axis,method:state.volumeMethod,revision:JSON.stringify([job.work,job.job_id,landmarks?[preview.path,preview.revision]:job.volumes?.[state.volumeMethod]]),autocontrast:state.volumeAutoContrast?'1':'0',contrast:state.volumeContrast ?? 1,brightness:state.volumeBrightness ?? 0}));
  if(landmarks && job.tilt_series){
    const refined=job.tilt_series;
    const tsIndex=matchingReconstructionTilt(refined.source_indices,refined.angles,preview.tilt_numbers.map(n=>n-1),preview.angles,index);
    prefetchReconstructionImage('/api/reconstruction-tilt-plane?'+new URLSearchParams({project_id:state.project.id,series_id:state.selected,index:tsIndex,revision:refined.revision}));
  }
  image.onerror=()=>{const error=$('recon-error') || $('recon-job-info');if(error)error.textContent=landmarks?'Could not load this landmark tilt.':'Could not load this tomogram slice.';};
}

async function submitReconstruction(event) {
  if(state.reconstructionViewsSaving){event.preventDefault();notify('Wait for the view selection to be saved.');return;}
  event.preventDefault();
  const all=event.submitter.dataset.submitRecon==='all';
  const ids=all?submittableSeries('reconstruction'):[state.selected];
  if(all&&!confirm(`Submit Greco Optimizer reconstruction for ${ids.length} tilt series? Preprocessing must be completed or submitted to Slurm.`))return;
  const denoisers=Array.from(document.querySelectorAll('[data-recon-denoiser]:checked')).map(input=>input.dataset.reconDenoiser);
  const buttons=Array.from(event.currentTarget.querySelectorAll('button'));buttons.forEach(b=>b.disabled=true);$('recon-error').textContent='';
  try {
    const optimizer='greco_optimizer';
    const advanced=reconstructionAdvancedSelection();
    const smoothness=Number($('recon-smoothness').value),rescue=advanced.enabled || $('recon-rescue').checked;
    const entrySigma=Number(state.project.reconstruction_entry_sigma_px ?? 8);
    const thickness=Number($('recon-thickness').value);
    if(!Number.isFinite(thickness) || thickness<30 || thickness>100)throw new Error('Tomogram thickness must be between 30% and 100%.');
    {
      let capabilities;
      try {capabilities=await api('/api/reconstruction-capabilities');}
      catch(error) {throw new Error('Restart the Greco web server and reload this page to enable the Greco Optimizer. No job was submitted.');}
      if(advanced.enabled && !capabilities.advanced_reconstruction)throw new Error('Restart the Greco web server to enable Advanced reconstruction.');
      if(advanced.refine_gamma && !capabilities.experimental_gamma_refinement)throw new Error('Restart the Greco web server and reload this page to enable experimental γ refinement.');
      if(!capabilities.final_binning || !capabilities.integrated_smoothness || capabilities.optimizer!==optimizer || capabilities.optimizer_version!==1)
        throw new Error('The server does not support the Greco Optimizer. Restart it and reload this page. No job was submitted.');
    }
    const result=await api('/api/submit-reconstruction',{project_id:state.project.id,series_ids:ids,denoisers,optimizer,advanced_options:advanced,entry_search_sigma_px:entrySigma,tomogram_thickness_percent:thickness,final_binning:Number($('recon-final-binning').value),smoothness_strength:smoothness,rescue_tilts:rescue,submit_all:all});
    saveProject(result);notify(result.submission_errors?.length?result.submission_errors.join(' · '):`${'Submitted'} ${ids.length} Greco Optimizer job(s).`);
  } catch(error){if($('recon-error'))$('recon-error').textContent=error.message;}
  finally{buttons.forEach(b=>b.disabled=false);}
}

function reconstructionAssembly(){return (state.project.reconstruction_assemblies || []).find(r=>r.id===state.project.active_reconstruction_assembly);}
function mountReconstructionAssemblies(){
  const project=state.project,heading=$('workspace').querySelector('.page-heading');
  const create=document.createElement('button');create.className='button primary';create.textContent='+ New reconstruction';heading.append(create);
  create.onclick=()=>{
    const dialog=document.createElement('dialog');dialog.className='assembly-job-dialog reconstruction-create-dialog';
    dialog.innerHTML=`<form><h3>New reconstruction</h3><label class="metadata-field"><span>Name</span><input name="name" required maxlength="100" value="Reconstruction ${String((project.reconstruction_assemblies || []).length+1).padStart(3,'0')}"></label><label class="metadata-field"><span>Preprocessing</span><select name="preprocessing" required>${(project.preprocessing_assemblies || []).map(p=>`<option value="${esc(p.id)}" ${p.id===project.active_preprocessing_assembly?'selected':''}>${esc(p.name)}</option>`).join('')}</select></label><fieldset class="reconstruction-mode-choice"><legend>Mode</legend>${[['standard','Simple','Aligns image shifts with a fixed tilt axis angle β. Set the correct axis angle in preprocessing.'],['advanced','Advanced','Angular refinement and final CTF re-estimation. Tolerates an imprecise initial β and allows defocus handedness to be adjusted after preprocessing.']].map(([value,label,description])=>`<label><input type="radio" name="mode" value="${value}" ${value==='standard'?'checked':''}><span><strong>${label}</strong><small>${description}</small></span></label>`).join('')}</fieldset><p class="form-error"></p><div class="submit-actions"><button type="button" class="button secondary">Cancel</button><button type="submit" class="button primary">Accept</button></div></form>`;
    document.body.append(dialog);dialog.onclose=()=>dialog.remove();dialog.querySelector('[type=button]').onclick=()=>dialog.close();
    dialog.querySelector('form').onsubmit=async event=>{
      event.preventDefault();event.submitter.disabled=true;
      try{const result=await api('/api/reconstruction-assemblies',{project_id:project.id,name:dialog.querySelector('[name=name]').value,preprocessing_id:dialog.querySelector('[name=preprocessing]').value,mode:dialog.querySelector('[name=mode]:checked').value});dialog.close();if(state.project?.id===project.id)saveProject(result);}
      catch(error){dialog.querySelector('.form-error').textContent=error.message;event.submitter.disabled=false;}
    };dialog.showModal();
  };
  const list=document.createElement('div');list.className='pick-assemblies';
  const content=$('workspace').querySelector('.reconstruct-step');
  for(const record of project.reconstruction_assemblies || []){
    const panel=document.createElement('details');panel.className='pick-assembly';panel.dataset.reconstructionId=record.id;panel.innerHTML=`<summary><strong>${esc(record.name)}</strong>${record.mode==='advanced'?'<span class="reconstruction-mode-badge active">Advanced</span>':''}<button type="button" class="button secondary assembly-delete-button" data-delete-reconstruction="${esc(record.id)}">Delete</button></summary>`;
    panel.querySelector('[data-delete-reconstruction]').onclick=event=>{event.preventDefault();event.stopPropagation();confirmReconstructionAssemblyDeletion(record);};
    const source=document.createElement('span');source.className='hint reconstruction-preprocessing-source';
    source.textContent=(project.preprocessing_assemblies || []).find(p=>p.id===record.preprocessing_assembly_id)?.name || 'Preprocessing';
    panel.querySelector('summary').insertBefore(source,panel.querySelector('[data-delete-reconstruction]'));
    if(record.id===project.active_reconstruction_assembly && content){panel.open=state.reconstructionCollapsed!==project.id+':'+record.id;const body=document.createElement('div');body.className='assembly-body';body.append(content);panel.append(body);}
    panel.addEventListener('toggle',async()=>{if(!panel.isConnected)return;if(record.id===state.project.active_reconstruction_assembly){state.reconstructionCollapsed=panel.open?null:project.id+':'+record.id;return;}if(!panel.open)return;try{const result=await api('/api/select-reconstruction-assembly',{project_id:project.id,assembly_id:record.id});if(state.project?.id===project.id)saveProject(result);}catch(error){panel.open=false;notify(error.message);}});
    list.append(panel);
  }
  heading.after(list);
}

function confirmReconstructionAssemblyDeletion(record){
  const projectId=state.project.id,dialog=document.createElement('dialog');dialog.className='assembly-job-dialog';
  dialog.innerHTML=`<form><h3>Delete reconstruction</h3><p>${esc(record.name)}</p><p>Delete this reconstruction for all tilt series, including its geometry refinements and manual curation steps.</p><label class="check-line"><span>I confirm permanent deletion</span><input type="checkbox" required></label><p class="form-error" role="alert"></p><div class="submit-actions"><button type="button" class="button secondary">Cancel</button><button type="submit" class="button primary" disabled>Delete</button></div></form>`;
  document.body.append(dialog);dialog.onclose=()=>dialog.remove();
  const check=dialog.querySelector('input'),submit=dialog.querySelector('[type=submit]'),cancel=dialog.querySelector('[type=button]');
  check.onchange=()=>submit.disabled=!check.checked;cancel.onclick=()=>dialog.close();
  dialog.oncancel=event=>{if(dialog.deleting)event.preventDefault();};
  dialog.querySelector('form').onsubmit=async event=>{
    event.preventDefault();if(!check.checked || dialog.deleting)return;
    dialog.deleting=true;check.disabled=true;submit.disabled=true;submit.textContent='Checking…';cancel.disabled=true;
    try{const result=await api('/api/reconstruction-assembly-delete',{project_id:projectId,assembly_id:record.id,confirmed:true});dialog.close();if(state.project?.id===projectId)saveProject(result);}
    catch(error){dialog.querySelector('.form-error').textContent=error.message;}
    finally{dialog.deleting=false;check.disabled=false;cancel.disabled=false;submit.disabled=!check.checked;submit.textContent='Delete';}
  };dialog.showModal();
}

function renderReconstructionAngles(job){
  const panel=$('recon-angle-table');if(!panel)return;
  const rows=(job.refined_angles || []).filter(row=>row && Number.isFinite(row.alpha_nominal) && Number.isFinite(row.alpha_refined) && Number.isFinite(row.beta_refined));
  const showGamma=rows.some(row=>Number.isFinite(row.gamma_refined));
  panel.innerHTML='<summary class="panel-header"><span>Refined angles</span><span class="reconstruction-angle-count">'+(rows.length?rows.filter(row=>row.included).length+' / '+rows.length+' tilts':'')+'</span></summary>'+(rows.length?
    '<div class="reconstruction-angle-grid"><table class="reconstruction-angle-table"><thead><tr><th scope="col">Tilt</th><th scope="col"><strong>α</strong> nominal (°)</th><th scope="col"><strong>α</strong> refined (°)</th><th scope="col"><strong>β</strong> refined (°)</th>'+(showGamma?'<th scope="col"><strong>γ</strong> refined (°)</th>':'')+'</tr></thead><tbody>'+rows.map(row=>`<tr${row.included?'':' class="angle-excluded"'}><td>${row.tilt}${row.included?'':'<span class="angle-excluded-label">Excluded</span>'}</td><td>${row.alpha_nominal.toFixed(2)}</td><td>${row.included?row.alpha_refined.toFixed(2):'—'}</td><td>${row.included?row.beta_refined.toFixed(2):'—'}</td>${showGamma?`<td>${row.included && Number.isFinite(row.gamma_refined)?row.gamma_refined.toFixed(3):'—'}</td>`:''}</tr>`).join('')+'</tbody></table></div>':
    '<p class="hint">Available after geometry refinement.</p>');
}

function matchingReconstructionTilt(targetSources,targetAngles,sourceSources,sourceAngles,index){
  const source=sourceSources[index];
  const exact=source==null?-1:targetSources.indexOf(source);
  if(exact>=0)return exact;
  const angle=sourceAngles[index];
  if(!Number.isFinite(angle) || !targetAngles.length)return 0;
  return targetAngles.reduce((best,a,i)=>Math.abs(a-angle)<Math.abs(targetAngles[best]-angle)?i:best,0);
}

function reconstructionLandmarkURL(index){
  const job=state.reconstructionStatus,preview=job.landmarks;
  return '/api/volume-plane?'+new URLSearchParams({project_id:state.project.id,series_id:state.selected,index,axis:'z',method:'landmarks',revision:JSON.stringify([job.work,job.job_id,[preview.path,preview.revision]]),autocontrast:state.volumeAutoContrast?'1':'0',contrast:state.volumeContrast ?? 1,brightness:state.volumeBrightness ?? 0});
}

function prefetchReconstructionImage(url){
  clearTimeout(state.reconstructionPrefetchTimer);
  const target=url.startsWith('/api/reconstruction-tilt-plane?')?$('tilt-plane-image'):$('volume-image');
  state.reconstructionPrefetchTimer=setTimeout(()=>{
    if(state.stage==='reconstruction' && $('recon-ts')?.checked && target?.isConnected)requestViewerImage(target,url);
  },120);
}

function reconstructionViewsEditable(){
  const row=selectedSeries();
  return reconstructionAssembly()?.mode==='advanced' && !row?.reconstruction_views_locked && !row?.reconstruction?.work && !row?.reconstruction?.job_id && !row?.queued_jobs?.reconstruction;
}
function renderReconstructionViewGrid(){
  const holder=$('recon-view-grid'),toggle=$('recon-ts'),row=selectedSeries();
  if(!holder || !toggle || !row)return;
  const editable=reconstructionViewsEditable(),angles=state.reconstructionTSStatus?.angles || [];
  toggle.textContent=editable?'Exclude views':toggle.checked?'Tilt series':'Tomogram';
  holder.hidden=!toggle.checked || !angles.length;
  const sliderRow=state.volumeMethod==='landmarks'?$('volume-slider')?.closest('.tilt-scroll'):$('recon-ts-scroll');
  if(sliderRow && sliderRow.nextElementSibling!==holder)sliderRow.after(holder);
  const grid=holder.firstElementChild,excluded=new Set(row.excluded_reconstruction_views || []);
  const active=state.reconstructionStatus?.tilt_series?.source_indices;
  const key=JSON.stringify([state.project.active_reconstruction_assembly,row.id,angles,editable,[...excluded],active]);
  if(grid.dataset.viewsKey!==key){
    grid.dataset.viewsKey=key;grid.style.setProperty('--curation-tilt-count',angles.length);grid.replaceChildren();
    angles.forEach((value,index)=>{
      const included=!excluded.has(index) && (!active || active.includes(index));
      const button=document.createElement('button');button.type='button';button.className='curation-tilt '+(included?'green':'gray');button.textContent=String(index+1);button.dataset.viewIndex=index;
      button.title='Tilt '+(index+1)+' · '+angle(value)+' · '+(included?'Included':'Excluded')+(editable?' · Double-click to toggle':'');button.setAttribute('aria-pressed',String(included));
      button.onclick=()=>{
        if(state.volumeMethod==='landmarks'){
          const target=(state.reconstructionStatus?.landmarks?.tilt_numbers || []).indexOf(index+1);
          if(target<0)return;
          state.volumeIndex=target;showVolumePlane();return;
        }
        const refined=state.reconstructionStatus?.tilt_series;
        const target=refined?.source_indices?refined.source_indices.indexOf(index):index;
        if(target<0)return;
        $('recon-ts-slider').value=target;showReconstructionTS();
      };
      button.ondblclick=async event=>{
        event.preventDefault();
        if(!reconstructionViewsEditable())return;
        const pid=state.project.id,sid=row.id,rid=state.project.active_reconstruction_assembly;const next=new Set(row.excluded_reconstruction_views || []);if(next.has(index))next.delete(index);else next.add(index);
        state.reconstructionViewsSaving=true;grid.querySelectorAll('button').forEach(b=>b.disabled=true);document.querySelectorAll('[data-submit-recon]').forEach(b=>b.disabled=true);
        try{const result=await api('/api/reconstruction-views',{project_id:pid,series_id:sid,excluded_views:[...next].sort((a,b)=>a-b)});if(state.project?.id===pid && state.selected===sid && state.project.active_reconstruction_assembly===rid)row.excluded_reconstruction_views=result.excluded_views;}
        catch(error){notify(error.message);}
        finally{state.reconstructionViewsSaving=false;if(state.project?.id===pid && state.selected===sid && state.project.active_reconstruction_assembly===rid){grid.dataset.viewsKey='';renderReconstructionViewGrid();document.querySelectorAll('[data-submit-recon]').forEach(b=>b.disabled=false);}}
      };grid.append(button);
    });
  }
}

function highlightReconstructionView(sourceIndex){
  const grid=$('recon-view-grid');if(!grid)return;
  grid.querySelectorAll('[data-view-index]').forEach(button=>{
    const active=Number(button.dataset.viewIndex)===sourceIndex;
    button.classList.toggle('active',active);
    if(active)button.setAttribute('aria-current','true');else button.removeAttribute('aria-current');
  });
}
