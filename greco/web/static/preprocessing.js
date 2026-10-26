"use strict";

const mcFields=[['output_bin1_sta','Output bin 1 for STA','Save the unweighted native MC sum for particle extraction; uses additional disk space',true],['mc_b_factor','B-factor','correlation px²',500],['mc_iter','Iterations','',15],['mc_tol','Tolerance','native px',.1],['mc_search_radius','Search radius','correlation px',64],['eer_group','EER group','raw subframes / frame',15],['eer_upscaling','EER upscaling','1× native / 2× superresolution',1],['eer_threads','EER decoder threads','per worker',8],['gpus','GPUs','per job',1],['cpus','CPU cores','per job',8],['flip_y','Flip Y','',false],['transpose','Transpose','',false]];
const ctfFields=[['ctf_res_low_A','Low resolution (Å)','CTF fit low-frequency edge; blank = automatic',null],['ctf_res_high_A','High resolution (Å)','CTF fit high-frequency edge; blank = automatic',null],['ctf_local_refine','Local CTF refinement','Fit handedness and alpha/beta offsets from native tile spectra using the detector-frame axis',true],['cs_mm','Cs','mm',2.7],['amp_contrast','Amplitude contrast','fraction',.1],['ctf_tile_size','Estimator tile size','native pixels',512],['ctf_regularization','Regularization','',.05],['ctf_spatial','Spatial CTF correction','defocus gradient',true],['df_hand','Defocus handedness','−1 or +1',1]];
function defocusHandExample() {
  // Fixed DISPLAY reference, not a prediction from the selected raw-frame
  // sign. In Greco's zero-offset, unflipped axis=0 frame this is hand=-1.
  return `<span class="defocus-hand-example-sign">Reference pattern · vertical tilt axis</span><div class="defocus-hand-columns"><span>Left</span><span>Right</span></div>`+[-40,0,40].map(tilt=>{
    const leftHigh=tilt<0;
    return `<div class="defocus-hand-row"><span class="defocus-hand-angle">${tilt<0?'−40':tilt>0?'+40':'0'}°</span><span class="defocus-hand-strip ${tilt===0?'equal':leftHigh?'high-left':'high-right'}"><span>${tilt===0?'Equal':leftHigh?'High':'Low'}</span><span>${tilt===0?'Equal':leftHigh?'Low':'High'}</span></span></div>`;
  }).join('');
}
function defocusHandHelp() {
  return `<details class="defocus-hand-help" ${storedValue('greco-defocus-help-open')==='1'?'open':''}><summary aria-label="About defocus handedness" title="About defocus handedness">ⓘ Defocus handedness</summary><div class="defocus-hand-popover"><strong>Defocus Handedness</strong><p>Tilting creates a defocus gradient perpendicular to the tilt axis: one side is farther from focus than the other.</p><div class="defocus-hand-diagram" role="group" aria-label="Fixed reference with vertical tilt axis; left and right sides">${defocusHandExample()}</div><p class="defocus-hand-legend"><span class="high">Blue: high</span> · <span class="low">Pink: low</span></p><p>Adjust the defocus handedness so the defocus gradient follows this convention.</p></div></details>`;
}
function preprocessFields(fields) {
  if(fields===ctfFields)return preprocessFields(fields.filter(([key])=>!['cs_mm','amp_contrast'].includes(key)));
  if(fields===mcFields){
    const basic=new Set(['output_bin1_sta','eer_group','eer_upscaling']);
    return preprocessFields(fields.filter(([key])=>basic.has(key)))+
      `<details class="mc-advanced"><summary>Advanced</summary><div class="metadata-grid">${preprocessFields(fields.filter(([key])=>!basic.has(key)))}</div></details>`;
  }
  return fields.map(([key,label,unit,value])=>{
    const current=state.preprocessOptions?.[key] ?? value;
    if(typeof value==='boolean') return `<label class="check-line" title="${esc(unit)}"><span>${label}</span><input type="checkbox" data-preprocess="${key}" ${current?'checked':''}></label>`;
    const choice=key==='eer_upscaling'?[1,2]:key==='df_hand'?[-1,1]:null;
    return `<div class="metadata-field" title="${esc(unit)}"><label for="prep-${key}">${label}</label>${choice ? `<select id="prep-${key}" data-preprocess="${key}" aria-label="${esc(label+' '+unit)}">${choice.map(n=>`<option value="${n}" ${Number(current)===n?'selected':''}>${n}</option>`).join('')}</select>` : `<input id="prep-${key}" data-preprocess="${key}" aria-label="${esc(label+' '+unit)}" type="number" step="any" ${value===null?'placeholder="Auto"':'required'} value="${esc(current)}">`}</div>`;
  }).join('');
}
function renderNext() {
  if(state.stage==='segmentation') {renderSegmentation();return;}
  if(state.stage==='picking') {renderPicking();return;}
  if(state.stage==='sta') {renderSTA();return;}
  if(!['preprocessing','reconstruction'].includes(state.stage)) {
    $('workspace').innerHTML=`<section class="panel"><div class="panel-header">${esc(state.stage==='segmentation'?'Segmentation':state.stage==='picking'?'Picking':'STA')}</div><div class="next-placeholder">Controls for this step will be added next.</div></section>`;
    return;
  }
  const row=selectedSeries();
  clearTimeout(state.curationTimer);
  clearTimeout(state.preprocessTimer);
  clearTimeout(state.reconstructionTimer);
  clearTimeout(state.refinementTimer);
  clearTimeout(state.angleRefinementTimer);
  clearTimeout(state.contrastTimer);
  if(!state.preprocessOptions || state.preprocessOptionsProjectId!==state.project.id) {
    state.preprocessOptionsProjectId=state.project.id;
    state.preprocessOptions={cs_mm:state.project?.metadata.cs_mm ?? 2.7,amp_contrast:state.project?.metadata.amplitude_contrast ?? .1,...(state.project?.preprocessing_options || {})};
    if(state.project?.ctf_axis_protocol!=='ctf-detector-frame-v1' && !state.project?.orientation)state.preprocessOptions.ctf_spatial_axis_deg=null;
    if(state.project.orientation)state.preprocessOptions.df_hand=state.project.orientation.df_hand;
  }
  $('workspace').innerHTML=`<div class="page-heading"><div><span class="eyebrow">${state.stage==='preprocessing'?'02 / PREPROCESSING':'03 / RECONSTRUCTION'}</span><h1>${state.stage==='preprocessing'?'Preprocessing':'Reconstruction'}</h1><p>Select a tilt series in the left sidebar.</p></div></div>${!row ? '<div class="empty-state"><h3>Select a tilt series</h3></div>' : state.stage==='reconstruction' ? reconstructionPanel(row) : `<div class="preprocessing-layout"><section class="panel tilt-viewer"><div class="panel-header"><span>${esc(row.name)} · <span id="viewer-bin-label">bin${row.preprocessing?.output_binning ?? row.preprocessing?.options?.output_binning ?? 8}</span> · 2D aligned</span><span id="prep-status" class="badge pending">Checking output…</span></div><div id="tilt-image-area" class="tilt-image-area"><div class="viewer-empty">The tilt series will appear here after preprocessing completes.</div></div><div class="tilt-scroll"><input id="tilt-slider" type="range" min="0" max="0" value="0" disabled aria-label="Tilt angle"><span id="tilt-angle">No processed tilt series</span></div><div id="prep-job-info" class="hint viewer-info"></div></section><form id="preprocessing-form" class="panel processing-parameters"><div class="panel-header">Motion correction</div><div class="parameter-section"><p class="hint">Native full-frame MC · fixed bin 8 before 2D alignment. EER pixel size refers to native 1× sampling.</p><div class="metadata-grid">${preprocessFields(mcFields)}</div></div><div class="panel-header">CTF estimation / correction</div><div class="parameter-section"><div class="metadata-grid">${preprocessFields(ctfFields)}</div><div id="ctf-orientation-fields"></div><p class="hint">Voltage ${esc(row.metadata.voltage_kv)} kV · native pixel ${esc(row.metadata.pixel_size_A)} Å · dose ${esc(row.metadata.dose_per_tilt)} e⁻ / Å²</p><p class="hint">MC/CTF and 2D alignment run together. CTF is estimated at full resolution before output binning. The viewer shows bin 8 with 2D shifts applied; CTF products are saved separately.</p></div><p id="prep-error" class="form-error viewer-info" role="alert"></p><div id="ctf-estimator-actions" class="ctf-estimator-actions"><button type="button" class="button secondary" id="orientation-hand-test">Test handedness</button><button type="button" class="button secondary" id="orientation-redo-ctf">Redo CTF</button><p id="orientation-hand-result" class="hint" role="status"></p><p id="ctf-redo-result" class="hint" role="status"></p></div><div class="submit-actions"><button type="submit" class="button primary" data-submit-prep="one">Submit</button><button type="submit" class="button secondary" data-submit-prep="all">Submit all</button></div></form></div>`}`;
  if(row && state.stage==='preprocessing') {
    const title=$('workspace').querySelector('.tilt-viewer .panel-header > span');
    title.classList.add('viewer-series-label');
    title.insertAdjacentHTML('afterbegin',`<input class="viewer-series-enabled" type="checkbox" data-series-enabled="${esc(row.id)}" ${row.enabled!==false?'checked':''} aria-label="Enable ${esc(row.name)}" title="Include this tilt series in processing">`);
    $('preprocessing-form').addEventListener('submit',submitPreprocessing);
    $('tilt-slider').addEventListener('input',()=>showTiltPlane());
    mountOrientation(row);
    pollPreprocessing(state.project.id,row.id);
  }
  if(row && state.stage==='reconstruction') {
    state.volumeMethod='original';state.volumeMethodChosen=false;state.volumeIndex=null;state.reconstructionStatus=null;
    if($('reconstruction-form'))$('reconstruction-form').addEventListener('submit',submitReconstruction);
    $('volume-image').insertAdjacentHTML('beforebegin',`<details class="volume-contrast picker-contrast-panel"><summary>Contrast</summary><div class="picker-contrast-controls"><div class="contrast-actions"><button id="volume-autocontrast" type="button" class="button secondary">Auto contrast</button><button id="volume-contrast-reset" type="button" class="button secondary">Reset contrast</button></div><label for="volume-contrast-slider">Contrast <input id="volume-contrast-slider" type="range" min="0.25" max="4" step="0.05" value="${state.volumeContrast ?? 1}" aria-label="Tomogram contrast"><output id="volume-contrast-value"></output></label><label for="volume-brightness-slider">Brightness <input id="volume-brightness-slider" type="range" min="-1" max="1" step="0.02" value="${state.volumeBrightness ?? 0}"><output id="volume-brightness-value"></output></label></div></details>`);
    $('volume-autocontrast').title='Compute a robust intensity window from the current MRC slice';
    $('volume-contrast-slider').addEventListener('input',()=>{state.volumeContrast=Number($('volume-contrast-slider').value);applyVolumeContrast();clearTimeout(state.contrastTimer);state.contrastTimer=setTimeout(showVolumePlane,120);});
    $('volume-brightness-slider').addEventListener('input',()=>{state.volumeBrightness=Number($('volume-brightness-slider').value);applyVolumeContrast();clearTimeout(state.contrastTimer);state.contrastTimer=setTimeout(showVolumePlane,120);});
    const resetContrast=automatic=>{state.volumeAutoContrast=automatic;state.volumeContrast=1;state.volumeBrightness=0;$('volume-contrast-slider').value=1;$('volume-brightness-slider').value=0;applyVolumeContrast();showVolumePlane();};
    $('volume-autocontrast').addEventListener('click',()=>resetContrast(true));
    $('volume-contrast-reset').addEventListener('click',()=>resetContrast(false));
    applyVolumeContrast();
    restoreReconstructionView(row);
    mountReconstructionZoom($('volume-image'));
    bindSliceWheel($('volume-image').parentElement,$('volume-slider'));
    $('volume-slider').addEventListener('input',()=>{state.volumeIndex=Number($('volume-slider').value);showVolumePlane();});
    $('volume-axis').addEventListener('change',()=>{state.volumeIndex=null;showVolumePlane();});
    document.querySelectorAll('[data-volume-method]').forEach(button=>button.addEventListener('click',()=>toggleReconstructionMethod(button.dataset.volumeMethod)));
    pollReconstruction(state.project.id,row.id);
    mountReconstructionSteps(row);
    mountPickingSource(row);
  }
}
function bindPreprocessingActions(job,kind='preprocessing',badgeId='prep-status') {
  const badge=$(badgeId);
  if(!badge || !job.work)return;
  badge.tabIndex=0;badge.setAttribute('role','button');badge.setAttribute('aria-label',kind+' job actions');badge.style.cursor='pointer';
  const toggle=()=>{
    const old=$('preprocessing-action-menu');if(old){old.closeMenu();return;}
    const menu=document.createElement('div');menu.id='preprocessing-action-menu';menu.className='particle-job-action-menu';menu.setAttribute('role','menu');
    const button=document.createElement('button');button.type='button';button.textContent=['submitted','running','submitting','unknown'].includes(job.state)?'Cancel job & delete':'Delete';button.setAttribute('role','menuitem');
    menu.append(button);document.body.appendChild(menu);
    const rect=badge.getBoundingClientRect();menu.style.left=Math.max(8,Math.min(rect.right-menu.offsetWidth,innerWidth-menu.offsetWidth-8))+'px';menu.style.top=(rect.bottom+6)+'px';
    const outside=event=>{if(!menu.contains(event.target) && event.target!==badge)menu.closeMenu();};
    const escape=event=>{if(event.key==='Escape')menu.closeMenu();};
    menu.closeMenu=()=>{menu.remove();document.removeEventListener('pointerdown',outside);document.removeEventListener('keydown',escape);window.removeEventListener('scroll',menu.closeMenu,true);window.removeEventListener('resize',menu.closeMenu);};
    document.addEventListener('pointerdown',outside);document.addEventListener('keydown',escape);window.addEventListener('scroll',menu.closeMenu,true);window.addEventListener('resize',menu.closeMenu);
    button.onclick=()=>{menu.closeMenu();confirmPreprocessingDeletion(job,kind);};button.focus();
  };
  badge.onclick=toggle;badge.onkeydown=event=>{if(event.key==='Enter' || event.key===' '){event.preventDefault();toggle();}};
}
function confirmPreprocessingDeletion(job,kind='preprocessing') {
  let dialog=$('preprocessing-delete-dialog');
  if(dialog?.deleting)return;
  if(!dialog){dialog=document.createElement('dialog');dialog.id='preprocessing-delete-dialog';dialog.className='assembly-job-dialog';document.body.appendChild(dialog);}
  const projectId=state.project.id,seriesId=state.selected;
  const active=['submitted','running','submitting','unknown'].includes(job.state);
  const label={preprocessing:'preprocessing',reconstruction:'reconstruction',geometry_refinement:'geometry refinement'}[kind];
  dialog.innerHTML=`<form><h3>Delete ${label}?</h3><p>${active?'Slurm will be cancelled before deleting files. ':''}The ${label} folder and results will be permanently deleted. ${kind==='preprocessing'?'Redo CTF revisions and Test handedness for this series are included; their running Slurm jobs will be cancelled. Source movies are preserved.':'Source data and other jobs are preserved.'} Dependent reconstruction and other downstream jobs must be deleted first.</p><label><input type="checkbox" required> I confirm permanent deletion</label><p class="form-error" role="alert"></p><div class="submit-actions"><button type="button" class="button secondary">Cancel</button><button type="submit" class="button primary" disabled>${active?'Cancel job & delete':'Delete'}</button></div></form>`;
  const check=dialog.querySelector('input'),submit=dialog.querySelector('[type=submit]');
  check.onchange=()=>{submit.disabled=!check.checked || !!dialog.deleting;};
  dialog.querySelector('[type=button]').onclick=()=>dialog.close();
  dialog.querySelector('form').onsubmit=async event=>{
    event.preventDefault();if(!check.checked || dialog.deleting)return;
    dialog.deleting=true;submit.disabled=true;check.disabled=true;
    try{
      let project;
      do{
        project=await api(kind==='preprocessing'?'/api/preprocessing-delete':'/api/processing-job-delete',{project_id:projectId,series_id:seriesId,work:job.work,confirmed:true,kind});
        if(project.deletion_pending){dialog.querySelector('.form-error').textContent='Waiting for Slurm cancellation before deleting files…';await new Promise(resolve=>setTimeout(resolve,1500));}
      }while(project.deletion_pending);
      dialog.close();
      if(state.project?.id===projectId){clearTimeout(state.preprocessTimer);clearTimeout(state.reconstructionTimer);clearTimeout(state.angleRefinementTimer);saveProject(project);}
      notify(label+' deleted. Source data preserved.');
    }catch(error){dialog.querySelector('.form-error').textContent=error.message;}
    finally{dialog.deleting=false;check.disabled=false;submit.disabled=!check.checked;}
  };
  dialog.showModal();
}
async function pollPreprocessing(projectId,seriesId) {
  try {
    let job=await api('/api/preprocessing-status?'+new URLSearchParams({project_id:projectId,series_id:seriesId}));
    if(state.project?.id!==projectId || state.selected!==seriesId || state.stage!=='preprocessing' || !$('prep-status')) return;
    job=preprocessingState(job,selectedSeries());
    state.preprocessStatus=job;state.refreshCTFButtons?.();
    const selectedHand=Number(document.querySelector('[data-preprocess="df_hand"]')?.value);
    state.orientationHandUnsaved=[-1,1].includes(selectedHand) && (job.ctf_local_geometry?.handedness ?? job.options?.df_hand)!==selectedHand;
    const row=selectedSeries(); row.preprocessing=job;renderStages();
    $('prep-status').textContent=job.state+(job.ctf_redo_active?' · Redo CTF':'');
    $('prep-status').className='badge '+(job.state==='completed'?'':'pending');
    bindPreprocessingActions(job);
    $('prep-job-info').textContent=[job.job_id?'Slurm job '+job.job_id:'',job.final_binning?`Final bin ${job.final_binning}`:'',job.work || '',job.error || job.status_error || ''].filter(Boolean).join(' · ');
    if(job.state==='completed') {
      if(job.tilt_display_version!==2) {
        $('prep-job-info').textContent='Restart the Greco web server, then reload this page to use the corrected tilt/defocus display.';
        return;
      }
      $('viewer-bin-label').textContent='bin'+(job.output_binning ?? job.options?.output_binning ?? 8);
      const area=$('tilt-image-area');
      area.querySelector('.viewer-empty')?.remove();
      if(!$('tilt-plane-image')) {const image=document.createElement('img');image.id='tilt-plane-image';image.alt='Motion-corrected tilt with 2D alignment';area.prepend(image);}
      const slider=$('tilt-slider'); slider.disabled=false; slider.max=job.angles.length-1;
      slider.value=job.angles.reduce((best,value,i)=>Math.abs(value)<Math.abs(job.angles[best])?i:best,0);
      showTiltPlane();
    } else if(['submitted','running','unknown'].includes(job.state)) {
      state.preprocessTimer=setTimeout(()=>pollPreprocessing(projectId,seriesId),8000);
    }
  } catch(error) {
    if(state.project?.id===projectId && state.selected===seriesId && $('prep-status')) {
      $('prep-status').textContent='Status unavailable';
      $('prep-job-info').textContent=error.message;
    }
  }
}
async function submitPreprocessing(event) {
  event.preventDefault();
  const all=event.submitter.dataset.submitPrep==='all';
  const identifiers=all?submittableSeries('preprocessing'):[state.selected];
  if(!identifiers.length){notify('No new jobs to submit. Existing jobs are preserved.');return;}
  if(all && !confirm(`Submit preprocessing for ${identifiers.length} tilt series using these parameters?`)) return;
  const buttons=Array.from(event.currentTarget.querySelectorAll('button'));
  buttons.forEach(button=>button.disabled=true);
  $('prep-error').textContent='';
  try {
    await requireCTFCoordinateProtocol();
    const options=await saveCTFPanelSettings();
    const result=await api('/api/submit-preprocessing',{project_id:state.project.id,series_ids:identifiers,options,submit_all:all});
    saveProject(result);
    if(result.submission_errors.length && $('prep-error')) $('prep-error').textContent=result.submission_errors.join('\n');
    notify(result.submission_errors.length?'Some submissions failed. Review the errors.':all?'Preprocessing submitted to Slurm.':'Preprocessing submitted to Slurm.');
  } catch(error) { if($('prep-error')) $('prep-error').textContent=error.message; }
  finally { buttons.forEach(button=>button.disabled=false); }
}
