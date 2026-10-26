"use strict";
let manualResizeInstalled=false;

function manualPipelineSource(row,step){
  if(step?.job)return 'curation_step_'+step.id;
  const prior=step?precedingGeometry(row,step):null;
  return !prior?'reconstruction':prior.kind==='manual_curation'?'curation_step_'+prior.id:prior.id==='geometry'?'geometry':'geometry_step_'+prior.id;
}
function manualStorageKey(){return ['greco-manual-priors',state.project.id,state.selected,state.activeCurationStep].join(':');}
function manualDrafts(){return state.manualDrafts || [];}
function manualDisabled(){return state.manualDisabled || [];}
function manualSelectionPending(){return manualDrafts().length>0 || manualDisabled().length>0;}
function manualRowActive(row){return !!row && (manualDrafts().some(edit=>edit.index===row.index) || (!row.discarded && !manualDisabled().includes(row.source_index)));}
function toggleManualTilt(row){
  if(state.manualSubmitting || ['submitted','running','unknown'].includes(state.manualJob?.state))return;
  resetManualOverlay();
  if(manualRowActive(row)){
    state.manualDrafts=manualDrafts().filter(edit=>edit.index!==row.index).map(edit=>edit.reference===row.index?{...edit,reference:null}:edit);
    if(!row.discarded)state.manualDisabled=[...new Set([...manualDisabled(),row.source_index])];
  }else if(!row.discarded)state.manualDisabled=manualDisabled().filter(t=>t!==row.source_index);
  else{
    const refs=(state.curationStatus?.rows || []).filter(other=>!other.discarded && manualRowActive(other));
    const reference=refs.sort((a,b)=>Math.abs(a.angle_deg-row.angle_deg)-Math.abs(b.angle_deg-row.angle_deg))[0];
    state.manualDrafts=[...manualDrafts(),{index:row.index,reference:reference?.index ?? null,shift_yx:[0,0]}];
  }
  state.manualDraftIdentity=state.curationStatus.input_identity;persistManualDrafts();renderManualDrafts();showCurationTilt();
}

function manualTiltNumber(index){
  if(index==null)return '—';
  return [...(state.curationStatus?.rows || []),...(state.curationStatus?.discarded_rows || [])].find(row=>row.index===index)?.number ?? index+1;
}
function persistManualDrafts(){
  localStorage.setItem(manualStorageKey(),JSON.stringify({input_identity:manualSelectionPending()?(state.manualDraftIdentity || state.curationStatus?.input_identity):null,edits:manualDrafts(),disabled_tilts:manualDisabled(),show_discarded:!!$('manual-show-discarded')?.checked,smoothness_strength:Number($('manual-smoothness')?.value ?? 0),gaussian_search_prior:!!$('manual-gaussian')?.checked,gaussian_sigma_px:Number($('manual-gaussian-sigma')?.value || 8)}));
}
function manualImageScale(){
  const image=$('curation-image'),rect=image.getBoundingClientRect();
  return Math.max(1e-6,Math.min(rect.width/Math.max(1,image.naturalWidth),rect.height/Math.max(1,image.naturalHeight)));
}
function resetManualOverlay(){
  state.manualEditing=null;
  $('curation-image-area').classList.remove('manual-overlay');
  $('manual-reference-image').hidden=true;$('manual-fix').disabled=true;$('manual-cancel-placement').hidden=true;
  $('manual-reference-image').style.transform='';
  $('manual-reference-label').hidden=true;
  $('manual-blend-label').hidden=true;
  paintManualOverlay();
}
function colorManualMovingLayer(){
  const image=$('curation-image'),canvas=$('manual-moving-layer');
  if(!image.complete || !image.naturalWidth){canvas.hidden=true;return;}
  if(canvas.dataset.imageKey!==image.dataset.imageKey){
    canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
    const context=canvas.getContext('2d');context.clearRect(0,0,canvas.width,canvas.height);context.drawImage(image,0,0);
    canvas.dataset.imageKey=image.dataset.imageKey;
  }
  canvas.hidden=false;
}
function paintManualOverlay(){
  const edit=state.manualEditing;
  const pose=edit || manualDrafts().find(draft=>draft.index===state.curationStatus?.rows?.[state.curationIndex]?.index);
  const scale=manualImageScale();
  const transform=pose?`translate(${pose.shift_yx[1]*scale}px,${pose.shift_yx[0]*scale}px)`:'';
  $('curation-image').style.transform=transform;
  $('manual-moving-layer').style.transform=transform;
  if(edit){
    colorManualMovingLayer();
    const blend=Math.max(0,Math.min(100,Number($('manual-blend').value || 0)));
    $('manual-moving-layer').style.opacity=String(blend/100);
    $('manual-blend-value').textContent=`${blend}% moving tilt`;
    const reference=manualDrafts().find(draft=>draft.index===edit.reference);
    $('manual-reference-image').style.transform=reference?`translate(${reference.shift_yx[1]*scale}px,${reference.shift_yx[0]*scale}px)`:'';
  }else $('manual-moving-layer').hidden=true;
  const running=state.manualSubmitting || ['submitted','running','unknown'].includes(state.manualJob?.state);
  if($('manual-apply'))$('manual-apply').disabled=!!running || !!edit || !manualSelectionPending();
  if($('manual-submit'))$('manual-submit').disabled=!!running || !!edit || manualSelectionPending() || !state.curationStatus?.rows?.length;
  $('manual-placement-info').textContent=pose?`Tilt ${manualTiltNumber(pose.index)} · X ${pose.shift_yx[1].toFixed(2)} · Y ${pose.shift_yx[0].toFixed(2)} bin8 px · ${edit?'Greyscale blend · 0% reference / 100% moving tilt':'Fixed prior · not refined yet'}`:'';
}
function manualReferenceOptions(target){
  return (state.curationStatus?.rows || []).filter(row=>row.index!==target && !row.discarded && manualRowActive(row));
}
function beginManualPlacement(){
  if(!manualRowActive(state.curationStatus?.rows?.[state.curationIndex])){notify('Double-click the tilt number to activate it first.');return;}
  if(!$('curation-image').complete || !$('curation-image').naturalWidth){notify('Wait for the selected tilt image to load.');return;}
  const target=state.curationStatus.rows[state.curationIndex].index,options=manualReferenceOptions(target);
    if(!options.length){notify('No fixed neighbouring tilt remains.');return;}
  const angles=state.curationStatus.rows;const targetRow=angles.find(row=>row.index===target);
  const preferred=options.filter(row=>row.color==='green' || manualDrafts().some(draft=>draft.index===row.index));
  const nearest=(preferred.length?preferred:options).slice().sort((a,b)=>Math.abs(a.angle_deg-targetRow.angle_deg)-Math.abs(b.angle_deg-targetRow.angle_deg))[0];
  const saved=manualDrafts().find(e=>e.index===target);
  state.manualEditing={index:target,reference:saved && options.some(row=>row.index===saved.reference)?saved.reference:nearest.index,shift_yx:saved?saved.shift_yx.slice():[0,0]};
  $('manual-reference').innerHTML=options.map(row=>`<option value="${row.index}" ${row.index===state.manualEditing.reference?'selected':''}>Tilt ${row.number} · ${angle(row.angle_deg)}${manualDrafts().some(draft=>draft.index===row.index)?' · fixed prior':''}</option>`).join('');
  $('manual-reference-label').hidden=false;$('manual-fix').disabled=false;$('manual-cancel-placement').hidden=false;
  $('manual-blend-label').hidden=false;
  $('curation-image-area').classList.add('manual-overlay');
  updateManualReference();paintManualOverlay();
}
function updateManualReference(){
  const edit=state.manualEditing;if(!edit)return;
  edit.reference=Number($('manual-reference').value);
  const image=$('manual-reference-image');image.hidden=false;
  image.src='/api/curation-plane?'+new URLSearchParams({project_id:state.project.id,series_id:state.selected,source:state.curationSource,index:edit.reference,display:'raw-mc-v1'});
  paintManualOverlay();
}
function renderManualDrafts(){
  if(!$('manual-drafts'))return;
  $('manual-drafts').innerHTML=manualDrafts().map(edit=>`<div class="manual-draft"><strong>Tilt ${manualTiltNumber(edit.index)}</strong><span>X ${edit.shift_yx[1].toFixed(2)} · Y ${edit.shift_yx[0].toFixed(2)} px · ref ${manualTiltNumber(edit.reference)}</span><button type="button" class="button secondary" data-remove-manual="${edit.index}">Remove</button></div>`).join('');
  document.querySelectorAll('[data-remove-manual]').forEach(button=>button.onclick=()=>{state.manualDrafts=manualDrafts().filter(edit=>edit.index!==Number(button.dataset.removeManual));persistManualDrafts();renderManualDrafts();});
  const running=state.manualSubmitting || ['submitted','running','unknown'].includes(state.manualJob?.state);
  if(manualDisabled().length)$('manual-drafts').insertAdjacentHTML('beforeend',`<p class="hint">Pending deactivation: ${manualDisabled().map(t=>t+1).join(', ')}. Double-click a tilt to undo.</p>`);
  document.querySelectorAll('[data-remove-manual]').forEach(button=>button.disabled=!!running);
  $('manual-apply').disabled=running || !manualSelectionPending();
  $('manual-submit').disabled=running || manualSelectionPending() || !state.curationStatus?.rows?.length;
  $('manual-submit').textContent='Submit';
  $('manual-show-discarded').disabled=!!running;
  $('manual-repeat').disabled=!!running || !!state.manualEditing || manualSelectionPending() || state.manualJob?.state!=='completed' || !state.manualJob?.target_reports?.length || !state.curationStatus?.rows?.length;
  $('manual-begin').disabled=running || !state.curationStatus?.rows?.length;
  $('manual-fix').disabled=running || !state.manualEditing;
  document.querySelectorAll('[data-curation-tilt]').forEach(button=>{
    const row=state.curationStatus?.rows?.find(row=>row.index===Number(button.dataset.curationTilt));
    button.classList.toggle('manual-fixed',manualDrafts().some(edit=>edit.index===row?.index));
    button.classList.toggle('tilt-disabled',!manualRowActive(row));
    button.title=`Tilt ${row?.number} · ${manualRowActive(row)?'Active':'Inactive'} · Double-click to toggle`;
  });
  paintManualOverlay();
}
function mountManualRefinement(pid,sid){
  mountTinycareReuse($('manual-tinycare'),$('manual-reuse-tinycare'));
  if(!manualResizeInstalled){
    window.addEventListener('resize',()=>{if($('curation-image') && $('manual-moving-layer'))paintManualOverlay();});
    manualResizeInstalled=true;
  }
  let saved;
  try{saved=JSON.parse(localStorage.getItem(manualStorageKey()) || '{}');}catch{saved={};}
  state.manualDrafts=Array.isArray(saved.edits)?saved.edits:[];state.manualDraftIdentity=saved.input_identity;state.manualDisabled=Array.isArray(saved.disabled_tilts)?saved.disabled_tilts:[];
  $('manual-show-discarded').checked=!!saved.show_discarded;
  if(saved.smoothness_strength!=null)$('manual-smoothness').value=saved.smoothness_strength;
  if(saved.gaussian_search_prior!=null)$('manual-gaussian').checked=saved.gaussian_search_prior;
  if(saved.gaussian_sigma_px!=null)$('manual-gaussian-sigma').value=saved.gaussian_sigma_px;
  const gaussianChanged=()=>{$('manual-gaussian-sigma').disabled=!$('manual-gaussian').checked;persistManualDrafts();};
  $('manual-gaussian-sigma').disabled=!$('manual-gaussian').checked;
  $('manual-gaussian').onchange=gaussianChanged;
  $('manual-gaussian-sigma').oninput=persistManualDrafts;
  $('manual-smoothness').oninput=persistManualDrafts;
  state.manualEditing=null;state.manualJob=processingSteps(selectedSeries()).find(s=>s.id===state.activeCurationStep)?.job || null;state.manualFinishedKey=null;state.manualSubmitting=false;
  $('manual-begin').onclick=beginManualPlacement;
  $('manual-reference').onchange=updateManualReference;
  $('manual-blend').oninput=paintManualOverlay;
  $('manual-cancel-placement').onclick=()=>{resetManualOverlay();$('manual-reference-label').hidden=true;};
  $('manual-fix').onclick=()=>{
    const edit=state.manualEditing;if(!edit)return;
    state.manualDrafts=manualDrafts().filter(e=>e.index!==edit.index).concat({...edit,shift_yx:edit.shift_yx.slice()});
    state.manualDraftIdentity=state.curationStatus.input_identity;
    persistManualDrafts();resetManualOverlay();$('manual-reference-label').hidden=true;renderManualDrafts();
  };
  const submitAction=async(final)=>{
    state.manualSubmitting=true;resetManualOverlay();renderManualDrafts();$('manual-refinement-error').textContent='';
    const stepId=state.activeCurationStep;
    try{
      const result=await api(final?'/api/submit-manual-export':'/api/submit-manual-refinement',{project_id:pid,series_id:sid,step_id:stepId,edits:final?[]:manualDrafts(),source_identity:final?state.curationStatus?.input_identity:state.manualDraftIdentity,options:{disabled_tilts:manualDisabled(),include_discarded:!!$('manual-show-discarded').checked,smoothness_strength:Number($('manual-smoothness').value),gaussian_search_prior:$('manual-gaussian').checked,gaussian_sigma_px:Number($('manual-gaussian-sigma').value || 8),denoisers:$('manual-tinycare').checked?['tinycare']:[],reuse_tinycare_model:$('manual-reuse-tinycare').checked,final_binning:Number($('manual-final-binning').value)}});
      if(state.project?.id!==pid || state.selected!==sid || state.activeCurationStep!==stepId)return;
      const updated=result.series.find(r=>r.id===sid)?.processing_steps?.find(s=>s.id===stepId);
      const current=processingSteps(selectedSeries()).find(s=>s.id===stepId);
      if(updated && current)Object.assign(current,updated);
      state.manualAwaitingWork=updated?.job?.work;
      state.manualJob=updated?.job;
      state.curationSource=manualPipelineSource(selectedSeries(),current);
      resetManualOverlay();pollManualRefinement(pid,sid,stepId);notify(final?'Submitted final reconstruction.':'Submitted fixes for neighbour-column XY refinement.');
    }catch(error){if(state.project?.id===pid && state.selected===sid && state.activeCurationStep===stepId && $('manual-refinement-error'))$('manual-refinement-error').textContent=error.message;}
    finally{if(state.project?.id===pid && state.selected===sid && state.activeCurationStep===stepId){state.manualSubmitting=false;renderManualDrafts();}}
  };
  $('manual-repeat').onclick=()=>{
    const rows=state.curationStatus?.rows || [],reports=state.manualJob?.target_reports || [];
    const targets=new Set(reports.map(report=>report.index));
    const fixed=rows.filter(row=>!row.discarded && manualRowActive(row) && !targets.has(row.source_index ?? row.number-1));
    if(fixed.length<3){$('manual-refinement-error').textContent='At least three fixed reference tilts are required.';return;}
    const drafts=reports.map(report=>{
      const row=rows.find(row=>(row.source_index ?? row.number-1)===report.index);
      if(!row)return null;
      const reference=fixed.reduce((best,next)=>Math.abs(next.angle_deg-row.angle_deg)<Math.abs(best.angle_deg-row.angle_deg)?next:best);
      return {index:row.index,reference:reference.index,shift_yx:[0,0]};
    });
    if(drafts.some(edit=>!edit)){$('manual-refinement-error').textContent='The previous tilt selection does not match the current alignment.';return;}
    state.manualDrafts=drafts;state.manualDraftIdentity=state.curationStatus.input_identity;
    persistManualDrafts();renderManualDrafts();
    notify('Previous tilts selected at their saved positions. Choose Smoothness, then Submit fixes.');
  };
  $('manual-apply').onclick=()=>submitAction(false);
  $('manual-submit').onclick=()=>submitAction(true);
  const area=$('curation-image-area');let drag=null;
  area.addEventListener('pointerdown',event=>{
    if(!state.manualEditing || event.button!==0)return;
    event.preventDefault();drag={x:event.clientX,y:event.clientY,shift:state.manualEditing.shift_yx.slice(),scale:manualImageScale()};area.setPointerCapture(event.pointerId);
  });
  area.addEventListener('pointermove',event=>{
    if(!drag || !state.manualEditing)return;
    state.manualEditing.shift_yx=[drag.shift[0]+(event.clientY-drag.y)/drag.scale,drag.shift[1]+(event.clientX-drag.x)/drag.scale];paintManualOverlay();
  });
  for(const name of ['pointerup','pointercancel','lostpointercapture'])area.addEventListener(name,()=>drag=null);
  $('curation-image').addEventListener('load',paintManualOverlay);
  $('manual-reference-image').addEventListener('load',paintManualOverlay);
  renderManualDrafts();pollManualRefinement(pid,sid,state.activeCurationStep);
}
function updateManualDraftIdentity(){
  if(state.manualDraftIdentity && state.curationStatus?.input_identity && state.manualDraftIdentity!==state.curationStatus.input_identity){
    state.manualDrafts=[];state.manualDisabled=[];state.manualDraftIdentity=null;resetManualOverlay();persistManualDrafts();
    notify('The alignment changed; old manual priors were cleared.');
  }
  renderManualDrafts();
}
async function pollManualRefinement(pid,sid,stepId){
  clearTimeout(state.manualRefinementTimer);
  try{
    const job=await api('/api/manual-refinement-status?'+new URLSearchParams({project_id:pid,series_id:sid,step_id:stepId}));
    if(state.project?.id!==pid || state.selected!==sid || state.activeCurationStep!==stepId || !$('manual-job-status'))return;
    state.manualJob=job;
    const step=processingSteps(selectedSeries()).find(s=>s.id===stepId);
    if(step && job.work)step.job=job;
    $('manual-job-status').textContent=[job.manual_action==='refine'?'Fixes':job.manual_action==='reconstruct'?'Final reconstruction':'',job.work?jobStateLabel(job.state):'No manual refinement submitted',job.job_id?'Slurm '+job.job_id:'',job.progress?`Tilt ${job.progress.tilt_number} · pass ${job.progress.pass_number} · ${job.progress.completed_tilts} / ${job.progress.total_tilts} finished`:'',job.converged===false?'Some tilts reached the pass limit':''].filter(Boolean).join(' · ');
    $('manual-refinement-error').textContent=job.error || '';
    renderManualDrafts();
    if(job.state==='completed' && state.manualFinishedKey!==job.work){
      state.manualFinishedKey=job.work;
      if(state.manualAwaitingWork===job.work){state.manualDrafts=[];state.manualDisabled=[];state.manualDraftIdentity=null;state.manualAwaitingWork=null;persistManualDrafts();}
      state.curationSource=manualPipelineSource(selectedSeries(),step);state.curationStatus=null;resetManualOverlay();pollManualCuration(pid,sid);
    }
    if(['submitted','running','unknown'].includes(job.state))state.manualRefinementTimer=setTimeout(()=>pollManualRefinement(pid,sid,stepId),5000);
  }catch(error){if($('manual-refinement-error'))$('manual-refinement-error').textContent=error.message.includes('Unknown route')?'Restart the Greco server to enable manual XY refinement.':error.message;}
}
