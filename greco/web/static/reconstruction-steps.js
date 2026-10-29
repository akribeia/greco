"use strict";

function mountPickingSource(row){
  const reconstructionId=state.project.active_reconstruction_assembly;
  const currentSource=['reconstruction','geometry_refinement','manual_curation'].includes(row.picking_source)?row.picking_source:'reconstruction';
  const card=document.querySelector(`[data-reconstruction-id="${state.project.active_reconstruction_assembly}"]`);
  const locations=[['reconstruction',card?.querySelector('.reconstruction-root-actions')]];
  for(const [kind,target] of locations){
    if(!target)continue;
    if(target.tagName==='SUMMARY')target.closest('details').dataset.pickingSourceFold=kind;
    const available=!!row[kind]?.work;
    target.insertAdjacentHTML('beforeend',`<label class="picking-source-choice" title="Select another output to change the picking source"><input type="checkbox" data-picking-source="${kind}" ${currentSource===kind?'checked':''} ${available?'':'disabled'}>Select reconstruction</label>`);
    target.insertAdjacentHTML('beforeend',`<button type="button" class="button secondary assembly-delete-button" id="reconstruction-delete" ${available?'':'disabled'}>Delete</button>`);
  }
  $('reconstruction-delete')?.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();confirmPreprocessingDeletion(selectedSeries().reconstruction,'reconstruction');});
  document.querySelectorAll('[data-picking-source]').forEach(input=>{
    input.closest('label').addEventListener('click',event=>event.stopPropagation());
    input.addEventListener('change',async()=>{
      if(!input.checked){input.checked=true;return;}
      const pid=state.project.id,sid=row.id;const choice=input.dataset.pickingSource;
      const previousSource=row.picking_source || 'reconstruction',previousStep=row.picking_step_id || 'geometry';
      state.pickSelectionVersion=(state.pickSelectionVersion || 0)+1;
      const stepId=choice==='reconstruction'?null:(input.dataset.pickingStep || 'geometry');
      const controls=Array.from(document.querySelectorAll('[data-picking-source]'));
      controls.forEach(c=>{c.disabled=true;c.checked=c.dataset.pickingSource===choice && (choice==='reconstruction' || c.dataset.pickingStep===stepId);});
      try{
        if(stepId)await requireLinearSteps();
        const project=await api('/api/picking-source',{project_id:pid,series_id:sid,source:choice,step_id:stepId,reconstruction_id:reconstructionId});
        if(state.project?.id!==pid || state.project.active_reconstruction_assembly!==reconstructionId)return;
        const saved=project.series.find(item=>item.id===sid);
        if(!saved)throw new Error('The tilt series no longer exists.');
        const current=state.project.series.find(item=>item.id===sid);
        for(const target of new Set([row,current].filter(Boolean))){
          target.picking_source=saved.picking_source;target.picking_step_id=saved.picking_step_id;
        }
        state.pickSelectionVersion=(state.pickSelectionVersion || 0)+1;
        rememberProject(state.project);
        if(state.selected===sid)notify(choice?'Picking source saved. Existing picking jobs are unchanged.':'Picking will use Reconstruction by default.');
      }catch(error){
        controls.forEach(c=>c.checked=c.dataset.pickingSource===previousSource && (previousSource==='reconstruction' || c.dataset.pickingStep===previousStep));
        notify(error.message);
      }finally{
        controls.forEach(c=>c.disabled=c.dataset.pickingStep?!processingStepInput(row,processingSteps(row).find(s=>s.id===c.dataset.pickingStep))?.work:!row[c.dataset.pickingSource]?.work);
      }
    });
  });
}
function processingSteps(row){
  const steps=row.processing_steps || (row.geometry_refinement?[{id:'geometry',kind:'geometry_refinement'}]:[]);
  return steps.filter(step=>!['geometry','curation'].includes(step.id) ||
    Object.keys(processingStepJob(row,step) || {}).length ||
    Object.keys(step).some(key=>!['id','kind','number'].includes(key)));
}
function processingStepJob(row,step){return step?.id==='geometry'?row.geometry_refinement:step?.job;}
function firstGeometrySeriesIds(){
  return state.project.series.filter(row=>{
    if(row.enabled===false || !['completed','partial','submitted','running'].includes(row.reconstruction?.state))return false;
    const first=processingSteps(row)[0];
    const job=first?processingStepJob(row,first):row.geometry_refinement;
    return (!first || first.kind==='geometry_refinement') && !job?.work && !job?.job_id;
  }).map(row=>row.id);
}
function precedingGeometry(row,step){const all=processingSteps(row);return all.slice(0,all.findIndex(s=>s.id===step.id)).reverse().find(s=>s.kind==='geometry_refinement' || (s.kind==='manual_curation' && s.job));}
function processingStepInput(row,step){
  if(step?.kind!=='manual_curation')return processingStepJob(row,step);
  if(step.job)return step.job;
  const prior=precedingGeometry(row,step);
  return prior?processingStepJob(row,prior):row.reconstruction;
}
function processingStepPanels(row){
  if(reconstructionAssembly()?.mode==='advanced')return '';
  return `<div class="processing-step-list">${processingSteps(row).map((step,index)=>{
    const geometry=step.kind==='geometry_refinement',job=processingStepInput(row,step);
    const checked=row.picking_source===step.kind && (row.picking_step_id || 'geometry')===step.id;
    return `<details class="refinement-dropdown geometry-dropdown pipeline-step" data-step-id="${esc(step.id)}" data-step-kind="${step.kind}" data-picking-source-fold="${esc(step.id)}"><summary><strong>${geometry?'Refine geometry':'Manual Curation'} ${String(step.number ?? index+2).padStart(3,'0')}</strong>${`<label class="picking-source-choice"><input type="checkbox" data-picking-source="${step.kind}" data-picking-step="${esc(step.id)}" ${checked?'checked':''} ${job?.work && job.manual_action!=='refine'?'':'disabled'}>Select reconstruction</label>`}<button type="button" class="button secondary assembly-delete-button" data-delete-step="${esc(step.id)}">Delete</button></summary><div class="processing-step-body"></div></details>`;
  }).join('')}</div><div class="processing-step-actions"><button type="button" class="button primary" data-add-step="geometry_refinement">＋ Refine geometry</button><button type="button" class="button primary" data-add-step="manual_curation">＋ Manual Curation</button></div>`;
}
function mountReconstructionSteps(row){
  state.activeGeometryStep=null;state.activeCurationStep=null;
  const reconstructionId=state.project.active_reconstruction_assembly,projectId=state.project.id;
  const container=document.querySelector(`[data-reconstruction-id="${reconstructionId}"]`) || $('workspace');
  container.querySelectorAll('[data-add-step]').forEach(button=>button.addEventListener('click',async()=>{
    button.disabled=true;
    try{
      const project=await api('/api/reconstruction-step-create',{project_id:projectId,series_id:row.id,kind:button.dataset.addStep,reconstruction_id:reconstructionId});
      if(state.project?.id!==projectId || state.project.active_reconstruction_assembly!==reconstructionId)return;
      saveProject(project);const last=processingSteps(selectedSeries()).at(-1);
      document.querySelector(`[data-step-id="${last.id}"]`).open=true;
    }catch(error){notify(error.message);button.disabled=false;}
  }));
  document.querySelectorAll('[data-delete-step]').forEach(button=>button.addEventListener('click',event=>{
    event.preventDefault();event.stopPropagation();confirmProcessingStepDeletion(row.id,button.dataset.deleteStep);
  }));
  document.querySelectorAll('[data-step-id]').forEach(panel=>panel.addEventListener('toggle',()=>{
    const step=processingSteps(row).find(s=>s.id===panel.dataset.stepId);const body=panel.querySelector('.processing-step-body');
    const geometry=step.kind==='geometry_refinement';
    if(!panel.open){
      if(geometry && state.activeGeometryStep===step.id){clearTimeout(state.angleRefinementTimer);clearTimeout(state.geometryContrastTimer);state.activeGeometryStep=null;}
      if(!geometry && state.activeCurationStep===step.id){clearTimeout(state.curationTimer);clearTimeout(state.manualRefinementTimer);state.activeCurationStep=null;}
      body.replaceChildren();body.removeAttribute('id');return;
    }
    document.querySelectorAll(`[data-step-kind="${step.kind}"]`).forEach(other=>{if(other!==panel){other.open=false;other.querySelector('.processing-step-body').replaceChildren();other.querySelector('.processing-step-body').removeAttribute('id');}});
    if(geometry){
      state.activeGeometryStep=step.id;body.innerHTML=geometryRefinementPanel(row);
      $('geometry-refinement-form').dataset.geometryStepId=step.id;
      $('geometry-refinement-form').addEventListener('submit',submitAngleRefinement);
      mountGeometryPatchScaling();
      $('geometry-refinement-form').querySelector('[data-submit-angle="all"]').hidden=processingSteps(row)[0]?.id!==step.id;
      state.geometryVolumeMethod='original';state.geometryVolumeIndex=null;state.geometryStatus=null;state.geometryVolumeSources=[];
      mountGeometryContrast();
      mountReconstructionZoom($('geometry-volume-image'));
      bindSliceWheel($('geometry-volume-image').parentElement,$('geometry-volume-slider'));
      $('geometry-volume-slider').addEventListener('input',()=>{state.geometryVolumeIndex=Number($('geometry-volume-slider').value);showGeometryVolume();});
      $('geometry-volume-axis').addEventListener('change',()=>{state.geometryVolumeIndex=null;showGeometryVolume();});
      state.geometryVolumeSource=step.id;state.geometryViewTinycare=false;
      $('geometry-view-tinycare').addEventListener('click',toggleGeometryTinycare);
      loadGeometryVolumeSources(state.project.id,row.id,step.id);
      pollAngleRefinement(state.project.id,row.id);
    }else{
      state.activeCurationStep=step.id;
      const temporary=document.createElement('div');temporary.innerHTML=manualCurationPanel(row);
      body.id='manual-curation';body.append(temporary.querySelector('.preprocessing-layout'));
      mountManualCuration();
    }
  }));
}
function confirmProcessingStepDeletion(sid,stepId){
  const pid=state.project.id;
  const dialog=document.createElement('dialog');dialog.className='assembly-job-dialog';document.body.append(dialog);
  dialog.innerHTML='<form><h3>Delete pipeline step</h3><p class="hint">Permanently delete this step and its outputs. A running Slurm job will be cancelled. Dependent jobs must be deleted first.</p><label class="check-line"><span>I confirm permanent deletion</span><input type="checkbox" required></label><p class="form-error"></p><div class="submit-actions"><button type="button" class="button secondary">Cancel</button><button type="submit" class="button primary" disabled>Delete</button></div></form>';
  const check=dialog.querySelector('input'),submit=dialog.querySelector('[type="submit"]');
  check.onchange=()=>submit.disabled=!check.checked;
  const cancel=dialog.querySelector('[type="button"]');cancel.onclick=()=>dialog.close();dialog.addEventListener('close',()=>dialog.remove());
  dialog.addEventListener('cancel',event=>{if(dialog.deleting)event.preventDefault();});
  dialog.querySelector('form').onsubmit=async event=>{
    event.preventDefault();submit.disabled=true;check.disabled=true;cancel.disabled=true;dialog.deleting=true;
    try{
      let project;
      do{
        project=await api('/api/reconstruction-step-delete',{project_id:pid,series_id:sid,step_id:stepId,confirmed:true});
        if(project.deletion_pending){dialog.querySelector('.form-error').textContent='Waiting for Slurm cancellation…';await new Promise(resolve=>setTimeout(resolve,1500));}
      }while(project.deletion_pending);
      dialog.close();if(state.project?.id===pid)saveProject(project);
    }catch(error){dialog.querySelector('.form-error').textContent=error.message;check.disabled=false;submit.disabled=!check.checked;cancel.disabled=false;dialog.deleting=false;}
  };dialog.showModal();
}
function finalReconstructionOptions(prefix,options){
  const tinycare=(options.denoisers || []).includes('tinycare');
  return `<div class="panel-header">Final reconstruction</div><div class="parameter-section"><div class="metadata-grid"><div class="metadata-field"><label for="${prefix}-final-binning">Final binning</label><select id="${prefix}-final-binning">${[1,2,4,8,16].map(bin=>`<option value="${bin}" ${bin===(options.final_binning ?? 8)?'selected':''}>Bin ${bin}</option>`).join('')}</select></div><label class="check-line"><span>TinyCARE</span><input id="${prefix}-tinycare" type="checkbox" ${prefix==='geometry'?'data-geometry-denoiser="tinycare"':''} ${tinycare?'checked':''}></label><label class="check-line"><span>Reuse TinyCARE model</span><input id="${prefix}-reuse-tinycare" type="checkbox" ${options.reuse_tinycare_model?'checked':''} ${tinycare?'':'disabled'}></label></div></div>`;
}
