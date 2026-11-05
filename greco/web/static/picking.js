"use strict";

function refreshPickerAssets(){
  for(const asset of ['template','mask']){
    const image=$('picker-'+asset+'-preview');if(!image)return;
    image.hidden=!state.project['picker_'+asset];
    const button=image.closest('button');if(button){button.disabled=image.hidden;button.hidden=image.hidden;}
    if(!image.hidden)image.src='/api/picker-asset-preview?'+new URLSearchParams({project_id:state.project.id,assembly_id:state.project.active_pick_assembly,asset,v:state.project['picker_'+asset].sha256});
  }
  const grid=$('picker-asset-previews');if(grid)grid.hidden=!state.project.picker_template && !state.project.picker_mask;
}

async function loadPickerMask(){
  const input=$('picker-mask-path'),projectId=state.project.id,assemblyId=state.project.active_pick_assembly;
  const path=input.value.trim();
  if(!path)throw new Error('Select a template mask MRC first.');
  const current=state.project.picker_mask;
  if(path===(current?.source_path || current?.path)){refreshPickerAssets();return true;}
  const version=(state.pickerMaskLoadVersion || 0)+1;state.pickerMaskLoadVersion=version;
  const mask=await api('/api/picker-mask-select',{project_id:projectId,path,assembly_id:assemblyId});
  if(state.project?.id!==projectId || state.project.active_pick_assembly!==assemblyId || !input.isConnected || state.pickerMaskLoadVersion!==version)return false;
  state.project.picker_mask=mask;
  const assembly=state.project.pick_assemblies.find(a=>a.id===assemblyId);if(assembly)assembly.picker_mask=mask;
  input.value=mask.source_path;
  refreshPickerAssets();
  return true;
}

function openPickerAsset(asset){
  const project=state.project,metadata=project['picker_'+asset];if(!metadata)return;
  let dialog=$('picker-asset-dialog');
  if(!dialog){dialog=document.createElement('dialog');dialog.id='picker-asset-dialog';dialog.className='sta-volume-dialog';document.body.appendChild(dialog);}
  let axis='z';const shape=metadata.shape,slices={x:Math.floor(shape[2]/2),y:Math.floor(shape[1]/2),z:Math.floor(shape[0]/2)};
  dialog.innerHTML=`<header class="sta-volume-header"><div><h3>${asset==='template'?'Template':'Mask'} · ${esc(metadata.name)}</h3><p>${shape.join(' × ')} px · ${metadata.pixel_size_A==null?'Unknown sampling':esc(metadata.pixel_size_A)+' Å / pixel'}</p></div><button type="button" class="sta-volume-close" aria-label="Close viewer">×</button></header><div class="sta-volume-toolbar"><div class="sta-volume-axes">${['x','y','z'].map(a=>`<button type="button" data-view-axis="${a}" class="${a==='z'?'active':''}" aria-pressed="${a==='z'}">${a.toUpperCase()}</button>`).join('')}</div></div><div class="sta-volume-canvas"><img alt="${asset==='template'?'Template':'Mask'} slice"></div><footer class="sta-volume-footer"><input type="range" min="0" max="${shape[0]-1}" value="${slices.z}" aria-label="Volume slice"><output></output></footer>`;
  const slider=dialog.querySelector('input'),image=dialog.querySelector('img');
  const refresh=()=>{slices[axis]=Number(slider.value);image.src='/api/picker-asset-preview?'+new URLSearchParams({project_id:project.id,assembly_id:project.active_pick_assembly,asset,axis,index:slices[axis],full:1,v:metadata.sha256});dialog.querySelector('output').textContent=`${axis.toUpperCase()} · ${slices[axis]+1} / ${shape[{z:0,y:1,x:2}[axis]]}`;};
  slider.oninput=refresh;
  dialog.querySelectorAll('[data-view-axis]').forEach(button=>button.onclick=()=>{axis=button.dataset.viewAxis;slider.max=shape[{z:0,y:1,x:2}[axis]]-1;slider.value=slices[axis];dialog.querySelectorAll('[data-view-axis]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});refresh();});
  dialog.querySelector('.sta-volume-close').onclick=()=>dialog.close();
  dialog.onclick=event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left || event.clientX>r.right || event.clientY<r.top || event.clientY>r.bottom)dialog.close();}};
  image.parentElement.onwheel=event=>{event.preventDefault();slider.value=Math.max(0,Math.min(Number(slider.max),Number(slider.value)+(event.deltaY>0?1:-1)));refresh();};
  refresh();dialog.showModal();
}

function pickAssemblyInfo(assembly){
  const records=Object.values(assembly.series || {}),count=records.reduce((n,r)=>n+Number(r.picking?.pick_count || 0),0);
  const states=records.map(r=>r.queued_picking?'queued':r.picking?.state);
  const active=['running','submitted','submitting','queued','failed'].find(s=>states.includes(s));
  const reconstruction=(state.project.reconstruction_assemblies || []).find(r=>r.id===assembly.reconstruction_assembly_id);
  return `${reconstruction?reconstruction.name+' · ':''}${assembly.picker_template?.name || 'No template'} · ${count} picks${active?' · '+jobStateLabel(active):''}`;
}

function updatePickAssemblySummaries(){
  for(const assembly of state.project.pick_assemblies || []){
    const summary=document.querySelector(`[data-pick-assembly-id="${assembly.id}"] summary .hint`);
    if(summary)summary.textContent=pickAssemblyInfo(assembly);
  }
}

function confirmPickAssemblyDeletion(id){
  const project=state.project,assembly=(project.pick_assemblies || []).find(a=>a.id===id);if(!assembly)return;
  const dialog=document.createElement('dialog');dialog.className='assembly-job-dialog';
  dialog.innerHTML=`<form><h3>Delete pick assembly</h3><p>${esc(assembly.name)}</p><p class="hint">Delete this assembly and its picking results permanently. Active picking jobs will be cancelled. Tomograms and source template files are preserved.</p><label class="check-line"><span>I confirm permanent deletion</span><input type="checkbox" required></label><p class="form-error" role="alert"></p><div class="submit-actions"><button type="button" class="button secondary">Cancel</button><button type="submit" class="button primary" disabled>Delete</button></div></form>`;
  document.body.appendChild(dialog);dialog.onclose=()=>dialog.remove();
  const check=dialog.querySelector('input'),submit=dialog.querySelector('[type=submit]'),cancel=dialog.querySelector('[type=button]'),error=dialog.querySelector('.form-error');
  check.onchange=()=>submit.disabled=!check.checked;cancel.onclick=()=>dialog.close();
  dialog.oncancel=event=>{if(dialog.deleting)event.preventDefault();};
  dialog.querySelector('form').onsubmit=async event=>{
    event.preventDefault();if(!check.checked || dialog.deleting)return;
    dialog.deleting=true;submit.disabled=true;check.disabled=true;cancel.disabled=true;
    try{
      let result;
      do{
        result=await api('/api/pick-assembly-delete',{project_id:project.id,assembly_id:id,confirmed:true});
        if(result.deletion_pending){error.textContent='Cancellation requested. Waiting for Slurm to stop picking before deleting files…';await new Promise(resolve=>setTimeout(resolve,1500));}
      }while(result.deletion_pending);
      dialog.close();if(state.project?.id===project.id)saveProject(result);notify('Pick assembly and picking results permanently deleted.');
    }catch(e){error.textContent=e.message;}
    finally{dialog.deleting=false;check.disabled=false;cancel.disabled=false;submit.disabled=!check.checked;}
  };
  dialog.showModal();
}

function bindPickAssemblies(){
  const project=state.project,heading=$('workspace').querySelector('.page-heading');
  const button=document.createElement('button');button.className='button primary';button.textContent='＋ New pick assembly';button.type='button';heading.appendChild(button);
  button.onclick=()=>{
    const dialog=document.createElement('dialog');dialog.className='assembly-job-dialog';
    dialog.innerHTML=`<form><h3>New pick assembly</h3><label class="metadata-field"><span>Name</span><input name="name" required maxlength="100" value="Pick assembly ${String((project.pick_assemblies || []).length+1).padStart(3,'0')}"></label><label class="metadata-field"><span>Reconstruction</span><select name="reconstruction" required>${(project.reconstruction_assemblies || []).map(r=>`<option value="${esc(r.id)}" ${r.id===project.active_reconstruction_assembly?'selected':''}>${esc(r.name)}</option>`).join('')}</select></label><p class="form-error"></p><div class="submit-actions"><button type="button" class="button secondary">Cancel</button><button type="submit" class="button primary">Create</button></div></form>`;
    document.body.appendChild(dialog);dialog.onclose=()=>dialog.remove();dialog.querySelector('[type=button]').onclick=()=>dialog.close();
    dialog.querySelector('form').onsubmit=async event=>{event.preventDefault();const submit=event.submitter;submit.disabled=true;try{const result=await api('/api/pick-assemblies',{project_id:project.id,name:dialog.querySelector('input').value,reconstruction_id:dialog.querySelector('[name=reconstruction]').value});dialog.close();if(state.project?.id===project.id)saveProject(result);}catch(e){dialog.querySelector('.form-error').textContent=e.message;submit.disabled=false;}};
    dialog.showModal();
  };
  const layout=$('workspace').querySelector('.picking-layout'),list=document.createElement('div');list.className='pick-assemblies';
  for(const assembly of project.pick_assemblies || []){
    const panel=document.createElement('details');panel.className='pick-assembly';panel.dataset.pickAssemblyId=assembly.id;
    panel.innerHTML=`<summary><strong>${esc(assembly.name)}</strong><span class="hint">${esc(pickAssemblyInfo(assembly))}</span><button type="button" class="button secondary assembly-delete-button" aria-label="Delete ${esc(assembly.name)}">Delete</button></summary>`;
    panel.querySelector('summary button').onclick=event=>{event.preventDefault();event.stopPropagation();confirmPickAssemblyDeletion(assembly.id);};
    if(assembly.id===project.active_pick_assembly && layout){const body=document.createElement('div');body.className='assembly-body';body.appendChild(layout);panel.appendChild(body);panel.open=state.pickCollapsed!==project.id+':'+assembly.id;}
    panel.addEventListener('toggle',async()=>{
      if(!panel.isConnected)return;
      if(assembly.id===state.project?.active_pick_assembly){state.pickCollapsed=panel.open?null:project.id+':'+assembly.id;return;}
      if(!panel.open)return;
      try{const result=await api('/api/select-pick-assembly',{project_id:project.id,assembly_id:assembly.id});state.pickCollapsed=null;if(state.project?.id===project.id)saveProject(result);}catch(e){panel.open=false;notify(e.message);}
    });
    list.appendChild(panel);
  }
  heading.after(list);
}

function pickerOptionsFromForm() {
  const options={};
  document.querySelectorAll('[data-picker-option]').forEach(field=>{
    options[field.dataset.pickerOption]=field.tagName==='SELECT'?field.value:field.type==='checkbox'?field.checked:field.value===''?null:Number(field.value);
  });
  return options;
}

function renderPicking() {
  const project=state.project;
  const assembly=(project.pick_assemblies || []).find(a=>a.id===project.active_pick_assembly);
  if(assembly?.reconstruction_assembly_id && assembly.reconstruction_assembly_id!==project.active_reconstruction_assembly){
    $('workspace').innerHTML=`<div class="page-heading">${workflowStageHeading('picking')}</div><p>Loading reconstruction…</p>`;
    api('/api/select-pick-assembly',{project_id:project.id,assembly_id:assembly.id}).then(result=>{if(state.project?.id===project.id)saveProject(result);}).catch(error=>notify(error.message));return;
  }
  const context=project.id+':'+(project.active_pick_assembly || '');
  if(state.pickerProject!==context){state.pickerDrafts ??= {};if(state.pickerProject)state.pickerDrafts[state.pickerProject]=state.pickerDraft;state.pickerProject=context;state.pickerDraft=state.pickerDrafts[context] || null;}
  if(!project.active_pick_assembly){
    $('workspace').innerHTML=`<div class="page-heading">${workflowStageHeading('picking')}</div><section class="panel empty-state"><h3>No pick assemblies yet</h3></section>`;
    bindPickAssemblies();return;
  }
  const template=project.picker_template;
  const options={template_pixel_size_A:template?.pixel_size_A ?? null,flip_z:false,invert_contrast:false,
    particle_diameter_A:300,angular_search_deg:10,max_particles:1000,candidate_count:2000,
    high_pass_A:400,low_pass_A:null,matching_mode:'exhaustive',threshold_factor:1,impulse_spectral_weighting:false,...project.picking_options,...state.pickerDraft};
  const fields=[['template_pixel_size_A','Template pixel size (Å)'],['particle_diameter_A','Particle diameter (Å)'],
    ['angular_search_deg','Angular sampling (°)'],['max_particles','Max particles'],['candidate_count','Candidate pool'],
    ['high_pass_A','High-pass (Å)'],['low_pass_A','Low-pass (Å)'],['threshold_factor','Threshold factor']];
  $('workspace').innerHTML=`<div class="page-heading">${workflowStageHeading('picking')}</div>
    <div class="picking-layout"><div class="picker-main"><section class="panel tilt-viewer">
    <div class="panel-header"><span id="picker-viewer-title">Tomogram · picks</span><span id="picker-viewer-count" class="badge"></span></div>
    <div class="tilt-image-area tomogram-area" id="picker-image-area">
    <div id="picker-viewer-empty" class="viewer-empty">Select a tilt series in the left sidebar.</div><img id="picker-volume-image" hidden alt="Tomogram slice with picked particles highlighted" draggable="false"><canvas id="picker-picks-overlay" hidden aria-hidden="true"></canvas>
    <details class="volume-contrast picker-contrast-panel"><summary>Contrast</summary><div class="picker-contrast-controls">
    <div class="contrast-actions"><button type="button" class="button secondary" id="picker-auto-contrast">Auto contrast</button><button type="button" class="button secondary" id="picker-contrast-reset">Reset contrast</button></div>
    <label>Contrast <input type="range" id="picker-contrast" min="0.25" max="4" step="0.05" value="1" aria-label="Contrast"><output id="picker-contrast-value">1.00×</output></label>
    <label>Brightness <input id="picker-brightness" type="range" min="-1" max="1" step="0.02" value="0"><output id="picker-brightness-value">0%</output></label>
    
    <label><input id="picker-mask" type="checkbox" disabled> Highlight picks</label></div></details>
    <div class="picker-view-tools" id="picker-view-tools"><button type="button" class="button secondary" id="picker-zoom-in" aria-label="Zoom in" title="Zoom in · drag to move">🔍 +</button><button type="button" class="button secondary" id="picker-zoom-out" aria-label="Zoom out" title="Zoom out">🔍 −</button></div>
    <div class="volume-variants"><div id="picker-variants" class="picker-variant-buttons"></div></div></div>
    <div class="tilt-scroll"><select id="picker-axis" aria-label="Slice axis"><option value="z">Z</option><option value="y">Y</option><option value="x">X</option></select><input id="picker-slice-slider" type="range" min="0" max="0" disabled value="0" aria-label="Tomogram slice"><span id="picker-slice-label">No tomogram</span></div>
    <p class="hint viewer-info">Picked particles: 100% · background: 20% · display diameter: 1.5 × particle diameter · scroll over the tomogram to change slice.</p><p id="picker-output-paths" class="hint viewer-info picker-path"></p></section></div>
    <form id="picking-form" class="panel processing-parameters"><div class="panel-header">Greco Picker</div><div class="parameter-section">
    <label class="field-label" for="picker-template-path">3D template on server · MRC</label><div class="path-control"><input id="picker-template-path" value="${esc(template?.source_path || template?.path || '')}" placeholder="/path/to/template.mrc" autocomplete="off"><button type="button" class="button secondary" id="picker-template-browse" data-browse="picker-template-path" data-kind="template" aria-label="Choose template file" title="Choose template file"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v2"/><path d="M3 7h6l2 3h10l-3 10H5L3 7Z"/></svg></button></div>
    <p id="picker-template-info" class="hint">${template?`${esc(template.name)} · ${template.shape.join(' × ')} · header pixel size: ${template.pixel_size_A==null?'unknown':esc(template.pixel_size_A)+' Å'}`:'No template selected. Choose a server file or paste its path. Pixel size is read from the MRC header and can be edited below.'}</p>
    <label class="field-label" for="picker-mask-path">Template mask on server · required MRC</label><div class="path-control"><input id="picker-mask-path" value="${esc(project.picker_mask?.source_path || project.picker_mask?.path || '')}" placeholder="/path/to/mask.mrc" required autocomplete="off"><button type="button" class="button secondary" data-browse="picker-mask-path" data-kind="template" aria-label="Choose mask file" title="Choose mask file"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v2"/><path d="M3 7h6l2 3h10l-3 10H5L3 7Z"/></svg></button></div>
    <p class="hint">Same box size and sampling as the template. Weights 0–1. Fast hybrid matching uses your supplied mask fixed in place, including non-spherical masks.</p>
    <div id="picker-asset-previews" class="picker-asset-previews"><button type="button" class="sta-volume-thumbnail" data-picker-asset="template"><img id="picker-template-preview" alt="Template central slice"><span>Template ↗</span></button><button type="button" class="sta-volume-thumbnail" data-picker-asset="mask"><img id="picker-mask-preview" alt="Mask central slice"><span>Mask ↗</span></button></div>
    </div><div class="panel-header">Matching</div><div class="parameter-section"><label class="metadata-field"><span>Matching mode</span><select id="picker-mode" data-picker-option="matching_mode"><option value="exhaustive" ${options.matching_mode==='exhaustive'?'selected':''}>Exhaustive</option><option value="fast" ${options.matching_mode==='fast'?'selected':''}>Fast</option></select></label>
    <p class="hint">Fast screens orientations before full refinement. Exhaustive searches every orientation. Both use symmetric noise correction. Candidate pool applies only to Fast.</p>
    <p class="hint">Threshold factor multiplies the automatic cutoff: 1 = original; 0.75 = 25% lower, more picks. Matching scores are unchanged.</p>
    <div class="metadata-grid">${fields.map(([key,label])=>{const integer=['max_particles','candidate_count'].includes(key);return `<div class="metadata-field"><label for="picker-${key}">${label}</label><input id="picker-${key}" data-picker-option="${key}" type="number" step="${integer?'1':'any'}" min="${integer?'1':'0.001'}" value="${esc(options[key])}" ${['low_pass_A','high_pass_A'].includes(key)?'placeholder="None"':'required'}></div>`}).join('')}
    <label class="check-line"><span>Flip template Z</span><input type="checkbox" data-picker-option="flip_z" ${options.flip_z?'checked':''}></label>
    <label class="check-line"><span>Invert template contrast</span><input type="checkbox" data-picker-option="invert_contrast" ${options.invert_contrast?'checked':''}></label></div>
    <p class="hint viewer-info">Particle diameter controls extraction exclusion and edge margins only. Your mask is resampled together with the template. Flip Z reflects both; contrast inversion changes only template density.</p>
    <label class="check-line picker-impulse-option"><span>Impulse-based spectral weighting</span><input type="checkbox" data-picker-option="impulse_spectral_weighting" ${options.impulse_spectral_weighting?'checked':''}></label>
    <p class="hint">Uses the radial spectrum of a reconstructed point to weight template frequencies.</p>
    </div><p class="hint viewer-info" id="picker-selected">Selected: ${esc(selectedSeries()?.name || 'none')}</p><p id="picker-error" class="form-error viewer-info" role="alert"></p>
    <div class="submit-actions"><button type="submit" class="button primary" data-picker-submit="one">Submit</button><button type="submit" class="button secondary" data-picker-submit="all">Submit all</button></div></form></div>`;
  bindPickAssemblies();refreshPickerAssets();
  document.querySelectorAll('[data-picker-asset]').forEach(button=>button.onclick=()=>openPickerAsset(button.dataset.pickerAsset));
  state.pickerViewer=null;state.pickerViewerLoading=null;
  state.pickerAutoContrast=true;state.pickerMethod='original';
  $('picker-auto-contrast').classList.toggle('active',true);
  state.pickerZoom=1;state.pickerPanX=0;state.pickerPanY=0;state.pickerPanEnabled=false;
  $('picker-zoom-in').addEventListener('click',()=>setPickerZoom(state.pickerZoom*1.5));
  $('picker-zoom-out').addEventListener('click',()=>setPickerZoom(state.pickerZoom/1.5));
  let drag=null;
  $('picker-image-area').addEventListener('pointerdown',event=>{if(!state.pickerPanEnabled || event.button!==0 || event.target!==$('picker-volume-image'))return;event.preventDefault();drag={x:event.clientX,y:event.clientY,px:state.pickerPanX,py:state.pickerPanY};$('picker-image-area').setPointerCapture(event.pointerId);});
  $('picker-image-area').addEventListener('pointermove',event=>{if(!drag)return;state.pickerPanX=drag.px+event.clientX-drag.x;state.pickerPanY=drag.py+event.clientY-drag.y;applyPickerTransform();});
  for(const name of ['pointerup','pointercancel','lostpointercapture'])$('picker-image-area').addEventListener(name,()=>{drag=null;});
  $('picker-brightness').addEventListener('input',()=>showPickerSlice());
  $('picker-contrast-reset').addEventListener('click',()=>{state.pickerAutoContrast=false;$('picker-auto-contrast').classList.toggle('active',false);$('picker-contrast').value=1;$('picker-brightness').value=0;showPickerSlice();});
  $('picker-slice-slider').addEventListener('input',()=>showPickerSlice());
  $('picker-axis').addEventListener('change',()=>{if(state.pickerViewer){const size=state.pickerViewer.shape[{z:0,y:1,x:2}[$('picker-axis').value]];$('picker-slice-slider').max=size-1;$('picker-slice-slider').value=Math.floor(size/2);showPickerSlice();}});
  $('picker-mask').addEventListener('change',()=>drawPickerPicks());
  $('picker-contrast').addEventListener('input',()=>showPickerSlice());
  $('picker-auto-contrast').addEventListener('click',()=>{state.pickerAutoContrast=true;$('picker-contrast').value=1;$('picker-brightness').value=0;$('picker-auto-contrast').classList.toggle('active',true);showPickerSlice();});
  $('picker-image-area').addEventListener('wheel',event=>{if(!state.pickerViewer)return;event.preventDefault();const slider=$('picker-slice-slider');slider.value=Math.max(0,Math.min(Number(slider.max),Number(slider.value)+(event.deltaY>0?1:-1)));showPickerSlice();},{passive:false});
  state.pickerResizeObserver?.disconnect();
  state.pickerResizeObserver=new ResizeObserver(()=>drawPickerPicks());
  state.pickerResizeObserver.observe($('picker-image-area'));
  if(state.selected)loadPickerViewer(state.selected);
  $('picking-form').addEventListener('input',()=>{state.pickerDraft=pickerOptionsFromForm();});
  let templateLoading=false;
  async function loadTemplate(force=false){
    if(templateLoading)return false;
    const path=$('picker-template-path').value.trim();
    if(!path){$('picker-error').textContent='Select a server template path.';return false;}
    const current=state.project.picker_template;
    if(!force && path===(current?.source_path || current?.path))return true;
    templateLoading=true;
    const projectId=project.id;const info=$('picker-template-info'),error=$('picker-error');
    const buttons=[...document.querySelectorAll('[data-picker-submit]'),$('picker-template-browse'),$('picker-template-path')];buttons.forEach(b=>b.disabled=true);
    info.textContent='Reading server template…';error.textContent='';
    try {
      const data=await api('/api/picker-template-select',{project_id:projectId,path,assembly_id:project.active_pick_assembly});
      if(state.project?.id!==projectId || !info.isConnected)return false;
      state.project.picker_template=data;
      const assembly=state.project.pick_assemblies.find(a=>a.id===project.active_pick_assembly);if(assembly)assembly.picker_template=data;
      $('picker-template-path').value=data.source_path;
      info.textContent=`${data.name} · ${data.shape.join(' × ')} · header pixel size: ${data.pixel_size_A==null?'unknown':data.pixel_size_A+' Å'}`;
      $('picker-template_pixel_size_A').value=data.pixel_size_A ?? '';
      state.pickerDraft=pickerOptionsFromForm();
      refreshPickerAssets();
      return true;
    } catch(e){if(error.isConnected){error.textContent=e.message;info.textContent='Template selection failed. Choose a valid 3D MRC.';}return false;}
    finally{templateLoading=false;buttons.forEach(b=>{if(b.isConnected)b.disabled=false;});}
  }
  $('picker-template-path').addEventListener('change',()=>loadTemplate(true));
  $('picker-template-path').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();loadTemplate(true);}});
  const maskChanged=async()=>{try{await loadPickerMask();$('picker-error').textContent='';}catch(error){if($('picker-error'))$('picker-error').textContent=error.message;}};
  $('picker-mask-path').addEventListener('change',maskChanged);
  $('picker-mask-path').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();maskChanged();}});
  $('picking-form').addEventListener('submit',async event=>{
    event.preventDefault();const projectId=project.id;
    const buttons=[...document.querySelectorAll('[data-picker-submit]')];buttons.forEach(b=>b.disabled=true);
    try {
      if(!await loadTemplate())return;
      if(!await loadPickerMask())return;
      buttons.forEach(b=>b.disabled=true);
      const options=pickerOptionsFromForm();state.pickerDraft=options;
      const all=event.submitter?.dataset.pickerSubmit==='all';
      const ids=all?project.series.filter(row=>row.enabled!==false&&['completed','submitted','running'].includes(pickingSourceJob(row)?.state)&&jobNeedsSubmission(row.picking)).map(row=>row.id):[state.selected].filter(Boolean);
      if(!ids.length){notify('No new jobs to submit. Existing jobs are preserved.');return;}
      const result=await api('/api/submit-picking',{project_id:projectId,series_ids:ids,options,submit_all:all,assembly_id:project.active_pick_assembly});
      if(state.project?.id===projectId)saveProject(result);
      notify(result.submission_errors?.length?result.submission_errors.join(' · '):`${'Submitted'} ${ids.length} Greco Picker job(s).`);
    } catch(error){if($('picker-error'))$('picker-error').textContent=error.message;}
    finally{buttons.forEach(b=>{if(b.isConnected)b.disabled=false;});}
  });
  pollPicking(project.id);
}

async function pollPicking(projectId) {
  try {
    const project=await api('/api/project-status?'+new URLSearchParams({project_id:projectId}));
    if(state.project?.id!==projectId || state.stage!=='picking')return;
    state.project=project;
    updatePickAssemblySummaries();
    const selected=selectedSeries();
    updatePickerHeader(selected);
    if(selected && !state.pickerViewerLoading && state.pickerViewer?.key!==pickerViewerKey(selected))loadPickerViewer(selected.id);
    renderStages();
  } catch(error){if(state.stage==='picking' && $('picker-error'))$('picker-error').textContent=error.message;}
  if(state.project?.id===projectId && state.stage==='picking')state.pickingTimer=setTimeout(()=>pollPicking(projectId),5000);
}

function pickingGeometryStepId(row){
  if(row.picking_source==='manual_curation'){
    const step=row.processing_steps?.find(s=>s.id===row.picking_step_id);
    if(step?.job)return step.id;
    return step?precedingGeometry(row,step)?.id:null;
  }
  return row.picking_source==='geometry_refinement'?(row.picking_step_id || 'geometry'):null;
}
function pickingSourceKind(row){return pickingGeometryStepId(row)?'geometry_refinement':'reconstruction';}
function pickingSourceJob(row){const id=pickingGeometryStepId(row);return id?(id==='geometry'?row.geometry_refinement:row.processing_steps?.find(s=>s.id===id)?.job):row.reconstruction;}
function pickerViewerKey(row){return [state.project.active_pick_assembly,row.id,row.picking?.job_id,row.picking?.work,row.picking?.state,row.picking?.pick_count,pickingSourceKind(row),row.picking_step_id,pickingSourceJob(row)?.work].join('|');}

function pickerTomogramLabel(row,info){
  if(!row)return '';
  const picked=row.picking?.state==='completed';
  const volume=info?.volumes?.original || (picked?row.picking?.volume:null);
  const marker=volume?.lastIndexOf('/output/');
  const work=marker>=0?volume.slice(0,marker):!picked?pickingSourceJob(row)?.work:null;
  const steps=processingSteps(row);
  const index=steps.findIndex(step=>processingStepJob(row,step)?.work===work && work);
  if(index>=0)return (steps[index].kind==='manual_curation'?'Manual curation':'Refine geometry')+' '+String(index+2).padStart(3,'0');
  if(work && work===row.geometry_refinement?.work)return 'Refine geometry';
  if(work && work===row.rescue_tilts?.work)return 'Rescue tilts';
  const kind=picked?row.picking?.picking_source_kind:pickingSourceKind(row);
  return kind==='geometry_refinement'?'Refine geometry':'Reconstruct 001';
}

function updatePickerHeader(row){
  if(!$('picker-viewer-title'))return;
  const job=row?.picking || {state:'pending'};
  const viewer=state.pickerViewer?.seriesId===row?.id?state.pickerViewer:null;
  const label=pickerTomogramLabel(row,viewer);
  $('picker-viewer-title').textContent=(row?.name || 'Select a tilt series')+(label?' · '+label:'');
  const queued=false; // Slurm state is reported by the job, never a virtual queue.
  $('picker-viewer-count').textContent=(queued?'Queued':job.state.charAt(0).toUpperCase()+job.state.slice(1))+(queued?'':job.state==='completed'?' · '+Number(job.pick_count ?? row.pick_count ?? 0)+' picks':job.job_id?' · Slurm '+job.job_id:'');
  $('picker-viewer-count').className='badge '+(queued?'pending':job.state==='completed'?'':job.state==='running'?'pending running':'pending');
  $('picker-output-paths').textContent=job.state==='completed'?'STAR: '+job.star:job.error || '';
}

async function loadPickerViewer(seriesId){
  clearTimeout(state.pickerSourceTimer);
  const projectId=state.project.id,row=state.project.series.find(r=>r.id===seriesId);
  const assemblyId=state.project.active_pick_assembly;
  const token={};state.pickerViewerLoading=token;
  state.pickerViewer=null;
  state.pickerDisplayedPlane=null;
  $('picker-picks-overlay').hidden=true;
  $('picker-mask').checked=false;$('picker-mask').disabled=true;
  const empty=$('picker-viewer-empty');if(!empty)return;
  empty.hidden=false;$('picker-volume-image').hidden=true;$('picker-slice-slider').disabled=true;
  updatePickerHeader(row);
  $('picker-volume-image').onerror=()=>{if($('picker-error'))$('picker-error').textContent='Could not load the picking tomogram slice. Restart the server if it has not loaded the updated viewer.';};
  const picked=row?.picking?.state==='completed';
  const source=pickingSourceKind(row);
  if(!picked && source==='reconstruction' && pickingSourceJob(row)?.state!=='completed'){empty.textContent='Complete the selected reconstruction to view this tomogram.';state.pickerViewerLoading=null;return;}
  empty.textContent='Loading tomogram…';
  try{
    const query=new URLSearchParams({project_id:projectId,series_id:seriesId,assembly_id:assemblyId});
    const stepId=!picked && source==='geometry_refinement'?pickingGeometryStepId(row):null;
    if(stepId)query.set('step_id',stepId);
    const info=await api((picked?'/api/picker-viewer-info?':stepId?'/api/geometry-refinement-status?':'/api/reconstruction-status?')+query);
    if(state.project?.id!==projectId || state.project.active_pick_assembly!==assemblyId || state.selected!==seriesId || state.stage!=='picking' || !empty.isConnected)return;
    if(stepId && info.step_id!==stepId)throw new Error('Restart the Greco server to enable linear refinement steps.');
    if(state.project?.id!==projectId || state.project.active_pick_assembly!==assemblyId || state.selected!==seriesId || state.stage!=='picking' || !empty.isConnected)return;
    if(stepId){
      if(stepId==='geometry')row.geometry_refinement=info;
      else{const step=row.processing_steps?.find(s=>s.id===stepId);if(step)step.job=info;}
      if(info.state!=='completed'){
        empty.textContent=info.error || 'Selected refined output: '+jobStateLabel(info.state);
        state.pickerViewerLoading=null;
        if(['submitted','running','unknown'].includes(info.state))state.pickerSourceTimer=setTimeout(()=>{
          if(state.project?.id===projectId && state.selected===seriesId && state.stage==='picking')loadPickerViewer(seriesId);
        },5000);
        return;
      }
    }
    if(stepId){info.shape=info.preview_shape;info.viewerStepId=stepId;}
    else if(!picked && source!=='reconstruction'){
      const key=source==='geometry_refinement'?'geometry':'rescue';
      if(!info.volumes?.[key])throw new Error('The selected picking volume is unavailable. Restart the server if needed.');
      const available=info.volumes;
      info.volumes={original:available[key]};info.viewerMethods={original:key};
      info.shape=info.volume_shapes?.[key];
      for(const method of ['tinycare'])if(available[key+'_'+method]){
        info.volumes[method]=available[key+'_'+method];info.viewerMethods[method]=key+'_'+method;
      }
    }
    if(!info.shape?.length)throw new Error('Tomogram dimensions are unavailable.');
    state.pickerViewer={...info,seriesId,key:pickerViewerKey(row),picked};
    updatePickerHeader(row);
    const volumes=info.volumes || {original:true};
    $('picker-variants').hidden=!['tinycare'].some(key=>volumes[key]);
    state.pickerMethodChoices ??= {};
    const methodKey=JSON.stringify([projectId,assemblyId,seriesId,stepId,volumes.original]);
    const chosen=state.pickerMethodChoices[methodKey];
    state.pickerMethod=chosen && volumes[chosen]?chosen:volumes.tinycare?'tinycare':'original';
    $('picker-variants').innerHTML=Object.entries({tinycare:'TinyCARE'}).filter(([key])=>volumes[key]).map(([key,label])=>`<button type="button" class="button secondary ${state.pickerMethod===key?'active':''}" data-picker-method="${key}" aria-pressed="${state.pickerMethod===key}" title="Click again to return to the selected source volume">${label}</button>`).join('');
    document.querySelectorAll('[data-picker-method]').forEach(button=>button.addEventListener('click',()=>{state.pickerMethod=state.pickerMethod===button.dataset.pickerMethod?'original':button.dataset.pickerMethod;state.pickerMethodChoices[methodKey]=state.pickerMethod;document.querySelectorAll('[data-picker-method]').forEach(b=>{b.classList.toggle('active',b.dataset.pickerMethod===state.pickerMethod);b.setAttribute('aria-pressed',String(b.dataset.pickerMethod===state.pickerMethod));});showPickerSlice();}));
    state.pickerPanX=0;state.pickerPanY=0;setPickerZoom(1);
    const size=info.shape[{z:0,y:1,x:2}[$('picker-axis').value]];
    $('picker-slice-slider').max=size-1;$('picker-slice-slider').value=Math.floor(size/2);$('picker-slice-slider').disabled=false;
    const hasPicks=picked && Number(info.pick_count)>0;
    $('picker-mask').disabled=!hasPicks;
    $('picker-mask').checked=hasPicks;
    empty.hidden=true;$('picker-volume-image').hidden=false;showPickerSlice();
  }catch(error){if(empty.isConnected && state.selected===seriesId)empty.textContent=error.message;}
  finally{if(state.pickerViewerLoading===token)state.pickerViewerLoading=null;}
}

function showPickerSlice(){
  if(!state.pickerViewer || !$('picker-volume-image'))return;
  const index=Number($('picker-slice-slider').value),axis=$('picker-axis').value;
  $('picker-slice-label').textContent=axis.toUpperCase()+' · '+(index+1)+' / '+(Number($('picker-slice-slider').max)+1);
  const brightness=$('picker-brightness').value || 0;
  $('picker-brightness-value').textContent=Math.round(Number(brightness)*100)+'%';
  $('picker-contrast-value').textContent=Number($('picker-contrast').value).toFixed(2)+'×';
  const query=new URLSearchParams({project_id:state.project.id,series_id:state.pickerViewer.seriesId,index,axis,method:state.pickerMethod || 'original',background:1,autocontrast:state.pickerAutoContrast?'1':'0',contrast:$('picker-contrast').value,brightness});
  if(!state.pickerViewer.picked && state.pickerViewer.viewerMethods)query.set('method',state.pickerViewer.viewerMethods[state.pickerMethod || 'original']);
  if(!state.pickerViewer.picked && state.pickerViewer.viewerStepId)query.set('step_id',state.pickerViewer.viewerStepId);
  query.set('assembly_id',state.project.active_pick_assembly);
  const endpoint=state.pickerViewer.picked?'/api/picker-plane?':'/api/volume-plane?';
  const viewer=state.pickerViewer;
  requestViewerImage($('picker-volume-image'),endpoint+query,()=>{
    if(state.pickerViewer===viewer){state.pickerDisplayedPlane={index,axis};drawPickerPicks();}
  });
}

function displayedPickerPicks(){
  const viewer=state.pickerViewer;
  return viewer?.picks_xyz || [];
}

function drawPickerPicks(){
  const canvas=$('picker-picks-overlay'),image=$('picker-volume-image'),area=$('picker-image-area');
  if(!canvas || !image || !area)return;
  const plane=state.pickerDisplayedPlane,viewer=state.pickerViewer;
  const points=displayedPickerPicks();
  if(!viewer?.picked || !plane || !$('picker-mask').checked || !points.length || !image.naturalWidth){canvas.hidden=true;return;}
  canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
  const scale=Math.min(area.clientWidth/canvas.width,area.clientHeight/canvas.height);
  canvas.style.width=canvas.width*scale+'px';canvas.style.height=canvas.height*scale+'px';
  canvas.style.left=(area.clientWidth-canvas.width*scale)/2+'px';canvas.style.top=(area.clientHeight-canvas.height*scale)/2+'px';
  canvas.style.transform=image.style.transform;
  const ctx=canvas.getContext('2d');ctx.fillStyle='rgba(0,0,0,0.8)';ctx.fillRect(0,0,canvas.width,canvas.height);
  const [normal,horizontal,vertical]={z:[2,0,1],y:[1,0,2],x:[0,1,2]}[plane.axis];
  const radius=1.5*viewer.radius_px;ctx.globalCompositeOperation='destination-out';
  ctx.beginPath();
  for(const point of points){
    const squared=radius*radius-(point[normal]-plane.index)**2;if(squared<0)continue;
    const r=Math.sqrt(squared);ctx.moveTo(point[horizontal]+r,point[vertical]);ctx.arc(point[horizontal],point[vertical],r,0,2*Math.PI);
  }
  ctx.fill();ctx.globalCompositeOperation='source-over';canvas.hidden=false;
}

function setPickerZoom(value){
  state.pickerZoom=Math.max(1,Math.min(8,value));
  state.pickerPanEnabled=state.pickerZoom>1;
  $('picker-image-area').classList.toggle('pan-enabled',state.pickerPanEnabled);
  if(state.pickerZoom===1){state.pickerPanX=0;state.pickerPanY=0;}
  applyPickerTransform();
}

function applyPickerTransform(){
  const image=$('picker-volume-image'),area=$('picker-image-area');
  if(!image?.style)return;
  const zoom=state.pickerZoom || 1;
  const maxX=(area.clientWidth || 0)*(zoom-1)/2,maxY=(area.clientHeight || 0)*(zoom-1)/2;
  state.pickerPanX=Math.max(-maxX,Math.min(maxX,state.pickerPanX || 0));
  state.pickerPanY=Math.max(-maxY,Math.min(maxY,state.pickerPanY || 0));
  image.style.transform=`translate(${state.pickerPanX}px, ${state.pickerPanY}px) scale(${zoom})`;
  if($('picker-picks-overlay'))$('picker-picks-overlay').style.transform=image.style.transform;
}
