"use strict";

function openAssemblyJobChooser(id, preset=null) {
  const assembly=state.project.particle_assemblies.find(row=>row.id===id);
  if(!assembly)return;
  const projectId=state.project.id;
  const busy=false; // Independent immutable jobs may be submitted concurrently.
  const available=job=>job?.state==='completed' || (['submitted','running','unknown'].includes(job?.state) && !!job?.job_id);
  const currentReady=available(assembly.extraction);
  const ready=currentReady || (assembly.job_history || []).some(job=>job.kind==='extraction' && available(job));
  let dialog=$('assembly-new-job-dialog');
  if(!dialog){dialog=document.createElement('dialog');dialog.id='assembly-new-job-dialog';dialog.className='assembly-job-dialog';document.body.appendChild(dialog);}
  const classResolution=2*Number(assembly.extraction?.pixel_size_A || assembly.pixel_size_A);
  dialog.innerHTML=`<form id="assembly-new-job-form"><h3>New job</h3><p class="hint">${esc(assembly.name)}</p><label class="metadata-field"><span>Job type</span><select id="assembly-new-job-type"><option value="extraction">Extract</option><option value="classification" ${ready?'':'disabled'}>Class3D • without alignment</option><option value="refinement3d" ${ready?'':'disabled'}>Refine3D • local refinement</option></select></label><label id="assembly-extract-source-field" class="metadata-field"><span>From:</span><select id="assembly-extract-source"><option value="assembly:${esc(assembly.id)}">${esc(assembly.name)}</option></select></label><label id="assembly-extract-binning-field" class="metadata-field"><span>Binning</span><select id="assembly-extract-binning">${[1,2,3,4,6,8,16].map(bin=>`<option value="${bin}" ${bin===4?'selected':''}>${bin}</option>`).join('')}</select></label><div id="assembly-class-options" hidden><label class="metadata-field"><span>Classes</span><input id="assembly-class-count" type="number" min="2" max="32" step="1" value="3"></label><label class="metadata-field"><span>Iterations</span><input id="assembly-class-iterations" type="number" min="1" max="200" step="1" value="25"></label><label class="metadata-field"><span>Regularization (T)</span><input id="assembly-class-t" type="number" min="0.1" max="100" step="any" value="4"></label><label class="metadata-field"><span>Resolution limit (Å)</span><input id="assembly-class-resolution" type="number" min="${2*assembly.pixel_size_A}" max="${assembly.output_box_size*assembly.pixel_size_A/4}" step="any" value="${classResolution}"></label><label class="metadata-field"><span>Particle diameter (Å, optional)</span><input id="assembly-class-diameter" type="number" step="any"></label><p class="hint">Automatic reference initialization. Angles, XYZ and shifts remain fixed. Sampling is inherited from Extract.</p></div><div id="assembly-refine-options" hidden><p class="hint">Greco local refinement · automatic convergence and resolution · independent half-maps · GPU.</p></div><p id="assembly-extract-sampling" class="hint"></p>${busy?'<p class="hint">Wait for the active assembly job to finish.</p>':''}<p id="assembly-new-job-error" class="form-error" role="alert"></p><div class="submit-actions"><button type="button" class="button secondary" id="assembly-new-job-cancel">Cancel</button><button class="button primary" type="submit" ${busy?'disabled':''}>Submit</button></div></form>`;
  $('assembly-new-job-cancel').onclick=()=>dialog.close();
  $('assembly-class-resolution').oninput=()=>{$('assembly-class-resolution').dataset.custom='true';};
  let refinePixel=Number(assembly.extraction?.pixel_size_A || assembly.pixel_size_A);
  let refineBox=Number(assembly.extraction?.box_size || assembly.output_box_size);
  const assemblyMask=assembly.default_mask_path || '';
  $('assembly-class-options').insertAdjacentHTML('beforeend','<label class="metadata-field"><span>Reference mask MRC (server path, optional)</span><input id="assembly-class-mask" type="text"><small>Defaults to the picking mask, including Flip Z; adapted to extraction sampling. Clear to use the spherical diameter.</small></label>');
  $('assembly-class-mask').value=assemblyMask;
  $('assembly-class-mask').oninput=()=>{$('assembly-class-mask').dataset.custom='true';};
  $('assembly-refine-options').insertAdjacentHTML('beforeend',`<label class="metadata-field"><span>Reference mask MRC (server path, optional)</span><input id="assembly-refine-mask" type="text" placeholder="/path/to/mask.mrc"></label><label class="metadata-field"><span>Spherical diameter (Å, when no mask)</span><input id="assembly-refine-diameter" type="number" min="${4*refinePixel}" max="${refineBox*refinePixel}" step="any" value="${.9*refineBox*refinePixel}"></label><label class="metadata-field"><span><input id="assembly-refine-per-tilt" type="checkbox"> Refine per-tilt particle shifts</span></label><p class="hint">Local XYZ and angle refinement with automatic resolution and convergence. Mask must match box and sampling.</p>`);
  $('assembly-refine-options').insertAdjacentHTML('beforeend',`<input id="assembly-refine-ctf" type="checkbox" hidden><p class="hint">Use separate CTFRefine and Polishing jobs with a reference mask: Refine3D → CTFRefine → Refine3D → Polishing → Refine3D.</p>`);
  $('assembly-refine-per-tilt').closest('label').hidden=true;
  const refinedSources=[assembly.refinement3d,...(assembly.job_history || []).filter(j=>j.kind==='refinement3d')].filter(available).sort((a,b)=>particleJobNumber(a,0)-particleJobNumber(b,0));
  $('assembly-refine-options').insertAdjacentHTML('afterend',`<div id="assembly-postprocess-options" hidden><label class="metadata-field"><span>Mask MRC (optional, server path)</span><input id="assembly-postprocess-mask" type="text" placeholder="/path/to/mask.mrc"><small>Adapted to the half-map sampling in physical coordinates. Resolution is measured within this mask.</small></label><label class="metadata-field"><span>Sharpening</span><select id="assembly-postprocess-mode"><option value="manual">Manual B-factor</option><option value="auto" selected>Automatic B-factor</option><option value="none">No sharpening</option></select></label><label class="metadata-field"><span>B-factor (Å²)</span><input id="assembly-postprocess-bfactor" type="number" min="-10000" max="10000" step="any" value="-20"></label><details><summary>Advanced</summary><label class="metadata-field"><span>B-factor fit: low-resolution limit (Å)</span><input id="assembly-postprocess-lowres" type="number" min="0.001" step="any" value="10"></label><label class="metadata-field"><span>B-factor fit: high-resolution limit (Å; 0 = automatic)</span><input id="assembly-postprocess-highres" type="number" min="0" step="any" value="0"></label><label class="metadata-field"><span>Low-pass (Å; −1 = FSC resolution, 0 = off)</span><input id="assembly-postprocess-lowpass" type="number" min="-1" step="any" value="-1"></label><label class="metadata-field"><span><input id="assembly-postprocess-interpolate" type="checkbox"> Interpolate the FSC resolution</span></label></details><p class="hint">FSC 0.143 with phase-randomization correction, FSC weighting and sharpened maps from unfiltered half-maps.</p></div>`);
  $('assembly-postprocess-mask').oninput=()=>{$('assembly-postprocess-mask').dataset.custom='true';};
  $('assembly-refine-mask').value=assemblyMask;
  $('assembly-refine-mask').oninput=()=>{$('assembly-refine-mask').dataset.custom='true';};
  $('assembly-refine-mask').closest('label').insertAdjacentHTML('beforeend','<small>Defaults to the picking mask (including Flip Z), adapted to extraction sampling. Clear to use a sphere.</small>');
  const refineMaskHint=Array.from($('assembly-refine-options').querySelectorAll('p.hint')).find(node=>node.textContent.includes('Mask must match'));
  if(refineMaskHint){refineMaskHint.id='assembly-refine-sampling-hint';refineMaskHint.textContent='Local XYZ and angle refinement. Masks are adapted in physical coordinates; the source files are unchanged.';}
  $('assembly-new-job-type').insertAdjacentHTML('beforeend',`<option value="ctf_refinement" ${refinedSources.length?'':'disabled'}>CTFRefine</option><option value="polishing" ${refinedSources.length?'':'disabled'}>Polishing</option><option value="postprocessing" ${refinedSources.length?'':'disabled'}>PostProcess</option>`);
  $('assembly-refine-options').insertAdjacentHTML('afterend',`<div id="assembly-observation-options" hidden><label class="metadata-field"><span>Own-half reference (Refine3D)</span><select id="assembly-observation-reference">${refinedSources.map(j=>`<option value="refinement3d:${esc(j.work)}">${esc(particleJobName(j,'Refine3D'))}</option>`).join('')}</select></label><label class="metadata-field" id="assembly-polishing-shift-field"><span>Maximum residual shift (Å)</span><input id="assembly-polishing-shift" type="number" min="0.001" step="any" value="${5*refinePixel}"></label><label class="metadata-field" id="assembly-polishing-velocity-field"><span>GP velocity σ (Å / e⁻ Å⁻²)</span><input id="assembly-polishing-velocity" type="number" min="0.001" step="any" value="0.2"></label><label class="metadata-field" id="assembly-ctf-defocus-field"><span>Maximum defocus offset (Å)</span><input id="assembly-ctf-defocus" type="number" min="1" max="5000" value="3000"></label><p class="hint"></p></div>`);
  $('assembly-polishing-shift').oninput=()=>{$('assembly-polishing-shift').dataset.custom='true';};
  $('assembly-polishing-velocity-field').insertAdjacentHTML('afterend',`<label class="metadata-field" id="assembly-polishing-divergence-field"><span>GP spatial σ (Å / e⁻ Å⁻²)</span><input id="assembly-polishing-divergence" type="number" min="0.001" step="any" value="5000"></label>`);
  $('assembly-polishing-divergence-field').insertAdjacentHTML('afterend',`<label class="metadata-field" id="assembly-polishing-iterations-field"><span>Maximum GP optimizer iterations</span><input id="assembly-polishing-iterations" type="number" min="1" max="10000" step="1" value="10000"></label>`);
  $('assembly-polishing-shift-field').insertAdjacentHTML('beforebegin',`<label class="metadata-field" id="assembly-polishing-mask-field"><span>Reference mask (optional, server path)</span><input id="assembly-polishing-mask" type="text" placeholder="/path/to/mask.mrc"><small>Applied before reference projection. Adapted to the particle sampling in physical coordinates.</small></label>`);
  $('assembly-polishing-mask').oninput=()=>{$('assembly-polishing-mask').dataset.custom='true';};
  const inheritPolishingMask=()=>{
    if($('assembly-polishing-mask').dataset.custom)return;
    const reference=refinedSources.find(j=>'refinement3d:'+j.work===$('assembly-observation-reference').value);
    $('assembly-polishing-mask').value=reference?.options?.mask_path || assemblyMask;
  };
  $('assembly-observation-reference').onchange=inheritPolishingMask;
  inheritPolishingMask();
  // Reuse the server-side MRC browser, including change events on selection.
  for(const id of ['assembly-class-mask','assembly-refine-mask','assembly-polishing-mask','assembly-postprocess-mask']){
    const input=$(id),control=document.createElement('div');control.className='path-control';
    input.before(control);control.appendChild(input);
    control.insertAdjacentHTML('beforeend',`<button type="button" class="button secondary" data-browse="${id}" data-kind="template">Choose file</button>`);
    input.onchange=input.oninput;
  }
  const selection=document.createElement('div');selection.id='assembly-extract-classes';selection.hidden=true;
  for(const kind of ['class','refine'])$(`assembly-${kind}-options`).insertAdjacentHTML('beforeend','<p class="hint sta-native-model-hint">Native beam-depth CTF per particle/view · CTF-premultiplied likelihood · frozen own-half residual noise. Existing extraction images are reused.</p>');
  $('assembly-extract-source-field').after(selection);
  const classSource=available(assembly.classification)?assembly.classification:null;
  const referenceSelect=$('assembly-observation-reference');
  Array.from(referenceSelect.options).forEach((option,i)=>{option.textContent=particleJobName(refinedSources[i],'Refine3D')+' · '+jobStateLabel(refinedSources[i].state);});
  if(refinedSources.length)referenceSelect.value='refinement3d:'+refinedSources[refinedSources.length-1].work;
  const sourceJobs=new Map();
  for(const item of [...(assembly.job_history || []),...['extraction','classification','refinement3d','polishing','ctf_refinement','postprocessing'].map(kind=>({...assembly[kind],kind}))]){
    if(item.work && available(item) && ['extraction','classification','refinement3d','polishing','ctf_refinement','postprocessing'].includes(item.kind))sourceJobs.set(item.kind+':'+item.work,item);
  }
  const selectedClasses=new Map();
  const showClasses=()=>{
    const enabled=$('assembly-new-job-type').value==='extraction' && $('assembly-extract-source').value.startsWith('classification:');
    selection.hidden=!enabled;
    if(!enabled)return;
    const source=$('assembly-extract-source').value;
    const sourceClass=sourceJobs.get(source) || classSource;
    if(!sourceJobs.has(source)){
      selection.textContent='Original Class3D input is unavailable. Choose an available From source.';
      return;
    }
    if(!selectedClasses.has(source))selectedClasses.set(source,new Set());
    const chosen=selectedClasses.get(source);
    const classes=sourceClass.classes || Array.from({length:Number(sourceClass.options?.classes || 0)},(_,i)=>({id:i+1,particle_count:'Pending'}));
    selection.innerHTML=`<p class="hint">Click classes to include or exclude their particles:</p><div class="sta-extract-class-grid">${classes.map(c=>`<button type="button" class="sta-extract-class" data-extract-class="${c.id}" aria-pressed="${chosen.has(Number(c.id))}" aria-label="Include Class ${c.id}">${sourceClass.state==='completed'?`<img src="${esc(particleVolumeUrl(assembly,sourceClass,'classification',c.id))}" alt="Class ${c.id} central slice" loading="lazy">`:''}<span>Class ${c.id}<small>${esc(c.particle_count)} particles</small></span></button>`).join('')}</div>`;
    selection.querySelectorAll('[data-extract-class]').forEach(button=>button.onclick=()=>{
      const id=Number(button.dataset.extractClass);
      if(chosen.has(id))chosen.delete(id);else chosen.add(id);
      button.setAttribute('aria-pressed',String(chosen.has(id)));
    });
  };
  const updateSharpening=()=>{
    $('assembly-postprocess-bfactor').disabled=$('assembly-new-job-type').value!=='postprocessing' || $('assembly-postprocess-mode').value!=='manual';
  };
  $('assembly-postprocess-mode').onchange=updateSharpening;
  const sourceSampling=()=>{
    const job=sourceJobs.get($('assembly-extract-source').value);
    if(!job || !job.pixel_size_A || !job.box_size)return;
    refinePixel=Number(job.pixel_size_A);refineBox=Number(job.box_size);
    if(!$('assembly-class-mask').dataset.custom)$('assembly-class-mask').value=job.options?.mask_path || assemblyMask;
    if($('assembly-new-job-type').value==='postprocessing' && !$('assembly-postprocess-mask').dataset.custom)$('assembly-postprocess-mask').value=job.options?.mask_path || assemblyMask;
    if($('assembly-new-job-type').value==='refinement3d' && !$('assembly-refine-mask').dataset.custom)$('assembly-refine-mask').value=job.options?.mask_path || assemblyMask;
    if(!$('assembly-polishing-shift').dataset.custom)$('assembly-polishing-shift').value=Math.min(5*refinePixel,refineBox*refinePixel/8);
    $('assembly-polishing-shift-field').querySelector('span').textContent=`Maximum residual shift (Å; 5 px = ${(5*refinePixel).toFixed(2)} Å)`;
    for(const kind of ['class']){
      const input=$('assembly-'+kind+'-resolution');input.min=2*refinePixel;input.max=refineBox*refinePixel/4;
      input.value=input.dataset.custom?Math.min(Math.max(Number(input.value),2*refinePixel),refineBox*refinePixel/4):2*refinePixel;
    }
    $('assembly-refine-diameter').min=4*refinePixel;$('assembly-refine-diameter').max=refineBox*refinePixel;
    $('assembly-refine-diameter').value=.9*refineBox*refinePixel;
    if($('assembly-refine-sampling-hint'))$('assembly-refine-sampling-hint').textContent=`Mask is adapted to the selected extraction in physical coordinates. One extracted pixel = ${refinePixel.toFixed(4)} Å.`;
    if($('assembly-new-job-type').value!=='extraction')$('assembly-extract-sampling').textContent=`Uses ${refineBox}² px extracted views · ${refinePixel.toFixed(4)} Å / pixel unchanged. Shift units are pixels at this sampling.`;
  };
  const showSampling=()=>{
    const extracting=$('assembly-new-job-type').value==='extraction';
    const kind=$('assembly-new-job-type').value;
    const allowed=kind==='postprocessing'?['refinement3d']:extracting?['classification','refinement3d','polishing','ctf_refinement']:['polishing','ctf_refinement'].includes(kind)?['refinement3d','polishing','ctf_refinement']:['extraction','refinement3d','polishing','ctf_refinement'];
    const choices=new Map();
    if(extracting)choices.set('assembly:'+assembly.id,{label:assembly.name});
    else if(currentReady && allowed.includes('extraction'))choices.set('extraction:'+assembly.extraction.work,{job:assembly.extraction});
    if(extracting && classSource)choices.set('classification:'+classSource.work,{job:classSource});
    for(const [value,job] of sourceJobs){
      if(allowed.includes(job.kind) && !choices.has(value))choices.set(value,{job});
    }
    const sorted=Array.from(choices).sort((a,b)=>particleJobNumber(a[1].job || {},0)-particleJobNumber(b[1].job || {},0));
    const select=$('assembly-extract-source');
    select.innerHTML=sorted.map(([value,{job,label}])=>`<option value="${esc(value)}">${esc(job?particleJobName(job,job.kind)+' · '+jobStateLabel(job.state):label)}</option>`).join('');
    if(select.options.length)select.selectedIndex=select.options.length-1;
    showClasses();
    $('assembly-extract-binning-field').hidden=!extracting;
    $('assembly-class-options').hidden=$('assembly-new-job-type').value!=='classification';
    $('assembly-refine-options').hidden=$('assembly-new-job-type').value!=='refinement3d';
    $('assembly-postprocess-options').hidden=kind!=='postprocessing';
    document.querySelectorAll('#assembly-postprocess-options input,#assembly-postprocess-options select').forEach(input=>input.disabled=kind!=='postprocessing');
    updateSharpening();
    const observation=['polishing','ctf_refinement'].includes(kind);
    $('assembly-observation-options').hidden=!observation;
    $('assembly-polishing-shift-field').hidden=kind!=='polishing';
    $('assembly-polishing-velocity-field').hidden=kind!=='polishing';
    $('assembly-polishing-mask-field').hidden=!observation;
    $('assembly-observation-options').querySelector('p.hint').textContent=kind==='ctf_refinement'?'Joint per-tilt defocus, astigmatism and contrast scale · native particle beam-depth gradient retained · original images and noise per tilt.':'GP3D joint tilt geometry, particle XYZ and motion · original images and noise per tilt. Use a reference mask to reproduce the validated protocol.';
    $('assembly-polishing-divergence-field').hidden=kind!=='polishing';
    $('assembly-polishing-iterations-field').hidden=kind!=='polishing';
    $('assembly-ctf-defocus-field').hidden=kind!=='ctf_refinement';
    document.querySelectorAll('#assembly-observation-options input,#assembly-observation-options select').forEach(input=>input.disabled=!observation || input.closest('label')?.hidden);
    document.querySelectorAll('#assembly-refine-options input').forEach(input=>input.disabled=$('assembly-refine-options').hidden);
    document.querySelectorAll('#assembly-class-options input').forEach(input=>input.disabled=$('assembly-class-options').hidden);
    if(extracting)showExtractionSampling();else $('assembly-extract-sampling').textContent='Uses the extracted sampling unchanged.';
    sourceSampling();
    if(kind==='postprocessing')$('assembly-extract-sampling').textContent=`Uses unfiltered half-maps · ${refineBox}³ px · ${refinePixel.toFixed(4)} Å / pixel.`;
  };
  const showExtractionSampling=()=>{
    const bin=Number($('assembly-extract-binning').value),native=Number(assembly.native_pixel_size_A);
    const box=2*Math.floor(assembly.box_size/(2*bin)+.5);
    $('assembly-extract-sampling').textContent=`${box}² px patches · ${(native*bin).toFixed(4)} Å / pixel`;
  };
  $('assembly-new-job-type').onchange=showSampling;
  $('assembly-extract-source').onchange=()=>{showClasses();sourceSampling();};
  $('assembly-extract-binning').onchange=showExtractionSampling;showSampling();
  if(preset){
    const type=$('assembly-new-job-type'),option=Array.from(type.options).find(o=>o.value===preset.kind);
    if(!option){notify('Unsupported job type.');return;}
    option.disabled=false;type.value=preset.kind;showSampling();
    const options=preset.options || {};
    const restoreSelect=(id,value)=>{
      if(value==null)return;
      const select=$(id),found=Array.from(select.options).find(o=>o.value===String(value));
      if(!found){const unavailable=document.createElement('option');unavailable.value=value;unavailable.textContent='Unavailable original source · '+String(value).split('/').pop();unavailable.disabled=true;select.appendChild(unavailable);}
      select.value=String(value);
    };
    restoreSelect('assembly-extract-source',options.from);
    restoreSelect('assembly-observation-reference',options.reference_from);
    sourceSampling();
    const fields=preset.kind==='classification'?{classes:'assembly-class-count',iterations:'assembly-class-iterations',regularization:'assembly-class-t',resolution_A:'assembly-class-resolution',particle_diameter_A:'assembly-class-diameter',mask_path:'assembly-class-mask'}:
      preset.kind==='refinement3d'?{mask_path:'assembly-refine-mask',particle_diameter_A:'assembly-refine-diameter',refine_per_tilt_shifts:'assembly-refine-per-tilt',refine_ctf:'assembly-refine-ctf'}:
      preset.kind==='polishing'?{maximum_shift_A:'assembly-polishing-shift',velocity_sigma_A:'assembly-polishing-velocity',divergence_sigma_A:'assembly-polishing-divergence',gp3d_iterations:'assembly-polishing-iterations',mask_path:'assembly-polishing-mask'}:
      preset.kind==='postprocessing'?{mask_path:'assembly-postprocess-mask',bfactor_mode:'assembly-postprocess-mode',bfactor_A2:'assembly-postprocess-bfactor',autob_lowres_A:'assembly-postprocess-lowres',autob_highres_A:'assembly-postprocess-highres',low_pass_A:'assembly-postprocess-lowpass',interpolate:'assembly-postprocess-interpolate'}:
      preset.kind==='ctf_refinement'?{maximum_defocus_A:'assembly-ctf-defocus',mask_path:'assembly-polishing-mask'}:{binning:'assembly-extract-binning'};
    for(const [key,id] of Object.entries(fields)){
      if(!Object.prototype.hasOwnProperty.call(options,key))continue;
      const input=$(id);if(input.type==='checkbox')input.checked=!!options[key];else input.value=options[key] ?? '';
    }
    if(preset.kind==='refinement3d' && Object.prototype.hasOwnProperty.call(options,'mask_path'))$('assembly-refine-mask').dataset.custom='true';
    if(preset.kind==='classification' && Object.prototype.hasOwnProperty.call(options,'mask_path'))$('assembly-class-mask').dataset.custom='true';
    if(preset.kind==='classification' && options.resolution_A!=null)$('assembly-class-resolution').dataset.custom='true';
    if(preset.kind==='postprocessing'){
      $('assembly-postprocess-mask').dataset.custom='true';
    }
    if(preset.kind==='polishing'){
      $('assembly-polishing-mask').value=options.mask_path || '';
      $('assembly-polishing-mask').dataset.custom='true';
      if(options.maximum_shift_A!=null)$('assembly-polishing-shift').dataset.custom='true';
    }
    if(preset.kind==='ctf_refinement' && options.mask_path)$('assembly-polishing-mask').dataset.custom='true';
    if(preset.kind==='extraction' && Array.isArray(options.classes))selectedClasses.set(options.from,new Set(options.classes.map(Number)));
    showClasses();if(preset.kind==='extraction')showExtractionSampling();
    dialog.querySelector('h3').textContent='New job · settings from '+preset.name;
  }
  $('assembly-new-job-form').onsubmit=async event=>{
    event.preventDefault();const button=event.submitter;button.disabled=true;
    let pendingId=null;
    const kind=$('assembly-new-job-type').value;
    if(!['extraction','classification','refinement3d','polishing','ctf_refinement','postprocessing'].includes(kind)){button.disabled=false;return;}
    try {
      const payload={project_id:projectId,assembly_id:id};
      payload.options={from:$('assembly-extract-source').value};
      if(!$('assembly-extract-source').value || $('assembly-extract-source').selectedOptions[0]?.disabled)throw new Error('Choose an available input in From.');
      if(kind==='extraction')payload.options.binning=Number($('assembly-extract-binning').value);
      if(kind==='extraction' && payload.options.from.startsWith('classification:')){
        payload.options.classes=Array.from(selection.querySelectorAll('[data-extract-class][aria-pressed="true"]'),button=>Number(button.dataset.extractClass));
        if(!payload.options.classes.length)throw new Error('Select at least one class.');
      }
      if(kind==='postprocessing')Object.assign(payload.options,{backend:'gpu',mask_path:$('assembly-postprocess-mask').value.trim() || null,bfactor_mode:$('assembly-postprocess-mode').value,bfactor_A2:Number($('assembly-postprocess-bfactor').value),autob_lowres_A:Number($('assembly-postprocess-lowres').value),autob_highres_A:Number($('assembly-postprocess-highres').value),low_pass_A:Number($('assembly-postprocess-lowpass').value),interpolate:$('assembly-postprocess-interpolate').checked});
      if(kind==='classification')Object.assign(payload.options,{mask_path:$('assembly-class-mask').value.trim() || null,classes:Number($('assembly-class-count').value),iterations:Number($('assembly-class-iterations').value),regularization:Number($('assembly-class-t').value),resolution_A:Number($('assembly-class-resolution').value),particle_diameter_A:$('assembly-class-diameter').value?Number($('assembly-class-diameter').value):null});
      if(kind==='refinement3d')Object.assign(payload.options,{auto_refine:true,local_marginalization:false,update_noise:false,signal_prior:'fsc',mask_corrected_fsc:true,mask_path:$('assembly-refine-mask').value.trim() || null,particle_diameter_A:$('assembly-refine-diameter').value?Number($('assembly-refine-diameter').value):null});
      if(kind==='refinement3d')Object.assign(payload.options,{observation_model:'ctf_premultiplied',noise_model:'half_residual'});
      if(kind==='classification')Object.assign(payload.options,{observation_model:'raw',noise_model:'half_residual'});
      if(kind==='refinement3d'){payload.options.alignment_mode='trust_region';payload.options.refine_per_tilt_shifts=false;payload.options.refine_ctf=false;}
      if(['polishing','ctf_refinement'].includes(kind)){
        if(!$('assembly-observation-reference').value || $('assembly-observation-reference').selectedOptions[0]?.disabled)throw new Error('Choose an available own-half reference.');
        Object.assign(payload.options,{reference_from:$('assembly-observation-reference').value});
        Object.assign(payload.options,{mask_path:$('assembly-polishing-mask').value.trim() || null,noise_model:'view_residual'});
        if(kind==='polishing')Object.assign(payload.options,{maximum_shift_A:Number($('assembly-polishing-shift').value),velocity_sigma_A:Number($('assembly-polishing-velocity').value),motion_model:'gp3d',divergence_sigma_A:Number($('assembly-polishing-divergence').value),gp3d_iterations:Number($('assembly-polishing-iterations').value),gp3d_dose_scaling:true,refine_rigid_geometry:true,observation_model:'raw'});
        else Object.assign(payload.options,{maximum_defocus_A:Number($('assembly-ctf-defocus').value),defocus_step_A:25,ctf_model:'joint'});
      }
      if(preset?.kind===kind){
        // Preserve original non-editable parameters, rather than silently
        // replacing historical jobs with today's defaults.
        const edited={postprocessing:['backend','from','mask_path','bfactor_mode','bfactor_A2','autob_lowres_A','autob_highres_A','low_pass_A','interpolate'],extraction:['from','binning','classes'],classification:['mask_path','from','classes','iterations','regularization','resolution_A','particle_diameter_A'],refinement3d:['from','mask_path','particle_diameter_A','refine_per_tilt_shifts','refine_ctf'],polishing:['from','reference_from','maximum_shift_A','velocity_sigma_A','motion_model','divergence_sigma_A','gp3d_iterations','gp3d_dose_scaling','refine_rigid_geometry','mask_path','observation_model','noise_model'],ctf_refinement:['from','reference_from','maximum_defocus_A','defocus_step_A','ctf_model','mask_path','noise_model']}[kind];
        for(const [key,value] of Object.entries(preset.options || {}))if(!edited.includes(key))payload.options[key]=value;
      }
      pendingId=crypto.randomUUID();
      state.staPendingSubmissions ??= new Map();
      state.staPendingSubmissions.set(pendingId,{projectId,assemblyId:id,kind,state:'submitting'});
      assemblyExpandedIds().add(id);
      dialog.close();
      renderSTA();
      const project=await api('/api/submit-particle-'+kind,payload);
      state.staPendingSubmissions.delete(pendingId);
      if(state.project?.id===projectId)saveProject(project);
    } catch(error){
      if(pendingId){
        const pending=state.staPendingSubmissions.get(pendingId);
        if(pending){pending.state='failed';pending.error=error.message;}
        if(state.project?.id===projectId && state.stage==='sta')renderPendingParticleSubmissions();
        notify('Job submission failed: '+error.message);
      }else $('assembly-new-job-error').textContent=error.message;
    }
    finally{if(button.isConnected)button.disabled=busy;}
  };
  updateSharpening();
  dialog.showModal();
}

