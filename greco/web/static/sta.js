"use strict";

function assemblyExpandedIds() {
  state.staExpandedAssemblies ??= {};
  return state.staExpandedAssemblies[state.project.id] ??= new Set();
}

function assemblySummary(assembly) {
  return `<span class="assembly-chevron" aria-hidden="true">›</span><span class="assembly-summary-name"><strong>${esc(assembly.name)}</strong><small>${assembly.tomograms.some(m=>m.picking_dependency)?'Awaiting picks':esc(assembly.particle_count)+' particles'} · ${assembly.tomograms.length} tomograms</small></span><span class="${particleTaskBadgeClass(assemblyTaskState(assembly))}">${esc(assemblyTaskLabel(assembly))}</span>`;
}

function particleTaskBadgeClass(status) {
  return status==='completed'?'badge':status==='running'?'badge pending running':'badge pending';
}

function assemblyLatestTask(assembly) {
  const jobs=['extraction','classification','refinement3d','polishing','ctf_refinement','postprocessing'].map((kind,i)=>({kind,job:assembly[kind],fallback:i+1})).filter(item=>item.job);
  jobs.sort((a,b)=>(b.job.number || b.fallback)-(a.job.number || a.fallback));
  return jobs.find(item=>['submitted','running','submitting','unknown'].includes(item.job.state)) || jobs[0];
}

function assemblyTaskState(assembly) {
  return assemblyLatestTask(assembly)?.job.state || 'pending';
}

function loadAssemblyPreviews(panel) {
  if(!panel.open)return;
  panel.querySelectorAll('[data-average-src]').forEach(image=>{
    if(!image.getAttribute('src'))image.src=image.dataset.averageSrc;
  });
}

function updateAssemblyCards(panel,assembly) {
  if(!panel.open)return false;
  const container=$('assembly-job-'+assembly.id);
  const signature=JSON.stringify([assembly.extraction,assembly.classification,assembly.refinement3d,assembly.polishing,assembly.ctf_refinement,assembly.postprocessing,assembly.job_history]);
  if(!container || container.dataset.signature===signature)return false;
  container.innerHTML=particleAssemblyCards(assembly);container.dataset.signature=signature;
  return true;
}

function assemblyTaskLabel(assembly) {
  const latest=assemblyLatestTask(assembly);
  if(!latest)return 'Configured';
  if(latest.kind==='extraction' && latest.job.state==='completed')return 'Extracted';
  return {extraction:'Extraction',classification:'Class3D',refinement3d:'Refine3D',polishing:'Polishing',ctf_refinement:'CTFRefine',postprocessing:'PostProcess'}[latest.kind]+' · '+jobStateLabel(latest.job.state);
}

async function pollParticleAssemblies() {
  clearTimeout(state.staTimer);
  if(state.stage!=='sta' || !state.project)return;
  const id=state.project.id;
  try {
    const project=await api('/api/project-status?project_id='+encodeURIComponent(id));
    if(state.stage!=='sta' || state.project?.id!==id)return;
    state.project=project;renderStages();
    for(const assembly of project.particle_assemblies || []) {
      const card=$('assembly-job-'+assembly.id);
      if(card){
        const panel=card.closest('.particle-assembly');
        if(updateAssemblyCards(panel,assembly))bindParticleJobs(panel);
        const badge=panel.querySelector('summary .badge');
        badge.textContent=assemblyTaskLabel(assembly);badge.className=particleTaskBadgeClass(assemblyTaskState(assembly));
      }
    }
    const volumeDialog=$('sta-volume-dialog');if(volumeDialog?.onStaPreviewUpdate)volumeDialog.onStaPreviewUpdate();
    if((project.particle_assemblies || []).some(a=>[...(a.job_history || []),...['extraction','classification','refinement3d','polishing','ctf_refinement','postprocessing'].map(k=>a[k])].some(j=>['submitted','running','submitting','unknown'].includes(j?.state))))state.staTimer=setTimeout(pollParticleAssemblies,5000);
  } catch(error){if(state.stage==='sta' && state.project?.id===id)state.staTimer=setTimeout(pollParticleAssemblies,8000);}
}

function confirmAssemblyDeletion(id) {
  const assembly=(state.project.particle_assemblies || []).find(row=>row.id===id);
  if(!assembly)return;
  const projectId=state.project.id,dialog=$('assembly-delete-dialog');
  $('assembly-delete-name').textContent=assembly.name;
  $('assembly-delete-confirm').checked=false;
  $('assembly-delete-submit').disabled=true;
  $('assembly-delete-error').textContent='';
  $('assembly-delete-confirm').onchange=()=>{$('assembly-delete-submit').disabled=!$('assembly-delete-confirm').checked;};
  $('assembly-delete-cancel').onclick=()=>dialog.close();
  $('assembly-delete-form').onsubmit=async event=>{
    event.preventDefault();
    if(!$('assembly-delete-confirm').checked)return;
    const button=$('assembly-delete-submit');button.disabled=true;
    try {
      const project=await api('/api/particle-assembly-delete',{project_id:projectId,assembly_id:id,confirmed:true});
      dialog.close();
      if(state.project?.id===projectId)saveProject(project);
      notify('Assembly and its files permanently deleted. Picks and tomograms were preserved.');
    } catch(error){$('assembly-delete-error').textContent=error.message;}
    finally{button.disabled=!$('assembly-delete-confirm').checked;}
  };
  dialog.showModal();
}

function assemblyEligible(row) {return row.enabled!==false && ((row.picking?.state==='completed' && Number(row.picking.pick_count)>0) || (['submitted','running'].includes(row.picking?.state) && !!row.picking?.job_id));}
function assemblyPickRows(){
  const id=$('assembly-pick-source')?.value || state.assemblyPickSource || state.project.active_pick_assembly;
  const source=(state.project.pick_assemblies || []).find(a=>a.id===id);
  return state.project.series.map(row=>source?{...row,picking:source.series?.[row.id]?.picking}:row);
}
function assemblyNativePixel() {
  const selected=[...document.querySelectorAll('[data-assembly-series]:checked')].map(input=>input.dataset.assemblySeries);
  const row=state.project.series.find(row=>selected.includes(row.id));
  return Number(row?.metadata?.pixel_size_A ?? state.project.metadata?.pixel_size_A);
}
function syncAssemblySize(source='box') {
  const pixel=assemblyNativePixel();
  let box=Number($('assembly-box').value);
  if(source==='dimensions') {
    const dimensions=Number($('assembly-dimensions').value);
    if(Number.isFinite(pixel) && pixel>0 && dimensions>0)box=Math.max(8,Math.min(2048,Math.round(dimensions/pixel)));
    $('assembly-box').value=box;
  }
  if(Number.isFinite(pixel) && pixel>0 && box>0) {
    if(source!=='dimensions')$('assembly-dimensions').value=Number((box*pixel).toFixed(6));
    $('assembly-sampling').textContent=`Original pixel size: ${pixel.toFixed(4)} Å / pixel · box extent: ${(box*pixel).toFixed(2)} Å per side. Choose output binning in the Extract job.`;
  } else $('assembly-sampling').textContent='Native pixel size is required.';
  const rows=assemblyPickRows().filter(row=>[...document.querySelectorAll('[data-assembly-series]:checked')].some(input=>input.dataset.assemblySeries===row.id));
  $('assembly-selection').textContent=`${rows.length} tomograms · ${rows.reduce((sum,row)=>sum+Number(row.picking?.pick_count || 0),0)} particles`;
}
function renderSTA() {
  if(state.assemblyPickProject!==state.project.id){state.assemblyPickProject=state.project.id;state.assemblyPickSource=state.project.active_pick_assembly;}
  if(!(state.project.pick_assemblies || []).some(a=>a.id===state.assemblyPickSource))state.assemblyPickSource=state.project.active_pick_assembly;
  const assemblies=state.project.particle_assemblies || [];
  const rows=assemblyPickRows();
  $('workspace').innerHTML=`<div class="page-heading"><div><span class="eyebrow">05 / SUBTOMOGRAM AVERAGING</span><h1>STA</h1><p>Organize picked particles into independent particle assemblies.</p></div><button id="new-particle-assembly" type="button" class="button primary" ${rows.some(assemblyEligible)?'':'disabled'}>＋ New particle assembly</button></div><section id="assembly-create-panel" class="panel hidden"><div class="panel-header">New particle assembly</div><form id="assembly-form" class="assembly-form"><div class="metadata-grid"><label class="metadata-field"><span>Name</span><input id="assembly-name" required maxlength="100" value="Particle assembly ${assemblies.length+1}"></label><label class="metadata-field"><span>Box size (original pixels)</span><input id="assembly-box" type="number" min="8" max="2048" step="1" required value="288"></label><label class="metadata-field"><span>Dimensions (Å per side)</span><input id="assembly-dimensions" type="number" min="0.001" step="any" required></label></div><p id="assembly-sampling" class="hint"></p><p class="hint">Box size and Å dimensions use the original acquisition pixel size. Include all picked particles from each selected tomogram.</p><div class="table-wrap"><table><thead><tr><th><input id="assembly-select-all" type="checkbox" checked aria-label="Select all available tomograms"></th><th>TOMOGRAM</th><th>PARTICLES</th><th>STATUS</th></tr></thead><tbody>${rows.map(row=>`<tr><td><input type="checkbox" data-assembly-series="${esc(row.id)}" ${assemblyEligible(row)?'checked':'disabled'} aria-label="Include ${esc(row.name)}"></td><td>${esc(row.name)}</td><td>${esc(row.picking?.pick_count ?? 'Pending')}</td><td>${assemblyEligible(row)?'Available':row.picking?.state==='completed'?'No particles':jobStateLabel(row.picking?.state)}</td></tr>`).join('')}</tbody></table></div><p id="assembly-selection" class="hint"></p><p id="assembly-error" class="form-error" role="alert"></p><div class="submit-actions"><button id="assembly-cancel" type="button" class="button secondary">Cancel</button><button type="submit" class="button primary">Create particle assembly</button></div></form></section><div class="particle-assemblies">${assemblies.map(assembly=>`<details class="panel particle-assembly" data-assembly-id="${esc(assembly.id)}" ${assemblyExpandedIds().has(assembly.id)?'open':''}><summary>${assemblySummary(assembly)}<button type="button" class="button secondary assembly-delete-button" data-delete-assembly="${esc(assembly.id)}" aria-label="Delete ${esc(assembly.name)}">Delete</button></summary></details>`).join('') || '<section class="panel empty-state"><h3>No particle assemblies yet</h3><p>Create an assembly by selecting tomograms with completed picks.</p></section>'}</div>`;
  const pickSources=state.project.pick_assemblies || [];
  if(pickSources.length){
    $('assembly-form').querySelector('.metadata-grid').insertAdjacentHTML('afterbegin',`<label class="metadata-field"><span>From pick assembly</span><select id="assembly-pick-source">${pickSources.map(a=>`<option value="${esc(a.id)}" ${a.id===(state.assemblyPickSource || state.project.active_pick_assembly)?'selected':''}>${esc(a.name)}</option>`).join('')}</select></label>`);
    $('assembly-pick-source').onchange=()=>{
      state.assemblyPickSource=$('assembly-pick-source').value;
      $('assembly-form').querySelector('tbody').innerHTML=assemblyPickRows().map(row=>`<tr><td><input type="checkbox" data-assembly-series="${esc(row.id)}" ${assemblyEligible(row)?'checked':'disabled'} aria-label="Include ${esc(row.name)}"></td><td>${esc(row.name)}</td><td>${esc(row.picking?.pick_count ?? 'Pending')}</td><td>${assemblyEligible(row)?'Available':jobStateLabel(row.picking?.state)}</td></tr>`).join('');
      document.querySelectorAll('[data-assembly-series]').forEach(input=>input.onchange=()=>syncAssemblySize());
      $('assembly-select-all').checked=true;$('assembly-select-all').indeterminate=false;syncAssemblySize();
    };
    $('new-particle-assembly').disabled=!pickSources.some(a=>Object.values(a.series || {}).some(assemblyEligible));
  }
  $('new-particle-assembly').addEventListener('click',()=>{$('assembly-create-panel').classList.remove('hidden');syncAssemblySize();$('assembly-name').focus();});
  document.querySelectorAll('.particle-assembly').forEach((panel,index)=>{
    const assembly=assemblies[index],container=document.createElement('div');
    const expanded=assemblyExpandedIds();
    panel.addEventListener('toggle',()=>{
      if(!panel.isConnected)return;
      if(panel.open)expanded.add(assembly.id);else expanded.delete(assembly.id);
      const current=state.project.particle_assemblies.find(a=>a.id===assembly.id);
      if(current && updateAssemblyCards(panel,current))bindParticleJobs(panel);
      loadAssemblyPreviews(panel);
    });
    container.className='assembly-body';
    container.id='assembly-job-'+assembly.id;panel.appendChild(container);
    updateAssemblyCards(panel,assembly);
    const badge=panel.querySelector('summary .badge');badge.textContent=assemblyTaskLabel(assembly);
  });
  bindParticleJobs();clearTimeout(state.staTimer);state.staTimer=setTimeout(pollParticleAssemblies,100);
  document.querySelectorAll('[data-delete-assembly]').forEach(button=>button.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();confirmAssemblyDeletion(button.dataset.deleteAssembly);}));
  $('assembly-cancel').addEventListener('click',()=>{$('assembly-create-panel').classList.add('hidden');});
  $('assembly-box').addEventListener('input',()=>syncAssemblySize('box'));
  $('assembly-dimensions').addEventListener('input',()=>syncAssemblySize('dimensions'));
  $('assembly-dimensions').addEventListener('change',()=>syncAssemblySize('box'));
  document.querySelectorAll('[data-assembly-series]').forEach(input=>input.addEventListener('change',()=>{const available=[...document.querySelectorAll('[data-assembly-series]:not(:disabled)')];$('assembly-select-all').checked=available.every(input=>input.checked);$('assembly-select-all').indeterminate=available.some(input=>input.checked)&&!available.every(input=>input.checked);syncAssemblySize();}));
  $('assembly-select-all').addEventListener('change',()=>{document.querySelectorAll('[data-assembly-series]:not(:disabled)').forEach(input=>input.checked=$('assembly-select-all').checked);syncAssemblySize();});
  $('assembly-form').addEventListener('submit',async event=>{
    event.preventDefault();const projectId=state.project.id,button=event.submitter;button.disabled=true;$('assembly-error').textContent='';
    const assembly={name:$('assembly-name').value,box_size:Number($('assembly-box').value),series_ids:[...document.querySelectorAll('[data-assembly-series]:checked')].map(input=>input.dataset.assemblySeries)};
    if($('assembly-pick-source'))assembly.pick_assembly_id=$('assembly-pick-source').value;
    try {const project=await api('/api/particle-assemblies',{project_id:projectId,assembly});if(state.project?.id!==projectId)return;saveProject(project);notify('Assembly created. Use + to submit an Extract job.');}
    catch(error){if(state.project?.id===projectId && $('assembly-error'))$('assembly-error').textContent=error.message;}
    finally{if(button.isConnected)button.disabled=false;}
  });
}

