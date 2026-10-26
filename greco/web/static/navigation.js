"use strict";

function stageStatus(stage) {
  const rows=(state.project?.series || []).filter(row=>row.enabled!==false);
  if(stage.id==='tilts')return {complete:stage.complete,label:stage.complete?'Import complete':'Data import'};
  if(!stage.enabled)return {complete:false,label:'Waiting for the previous stage'};
  if(stage.id==='sta') {
    const counts={};
    for(const assembly of state.project.particle_assemblies || []) {
      const value=assemblyTaskState(assembly);counts[value]=(counts[value] || 0)+1;
    }
    const labels={running:'Running',submitted:'Submitted',submitting:'Submitting',unknown:'Unknown',failed:'Failed',completed:'Jobs completed',extracted:'Extracted',configured:'Configured',pending:'Configured'};
    return {complete:!!stage.complete,label:Object.keys(labels).filter(key=>counts[key]).map(key=>labels[key]+' '+counts[key]).join(' · ') || 'Ready to configure'};
  }
  if(stage.id==='segmentation') {
    const steps=(state.project.segmentations || []).flatMap(a=>a.tomograms.flatMap(t=>t.steps));
    const complete=steps.length>0 && (state.project.segmentations || []).every(a=>a.tomograms.length>0 && a.tomograms.every(t=>t.steps.length>0 && t.steps.every(s=>s.state==='completed')));
    const busy=steps.filter(s=>['submitted','running','submitting','unknown'].includes(s.state)).length;
    const failed=steps.filter(s=>['failed','cancelled'].includes(s.state)).length;
    return {complete,label:complete?'Jobs completed':busy?'Active jobs '+busy:failed?'Failed jobs '+failed:steps.length?'Jobs pending':'Ready to configure'};
  }
  if(!['preprocessing','reconstruction','picking'].includes(stage.id))return {complete:stage.complete,label:'Ready to configure'};
  const counts={};
  for(const row of rows){
    const value=(stage.id==='preprocessing'?preprocessingState(row.preprocessing,row):row[stage.id])?.state || 'pending';
    counts[value]=(counts[value] || 0)+1;
  }
  const complete=rows.length>0 && counts.completed===rows.length;
  const labels={running:'Running',submitted:'Submitted',submitting:'Submitting',queued:'Queued',unknown:'Unknown',failed:'Failed',partial:'Partial',cancelled:'Cancelled',completed:'Completed'};
  const label=Object.keys(labels).filter(key=>counts[key]).map(key=>labels[key]+' '+counts[key]).join(' · ');
  return {complete,label:label || 'Ready to configure'};
}
function renderStages() {
  const stages=state.project?.stages || [{id:"tilts",label:"Tilt series",enabled:true,complete:false},{id:"preprocessing",label:"Preprocessing",enabled:false},{id:"reconstruction",label:"Reconstruction",enabled:false},{id:"picking",label:"Picking",enabled:false},{id:"sta",label:"STA",enabled:false}];
  $("stages").innerHTML=stages.map((s,i)=>{
    const status=stageStatus(s);
    return `<button class="stage ${status.complete?'complete':''} ${s.id===state.stage?'selected':''} ${!s.enabled?'locked':''}" data-stage="${s.id}" ${!s.enabled?'disabled':''} aria-current="${s.id===state.stage?'step':'false'}"><span class="stage-number">${status.complete?'✓':String(i+1).padStart(2,'0')}</span><span><strong>${esc(s.label)}</strong><small>${esc(status.label)}</small></span><span class="stage-state">${status.complete?'✓':!s.enabled?'○':'→'}</span></button>`;
  }).join("");
}
function renderSidebar() {
  const query=$("series-search").value.toLowerCase();
  const rows=(state.project?.series || []).filter(row=>row.name.toLowerCase().includes(query));
  $("series-list").innerHTML=rows.length ? rows.map(row=>`<button class="series-item ${row.id===state.selected?'selected':''}" data-select="${row.id}" aria-pressed="${row.id===state.selected}"><span class="series-symbol">▤</span><span><strong>${esc(row.name)}</strong><small>${row.found}/${row.expected} tilts · ${angle(row.angle_min)} to ${angle(row.angle_max)}</small></span><span class="mini-dot"></span></button>`).join("") : `<div class="sidebar-empty">${query?'No matching tilt series.':'Imported tilt series will appear here and remain visible throughout the workflow.'}</div>`;
  for(const button of $('series-list').querySelectorAll('[data-select]')) {
    const inactive=rows.find(row=>row.id===button.dataset.select)?.enabled===false;
    button.classList.toggle('inactive',inactive);
    button.querySelector('.mini-dot').title=inactive?'Disabled tilt series':'Enabled tilt series';
  }
}
async function toggleSeriesEnabled(input,shiftKey=false) {
  const projectId=state.project.id,enabled=input.checked;
  const current=input.dataset.seriesEnabled;
  let ids=[current];
  if(input.closest('table')) {
    const order=[...input.closest('table').querySelectorAll('[data-series-enabled]')].map(item=>item.dataset.seriesEnabled);
    const previous=state.seriesCheckAnchor;
    const a=previous?.projectId===projectId?order.indexOf(previous.id):-1,b=order.indexOf(current);
    if(shiftKey && a>=0 && b>=0)ids=order.slice(Math.min(a,b),Math.max(a,b)+1);
  }
  const checks=[...document.querySelectorAll('[data-series-enabled]')];
  const before=new Map(state.project.series.map(row=>[row.id,row.enabled!==false]));
  checks.forEach(check=>{if(ids.includes(check.dataset.seriesEnabled))check.checked=enabled;check.disabled=true;});
  try {
    const project=await api('/api/series-enabled',{project_id:projectId,series_ids:ids,enabled});
    if(state.project?.id===projectId){
      state.seriesCheckAnchor={projectId,id:current};
      if(state.stage==='preprocessing') {
        // Keep the current slice and unsaved orientation/CTF preview in place.
        state.project=project;rememberProject(project);renderSidebar();renderStages();
      }else saveProject(project);
    }
  }catch(error){checks.forEach(check=>check.checked=before.get(check.dataset.seriesEnabled));notify(error.message);}
  finally{checks.forEach(check=>check.disabled=false);}
}
function projectPickAssemblySelector() {
  const assemblies=state.project?.pick_assemblies || [];
  return `<label class="project-pick-selector"><span>Pick assembly</span><select id="project-pick-assembly" aria-label="Pick assembly shown in the Picks column" ${assemblies.length?'':'disabled'}>${assemblies.length?assemblies.map(a=>`<option value="${esc(a.id)}" ${a.id===state.project.active_pick_assembly?'selected':''}>${esc(a.name)}</option>`).join(''):'<option>No pick assemblies</option>'}</select></label>`;
}
async function selectProjectPickAssembly(input) {
  const project=state.project;
  if(!project || !input.value || input.value===project.active_pick_assembly)return;
  state.pickSelectionVersion=(state.pickSelectionVersion || 0)+1;
  clearTimeout(state.projectStatusTimer);input.disabled=true;
  try {
    const result=await api('/api/select-pick-assembly',{project_id:project.id,assembly_id:input.value});
    if(state.project?.id!==project.id)return;
    state.pickCollapsed=null;saveProject(result);
  }catch(error){
    if(state.project?.id===project.id){input.value=project.active_pick_assembly;notify(error.message);renderTilts(state.project.series);}
  }finally{input.disabled=false;}
}
function renderTilts(series) {
  const tilts=series.reduce((n,s)=>n+s.expected,0);
  const metadata=state.project?.metadata || {};
  const metadataItems=[["Voltage","voltage_kv","kV"],["Cs","cs_mm","mm"],["Tilt axis","tilt_axis_deg","°"],["Pixel size","pixel_size_A","Å / pixel"],["Dose","dose_per_tilt","e⁻ / Å²"]];
  $("workspace").innerHTML=`<div class="page-heading"><div><span class="eyebrow">01 / ACQUISITION</span><h1>Tilt series</h1><p>Import your data and review acquisition metadata.</p></div><div class="action-group"><button class="button secondary" data-edit="project" ${!state.project?'disabled':''}>Project metadata</button><button class="button primary" data-action="import" ${!state.project?'disabled':''}>＋ Import tilt series</button></div></div>
  <div class="stats" aria-label="Import summary"><div class="stat"><span class="stat-label">TILT SERIES</span><span class="stat-value">${series.length}</span></div><div class="stat"><span class="stat-label">MATCHED TILTS</span><span class="stat-value">${tilts}</span></div><div class="stat"><span class="stat-label">IMPORT STATUS</span><span class="badge ${series.length?'':'pending'}">${series.length?'Complete':'Pending'}</span></div>
  <div class="project-metadata-strip" aria-label="Project metadata">${metadataItems.map(([label,key,unit])=>`<span>${label}<strong>${metadata[key]===null || metadata[key]===undefined ? '—' : esc(metadata[key])+' '+unit}</strong></span>`).join('')}</div></div>
<section class="panel"><div class="panel-header project-tilt-table-header"><span>Project tilt series</span><div class="project-tilt-table-controls"><small>${series.length} tilt series · original binning</small>${projectPickAssemblySelector()}</div></div>${series.length ? `<div class="table-wrap"><table><thead><tr><th>TILT SERIES</th><th>TILTS</th><th>DEGREES</th><th>PREPROCESSING</th><th>RECONSTRUCTION</th><th>PICKS</th></tr></thead><tbody>${series.map(row=>`<tr class="${row.id===state.selected?'selected':''}"><td><button class="table-id" data-select="${row.id}">${esc(row.name)}<small></small></button></td><td>${row.found}<span class="muted"> / ${row.expected}</span></td><td>${angle(row.angle_min)} <span class="muted">→</span> ${angle(row.angle_max)}</td><td><span class="badge ${row.preprocessing?.state==='completed'?'':'pending'}">${jobStateLabel(row.preprocessing?.state)}</span></td><td><span class="badge ${row.reconstruction?.state==='completed'?'':'pending'}">${reconstructionStateLabel(row.reconstruction)}</span> ${queuedRefinementLabels(row)}</td><td>${(row.picking && row.picking.state!=='completed')?`<span class="badge pending">${jobStateLabel(row.picking?.state)}</span>`:esc(row.picking?.pick_count ?? row.pick_count ?? 0)}</td></tr>`).join("")}</tbody></table></div>` : `<div class="empty-state"><div class="empty-symbol">▤</div><h3>Your first tilt series starts here</h3><p>Select your mdocs and TIFF, EER or MRC files. Greco will match the files and ask you for missing metadata.</p><button class="button secondary" data-action="${state.project?'import':'project'}">${state.project?'Select files':'Choose a project folder'} →</button></div>`}</section>
  <div class="info-strip">${series.length?'Import complete. Open Reconstruction and select a tilt series on the left. CTF will be estimated during processing.':'Mdocs provide tilt angles and acquisition parameters. Original data stays in place.'}</div>`;
  const table=$('workspace').querySelector('.table-wrap table');
  if(table) {
    $('workspace').querySelector('.project-tilt-table-controls small').textContent=`${series.filter(row=>row.enabled!==false).length} enabled / ${series.length} tilt series · original binning`;
    table.querySelector('thead tr').insertAdjacentHTML('afterbegin','<th class="series-enable-column" title="Enabled · Shift-click to change a range" aria-label="Enabled"></th>');
    table.querySelectorAll('tbody tr').forEach((tr,index)=>{
      const row=series[index];tr.classList.toggle('series-inactive',row.enabled===false);
      tr.querySelector('.table-id small').textContent=`TS ${row.slurm_array_index}`;
      tr.insertAdjacentHTML('afterbegin',`<td class="series-enable-column"><input type="checkbox" data-series-enabled="${esc(row.id)}" ${row.enabled!==false?'checked':''} aria-label="Enable ${esc(row.name)}" title="Include in processing · Shift-click to change a range"></td>`);
    });
  }
  for(const badge of $("workspace").querySelectorAll('.badge'))if(badge.textContent==='Running')badge.classList.add('running');
  clearTimeout(state.projectStatusTimer);
  if(series.some(row=>(row.picking?.state==='completed' && row.pick_count!==row.picking.pick_count) || ['preprocessing','reconstruction','refinement','angle_refinement','geometry_refinement','picking','handedness_test','ctf_redo'].some(kind=>['submitted','running','submitting','unknown'].includes(row[kind]?.state))))state.projectStatusTimer=setTimeout(pollProjectStatus,5000);
}
