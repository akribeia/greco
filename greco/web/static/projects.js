"use strict";

function storedValue(key) {try{return localStorage.getItem(key);}catch{return null;}}
function rememberProjectView() {
  if(!state.project)return;
  try {
    localStorage.setItem('greco-view:'+state.project.path,JSON.stringify({
      stage:state.stage,selected:state.selected,search:$('series-search').value,
      expandedAssemblies:Array.from(state.staExpandedAssemblies?.[state.project.id] || [])
    }));
  }catch{} // Navigation still works when browser storage is disabled.
}
function restoreProjectView(project) {
  try {
    const view=JSON.parse(storedValue('greco-view:'+project.path) || '{}');
    if(project.stages.some(s=>s.id===view.stage && s.enabled))state.stage=view.stage;
    if(project.series.some(s=>s.id===view.selected))state.selected=view.selected;
    if(typeof view.search==='string')$('series-search').value=view.search;
    const ids=new Set((project.particle_assemblies || []).map(a=>a.id));
    state.staExpandedAssemblies ??= {};
    state.staExpandedAssemblies[project.id]=new Set(Array.isArray(view.expandedAssemblies)?view.expandedAssemblies.filter(id=>ids.has(id)):[]);
  }catch{} // Missing or obsolete saved views fall back to Tilt series.
}
let serverRecentProjects=null;
function recentProjects() {
  if(serverRecentProjects!==null)return serverRecentProjects;
  try {
    const rows=JSON.parse(storedValue('greco-recent-projects') || '[]');
    if(!Array.isArray(rows))return [];
    const seen=new Set();
    return rows.filter(row=>row && typeof row.path==='string' && row.path && !seen.has(row.path) && seen.add(row.path)).slice(0,10);
  } catch {return [];}
}
function rememberProject(project) {
  const recent=[{path:project.path,name:project.name || project.path.split('/').pop()},...recentProjects().filter(row=>row.path!==project.path)].slice(0,10);
  serverRecentProjects=recent;
  try {
    localStorage.setItem('greco-project',project.path);
    localStorage.setItem('greco-recent-projects',JSON.stringify(recent));
  } catch {} // Storage restrictions must never prevent opening a project.
}
function closeProjectMenu() {$('recent-project-menu').hidden=true;$('project-switch').setAttribute('aria-expanded','false');}
function toggleProjectMenu() {
  const menu=$('recent-project-menu');
  if(!menu.hidden){closeProjectMenu();return;}
  menu.innerHTML=`<div class="recent-project-heading">Recent projects</div>${recentProjects().map(row=>`<button type="button" class="recent-project-item ${row.path===state.project?.path?'current':''}" data-recent-project="${esc(row.path)}"><strong>${esc(row.name || row.path.split('/').pop())}</strong><small>${esc(row.path)}</small></button>`).join('') || '<p class="recent-project-empty">No recent projects yet.</p>'}<button type="button" class="recent-project-open" data-open-project>＋ Open another project…</button>`;
  menu.hidden=false;$('project-switch').setAttribute('aria-expanded','true');
}
async function activateProject(path,create=false) {
  rememberProjectView();
  const project=await api('/api/open',{path,create});
  state.stage='tilts';state.selected=null;state.preprocessOptions=null;
  state.preprocessStatus=null;state.reconstructionStatus=null;state.pickerDraft=null;
  $('series-search').value='';
  restoreProjectView(project);
  saveProject(project);
  closeProjectMenu();
}
async function openRecentProject(path) {
  if(state.openingProject)return;
  state.openingProject=true;
  try {await activateProject(path);}
  catch(error){notify('Could not open project: '+error.message);}
  finally{state.openingProject=false;}
}
async function restoreProject() {
  let path=storedValue('greco-project');
  if(serverRecentProjects!==null && !recentProjects().some(row=>row.path===path))path=null;
  if(path) {
    state.openingProject=true;$('project-name').textContent='Opening project…';
    try {await activateProject(path);}
    catch(error){render();notify('Could not restore the last project: '+error.message);toggleProjectMenu();}
    finally{state.openingProject=false;}
  } else if(recentProjects().length)toggleProjectMenu();
  else showProject();
}
function showProject() { closeProjectMenu();$("project-error").textContent=""; $("project-path").value=state.project?.path || storedValue("greco-project") || ""; $("project-cancel").hidden=!state.project; $("project-dialog").showModal(); }
async function openProject(event) {
  event.preventDefault();
  const submit=event.submitter; submit.disabled=true;
  try { await activateProject($("project-path").value,$("project-create").checked); $("project-dialog").close(); }
  catch(error){$("project-error").textContent=error.message;} finally{submit.disabled=false;}
}
async function browseFor(target, kind="folder") {
  state.browse={target,kind,path:null,parent:null}; $("browse-error").textContent="";
  $("browse-title").textContent=kind==="template"?'Select a 3D template':kind==="gain"?'Select a gain reference':'Select a folder';
  $("browse-select").hidden=kind!=="folder"; $("browse-dialog").showModal();
  let path=$(target).value || state.project?.path || undefined;
  if(kind!=="folder" && $(target).value) path=path.slice(0,path.lastIndexOf('/'));
  try{await browseLoad(path);}catch(error){$("browse-error").textContent=error.message;}
}
async function browseLoad(path) {
  const data=await api('/api/browse?kind='+state.browse.kind+(path?'&path='+encodeURIComponent(path):''));
  state.browse.path=data.path; state.browse.parent=data.parent;
  $("browse-location").value=data.path; $("browse-up").disabled=!data.parent;
  $("browse-list").innerHTML=data.folders.length?data.folders.map(f=>`<button class="folder-row" data-folder="${esc(f.path)}"><span>▰</span>${esc(f.name)}<span>→</span></button>`).join(""):'<div class="sidebar-empty">No visible subfolders in this folder.</div>';
  if(state.browse.kind!=="folder") {
    const attribute=state.browse.kind==="template"?'data-template-file':'data-gain-file';
    $("browse-list").insertAdjacentHTML('beforeend',(data.files || []).map(f=>`<button class="folder-row" ${attribute}="${esc(f.path)}"><span>▤</span>${esc(f.name)}<span>Select</span></button>`).join(''));
    if(!data.files?.length) $("browse-list").insertAdjacentHTML('beforeend',`<div class="hint detail-card">${state.browse.kind==="template"?'No MRC template files found here.':'No TIFF, MRC or .gain files found here.'} Open a subfolder.</div>`);
  }
  $("browse-select").disabled=false;
}
function showImport() {
  state.step=0; state.scan=null; state.preview=null; state.sources={mdoc:state.project.path,movie:state.project.path,gain:state.project.gain_path || "",gain_flip_y:state.project.gain_flip_y || false};
  $("import-error").textContent=""; renderImport(); $("import-dialog").showModal();
}
function metadataFields(row, prefix) {
  return fields.map(([key,label,unit,required])=>`<div class="metadata-field"><label for="${prefix}-${key}">${label}${required?' *':''}</label><input id="${prefix}-${key}" data-meta="${key}" type="number" step="any" ${required?'required':''} value="${esc(row.metadata[key])}" class="${required && row.metadata[key]===null?'missing':''}" placeholder="${key==='cs_mm'?'e.g. 2.7':key==='tilt_axis_deg'?'Enter tilt axis':'Required'}"><small>${unit} · ${esc(row.metadata_sources?.[key] || 'editable')}</small></div>`).join("") + gainControl(row.gain_path || "", prefix+"-gain",row.gain_flip_y || false);
}
function collectMetadata(container) {
  const values={}; container.querySelectorAll('[data-meta]').forEach(input=>values[input.dataset.meta]=input.value===''?null:Number(input.value)); return values;
}
function projectDraft(rows) {
  const values={}, sources={}, conflicts=[];
  for(const [key,label] of fields) {
    const found=[...new Set(rows.map(row=>row.metadata[key]).filter(value=>value!==null && value!==undefined))];
    const saved=state.project.metadata?.[key];
    values[key]=saved!==null && saved!==undefined ? saved : found.length===1 ? found[0] : null;
    if(key==='dose_per_tilt' && (saved===null || saved===undefined)) values[key]=null;
    sources[key]=saved!==null && saved!==undefined?'project':found.length>1?'conflicting mdoc values':key==='tilt_axis_deg'?rows.map(row=>row.metadata_sources[key]).filter((value,index,array)=>array.indexOf(value)===index).join(' / '):key==='dose_per_tilt'?'no initial value':'shared mdoc value';
    if(found.length>1) conflicts.push(`${label}: ${found.join(', ')}. Choose a shared value.`);
  }
  return {metadata:values,metadata_sources:sources,gain_path:state.sources?.gain ?? state.project.gain_path ?? '',gain_flip_y:state.sources?.gain_flip_y ?? state.project.gain_flip_y ?? false,conflicts};
}
function renderImport() {
  $("import-progress").innerHTML=['01 · Folders','02 · Files','03 · Metadata'].map((s,i)=>`<span class="wizard-step ${i===state.step?'active':i<state.step?'done':''}">${s}</span>`).join("");
  $("import-back").hidden=state.step===0;
  $("import-next").textContent=['Find files →','Review metadata →','Complete import ✓'][state.step];
  $("import-summary").textContent=state.step===2?`${state.preview.series.length} selected tilt series`:'';
  if(state.step===0) {
    $("import-body").innerHTML=`<div class="source-field"><label class="field-label" for="mdoc-dir">Mdoc folder</label><div class="path-control"><input id="mdoc-dir" value="${esc(state.sources.mdoc)}"><button class="button secondary" data-browse="mdoc-dir">Browse</button></div><div class="hint">Search includes subfolders.</div></div><div class="source-field"><label class="field-label" for="movie-dir">TIFF / EER / MRC data folder</label><div class="path-control"><input id="movie-dir" value="${esc(state.sources.movie)}"><button class="button secondary" data-browse="movie-dir">Browse</button></div><div class="hint">A shared folder can contain several positions.</div></div>${gainControl(state.sources.gain,"source-gain",state.sources.gain_flip_y)}<div class="source-note">Files are read directly from the server. SubFramePath names in each mdoc are matched to the selected files, including Windows acquisition paths.</div>`;
  } else if(state.step===1) {
    $("import-body").innerHTML=`<p class="hint">Select the mdocs and data files to import. If several files share a name, select only the correct copy.</p><div class="file-columns">${[['mdocs','MDOC'],['movies','TIFF / EER / MRC']].map(([key,label])=>`<section class="file-panel"><label class="file-panel-header"><input type="checkbox" checked data-select-all="${key}">${label} · ${state.scan[key].length} files</label><div class="file-scroll">${state.scan[key].map(path=>`<label class="file-line" title="${esc(path)}"><input type="checkbox" checked data-file="${key}" value="${esc(path)}"><span>${esc(path.split('/').pop())}</span></label>`).join('') || '<div class="hint detail-card">No files found.</div>'}</div></section>`).join('')}</div>`;
  } else {
    const common=projectDraft(state.preview.series);
    $("import-body").innerHTML=`<p class="hint">One set of metadata for every tilt series in the project. Dose and tilt axis are required; enter the dose and review the axis suggested by the mdoc.</p><section class="metadata-card"><div class="metadata-card-title"><span>Project metadata</span><span class="badge pending">${state.preview.series.length} series</span></div><div class="metadata-grid">${metadataFields(common,'common')}</div>${common.conflicts.map(message=>`<div class="warning-note">${esc(message)}</div>`).join('')}</section><section class="panel"><div class="panel-header">File matching</div><div class="table-wrap"><table><thead><tr><th>TILT SERIES</th><th>MATCHED TILTS</th><th>FORMAT</th><th>CHECK</th></tr></thead><tbody>${state.preview.series.map(row=>`<tr><td>${esc(row.name)}</td><td>${row.found} / ${row.expected}</td><td>${esc(row.input_format)}</td><td>${row.errors.length || row.missing.length || row.ambiguous.length ? '<span class="badge pending">Check files</span>' : '<span class="badge">✓ Complete</span>'}</td></tr>`).join('')}</tbody></table></div></section>${state.preview.series.map(row=>[...row.errors,...row.missing.map(name=>'Missing '+name),...row.ambiguous.map(name=>'Ambiguous file: '+name)].map(message=>`<div class="error-note">${esc(row.name)}: ${esc(message)}</div>`).join('')).join('')}`;
  }
}
async function nextImport() {
  $("import-error").textContent=""; $("import-next").disabled=true;
  try {
    if(state.step===0) {
      state.sources={mdoc:$("mdoc-dir").value,movie:$("movie-dir").value,gain:$("source-gain").value,gain_flip_y:gainFlipY("source-gain")};
      state.scan=await api('/api/scan',{project_id:state.project.id,mdoc_dir:state.sources.mdoc,movie_dir:state.sources.movie});
      state.step=1; renderImport();
    } else if(state.step===1) {
      const chosen=key=>Array.from(document.querySelectorAll(`[data-file="${key}"]:checked`),input=>input.value);
      state.preview=await api('/api/preview',{project_id:state.project.id,scan_id:state.scan.scan_id,mdocs:chosen('mdocs'),movies:chosen('movies'),gain_path:state.sources.gain || null,gain_flip_y:state.sources.gain_flip_y});
      state.step=2; renderImport();
    } else {
      const missing=Array.from($("import-body").querySelectorAll('input[required]')).find(input=>!input.value || !input.checkValidity());
      if(missing){missing.focus(); throw new Error('Complete the required project metadata.');}
      const shared=collectMetadata($('import-body')), gain=$('common-gain').value || null;
      const project=await api('/api/import',{project_id:state.project.id,preview_id:state.preview.preview_id,project_metadata:shared,gain_path:gain,gain_flip_y:gainFlipY("common-gain")});
      saveProject(project); $("import-dialog").close(); notify('Import complete. Reconstruction is now available.');
    }
  } catch(error){$("import-error").textContent=error.message;} finally{$("import-next").disabled=false;}
}
function showEdit(id) {
  state.edit=null;
  const row={metadata:state.project.metadata,metadata_sources:{},gain_path:state.project.gain_path,gain_flip_y:state.project.gain_flip_y};
  $("edit-title").textContent='Metadata · '+state.project.name; $("edit-fields").innerHTML=metadataFields(row,'edit'); $("edit-error").textContent=''; $("edit-dialog").showModal();
}
function resizeSlurmEditor() {
  const editor=$('slurm-template');
  if(!$('settings-dialog').open)return;
  editor.rows=1;
  const style=getComputedStyle(editor);
  editor.rows=Math.max(1,Math.ceil((editor.scrollHeight-parseFloat(style.paddingTop)-parseFloat(style.paddingBottom))/parseFloat(style.lineHeight)));
}
function showSettings() {
  if(!state.project) return;
  const slurm=state.project.slurm || {};
  $('submission-host').value=slurm.submission_host || '';
  $('slurm-template').value=slurm.template || '#!/usr/bin/env bash\n#SBATCH --job-name={{job_name}}\n#SBATCH --output={{log_path}}\n\n{{environment}}\ncd "{{work_dir}}"\n{{command}}\n';
  $('slurm-warning').textContent=(slurm.warnings || []).join('\n');
  $('settings-error').textContent='';
  $('settings-dialog').showModal();
  resizeSlurmEditor();
}
