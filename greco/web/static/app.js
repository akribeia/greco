"use strict";

function saveProject(project) {
  state.project=project;
  if (!project.series.some(row=>row.id===state.selected)) state.selected=project.series[0]?.id || null;
  if (!project.stages.find(s=>s.id===state.stage)?.enabled) state.stage="tilts";
  rememberProject(project);
  render();
}
function render() {
  rememberReconstructionView();
  rememberProjectView();
  clearTimeout(state.projectStatusTimer);
  clearTimeout(state.preprocessTimer);
  clearTimeout(state.reconstructionTimer);
  clearTimeout(state.refinementTimer);
  clearTimeout(state.angleRefinementTimer);
  clearTimeout(state.curationTimer);
  clearTimeout(state.manualRefinementTimer);
  clearTimeout(state.contrastTimer);
  clearTimeout(state.pickingTimer);
  clearTimeout(state.pickerSourceTimer);
  clearTimeout(state.staTimer);
  const p=state.project, series=p?.series || [];
  $("settings-open").disabled=!p;
  $("project-name").textContent=p?.name || "Select project";
  $("project-location").textContent=p?.path || "Server folder";
  $("project-location").title=p?.path || "";
  $("breadcrumb-project").textContent=p?.name || "No project";
  $("series-count").textContent=series.length;
  $("project-save-status").textContent=p ? "Settings saved in project.json" : "Choose a project to get started";
  renderSidebar();
  renderStages();
  if (state.stage==="tilts") renderTilts(series); else renderNext();
  if(!['tilts','picking'].includes(state.stage) && projectHasActiveJobs(p))state.projectStatusTimer=setTimeout(pollProjectStatus,5000);
}

function startGreco() {
window.addEventListener('pagehide',rememberProjectView);
document.addEventListener('click', async event=>{
  const helpSummary=event.target.closest('.defocus-hand-help>summary');
  if(helpSummary){
    event.preventDefault();const help=helpSummary.parentElement;help.open=!help.open;
    try{localStorage.setItem('greco-defocus-help-open',help.open?'1':'0');}catch{}
    return;
  }
  const enabled=event.target.closest('[data-series-enabled]');if(enabled){toggleSeriesEnabled(enabled,event.shiftKey);return;}
  const gainFlip=event.target.closest('[data-gain-flip]');
  if(gainFlip){
    const pressed=gainFlip.getAttribute('aria-pressed')!=='true';gainFlip.setAttribute('aria-pressed',String(pressed));
    if(['source-gain','common-gain'].includes(gainFlip.dataset.gainFlip) && state.sources)state.sources.gain_flip_y=pressed;
    return;
  }
  const clear=event.target.closest('[data-clear]'); if(clear){$(clear.dataset.clear).value='';return;}
  const template=event.target.closest('[data-template-file]'); if(template){const target=$(state.browse.target);target.value=template.dataset.templateFile;$("browse-dialog").close();target.dispatchEvent(new Event('change',{bubbles:true}));return;}
  const gain=event.target.closest('[data-gain-file]'); if(gain){$(state.browse.target).value=gain.dataset.gainFile;$("browse-dialog").close();return;}
  const browse=event.target.closest('[data-browse]'); if(browse){browseFor(browse.dataset.browse,browse.dataset.kind || 'folder');return;}
  const folder=event.target.closest('[data-folder]'); if(folder){try{await browseLoad(folder.dataset.folder);}catch(error){$("browse-error").textContent=error.message;}return;}
  const select=event.target.closest('[data-select]'); if(select){state.selected=select.dataset.select;render();return;}
  const stage=event.target.closest('[data-stage]'); if(stage&&!stage.disabled){state.stage=stage.dataset.stage;render();return;}
  const edit=event.target.closest('[data-edit]'); if(edit){showEdit(edit.dataset.edit);return;}
  const action=event.target.closest('[data-action]'); if(action){action.dataset.action==='import'?showImport():showProject();}
});
document.addEventListener('change',event=>{
  if(event.target.id==='project-pick-assembly'){selectProjectPickAssembly(event.target);return;}
  if(event.target.dataset.preprocess){state.preprocessOptions[event.target.dataset.preprocess]=event.target.type==='checkbox'?event.target.checked:(event.target.value.trim()===''?null:Number(event.target.value));}
  if(event.target.dataset.selectAll){document.querySelectorAll(`[data-file="${event.target.dataset.selectAll}"]`).forEach(input=>input.checked=event.target.checked);}
});
startGrecoUpdates();
$('settings-open').addEventListener('click',showSettings);
$('settings-cancel').addEventListener('click',()=>$('settings-dialog').close());
$('slurm-template').addEventListener('input',resizeSlurmEditor);
window.addEventListener('resize',resizeSlurmEditor);
$('settings-form').addEventListener('submit',async event=>{
  event.preventDefault(); event.submitter.disabled=true; $('settings-error').textContent='';
  try {
    const project=await api('/api/slurm-settings',{project_id:state.project.id,template:$('slurm-template').value,submission_host:$('submission-host').value.trim()});
    saveProject(project);
    $('slurm-template').value=project.slurm.template;
    resizeSlurmEditor();
    $('slurm-warning').textContent=project.slurm.warnings.join('\n');
    notify(project.slurm.ready?'Slurm template saved for all your projects.':'Saved draft: review cluster option values.');
  } catch(error) { $('settings-error').textContent=error.message; }
  finally { event.submitter.disabled=false; }
});
$("project-switch").addEventListener('click',toggleProjectMenu);
document.addEventListener('click',event=>{
  const recent=event.target.closest('[data-recent-project]');
  if(recent){openRecentProject(recent.dataset.recentProject);return;}
  if(event.target.closest('[data-open-project]')){showProject();return;}
  if(!event.target.closest('.project-selector'))closeProjectMenu();
});
document.addEventListener('keydown',event=>{if(event.key==='Escape')closeProjectMenu();});
$("project-form").addEventListener('submit',openProject);
$("project-cancel").addEventListener('click',()=>$("project-dialog").close());
$("project-dialog").addEventListener('cancel',event=>{if(!state.project)event.preventDefault();});
$("series-search").addEventListener('input',renderSidebar);
$("browse-cancel").addEventListener('click',()=>$("browse-dialog").close());
$("browse-path-form").addEventListener('submit',async event=>{event.preventDefault();$("browse-error").textContent='';try{await browseLoad($("browse-location").value.trim());}catch(error){$("browse-error").textContent=error.message;}});
$("browse-up").addEventListener('click',async()=>{try{await browseLoad(state.browse.parent);}catch(error){$("browse-error").textContent=error.message;}});
$("browse-select").addEventListener('click',()=>{const path=$("browse-location").value.trim();if(path){$(state.browse.target).value=path;$("browse-dialog").close();}});
$("import-close").addEventListener('click',()=>$("import-dialog").close());
$("import-next").addEventListener('click',nextImport);
$("import-back").addEventListener('click',()=>{if(state.step>0){state.step--;$("import-error").textContent='';renderImport();}});
$("edit-cancel").addEventListener('click',()=>$("edit-dialog").close());
$("edit-form").addEventListener('submit',async event=>{event.preventDefault();event.submitter.disabled=true;try{saveProject(await api('/api/project-metadata',{project_id:state.project.id,metadata:collectMetadata($("edit-fields")),gain_path:$("edit-gain").value || null,gain_flip_y:gainFlipY("edit-gain")}));$("edit-dialog").close();notify('Project metadata saved.');}catch(error){$("edit-error").textContent=error.message;}finally{event.submitter.disabled=false;}});
render();
api('/api/settings').then(settings=>{
  serverRecentProjects=Array.isArray(settings.recent_projects)?settings.recent_projects:[];
  if(!$("project-path").value)$("project-path").placeholder=settings.browse_directory+'/my_project';
  return restoreProject();
}).catch(error=>{showProject();$("project-error").textContent=error.message;});
}

startGreco();
