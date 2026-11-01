"use strict";

function requestViewerImage(image,url,onload=null) {
  if(!image?.isConnected)return;
  const queue=image.grecoImageQueue ??= {active:null,pending:null};
  if(queue.active?.url===url){queue.active.onload=onload;queue.pending=null;return;}
  queue.pending={url,onload};
  const next=()=>{
    if(queue.active || !queue.pending || !image.isConnected)return;
    const frame=queue.pending;queue.pending=null;
    if(image.getAttribute('src')===frame.url && image.complete && image.naturalWidth){frame.onload?.();return;}
    queue.active=frame;
    const finished=event=>{
      image.removeEventListener('load',finished);image.removeEventListener('error',finished);
      queue.active=null;
      try{if(image.isConnected && event.type==='load')frame.onload?.();}
      finally{next();}
    };
    image.addEventListener('load',finished);image.addEventListener('error',finished);
    image.decoding='async';image.src=frame.url;
  };
  next();
}
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const state = {project:null, selected:null, stage:"tilts", step:0, scan:null, preview:null, browse:null, edit:null, volumeAutoContrast:true, geometryAutoContrast:true};
const fields = [
  ["pixel_size_A", "Pixel size", "Å / pixel", true], ["voltage_kv", "Voltage", "kV", true],
  ["dose_per_tilt", "Dose per tilt", "e⁻ / Å² · enter dose for the complete movie", true],
  ["cs_mm", "Spherical aberration · Cs", "mm", true], ["amplitude_contrast", "Amplitude contrast", "fraction · 0–1", true],
  ["tilt_axis_deg", "Tilt axis", "° · required; unmodified TiltAxisAngle from the mdoc", true]
];
const grecoWaitingRequests=new Set();
let grecoWaitingDialog=null;
function beginGrecoWaiting(path,body){
  const count=Array.isArray(body?.series_ids)?body.series_ids.length:0;
  const title=path.includes('submit') || path.includes('queue')?'Submitting jobs':'Please wait';
  const message=title==='Submitting jobs'?`Preparing inputs and submitting${count?' '+count+' job'+(count===1?'':'s'):''} to Slurm…`:
    path.includes('delete')?'Checking dependencies and deleting the selected job…':
    path==='/api/open'?'Opening project…':'Applying your changes…';
  const request={title,message,visible:false};grecoWaitingRequests.add(request);
  const update=()=>{
    const visible=[...grecoWaitingRequests].find(item=>item.visible);
    if(!visible){if(grecoWaitingDialog?.open)grecoWaitingDialog.close();return;}
    if(!grecoWaitingDialog){
      grecoWaitingDialog=document.createElement('dialog');grecoWaitingDialog.className='greco-waiting-dialog';
      grecoWaitingDialog.setAttribute('aria-labelledby','greco-waiting-title');
      grecoWaitingDialog.innerHTML='<span class="greco-waiting-spinner" aria-hidden="true"></span><div role="status" aria-live="polite"><h3 id="greco-waiting-title"></h3><p></p></div>';
      grecoWaitingDialog.addEventListener('cancel',event=>event.preventDefault());document.body.append(grecoWaitingDialog);
    }
    grecoWaitingDialog.querySelector('h3').textContent=visible.title;
    grecoWaitingDialog.querySelector('p').textContent=visible.message;
    if(!grecoWaitingDialog.open)grecoWaitingDialog.showModal();
  };
  const timer=setTimeout(()=>{request.visible=true;update();},1000);
  return ()=>{clearTimeout(timer);grecoWaitingRequests.delete(request);update();};
}
async function api(path, body) {
  const finished=body===undefined?()=>{}:beginGrecoWaiting(path,body);
  try{
    const response = await fetch(path, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    let data;
    try { data = await response.json(); } catch { throw new Error("Invalid server response."); }
    if (!response.ok) throw new Error(response.status===404 && (path.startsWith('/api/queue-') || path==='/api/submit-all-queued' || path==='/api/clean-queued' || path==='/api/submit-reconstruction' || path.startsWith('/api/reconstruction-status') || path==='/api/orientation' || path==='/api/slurm-settings' || path==='/api/submit-preprocessing' || path.startsWith('/api/preprocessing-status')) ? 'The running server does not have the latest processing routes yet. Restart the Greco web server, then reload this page. Your edits remain in the form.' : data.error || "The operation could not be completed.");
    return data;
  }finally{finished();}
}

function notify(message) { $("toast").textContent=message; $("toast").classList.remove("hidden"); clearTimeout(notify.timer); notify.timer=setTimeout(()=>$("toast").classList.add("hidden"),5000); }
function angle(value) { return Number(value).toFixed(1)+"°"; }
async function requireCTFCoordinateProtocol() {
  let capabilities;
  try {capabilities=await api('/api/preprocessing-capabilities');}
  catch {throw new Error('Restart the Greco server and reload this page before submitting: the CTF coordinate protocol changed. No job was submitted.');}
  if(capabilities.ctf_axis_protocol!=='ctf-detector-frame-v1')throw new Error('Restart the Greco server before submitting; its CTF coordinate protocol is outdated. No job was submitted.');
  if(capabilities.explicit_ctf_handedness!==true)throw new Error('Restart Greco and reload this page to enable explicit CTF sign selection. No job was submitted.');
}
function gainControl(path, id, flipY=false) {
  return `<div class="metadata-field gain-field"><label for="${esc(id)}">Gain reference · optional</label><div class="path-control"><input id="${esc(id)}" value="${esc(path)}" placeholder="No gain" autocomplete="off"><button type="button" class="button secondary" data-browse="${esc(id)}" data-kind="gain">Choose file</button><button type="button" class="button secondary" id="${esc(id)}-flip-y" data-gain-flip="${esc(id)}" aria-pressed="${flipY}" title="Flip only the gain reference along Y">Flip Y</button></div><small>TIFF, .gain or MRC. Leave blank if gain correction is not needed.</small></div>`;
}
function selectedSeries() { return state.project?.series.find(row=>row.id===state.selected); }
function preprocessingState(job,row) {
  const redo=row?.ctf_redo;
  return ['queued','submitting','submitted','running','unknown'].includes(redo?.state)?
    {...job,state:redo.state,ctf_redo_active:true,ctf_redo_job_id:redo.job_id}:job;
}
function jobStateLabel(value,queued=false){const label=queued || value==='queued'?'Queued':value || 'Pending';return esc(label.charAt(0).toUpperCase()+label.slice(1));}
function projectHasActiveJobs(project){
  const busy=job=>['submitted','running','submitting','unknown'].includes(job?.state);
  const rowBusy=row=>['preprocessing','reconstruction','refinement','angle_refinement','geometry_refinement','picking','handedness_test','ctf_redo'].some(key=>busy(row[key])) || (row.processing_steps || []).some(step=>busy(step.job));
  return (project?.preprocessing_assemblies || []).some(record=>record.id!==project.active_preprocessing_assembly && Object.values(record.series || {}).some(rowBusy)) ||
    (project?.series || []).some(rowBusy) ||
    (project?.reconstruction_assemblies || []).some(record=>record.id!==project.active_reconstruction_assembly && Object.values(record.series || {}).some(rowBusy)) ||
    (project?.pick_assemblies || []).some(record=>record.id!==project.active_pick_assembly && Object.values(record.series || {}).some(rowBusy));
}
async function pollProjectStatus(){
  const id=state.project?.id;
  const pickVersion=state.pickSelectionVersion || 0;
  const reconstructionId=state.project?.active_reconstruction_assembly;
  if(!id)return;
  try{const project=await api('/api/project-status?project_id='+encodeURIComponent(id));if(state.project?.id!==id || (state.pickSelectionVersion || 0)!==pickVersion || state.project?.active_reconstruction_assembly!==reconstructionId)return;state.project=project;renderStages();if(state.stage==='tilts')renderTilts(project.series);else if(state.stage!=='picking' && projectHasActiveJobs(project))state.projectStatusTimer=setTimeout(pollProjectStatus,5000);}
  catch(error){if(state.project?.id===id)state.projectStatusTimer=setTimeout(pollProjectStatus,8000);}
}
function reconstructionStateLabel(job){return ['completed','partial'].includes(job?.state)?`Completed${job.active_count!=null && job.total_tilts!=null?' · '+job.active_count+'/'+job.total_tilts:''}`:jobStateLabel(job?.state);}
function queuedRefinementLabels(row){return [['geometry_refinement','Geometry']].filter(([key])=>['submitted','running'].includes(row[key]?.state)).map(([key,label])=>`<small class="badge pending">${label} · ${jobStateLabel(row[key].state)}</small>`).join(' ');}
function jobNeedsSubmission(job){if(['completed','submitting','submitted','running','unknown','queued'].includes(job?.state))return false;return ['pending','failed','cancelled'].includes(job?.state) || (!job?.work && !job?.job_id);}
function submittableSeries(kind){return state.project.series.filter(row=>row.enabled!==false && jobNeedsSubmission(row[kind]) && (kind==='preprocessing' || ['completed','submitted','running'].includes(row[kind==='reconstruction'?'preprocessing':'reconstruction']?.state))).map(row=>row.id);}

function bindSliceWheel(area,slider){
  area.addEventListener('wheel',event=>{
    if(!event.deltaY || event.ctrlKey || event.metaKey || slider.disabled || event.target.closest('button,input,select,textarea,.volume-contrast'))return;
    event.preventDefault();
    event.stopPropagation();
    const next=Math.max(Number(slider.min),Math.min(Number(slider.max),Number(slider.value)+(event.deltaY>0?1:-1)));
    if(next===Number(slider.value))return;
    slider.value=next;
    slider.dispatchEvent(new Event('input',{bubbles:true}));
  },{passive:false});
}

function gainFlipY(id) { return $(id+'-flip-y')?.getAttribute('aria-pressed')==='true'; }
