"use strict";

function manualCurationPanel(row){
  const options=processingSteps(row).find(step=>step.id===state.activeCurationStep)?.job?.options || {};
  return `<details id="manual-curation" class="refinement-dropdown geometry-dropdown"><summary>Manual curation</summary><div class="preprocessing-layout geometry-refinement-layout"><div class="reconstruction-viewer-column"><section class="panel tilt-viewer"><div class="panel-header"><span>${esc(row.name)} · Aligned tilt series</span><span id="curation-plane-label" class="muted"></span></div><div class="tilt-image-area" id="curation-image-area" tabindex="0"><div id="curation-empty" class="viewer-empty">No aligned tilt series yet.</div><img id="manual-reference-image" hidden alt="Fixed neighbouring tilt"><img id="curation-image" hidden alt="Aligned tilt selected for manual review"><canvas id="manual-moving-layer" hidden aria-label="Greyscale movable tilt overlay"></canvas></div><div class="curation-strip-scroll"><div id="curation-tilts" class="curation-tilts" role="group" aria-label="Tilt alignment review"></div></div><div class="tilt-scroll curation-slider-row"><input id="curation-slider" type="range" min="0" max="0" value="0" disabled aria-label="Aligned tilt"><span id="curation-slider-label">No tilts</span></div><p id="curation-tilt-info" class="hint viewer-info"></p></section></div><section class="panel processing-parameters"><div class="panel-header"><span>2D transition diagnostic</span><span id="curation-status" class="badge pending">Not calculated</span></div><div class="reconstruction-details"><p>No reconstruction or landmarks. Gradient cross-correlation compares adjacent images against a robust local angular trend.</p><p><span class="curation-legend green"></span> Smooth transitions<br><span class="curation-legend yellow"></span> Weak evidence or unusual transition<br><span class="curation-legend red"></span> Opposing jumps: inspect this tilt<br><span class="curation-legend gray"></span> Not evaluated / no usable texture</p><p>Parallax and changing specimen appearance can also cause jumps. These are review flags, not probabilities. No shifts are changed and no tilt is automatically excluded.</p></div><div class="manual-refinement-controls"><label class="check-line"><span>Show discarded tilts</span><input id="manual-show-discarded" type="checkbox"></label><p class="hint">Double-click a tilt number to activate or deactivate it. Submit fixes saves the selection.</p><h3>Manual XY refinement</h3><p class="hint">Drag the selected tilt over a fixed neighbour. Fix several positions, then submit one batch. Angles, XYZ and unselected tilts remain fixed.</p><div class="manual-placement-actions"><button id="manual-begin" type="button" class="button secondary" disabled>Manual refine</button><button id="manual-fix" type="button" class="button secondary" disabled>Fix position</button><button id="manual-cancel-placement" type="button" class="button secondary" hidden>Cancel placement</button></div><label id="manual-reference-label" class="metadata-field" hidden><span>Fixed neighbour</span><select id="manual-reference"></select></label><label id="manual-blend-label" class="metadata-field" hidden><span>Reference ↔ Moving tilt · <span id="manual-blend-value">50% moving tilt</span></span><input id="manual-blend" type="range" min="0" max="100" value="50" aria-label="Moving tilt blend"></label><p id="manual-placement-info" class="hint"></p><div id="manual-drafts"></div><label class="metadata-field"><span>Smoothness</span><input id="manual-smoothness" type="number" min="0" max="5" step="0.05" value="${esc(options.smoothness_strength ?? 0)}"><small>Image transition guidance for XY refinement. 0 disables it.</small></label><label class="check-line"><span>Use Gaussian search prior</span><input id="manual-gaussian" type="checkbox" ${options.gaussian_search_prior?'checked':''}></label><label class="metadata-field"><span>Gaussian sigma (bin8 px)</span><input id="manual-gaussian-sigma" type="number" min="0.1" max="128" step="0.1" value="${esc(options.gaussian_sigma_px ?? 8)}"><small>Favours your manual position in the first wide search.</small></label><button id="manual-apply" type="button" class="button primary" disabled>Submit fixes</button><button id="manual-repeat" type="button" class="button secondary" disabled>Refine these tilts again</button><p id="manual-job-status" class="hint"></p><p id="manual-refinement-error" class="form-error"></p></div><p id="curation-info" class="hint viewer-info"></p><p id="curation-error" class="form-error viewer-info"></p><div class="submit-actions curation-diagnostic-actions"><button id="curation-run" type="button" class="button primary">Run 2D diagnostic</button></div>${finalReconstructionOptions('manual',options)}<div class="submit-actions"><button id="manual-submit" type="button" class="button primary" disabled>Submit</button></div></section></div></details>`;
}
function mountManualCuration(){
  if(!$('manual-curation'))return;
  const pid=state.project.id,sid=state.selected;
  const step=processingSteps(selectedSeries()).find(s=>s.id===state.activeCurationStep);
  const pipelineSource=manualPipelineSource(selectedSeries(),step);
  clearTimeout(state.curationTimer);state.curationSource=pipelineSource;state.curationIndex=null;state.curationStatus=null;
  $('manual-curation').addEventListener('toggle',()=>{clearTimeout(state.curationTimer);if($('manual-curation').open)pollManualCuration(pid,sid);});
  $('curation-slider').addEventListener('input',()=>{state.curationIndex=Number($('curation-slider').value);showCurationTilt();});
  const navigateTilts=event=>{
    if(!['ArrowLeft','ArrowRight'].includes(event.key))return;
    const rows=state.curationStatus?.rows || [];if(!rows.length)return;
    event.preventDefault();
    state.curationIndex=Math.max(0,Math.min(rows.length-1,(state.curationIndex ?? 0)+(event.key==='ArrowRight'?1:-1)));
    showCurationTilt();
    const button=$('curation-tilts').querySelector(`[data-curation-tilt="${rows[state.curationIndex].index}"]`);
    if(event.currentTarget===$('curation-tilts'))button?.focus({preventScroll:true});
    button?.scrollIntoView({block:'nearest',inline:'nearest'});
  };
  $('curation-image-area').addEventListener('keydown',navigateTilts);
  $('curation-tilts').addEventListener('keydown',navigateTilts);
  $('curation-run').addEventListener('click',async()=>{
    const button=$('curation-run');button.disabled=true;$('curation-error').textContent='';
    try{await api('/api/curation-test',{project_id:pid,series_id:sid,source:state.curationSource});pollManualCuration(pid,sid);}
    catch(error){if($('curation-error'))$('curation-error').textContent=error.message;if($('curation-run'))button.disabled=false;}
  });
  $('manual-show-discarded').onchange=()=>{
    if(!$('manual-show-discarded').checked && manualDrafts().some(edit=>edit.index>=(state.curationStatus?.accepted_rows?.length ?? Infinity))){$('manual-show-discarded').checked=true;notify('Submit or remove the discarded tilt fixes before hiding them.');return;}
    resetManualOverlay();persistManualDrafts();pollManualCuration(pid,sid);
  };
  mountManualRefinement(pid,sid);pollManualCuration(pid,sid);
}
async function pollManualCuration(pid,sid){
  clearTimeout(state.curationTimer);
  const source=state.curationSource;
  try{
    const report=await api('/api/curation-status?'+new URLSearchParams({project_id:pid,series_id:sid,source}));
    if(state.project?.id!==pid || state.selected!==sid || state.stage!=='reconstruction' || !$('curation-status') || source!==state.curationSource)return;
    const available=Object.keys(report.sources || {});
    const selectedSource=state.curationStatus?.rows?.[state.curationIndex]?.source_index;
    report.accepted_rows=report.rows || [];
    if($('manual-show-discarded').checked)report.rows=[...report.accepted_rows,...(report.discarded_rows || [])].sort((a,b)=>a.source_index-b.source_index);
    state.curationStatus=report;
    if(selectedSource!=null){const position=report.rows.findIndex(row=>row.source_index===selectedSource);state.curationIndex=position<0?null:position;}
    $('curation-status').textContent=report.state;$('curation-run').disabled=report.state==='running' || !report.sources?.[source];
    $('curation-error').textContent=report.error || '';
    const rows=report.rows || [];
    $('curation-tilts').style.setProperty('--curation-tilt-count',Math.max(1,rows.length));
    $('curation-slider').disabled=!rows.length;
    $('curation-slider').max=Math.max(0,rows.length-1);
    $('curation-tilts').innerHTML=rows.map(row=>`<button type="button" class="curation-tilt ${row.color} ${row.discarded?'discarded':''}" data-curation-tilt="${row.index}" title="${esc('Tilt '+row.number+' · '+angle(row.angle_deg)+' · '+row.reason)}" aria-label="${esc('Tilt '+row.number+' '+row.color)}">${row.number}</button>`).join('');
    document.querySelectorAll('[data-curation-tilt]').forEach(button=>button.addEventListener('click',()=>{state.curationIndex=rows.findIndex(row=>row.index===Number(button.dataset.curationTilt));showCurationTilt();}));
    document.querySelectorAll('[data-curation-tilt]').forEach(button=>button.addEventListener('dblclick',event=>{event.preventDefault();toggleManualTilt(rows.find(row=>row.index===Number(button.dataset.curationTilt)));}));
    if(rows.length && state.curationIndex==null)state.curationIndex=Math.floor(rows.length/2);
    const counts={};for(const row of rows)counts[row.color]=(counts[row.color] || 0)+1;
    $('curation-info').textContent=report.state==='completed'?Object.entries(counts).map(([key,n])=>n+' '+key).join(' · ')+` · ${Number(report.runtime_s).toFixed(1)} s`:'Results persist when the page is reloaded. This test runs on CPU, without Slurm or GPU allocation.';
    showCurationTilt();
    updateManualDraftIdentity();
    if(report.state==='running' || (state.activeCurationStep && !available.includes(source)))state.curationTimer=setTimeout(()=>pollManualCuration(pid,sid),report.state==='running'?1500:5000);
  }catch(error){if($('curation-error'))$('curation-error').textContent=error.message;if($('curation-run'))$('curation-run').disabled=false;}
}
function showCurationTilt(){
  const report=state.curationStatus,row=report?.rows?.[state.curationIndex];
  if(!row || !$('curation-image'))return;
  if(state.manualEditing && state.manualEditing.index!==row.index)resetManualOverlay();
  const viewSource=state.curationSource;
  $('curation-plane-label').textContent=`Tilt ${row.number} · ${angle(row.angle_deg)}`;
  document.querySelectorAll('[data-curation-tilt]').forEach(button=>{
    const active=Number(button.dataset.curationTilt)===row.index;
    button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));
  });
  $('curation-slider').value=state.curationIndex;
  $('curation-slider-label').textContent=`Tilt ${row.number} · ${state.curationIndex+1}/${report.rows.length} ${manualRowActive(row)?'active':'inactive'} · ${angle(row.angle_deg)}`;
  const pairs=(report.pairs || []).filter(pair=>pair.left_index===row.index || pair.right_index===row.index);
  $('curation-tilt-info').textContent=[row.reason,row.max_transition_jump_px!=null?'Trend deviation '+row.max_transition_jump_px.toFixed(2)+' px':'',...pairs.map(pair=>`${pair.left_number}–${pair.right_number}: ${pair.measured?'CC '+pair.correlation.toFixed(3)+' · correction ['+pair.correction_yx_px.map(v=>v.toFixed(2)).join(', ')+'] px':'no texture'}`)].filter(Boolean).join(' · ');
  $('curation-empty').hidden=true;const image=$('curation-image');image.hidden=false;
  const imageKey=state.project.id+'|'+state.selected+'|'+viewSource+'|'+report.input_identity+'|'+row.index;
  paintManualOverlay();
  if(image.dataset.imageKey===imageKey)return;
  image.dataset.imageKey=imageKey;
  image.src='/api/curation-plane?'+new URLSearchParams({project_id:state.project.id,series_id:state.selected,source:viewSource,index:row.discarded?row.source_index:row.index,discarded:row.discarded?'1':'0',display:'raw-mc-v1'});
  image.onerror=()=>{if($('curation-error'))$('curation-error').textContent='Could not load the selected aligned tilt.';};
}
