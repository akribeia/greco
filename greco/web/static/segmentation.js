function segmentationMembranesReady(member){
  const parent=[...member.steps].reverse().find(s=>s.work && ['reconstruction','geometry_refinement','denoise'].includes(s.kind));
  return !!(parent?.kind==='denoise' && parent.state==='completed' && parent.volumes?.original);
}
function segmentationVolumeChoices(done,step){
  const history=done.filter(s=>s.number<=step.number);
  const bases=history.filter(s=>s.kind!=='denoise');
  const original=bases[0],refined=bases.at(-1);
  const denoised=base=>{
    if(!base)return null;
    const next=bases.find(s=>s.number>base.number);
    return history.filter(s=>s.kind==='denoise' && s.number>base.number && (!next || s.number<next.number)).at(-1) || null;
  };
  return {original,refined,denoised};
}

/* Independent segmentation workspaces; tomography sources remain immutable. */
function segmentationLabel(kind){return {reconstruction:'Reconstruct',geometry_refinement:'Refine geometry',denoise:'Denoise tomogram',membranes:'Segment membranes',insert_particle:'Insert particle'}[kind] || kind;}
function segmentationStaSources(member,tool='geometry_refinement'){
  const sources=new Map();
  const kinds=tool==='insert_particle'?['refinement3d','postprocessing']:['refinement3d','polishing'];
  for(const assembly of state.project.particle_assemblies || []){
    if(!assembly.tomograms.some(t=>t.series_id===member.series_id))continue;
    const jobs=[...(assembly.job_history || []),...kinds.map(kind=>({...assembly[kind],kind}))];
    for(const job of jobs){
      if(!kinds.includes(job.kind) || job.state!=='completed' || !job.work)continue;
      const value=assembly.id+':'+job.kind+':'+job.work.split('/').pop();
      sources.set(value,{value,assemblyId:assembly.id,assemblyName:assembly.name,label:particleJobName(job,job.kind),number:particleJobNumber(job,0)});
    }
  }
  return [...sources.values()].sort((a,b)=>a.number-b.number);
}
function segmentationStaSelectors(sources,selected){
  const assemblies=[...new Map(sources.map(s=>[s.assemblyId,s.assemblyName]))];
  const source=sources.find(s=>s.value===selected) || sources.at(-1);
  const jobs=sources.filter(s=>s.assemblyId===source?.assemblyId);
  return `<label class="metadata-field"><span>STA project</span><select data-segmentation-sta-project ${assemblies.length?'':'disabled'}>${assemblies.length?assemblies.map(([id,name])=>`<option value="${esc(id)}" ${id===source?.assemblyId?'selected':''}>${esc(name)}</option>`).join(''):'<option>No available STA assemblies</option>'}</select></label><label class="metadata-field"><span>Job</span><select data-segmentation-sta-source ${jobs.length?'':'disabled'}>${jobs.length?jobs.map(s=>`<option value="${esc(s.value)}" ${s.value===source?.value?'selected':''}>${esc(s.label)}</option>`).join(''):'<option value="">No compatible completed STA jobs</option>'}</select></label>`;
}
function segmentationHsvRgb(h,s,v){
  const f=n=>{const k=(n+h/60)%6;return Math.round(255*v*(1-s*Math.max(0,Math.min(k,4-k,1))));};
  return [f(5),f(3),f(1)];
}
function segmentationHexHsv(hex){
  const [r,g,b]=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255),v=Math.max(r,g,b),lo=Math.min(r,g,b),d=v-lo;
  const h=d===0?0:((v===r?(g-b)/d:v===g?(b-r)/d+2:(r-g)/d+4)*60+360)%360;
  return {h,s:v===0?0:d/v,v};
}
function bindSegmentationColorWheel(section,input,key){
  const canvas=section.querySelector('[data-segmentation-wheel]'),ctx=canvas.getContext('2d');
  const marker=section.querySelector('[data-segmentation-wheel-marker]');
  const hex=section.querySelector('[data-segmentation-color-hex]');
  const trigger=section.querySelector('[data-segmentation-color-toggle]');
  const popup=section.querySelector('.segmentation-palette');
  let popupAbort;
  const positionPopup=()=>{
    const rect=trigger.getBoundingClientRect(),size=popup.getBoundingClientRect();
    popup.style.left=Math.max(8,Math.min(rect.left,window.innerWidth-size.width-8))+'px';
    popup.style.top=Math.max(8,rect.bottom+8+size.height>window.innerHeight-8?rect.top-size.height-8:rect.bottom+8)+'px';
  };
  const popupChanged=open=>{
    trigger.setAttribute('aria-expanded',String(open));popupAbort?.abort();
    if(!open)return;
    state.segmentationPopoverAbort?.abort();
    popupAbort=new AbortController();state.segmentationPopoverAbort=popupAbort;
    positionPopup();
    window.addEventListener('resize',positionPopup,{signal:popupAbort.signal});
    window.addEventListener('scroll',positionPopup,{capture:true,passive:true,signal:popupAbort.signal});
    if(!popup.showPopover){
      const close=()=>{popup.removeAttribute('data-open');popupChanged(false);};
      document.addEventListener('pointerdown',e=>{if(!popup.contains(e.target) && !trigger.contains(e.target))close();},{signal:popupAbort.signal});
      document.addEventListener('keydown',e=>{if(e.key==='Escape')close();},{signal:popupAbort.signal});
    }
  };
  trigger.onclick=()=>{
    if(popup.togglePopover)popup.togglePopover();
    else{const open=popup.dataset.open!=='true';popup.dataset.open=String(open);popupChanged(open);}
  };
  popup.ontoggle=e=>popupChanged(e.newState==='open');
  let hsv=segmentationHexHsv(input.value);
  const paint=()=>{
    const n=canvas.width,key=n+':'+hsv.v;
    const cache=bindSegmentationColorWheel.frames ??= new Map();
    let frame=cache.get(key);
    if(!frame){
      const r=n/2-4;frame=ctx.createImageData(n,n);
      for(let y=0;y<n;y++)for(let x=0;x<n;x++){
        const dx=x+.5-n/2,dy=y+.5-n/2,s=Math.hypot(dx,dy)/r;
        if(s>1)continue;
        const h=(Math.atan2(dy,dx)*180/Math.PI+360)%360,i=4*(y*n+x);
        frame.data.set(segmentationHsvRgb(h,s,hsv.v),i);frame.data[i+3]=255;
      }
      if(cache.size>=8)cache.delete(cache.keys().next().value);
      cache.set(key,frame);
    }
    ctx.putImageData(frame,0,0);
  };
  const update=(repaint=false)=>{
    input.value='#'+segmentationHsvRgb(hsv.h,hsv.s,hsv.v).map(v=>v.toString(16).padStart(2,'0')).join('');
    state.segmentationColors ??= {};
    if(state.segmentationColors[key]!==input.value){
      state.segmentationColors[key]=input.value;
      try{localStorage.setItem('greco-segmentation-colors-'+state.project.id,JSON.stringify(state.segmentationColors));}catch{}
    }
    section.querySelector('[data-segmentation-color-preview]').style.background=input.value;
    hex.value=input.value.toUpperCase();
    marker.style.left=(50+48.333*hsv.s*Math.cos(hsv.h*Math.PI/180))+'%';
    marker.style.top=(50+48.333*hsv.s*Math.sin(hsv.h*Math.PI/180))+'%';
    canvas.setAttribute('aria-label','Hue and saturation: '+input.value+'. Arrow keys change hue and saturation.');
    if(repaint)paint();
  };
  const pick=e=>{
    const rect=canvas.getBoundingClientRect(),dx=(e.clientX-rect.left)/rect.width-.5,dy=(e.clientY-rect.top)/rect.height-.5;
    hsv.h=(Math.atan2(dy,dx)*180/Math.PI+360)%360;hsv.s=Math.min(1,Math.hypot(dx,dy)/.483333);update();
  };
  canvas.onpointerdown=e=>{e.preventDefault();canvas.setPointerCapture(e.pointerId);pick(e);};
  canvas.onpointermove=e=>{if(canvas.hasPointerCapture(e.pointerId))pick(e);};
  canvas.onpointerup=e=>{if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);};
  canvas.onkeydown=e=>{
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;
    e.preventDefault();
    if(e.key==='ArrowLeft' || e.key==='ArrowRight')hsv.h=(hsv.h+(e.key==='ArrowLeft'?-2:2)+360)%360;
    else hsv.s=Math.max(0,Math.min(1,hsv.s+(e.key==='ArrowUp'?.02:-.02)));
    update();
  };
  hex.onchange=()=>{if(/^#[0-9a-fA-F]{6}$/.test(hex.value)){hsv=segmentationHexHsv(hex.value);update(true);}else hex.value=input.value.toUpperCase();};
  input.oninput=()=>{hsv=segmentationHexHsv(input.value);update(true);};
  update(true);
}
function segmentationColorControl(color,label){
  return `<div class="metadata-field"><span>Color</span><input type="hidden" data-segmentation-color-input value="${esc(color)}"><div class="segmentation-color-picker"><button type="button" data-segmentation-color-toggle aria-label="Choose ${esc(label)} color" aria-expanded="false"><span data-segmentation-color-preview></span></button><div class="segmentation-palette" popover="auto" role="dialog" aria-label="${esc(label)} color"><div class="segmentation-color-disk"><canvas width="240" height="240" data-segmentation-wheel tabindex="0" role="group" aria-label="Hue and saturation"></canvas><span data-segmentation-wheel-marker></span></div><div class="segmentation-color-choice"><input type="text" maxlength="7" pattern="#[0-9a-fA-F]{6}" data-segmentation-color-hex aria-label="Hex color"></div></div></div></div>`;
}
function segmentationToolContent(kind,sources,selected,key){
  if(kind==='membranes')return segmentationColorControl(state.segmentationColors?.[key+':membranes'] || '#f472b6','membrane')+'<p class="hint">MemBrain-seg · quantized weights · batch size 8 · 50% overlap · 8 mirror views. Requires TinyCARE denoising for the current geometry. Adds a colored membrane layer; the tomogram is unchanged.</p>';
  if(kind==='geometry_refinement')return segmentationStaSelectors(sources,selected)+'<p class="hint">Use this tomogram’s refined particle coordinates as landmarks. Joint rigid tilt geometry + static XYZ.</p>';
  if(kind==='insert_particle'){
    const color=state.segmentationColors?.[key] || '#38bdf8';
    return segmentationStaSelectors(sources,selected)+segmentationColorControl(color,'particle')+'<p class="hint">Insert the refined density at each particle’s XYZ and Euler orientation. Adds an independent colored layer.</p>';
  }
  if(kind==='denoise'){
    const models=(state.project.segmentations || []).flatMap(a=>a.tomograms.flatMap(t=>t.steps.filter(s=>s.state==='completed' && s.tinycare_model).map(s=>({work:s.work,label:a.name+' · '+t.name+' · Denoise '+String(s.number).padStart(3,'0')}))));
    return `<div class="metadata-field"><span>Model</span><strong>TinyCARE</strong></div><label class="metadata-field" data-segmentation-tinycare-options><span>TinyCARE training</span><select data-segmentation-tinycare-model><option value="">Train on this tomogram (even/odd)</option>${models.map(m=>`<option value="${esc(m.work)}" ${state.segmentationTinycareModels?.[key]===m.work?'selected':''}>Reuse: ${esc(m.label)}</option>`).join('')}</select><small class="hint">3D U-Net 12×2 · 3000 steps · GPU. First run reconstructs even/odd with the current geometry; reuse requires matching pixel size.</small></label>`;
  }
  return '';
}
function bindSegmentationLayerName(input,segmentationId,seriesId){
  const pid=state.project.id,number=Number(input.dataset.segmentationLayerName);
  let saved=input.value;
  input.onkeydown=event=>{
    if(event.key==='Enter'){event.preventDefault();input.blur();}
    else if(event.key==='Escape'){event.preventDefault();input.value=saved;input.blur();}
  };
  input.onblur=async()=>{
    const name=input.value.trim();
    if(!name || name===saved){input.value=saved;return;}
    input.disabled=true;state.segmentationRenaming=true;
    state.segmentationNameVersion=(state.segmentationNameVersion || 0)+1;
    try{
      const result=await api('/api/segmentation-layer-name',{project_id:pid,segmentation_id:segmentationId,series_id:seriesId,number,name});
      if(state.project?.id!==pid)return;
      const layer=state.project.segmentations?.find(s=>s.id===segmentationId)?.tomograms.find(t=>t.series_id===seriesId)?.steps.find(s=>s.number===number);
      if(layer)layer.name=result.name;
      saved=result.name;input.value=saved;
    }catch(error){input.value=saved;notify(error.message);}
    finally{input.disabled=false;state.segmentationRenaming=false;}
  };
}
function renderSegmentationLayers(section,record,member,active){
  const layers=member.steps.filter(s=>s.kind==='insert_particle');
  const layersByNumber=new Map(layers.map(layer=>[layer.number,layer]));
  section.querySelector('.segmentation-tool-body').insertAdjacentHTML('beforeend',`<div class="segmentation-layer-list">${layers.map((s,index)=>`<div class="segmentation-layer"><span class="segmentation-layer-number">${String(s.layer_number ?? index+1).padStart(3,'0')}</span><span class="segmentation-layer-details"><input type="text" class="segmentation-layer-name" data-segmentation-layer-name="${s.number}" value="${esc(s.name || '')}" placeholder="Name" maxlength="100" aria-label="Layer name" title="Rename layer · Enter to save · Escape to cancel"><small>${esc(s.inserted_particles ?? '—')} particles · ${esc(jobStateLabel(s.state))}</small></span><span class="segmentation-layer-color" data-segmentation-layer-color="${s.number}" title="Layer color"></span><button type="button" class="button secondary" data-clear-particle-layer="${s.number}" ${active?'disabled':''}>Clear</button></div>`).join('')}</div>`);
  section.querySelectorAll('[data-segmentation-layer-color]').forEach(swatch=>{
    const layer=layersByNumber.get(Number(swatch.dataset.segmentationLayerColor));
    swatch.style.backgroundColor=layer.color || layer.options?.color || '#38bdf8';
  });
  section.querySelectorAll('[data-segmentation-layer-name]').forEach(input=>bindSegmentationLayerName(input,record.id,member.series_id));
  section.querySelectorAll('[data-clear-particle-layer]').forEach(button=>button.onclick=()=>clearSegmentationStep(record,member,layersByNumber.get(Number(button.dataset.clearParticleLayer))));
}
function saveSegmentationExpansion(projectId){
  if(state.segmentationProject!==projectId)return;
  try{
    localStorage.setItem('greco-segmentation-open-'+projectId,JSON.stringify([...state.segmentationOpen]));
    localStorage.setItem('greco-segmentation-sections-'+projectId,JSON.stringify(state.segmentationSections));
  }catch{}
}
function segmentationExpanded(){
  if(state.segmentationProject!==state.project.id){
    state.segmentationProject=state.project.id;
    try{
      const colors=JSON.parse(localStorage.getItem('greco-segmentation-colors-'+state.project.id) || '{}');
      state.segmentationColors=Object.fromEntries(Object.entries(colors || {}).filter(([key,value])=>typeof value==='string' && /^#[0-9a-fA-F]{6}$/.test(value)));
    }catch{state.segmentationColors={};}
    for(const record of state.project.segmentations || [])for(const member of record.tomograms){
      for(const kind of ['insert_particle','membranes']){
        const key=record.id+':'+member.series_id+(kind==='membranes'?':membranes':'');
        const job=[...member.steps].reverse().find(step=>step.kind===kind);
        const color=job?.color || job?.options?.color;
        if(!state.segmentationColors[key] && typeof color==='string' && /^#[0-9a-fA-F]{6}$/.test(color))state.segmentationColors[key]=color;
      }
    }
    try{state.segmentationOpen=new Set(JSON.parse(localStorage.getItem('greco-segmentation-open-'+state.project.id) || '[]'));}catch{state.segmentationOpen=new Set();}
    try{
      const sections=JSON.parse(localStorage.getItem('greco-segmentation-sections-'+state.project.id) || '{}');
      state.segmentationSections=sections && typeof sections==='object' && !Array.isArray(sections)?sections:{};
    }catch{state.segmentationSections={};}
  }
  return state.segmentationOpen;
}
function bindSegmentationZoom(image,key){
  const area=image.parentElement;
  state.segmentationZoom ??= {};
  const view=state.segmentationZoom[key] ??= {zoom:1,x:0,y:0};
  area.insertAdjacentHTML('beforeend','<div class="picker-view-tools"><button type="button" class="button secondary" data-segmentation-zoom="in" aria-label="Zoom in" title="Zoom in · drag to move">🔍 +</button><button type="button" class="button secondary" data-segmentation-zoom="out" aria-label="Zoom out" title="Zoom out">🔍 −</button></div>');
  const apply=()=>{
    view.zoom=Math.max(1,Math.min(8,view.zoom));
    const maxX=area.clientWidth*(view.zoom-1)/2,maxY=area.clientHeight*(view.zoom-1)/2;
    view.x=Math.max(-maxX,Math.min(maxX,view.x));view.y=Math.max(-maxY,Math.min(maxY,view.y));
    image.style.transform=`translate(${view.x}px, ${view.y}px) scale(${view.zoom})`;
    area.classList.toggle('segmentation-pan-enabled',view.zoom>1);
    area.querySelector('[data-segmentation-zoom="out"]').disabled=view.zoom===1;
    area.querySelector('[data-segmentation-zoom="in"]').disabled=view.zoom===8;
  };
  area.querySelectorAll('[data-segmentation-zoom]').forEach(button=>button.onclick=()=>{view.zoom*=button.dataset.segmentationZoom==='in'?1.5:1/1.5;apply();});
  let drag=null;
  image.draggable=false;
  area.addEventListener('pointerdown',e=>{
    if(view.zoom<=1 || e.button!==0 || e.target!==image)return;
    e.preventDefault();drag={x:e.clientX,y:e.clientY,px:view.x,py:view.y};area.setPointerCapture(e.pointerId);
  });
  area.addEventListener('pointermove',e=>{if(!drag)return;view.x=drag.px+e.clientX-drag.x;view.y=drag.py+e.clientY-drag.y;apply();});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])area.addEventListener(event,()=>{drag=null;});
  apply();
}
function segmentationToolsHtml(member,key,step,active,staSources){
  return Object.entries({geometry_refinement:'Refine geometry',denoise:'Denoise tomogram',membranes:'Segment membranes',insert_particle:'Insert particle'}).map(([kind,label])=>{
    const sources=staSources(member,kind);
    const disabled=!step || active || (kind==='membranes' && !segmentationMembranesReady(member)) || (['geometry_refinement','insert_particle'].includes(kind) && !sources.length);
    return `<details class="segmentation-tool" data-segmentation-tool="${kind}" ${state.segmentationSections?.[key+':'+kind]?'open':''}>
      <summary>${label}</summary>
      <div class="segmentation-tool-body">
        ${segmentationToolContent(kind,sources,state.segmentationStaSources?.[key+':'+kind],key)}
        ${kind==='membranes' && !segmentationMembranesReady(member)?'<p class="hint">Complete TinyCARE denoising before submitting membranes.</p>':''}
        <div class="segmentation-tool-actions"><button type="button" class="button primary" data-segmentation-add="${kind}" ${disabled?'disabled':''}>${kind==='insert_particle'?'＋ Insert particle':'Submit'}</button></div>
      </div>
    </details>`;
  }).join('');
}
function segmentationViewerHtml(member,binning,step){
  return `<section class="panel tilt-viewer"><div class="panel-header"><span>${esc(member.name)} · bin${binning}</span></div>${step?`<div class="tilt-image-area tomogram-area"><img data-segmentation-image alt="Segmentation tomogram"></div><div class="tilt-scroll"><select data-segmentation-axis aria-label="Slice axis"><option>Z</option><option>Y</option><option>X</option></select><input data-segmentation-slice type="range" min="0" value="${Math.floor(step.shape[0]/2)}" max="${step.shape[0]-1}" aria-label="Tomogram slice"><span data-segmentation-position></span></div>`:'<div class="tilt-image-area tomogram-area"><div class="viewer-empty">The reconstructed tomogram will appear here.</div></div><div class="tilt-scroll"><select disabled aria-label="Slice axis"><option>Z</option></select><input type="range" min="0" max="0" value="0" disabled aria-label="Tomogram slice"><span>No tomogram</span></div>'}</section>`;
}
function segmentationWorkspaceHtml(record,opened,staSources){
  const member=record.tomograms.find(t=>t.series_id===state.selected) || record.tomograms[0];
  const completed=member.steps.filter(s=>s.state==='completed' && s.volumes?.original);
  state.segmentationView ??= {};
  const key=record.id+':'+member.series_id;
  const step=completed.find(s=>s.number===state.segmentationView[key]) || completed[completed.length-1];
  const active=member.steps.some(s=>['submitted','running','unknown'].includes(s.state)) || state.segmentationSubmitting?.[state.project.id+':'+key];
  return `<details class="pick-assembly segmentation-workspace" data-segmentation="${esc(record.id)}" ${opened.has(record.id)?'open':''}><summary><strong>${esc(record.name)}</strong><span class="hint">${record.tomograms.length} tomograms · bin${record.binning}</span><button type="button" class="button secondary assembly-delete-button" data-delete-segmentation="${esc(record.id)}" aria-label="Delete ${esc(record.name)}">Delete</button></summary><div class="assembly-body"><div class="picking-layout segmentation-content"><div class="picker-main segmentation-main">
    ${segmentationViewerHtml(member,record.binning,step)}
    </div><aside class="panel processing-parameters segmentation-sidebar" aria-label="Segmentation tools"><div class="panel-header">Segmentation</div>${segmentationToolsHtml(member,key,step,active,staSources)}</aside><p class="form-error" data-segmentation-error></p></div></div></details>`;
}
function renderSegmentation(){
  state.segmentationPopoverAbort?.abort();
  clearTimeout(state.segmentationTimer);
  const project=state.project,records=project.segmentations || [],opened=segmentationExpanded();
  const sourceCache=new Map();
  const staSources=(member,kind)=>{
    if(!['geometry_refinement','insert_particle'].includes(kind))return [];
    const cacheKey=member.series_id+':'+kind;
    if(!sourceCache.has(cacheKey))sourceCache.set(cacheKey,segmentationStaSources(member,kind));
    return sourceCache.get(cacheKey);
  };
  renderStages();
  $('workspace').innerHTML=`<div class="page-heading"><div><span class="eyebrow">05 / SEGMENTATION</span><h1>Segmentation</h1><p>Independent volumes and downstream segmentation steps.</p></div><button class="button primary" id="new-segmentation">＋ New segmentation</button></div>`+
    (records.length?'<div class="pick-assemblies segmentation-assemblies">'+records.map(record=>{
      return segmentationWorkspaceHtml(record,opened,staSources);
    }).join('')+'</div>':'<section class="panel empty-state"><h3>No segmentations yet</h3><p>Create a workspace and choose the tomograms and binning.</p></section>');
  $('new-segmentation').onclick=newSegmentation;
  document.querySelectorAll('[data-delete-segmentation]').forEach(button=>button.onclick=event=>{
    event.preventDefault();event.stopPropagation();confirmSegmentationDeletion(button.dataset.deleteSegmentation);
  });
  document.querySelectorAll('[data-segmentation]').forEach(panel=>{
    const record=records.find(a=>a.id===panel.dataset.segmentation);
    const member=record.tomograms.find(t=>t.series_id===state.selected) || record.tomograms[0];
    const key=record.id+':'+member.series_id;
    const active=member.steps.some(s=>['submitted','running','unknown'].includes(s.state));
    panel.ontoggle=()=>{
      if(state.segmentationProject!==project.id)return;
      if(panel.open===opened.has(record.id))return;
      panel.open?opened.add(record.id):opened.delete(record.id);
      saveSegmentationExpansion(project.id);
    };
    panel.querySelectorAll('[data-segmentation-tool]').forEach(section=>{
      const kind=section.dataset.segmentationTool;
      section.ontoggle=()=>{
        if(state.segmentationProject!==project.id)return;
        state.segmentationSections ??= {};
        const sectionKey=key+':'+section.dataset.segmentationTool;
        if(!!state.segmentationSections[sectionKey]===section.open)return;
        state.segmentationSections[sectionKey]=section.open;
        saveSegmentationExpansion(project.id);
      };
      const job=[...member.steps].reverse().find(s=>s.kind===section.dataset.segmentationTool);
      if(job){
        section.querySelector('summary').insertAdjacentHTML('beforeend',`<span class="badge ${job.state==='completed'?'':'pending'}">${esc(jobStateLabel(job.state))}</span>`);
        section.querySelector('.segmentation-tool-body').insertAdjacentHTML('afterbegin',`<p class="hint segmentation-job-status" role="status">${esc(segmentationLabel(job.kind))} ${String(job.number).padStart(3,'0')}${job.job_id?' · Slurm '+esc(job.job_id):''}</p>${job.error?`<p class="form-error">${esc(job.error)}</p>`:''}`);
        if(job.progress){
          const progress=job.progress;
          const epoch=Number.isFinite(progress.epoch)?' · '+esc(progress.epoch)+' / '+esc(progress.epochs):'';
          const loss=Number.isFinite(progress.val_loss)?' · validation MSE '+esc(progress.val_loss.toFixed(4)):'';
          section.querySelector('.segmentation-job-status').insertAdjacentHTML('beforeend',`<br>${esc(progress.phase)}${epoch}${loss}`);
        }
        if(job.kind==='membranes' && job.state==='completed')section.querySelector('.segmentation-job-status').insertAdjacentHTML('beforeend',`<br>${esc(Number(job.membrane_voxels || 0).toLocaleString())} membrane voxels · ${esc((100*(job.membrane_fraction || 0)).toFixed(2))}%`);
        const submit=section.querySelector('[data-segmentation-add]');
        if(kind!=='insert_particle' && job.work){submit.disabled=true;submit.title='Clear this job before submitting another.';}
        const actions=submit.closest('.segmentation-tool-actions');
        actions.insertAdjacentHTML('beforeend',`<button type="button" class="button secondary" data-segmentation-clear ${state.segmentationSubmitting?.[project.id+':'+key] || active?'disabled':''}>Clear</button>`);
        actions.querySelector('[data-segmentation-clear]').onclick=()=>clearSegmentationStep(record,member,job);
      }
      const staSelect=section.querySelector('[data-segmentation-sta-source]');
      const staProject=section.querySelector('[data-segmentation-sta-project]');
      if(staSelect)staSelect.onchange=()=>{state.segmentationStaSources ??= {};state.segmentationStaSources[key+':'+kind]=staSelect.value;};
      if(staProject)staProject.onchange=()=>{
        const jobs=staSources(member,kind).filter(s=>s.assemblyId===staProject.value);
        staSelect.innerHTML=jobs.map(s=>`<option value="${esc(s.value)}">${esc(s.label)}</option>`).join('');
        staSelect.value=jobs.at(-1)?.value || '';staSelect.onchange();
      };
      const colorInput=section.querySelector('[data-segmentation-color-input]');
      if(colorInput){
        bindSegmentationColorWheel(section,colorInput,kind==='membranes'?key+':membranes':key);
        if(kind==='insert_particle')renderSegmentationLayers(section,record,member,active);
      }
      const submit=section.querySelector('[data-segmentation-add]');
      const tinyModel=section.querySelector('[data-segmentation-tinycare-model]');
      if(tinyModel)tinyModel.onchange=()=>{state.segmentationTinycareModels ??= {};state.segmentationTinycareModels[key]=tinyModel.value;};
      submit.onclick=()=>addSegmentationStep(record,member,kind,staSelect?.value,colorInput?.value,tinyModel?.value);
    });
    const image=panel.querySelector('[data-segmentation-image]');
    if(image){
      bindSegmentationZoom(image,project.id+':'+key);
      const done=member.steps.filter(s=>s.state==='completed' && s.volumes?.original);
      const step=done.find(s=>s.number===state.segmentationView[key]) || done[done.length-1];
      const slider=panel.querySelector('[data-segmentation-slice]'),axis=panel.querySelector('[data-segmentation-axis]');
      const choices=segmentationVolumeChoices(done,step);
      const original=choices.original,refined=choices.refined;
      const comparisonKey=project.id+':'+key+':'+step.number;
      const saved=state.segmentationComparison?.[comparisonKey];
      let base=saved?.base===original?.number?original:refined;
      let tinycare=typeof saved==='object'?!!saved.tinycare:step.kind==='denoise';
      if(typeof saved==='number'){
        const previous=done.find(s=>s.number===saved);
        if(previous?.kind!=='denoise' && previous){base=previous;tinycare=false;}
      }
      const selectedVolume=()=>tinycare && choices.denoised(base) || base || step;
      let displayed=selectedVolume();
      const variants=[];
      if(original && refined && original.number!==refined.number){
        variants.push('<button type="button" class="button secondary" data-segmentation-compare="refined">Refined</button><button type="button" class="button secondary" data-segmentation-compare="original">Original</button>');
      }
      if(done.some(s=>s.kind==='denoise' && s.number<=step.number)){
        variants.unshift('<button type="button" class="button secondary" data-segmentation-compare="tinycare" aria-pressed="false" title="Toggle TinyCARE for the selected geometry">TinyCARE</button>');
      }
      if(variants.length)image.parentElement.insertAdjacentHTML('beforeend',`<div class="volume-variants">${variants.join('')}</div>`);
      const layers=member.steps.filter(s=>s.state==='completed' && (s.particle_layer || s.membrane_layer));
      if(layers.length){
        if(!image.parentElement.querySelector('.volume-variants'))image.parentElement.insertAdjacentHTML('beforeend','<div class="volume-variants"></div>');
        image.parentElement.querySelector('.volume-variants').insertAdjacentHTML('beforeend','<button type="button" class="button secondary" data-segmentation-segment>Segment</button>');
      }
      state.segmentationContrast ??= {};
      const contrastKey=project.id+':'+key;
      const contrast=state.segmentationContrast[contrastKey] ??= {automatic:true,value:1,brightness:0};
      image.insertAdjacentHTML('beforebegin',`<details class="volume-contrast picker-contrast-panel"><summary>Contrast</summary><div class="picker-contrast-controls"><div class="contrast-actions"><button type="button" class="button secondary" data-segmentation-autocontrast>Auto contrast</button><button type="button" class="button secondary" data-segmentation-reset-contrast>Reset contrast</button></div><label>Contrast <input data-segmentation-contrast type="range" min="0.25" max="4" step="0.05" value="${contrast.value}" aria-label="Segmentation contrast"><output data-segmentation-contrast-value></output></label><label>Brightness <input data-segmentation-brightness type="range" min="-1" max="1" step="0.02" value="${contrast.brightness}" aria-label="Segmentation brightness"><output data-segmentation-brightness-value></output></label></div></details>`);
      panel.querySelector('.volume-contrast').onwheel=e=>e.stopPropagation();
      const overlayKey=project.id+':'+key;
      const refresh=()=>{
        slider.max=displayed.shape[{Z:0,Y:1,X:2}[axis.value]]-1;
        slider.value=Math.min(Number(slider.value),Number(slider.max));
        const a=axis.value.toLowerCase();panel.querySelector('[data-segmentation-position]').textContent=`${a.toUpperCase()} · ${Number(slider.value)+1} / ${Number(slider.max)+1}`;
        requestViewerImage(image,'/api/segmentation-plane?'+new URLSearchParams({project_id:project.id,segmentation_id:record.id,series_id:member.series_id,number:displayed.number,index:slider.value,axis:a,segment:state.segmentationOverlay?.[overlayKey]?'1':'0',layers:layers.map(s=>s.number).join(','),autocontrast:contrast.automatic?'1':'0',contrast:contrast.value,brightness:contrast.brightness}));
        panel.querySelector('[data-segmentation-segment]')?.classList.toggle('active',!!state.segmentationOverlay?.[overlayKey]);
        panel.querySelectorAll('[data-segmentation-compare]').forEach(button=>{
          const kind=button.dataset.segmentationCompare;
          const available=!!choices.denoised(base);
          const selected=kind==='tinycare'?tinycare && available:kind==='original'?base===original:base===refined;
          if(kind==='tinycare'){
            button.disabled=!(available || choices.denoised(refined));
            button.title=available || choices.denoised(refined)?'Toggle TinyCARE; selects Refined when available':'No TinyCARE result for the selected geometry';
          }
          button.classList.toggle('active',selected);button.setAttribute('aria-pressed',String(selected));
        });
      };
      const contrastSlider=panel.querySelector('[data-segmentation-contrast]');
      const brightnessSlider=panel.querySelector('[data-segmentation-brightness]');
      const updateContrast=()=>{
        panel.querySelector('[data-segmentation-contrast-value]').textContent=contrast.value.toFixed(2)+'×';
        panel.querySelector('[data-segmentation-brightness-value]').textContent=Math.round(contrast.brightness*100)+'%';
        const auto=panel.querySelector('[data-segmentation-autocontrast]');
        auto.classList.toggle('active',contrast.automatic);auto.setAttribute('aria-pressed',String(contrast.automatic));
      };
      const resetContrast=automatic=>{contrast.automatic=automatic;contrast.value=1;contrast.brightness=0;contrastSlider.value=1;brightnessSlider.value=0;updateContrast();refresh();};
      panel.querySelector('[data-segmentation-autocontrast]').onclick=()=>resetContrast(true);
      panel.querySelector('[data-segmentation-reset-contrast]').onclick=()=>resetContrast(false);
      contrastSlider.oninput=()=>{contrast.value=Number(contrastSlider.value);updateContrast();refresh();};
      brightnessSlider.oninput=()=>{contrast.brightness=Number(brightnessSlider.value);updateContrast();refresh();};
      updateContrast();
      const segment=panel.querySelector('[data-segmentation-segment]');
      if(segment)segment.onclick=()=>{state.segmentationOverlay ??= {};state.segmentationOverlay[overlayKey]=!state.segmentationOverlay[overlayKey];refresh();};
      panel.querySelectorAll('[data-segmentation-compare]').forEach(button=>button.onclick=()=>{
        const kind=button.dataset.segmentationCompare;
        if(kind==='tinycare'){
          if(tinycare && choices.denoised(base))tinycare=false;
          else{
            if(choices.denoised(refined))base=refined;
            tinycare=true;
          }
        }
        else base=kind==='original'?original:refined;
        displayed=selectedVolume();
        state.segmentationComparison ??= {};state.segmentationComparison[comparisonKey]={base:base.number,tinycare};refresh();
      });
      slider.oninput=refresh;axis.onchange=()=>{slider.max=step.shape[{Z:0,Y:1,X:2}[axis.value]]-1;slider.value=Math.floor(Number(slider.max)/2);refresh();};refresh();
      image.parentElement.onwheel=e=>{e.preventDefault();slider.value=Math.max(0,Math.min(Number(slider.max),Number(slider.value)+(e.deltaY>0?1:-1)));refresh();};
    }
  });
  if(records.some(a=>a.tomograms.some(t=>t.steps.some(s=>['submitted','running','unknown'].includes(s.state)))))state.segmentationTimer=setTimeout(pollSegmentation,5000);
}
function confirmSegmentationDeletion(id){
  const project=state.project,record=project.segmentations?.find(r=>r.id===id);
  if(!record || state.segmentationDeleting)return;
  if(Object.entries(state.segmentationSubmitting || {}).some(([key,busy])=>busy && key.startsWith(project.id+':'+id+':'))){notify('Wait for the current submission to finish before deleting.');return;}
  const dialog=document.createElement('dialog');dialog.className='assembly-job-dialog';
  dialog.innerHTML=`<form><h3>Delete segmentation</h3><p>${esc(record.name)}</p><p class="hint">Permanently delete this segmentation and its results for all included tomograms. Active jobs will be cancelled. Source reconstructions and STA results are preserved.</p><label class="check-line"><span>I confirm permanent deletion</span><input type="checkbox" required></label><p class="form-error" role="alert"></p><div class="submit-actions"><button type="button" class="button secondary">Cancel</button><button type="submit" class="button primary" disabled>Delete</button></div></form>`;
  document.body.appendChild(dialog);dialog.onclose=()=>dialog.remove();
  const check=dialog.querySelector('input'),submit=dialog.querySelector('[type=submit]'),cancel=dialog.querySelector('[type=button]'),error=dialog.querySelector('.form-error');
  check.onchange=()=>submit.disabled=!check.checked;cancel.onclick=()=>dialog.close();
  dialog.oncancel=event=>{if(dialog.deleting)event.preventDefault();};
  dialog.querySelector('form').onsubmit=async event=>{
    event.preventDefault();if(!check.checked || dialog.deleting)return;
    dialog.deleting=true;state.segmentationDeleting=project.id+':'+id;
    submit.disabled=true;check.disabled=true;cancel.disabled=true;
    try{
      let result;
      do{
        result=await api('/api/segmentation-delete',{project_id:project.id,segmentation_id:id,confirmed:true});
        if(result.deletion_pending){error.textContent='Cancellation requested. Waiting for Slurm to stop before deleting files…';await new Promise(resolve=>setTimeout(resolve,1500));}
      }while(result.deletion_pending);
      dialog.close();
      if(state.project?.id===project.id){
        clearTimeout(state.segmentationTimer);
        state.segmentationOpen?.delete(id);
        for(const key of Object.keys(state.segmentationSections || {}))if(key.startsWith(id+':'))delete state.segmentationSections[key];
        saveSegmentationExpansion(project.id);saveProject(result);
      }
      notify('Segmentation and its results permanently deleted.');
    }catch(e){error.textContent=e.message==='Unknown route.'?'Restart the Greco server and reload this page to enable segmentation deletion.':e.message;}
    finally{state.segmentationDeleting=null;dialog.deleting=false;check.disabled=false;cancel.disabled=false;submit.disabled=!check.checked;}
  };
  dialog.showModal();
}
async function clearSegmentationStep(record,member,job){
  if(!confirm(['insert_particle','membranes'].includes(job.kind)?'Clear this segmentation layer? Other layers and the tomogram will remain unchanged.':`Clear ${segmentationLabel(job.kind)} ${String(job.number).padStart(3,'0')} and any later steps? The previous volume will be restored. Files remain archived.`))return;
  const pid=state.project.id;
  try{
    const result=await api('/api/segmentation-clear',{project_id:pid,segmentation_id:record.id,series_id:member.series_id,number:job.number,confirmed:true});
    if(state.project?.id===pid){state.project=result;delete state.segmentationView?.[record.id+':'+member.series_id];if(state.stage==='segmentation')renderSegmentation();notify('Step cleared. Previous volume restored.');}
  }catch(error){notify(error.message);}
}
async function pollSegmentation(){
  if(state.stage!=='segmentation')return;
  if(Object.keys(state.segmentationSubmitting || {}).some(key=>key.startsWith(state.project.id+':'))){state.segmentationTimer=setTimeout(pollSegmentation,1500);return;}
  const pid=state.project.id,nameVersion=state.segmentationNameVersion || 0,actionVersion=state.segmentationActionVersion || 0;
  try{
    const result=await api('/api/segmentation-status?'+new URLSearchParams({project_id:pid}));
    if(state.project?.id!==pid || state.stage!=='segmentation')return;
    if(actionVersion!==(state.segmentationActionVersion || 0)){state.segmentationTimer=setTimeout(pollSegmentation,1500);return;}
    if(state.segmentationDeleting?.startsWith(pid+':')){state.segmentationTimer=setTimeout(pollSegmentation,1500);return;}
    if(Object.keys(state.segmentationSubmitting || {}).some(key=>key.startsWith(pid+':')) || state.segmentationRenaming || document.activeElement?.matches('[data-segmentation-layer-name]') || nameVersion!==(state.segmentationNameVersion || 0)){state.segmentationTimer=setTimeout(pollSegmentation,5000);return;}
    const changed=JSON.stringify(state.project.segmentations)!==JSON.stringify(result.segmentations);
    state.project=result;
    if(changed)renderSegmentation();else state.segmentationTimer=setTimeout(pollSegmentation,5000);
  }catch(error){notify(error.message);}
}
function newSegmentation(){
  const project=state.project;let dialog=$('segmentation-create-dialog');
  if(!dialog){dialog=document.createElement('dialog');dialog.id='segmentation-create-dialog';dialog.className='modal';document.body.appendChild(dialog);}
  const rows=project.series.filter(r=>r.enabled!==false && (pickingSourceJob(r)?.state==='completed'));
  dialog.innerHTML=`<form><h3>New segmentation</h3><div class="metadata-grid"><label class="metadata-field"><span>Name</span><input name="name" required maxlength="100" value="Segmentation ${String((project.segmentations || []).length+1).padStart(3,'0')}"></label><label class="metadata-field"><span>Binning</span><select name="binning">${[1,2,4,8,16].map(b=>`<option ${b===8?'selected':''}>${b}</option>`).join('')}</select></label></div><p class="hint">Rebuild with final Wiener CTF correction and Greco WBP, retaining the selected geometry and tomogram thickness. Finer binning may require reading native movies.</p><label class="check-line"><input type="checkbox" data-select-all checked> All available tomograms</label><div class="segmentation-selection">${rows.map(r=>`<label class="check-line"><input type="checkbox" name="series" value="${esc(r.id)}" checked>${esc(r.name)}</label>`).join('') || '<p>No completed reconstructions available.</p>'}</div><p class="form-error"></p><div class="submit-actions"><button type="button" class="button secondary">Cancel</button><button type="submit" class="button primary" ${rows.length?'':'disabled'}>Create &amp; reconstruct</button></div></form>`;
  dialog.querySelector('button[type=button]').onclick=()=>dialog.close();
  dialog.querySelector('[data-select-all]').onchange=e=>dialog.querySelectorAll('[name=series]').forEach(c=>c.checked=e.target.checked);
  dialog.querySelector('form').onsubmit=async e=>{
    e.preventDefault();const button=dialog.querySelector('button[type=submit]');button.disabled=true;button.textContent='Creating…';dialog.querySelector('.form-error').textContent='';
    try{
      const result=await api('/api/segmentation-create',{project_id:project.id,segmentation:{name:dialog.querySelector('[name=name]').value,binning:Number(dialog.querySelector('[name=binning]').value),series_ids:[...dialog.querySelectorAll('[name=series]:checked')].map(c=>c.value)}});
      if(state.project.id!==project.id)return;state.project=result;segmentationExpanded().add(result.segmentations.at(-1).id);saveSegmentationExpansion(project.id);dialog.close();renderSegmentation();
    }catch(error){const message=dialog.querySelector('.form-error');message.textContent=error.message;message.scrollIntoView({block:'nearest'});}finally{button.disabled=false;button.textContent='Create & reconstruct';}
  };dialog.showModal();
}
async function addSegmentationStep(record,member,kind,staSource=null,color='#38bdf8',modelFrom=''){
  const pid=state.project.id;
  const key=pid+':'+record.id+':'+member.series_id;
  state.segmentationSubmitting ??= {};
  if(state.segmentationSubmitting[key])return;
  state.segmentationSubmitting[key]=true;
  state.segmentationActionVersion=(state.segmentationActionVersion || 0)+1;
  const panel=document.querySelector(`[data-segmentation="${record.id}"]`);
  panel?.querySelectorAll('[data-segmentation-add]').forEach(button=>{
    button.disabled=true;
    if(button.dataset.segmentationAdd===kind)button.textContent='Submitting…';
  });
  try{
    const result=await api('/api/segmentation-step',{project_id:pid,segmentation_id:record.id,series_id:member.series_id,kind,options:kind==='denoise'?{method:'tinycare',...(modelFrom?{model_from:modelFrom}:{})}:kind==='geometry_refinement'?{sta_source:staSource}:kind==='insert_particle'?{sta_source:staSource,color}:kind==='membranes'?{color}:{}});
    if(state.project.id===pid){
      state.project=result;
      const job=result.segmentations.find(s=>s.id===record.id)?.tomograms.find(t=>t.series_id===member.series_id)?.steps.at(-1);
      notify(job?.job_id?`${segmentationLabel(kind)} submitted · Slurm ${job.job_id}`:`${segmentationLabel(kind)} ${jobStateLabel(job?.state || 'configured')}`);
    }
  }
  catch(error){notify(error.message);}
  finally{
    delete state.segmentationSubmitting[key];
    state.segmentationActionVersion=(state.segmentationActionVersion || 0)+1;
    if(state.project?.id===pid && state.stage==='segmentation')renderSegmentation();
  }
}
