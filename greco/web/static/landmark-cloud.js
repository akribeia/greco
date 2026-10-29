"use strict";

function landmarkCloudPanel(){
  return '<details id="recon-landmark-cloud" class="panel landmark-cloud-panel"><summary class="panel-header"><span>Landmark positions</span><span class="landmark-cloud-count"></span></summary><div class="landmark-cloud-body"><p class="hint">Available after reconstruction.</p></div></details><details id="recon-landmark-densities" class="panel landmark-cloud-panel"><summary class="panel-header"><span>Landmark densities</span><span class="landmark-cloud-count"></span></summary><div class="landmark-cloud-body"><p class="hint">Open to show local tomogram density at the final landmark positions.</p></div></details>';
}
async function loadReconstructionLandmarks(job){
  loadReconstructionLandmarkDensities(job);
  const panel=$('recon-landmark-cloud');if(!panel || !['completed','partial'].includes(job.state))return;
  if(!panel.dataset.cloudBound){
    panel.dataset.cloudBound='1';
    panel.addEventListener('toggle',()=>{if(panel.open && state.reconstructionStatus)loadReconstructionLandmarks(state.reconstructionStatus);});
  }
  if(!panel.open)return;
  const projectId=state.project.id,seriesId=state.selected,assemblyId=state.project.active_reconstruction_assembly;
  const key=JSON.stringify([projectId,seriesId,assemblyId,job.work,job.job_id,job.tilt_series?.revision]);
  if(panel.dataset.cloudKey===key)return;
  panel.dataset.cloudKey=key;
  try{
    const data=await api('/api/reconstruction-landmarks?'+new URLSearchParams({project_id:projectId,series_id:seriesId,assembly_id:assemblyId}));
    if(!panel.isConnected || state.project?.id!==projectId || state.selected!==seriesId || state.project.active_reconstruction_assembly!==assemblyId)return;
    renderLandmarkCloud(panel,data);
  }catch(error){delete panel.dataset.cloudKey;if(panel.isConnected)panel.querySelector('.landmark-cloud-body').textContent=error.message;}
}
function landmarkBoxProject(point){return [point[0]+.52*point[1],-point[2]-.34*point[1]];}
function renderLandmarkCloud(panel,data){
  const points=data.points,half=data.dimensions_dqh.map(n=>n/2),[D,Q,H]=half;
  const corners=[];for(const d of [-D,D])for(const q of [-Q,Q])for(const h of [-H,H])corners.push([d,q,h]);
  const projected=[...corners,...points.map(p=>[p.d,p.q,p.h])].map(landmarkBoxProject);
  const lo=[Math.min(...projected.map(p=>p[0]))-45,Math.min(...projected.map(p=>p[1]))-35];
  const hi=[Math.max(...projected.map(p=>p[0]))+55,Math.max(...projected.map(p=>p[1]))+45];
  const project=p=>landmarkBoxProject(p).map(n=>n.toFixed(2)).join(',');
  const face=vertices=>`<polygon points="${vertices.map(project).join(' ')}"/>`;
  const line=(a,b,front)=>`<line x1="${landmarkBoxProject(a)[0]}" y1="${landmarkBoxProject(a)[1]}" x2="${landmarkBoxProject(b)[0]}" y2="${landmarkBoxProject(b)[1]}" class="${front?'cloud-edge':'cloud-edge cloud-edge-back'}"/>`;
  let back='',front='';
  for(let axis=0;axis<3;axis++){
    const fixed=[0,1,2].filter(i=>i!==axis);
    for(const s of [-1,1])for(const t of [-1,1]){
      const a=[0,0,0];a[fixed[0]]=s*half[fixed[0]];a[fixed[1]]=t*half[fixed[1]];
      const b=a.slice();a[axis]=-half[axis];b[axis]=half[axis];
      const visible=(axis!==1&&a[1]===-Q)||(axis!==0&&a[0]===D)||(axis!==2&&a[2]===H);
      if(visible)front+=line(a,b,true);else back+=line(a,b,false);
    }
  }
  const radius=Math.max(1.8,D/125);
  const dots=points.slice().sort((a,b)=>b.q-a.q).map(p=>{
    const [x,y]=landmarkBoxProject([p.d,p.q,p.h]);
    return `<circle cx="${x}" cy="${y}" r="${radius}" data-landmark="${p.id}" tabindex="0" aria-label="Landmark ${p.id}: d ${p.d.toFixed(1)}, q ${p.q.toFixed(1)}, h ${p.h.toFixed(1)} pixels"><title>Landmark ${p.id} · d ${p.d.toFixed(1)} · q ${p.q.toFixed(1)} · h ${p.h.toFixed(1)} px</title></circle>`;
  }).join('');
  const label=(text,p)=>{const [x,y]=landmarkBoxProject(p);return `<text x="${x}" y="${y}" class="cloud-axis-label">${text}</text>`;};
  const axis=data.tilt_axis_dqh || [0,1,0];
  const axisLength=Math.min(...half.map((v,i)=>Math.abs(axis[i])>1e-10?v/Math.abs(axis[i]):Infinity));
  const axisStart=landmarkBoxProject(axis.map(v=>-v*axisLength)),axisEnd=landmarkBoxProject(axis.map(v=>v*axisLength));
  const tiltAxis=`<g class="cloud-tilt-axis"><title>Original tilt axis through the origin</title><line x1="${axisStart[0]}" y1="${axisStart[1]}" x2="${axisEnd[0]}" y2="${axisEnd[1]}"/><text x="${axisEnd[0]+8}" y="${axisEnd[1]-7}">${data.leveled?'Tilt axis':'Tilt axis · q'}</text></g>`;
  panel.querySelector('.landmark-cloud-count').textContent=points.length+' landmarks';
  const body=panel.querySelector('.landmark-cloud-body');
  body.innerHTML=`<svg class="landmark-cloud-svg" viewBox="${lo[0]} ${lo[1]} ${hi[0]-lo[0]} ${hi[1]-lo[1]}" role="group" aria-label="Final fitted landmark positions in a perspective box"><g class="cloud-faces">${face([[-D,-Q,-H],[D,-Q,-H],[D,-Q,H],[-D,-Q,H]])}${face([[D,-Q,-H],[D,Q,-H],[D,Q,H],[D,-Q,H]])}${face([[-D,-Q,H],[D,-Q,H],[D,Q,H],[-D,Q,H]])}</g>${back}<g class="cloud-points">${dots}</g>${front}${tiltAxis}${label('d',[D+20,-Q,-H])}${label('q',[D+20,Q,-H])}${label('h',[D+20,Q,H])}<g class="cloud-origin"><path d="M-5,0 H5 M0,-5 V5"/></g></svg><div class="landmark-cloud-tip" role="status" hidden></div><p class="landmark-cloud-caption">${data.dimensions_dqh.map(n=>Math.round(n)).join(' × ')} px · bin ${data.binning} · ${data.leveled?'after leveling':'centred internal coordinates'}<br>Hover over a landmark for d, q, h.</p>`;
  const tip=body.querySelector('.landmark-cloud-tip'),lookup=new Map(points.map(p=>[String(p.id),p]));
  const show=target=>{
    const point=lookup.get(target?.dataset?.landmark);if(!point)return;
    tip.textContent=`Landmark ${point.id}   ·   d ${point.d.toFixed(1)}   q ${point.q.toFixed(1)}   h ${point.h.toFixed(1)} px   ·   ${point.views} views`;tip.hidden=false;
  };
  body.onpointerover=event=>show(event.target);
  body.onpointerout=event=>{if(event.target.dataset?.landmark)tip.hidden=true;};
  body.onfocusin=event=>show(event.target);
  body.onfocusout=()=>tip.hidden=true;
}

async function loadReconstructionLandmarkDensities(job){
  const panel=$('recon-landmark-densities');
  if(!panel || !['completed','partial'].includes(job.state))return;
  if(!panel.dataset.densityBound){
    panel.dataset.densityBound='1';
    panel.addEventListener('toggle',()=>{if(panel.open && state.reconstructionStatus)loadReconstructionLandmarkDensities(state.reconstructionStatus);});
  }
  if(!panel.open)return;
  const projectId=state.project.id,seriesId=state.selected,assemblyId=state.project.active_reconstruction_assembly;
  const key=JSON.stringify([projectId,seriesId,assemblyId,job.work,job.job_id,job.tilt_series?.revision]);
  if(panel.dataset.densityKey===key)return;
  panel.dataset.densityKey=key;
  panel.querySelector('.landmark-cloud-body').innerHTML='<p class="hint landmark-density-loading">Preparing landmark densities…</p>';
  try{
    const data=await api('/api/reconstruction-landmark-densities?'+new URLSearchParams({project_id:projectId,series_id:seriesId,assembly_id:assemblyId}));
    if(!panel.isConnected || panel.dataset.densityKey!==key || state.project?.id!==projectId || state.selected!==seriesId || state.project.active_reconstruction_assembly!==assemblyId)return;
    renderLandmarkDensities(panel,data);
  }catch(error){delete panel.dataset.densityKey;if(panel.isConnected)panel.querySelector('.landmark-cloud-body').textContent=error.message;}
}
function renderLandmarkDensities(panel,data){
  if(!data.meshes.length){
    panel.querySelector('.landmark-cloud-body').textContent='No central density surfaces are available at these landmark positions.';
    return;
  }
  renderLandmarkCloud(panel,{...data,points:data.meshes});
  const body=panel.querySelector('.landmark-cloud-body'),svg=body.querySelector('svg');
  svg.querySelector('.cloud-points').remove();svg.setAttribute('aria-label','Local tomogram densities at fitted landmark positions');
  const stage=document.createElement('div');stage.className='landmark-density-stage';svg.replaceWith(stage);stage.append(svg);
  const canvas=document.createElement('canvas');canvas.className='landmark-density-canvas';canvas.setAttribute('aria-label','Density surfaces');stage.append(canvas);
  panel.querySelector('.landmark-cloud-count').textContent=data.displayed_count+' / '+data.total_count+' landmarks';
  body.querySelector('.landmark-cloud-caption').textContent=`Local tomogram density${data.leveled?' · after leveling':''} · representative landmarks · thresholded display. Hover for d, q, h.`;
  const triangles=[];
  for(const mesh of data.meshes){
    const vertices=mesh.vertices.map(v=>[v[0]+mesh.d,v[1]+mesh.q,v[2]+mesh.h]);
    for(const face of mesh.faces){
      const p=face.map(i=>vertices[i]),a=p[1].map((v,i)=>v-p[0][i]),b=p[2].map((v,i)=>v-p[0][i]);
      const n=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],length=Math.hypot(...n)||1;
      const shade=.38+.62*Math.abs((-.4*n[0]-.7*n[1]+.6*n[2])/length);
      const colour=[101,120,149].map(v=>Math.round(Math.min(255,v*shade+32)));
      triangles.push({p:p.map(landmarkBoxProject),depth:p.reduce((sum,v)=>sum+v[1]-.52*v[0]-.34*v[2],0),colour:`rgb(${colour.join(',')})`});
    }
  }
  triangles.sort((a,b)=>b.depth-a.depth);
  let mapping=null;
  const draw=()=>{
    if(!panel.isConnected || !panel.open)return;
    const rect=svg.getBoundingClientRect(),box=svg.viewBox.baseVal;
    if(!rect.width || !rect.height)return;
    const pixelRatio=Math.min(window.devicePixelRatio || 1,2);
    canvas.width=Math.round(rect.width*pixelRatio);canvas.height=Math.round(rect.height*pixelRatio);
    const context=canvas.getContext('2d');context.scale(pixelRatio,pixelRatio);
    const scale=Math.min(rect.width/box.width,rect.height/box.height);
    mapping={scale,x:(rect.width-box.width*scale)/2-box.x*scale,y:(rect.height-box.height*scale)/2-box.y*scale};
    for(const triangle of triangles){
      context.beginPath();triangle.p.forEach((v,i)=>context[i?'lineTo':'moveTo'](v[0]*scale+mapping.x,v[1]*scale+mapping.y));
      context.closePath();context.fillStyle=triangle.colour;context.fill();
    }
  };
  const tip=body.querySelector('.landmark-cloud-tip');
  canvas.onpointermove=event=>{
    if(!mapping)return;
    const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left-mapping.x)/mapping.scale,y=(event.clientY-rect.top-mapping.y)/mapping.scale;
    let nearest=null,distance=18**2;
    for(const mesh of data.meshes){const p=landmarkBoxProject([mesh.d,mesh.q,mesh.h]),d=(p[0]-x)**2+(p[1]-y)**2;if(d<distance){distance=d;nearest=mesh;}}
    tip.hidden=!nearest;
    if(nearest)tip.textContent=`Landmark ${nearest.id} · d ${nearest.d.toFixed(1)} · q ${nearest.q.toFixed(1)} · h ${nearest.h.toFixed(1)} px · ${nearest.views} views`;
  };
  canvas.onpointerleave=()=>tip.hidden=true;
  panel.grecoDensityObserver?.disconnect();panel.grecoDensityObserver=new ResizeObserver(draw);panel.grecoDensityObserver.observe(stage);
  draw();
}
