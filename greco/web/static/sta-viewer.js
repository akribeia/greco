"use strict";

function particleVolumeUrl(assembly,job,kind,cl=1,axis='z',index=null) {
  const route=kind==='classification'?'particle-class-plane':'particle-average-plane';
  return `/api/${route}?project_id=${encodeURIComponent(state.project.id)}&assembly_id=${encodeURIComponent(assembly.id)}&job=${encodeURIComponent((job.work || '').split('/').pop())}&kind=${kind}&class_id=${cl}&axis=${axis}&index=${index ?? Math.floor((job.box_size || 1)/2)}&iteration=${job.latest_iteration || 0}&display=class-support-4`;
}

function particleVolumeThumbnails(assembly,job,kind) {
  const classes=kind==='classification'?(job.classes || []):[{id:1}];
  return `<div class="sta-volume-thumbnails">${classes.map(c=>`<button type="button" class="sta-volume-thumbnail" data-volume-assembly="${esc(assembly.id)}" data-volume-job="${esc((job.work || '').split('/').pop())}" data-volume-kind="${kind}" data-volume-class="${c.id}" aria-label="Open ${esc(particleJobName(job,''))}${kind==='classification'?' class '+c.id:''}"><img data-average-src="${esc(particleVolumeUrl(assembly,job,kind,c.id))}" alt="Central slice" loading="lazy" decoding="async" width="92" height="92"><span>${kind==='classification'?'Class '+c.id+'<br><small class="sta-class-caption"><strong>'+esc(c.particle_count)+'</strong>particles</small>':job.latest_iteration?'Iteration '+esc(job.latest_iteration):'View volume ↗'}</span></button>`).join('')}</div>`;
}

function openParticleVolume(assemblyId,jobName,kind,selectedClass=1) {
  const assembly=state.project.particle_assemblies.find(a=>a.id===assemblyId);
  if(!assembly)return;
  const candidates=[assembly[kind],...(assembly.job_history || [])].filter(Boolean);
  const job=candidates.find(j=>(j.work || '').split('/').pop()===jobName);
  if(!job || (job.state!=='completed' && !(kind==='refinement3d' && job.latest_iteration)))return;
  let dialog=$('sta-volume-dialog');
  if(!dialog){dialog=document.createElement('dialog');dialog.id='sta-volume-dialog';dialog.className='sta-volume-dialog';document.body.appendChild(dialog);}
  const n=Number(job.box_size || 1);let axis='z',cl=Number(selectedClass);
  const slices={x:Math.floor(n/2),y:Math.floor(n/2),z:Math.floor(n/2)};
  dialog.innerHTML=`<header class="sta-volume-header"><div><h3>${esc(particleJobName(job,''))}</h3><p>${esc(assembly.name)} · ${n}³ px · ${esc(particlePixelSize(job.pixel_size_A))} Å / pixel</p></div><button type="button" class="sta-volume-close" aria-label="Close viewer">×</button></header><div class="sta-volume-toolbar"><div class="sta-volume-axes">${['x','y','z'].map(a=>`<button type="button" data-view-axis="${a}" class="${a==='z'?'active':''}" aria-pressed="${a==='z'}">${a.toUpperCase()}</button>`).join('')}</div>${kind==='classification'?`<label>Class <select class="sta-volume-class">${(job.classes || []).map(c=>`<option value="${c.id}" ${c.id===cl?'selected':''}>${c.id} · ${c.particle_count} particles</option>`).join('')}</select></label>`:''}</div><div class="sta-volume-canvas"><img alt="Volume slice"></div><footer class="sta-volume-footer"><input type="range" min="0" max="${n-1}" value="${slices.z}" aria-label="Volume slice"><output></output></footer>`;
  const slider=dialog.querySelector('input'),image=dialog.querySelector('.sta-volume-canvas img');
  const refresh=()=>{
    const current=state.project?.particle_assemblies?.find(a=>a.id===assemblyId);
    const updated=[current?.[kind],...(current?.job_history || [])].find(j=>j && (j.work || '').split('/').pop()===jobName) || job;
    slices[axis]=Number(slider.value);
    const url=particleVolumeUrl(assembly,updated,kind,cl,axis,slices[axis]);
    requestViewerImage(image,url);
    dialog.querySelector('output').textContent=`${axis.toUpperCase()} · ${slices[axis]+1} / ${n}${kind==='refinement3d' && updated.latest_iteration?' · '+(updated.preview_label || 'Half 1')+' · iteration '+updated.latest_iteration:''}`;
  };
  dialog.onStaPreviewUpdate=()=>{if(dialog.open && kind==='refinement3d')refresh();};
  slider.oninput=refresh;
  dialog.querySelectorAll('[data-view-axis]').forEach(button=>button.onclick=()=>{
    axis=button.dataset.viewAxis;slider.value=slices[axis];
    dialog.querySelectorAll('[data-view-axis]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});refresh();
  });
  const select=dialog.querySelector('.sta-volume-class');if(select)select.onchange=()=>{cl=Number(select.value);refresh();};
  dialog.querySelector('.sta-volume-close').onclick=()=>dialog.close();
  dialog.onclick=event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left || event.clientX>r.right || event.clientY<r.top || event.clientY>r.bottom)dialog.close();}};
  refresh();dialog.showModal();
}

