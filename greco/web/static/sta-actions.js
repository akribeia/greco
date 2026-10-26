"use strict";

function bindParticleJobs(root=document) {
  const panels=root.matches?.('.particle-assembly')?[root]:[...root.querySelectorAll('.particle-assembly')];
  const menu=$('particle-job-action-menu');if(menu && !menu.anchor?.isConnected)menu.closeMenu();
  panels.forEach(panel=>{
    const assembly=state.project.particle_assemblies.find(a=>a.id===panel.dataset.assemblyId);
    if(!assembly)return;
    const jobs=[...['extraction','classification','refinement3d','polishing','ctf_refinement','postprocessing'].map(k=>assembly[k]),...(assembly.job_history || [])].filter(Boolean);
    panel.querySelectorAll('[data-job-number]').forEach(card=>{
      const job=jobs.find(j=>particleJobNumber(j,0)===Number(card.dataset.jobNumber));
      const details=card.querySelector('.sta-job-details');
      if(details && job){
        const key=state.project.id+':'+job.work;
        state.staJobDetails ??= {};
        details.open=Boolean(state.staJobDetails[key]);
        details.ontoggle=()=>{state.staJobDetails[key]=details.open;};
      }
      const status=card.querySelector('.assembly-card-header > span:last-child');
      if(!job || !status || status.dataset.jobDeleteBound)return;
      status.dataset.jobDeleteBound='true';status.tabIndex=0;status.setAttribute('role','button');
      status.setAttribute('aria-label','Job actions');status.style.cursor='pointer';
      const toggle=()=>{
        const existing=$('particle-job-action-menu');
        const same=existing?.anchor===status;
        if(existing)existing.closeMenu();
        if(same)return;
        const menu=document.createElement('div');menu.id='particle-job-action-menu';menu.className='particle-job-action-menu';menu.setAttribute('role','menu');menu.anchor=status;
        const button=document.createElement('button');button.type='button';button.setAttribute('role','menuitem');
        button.dataset.deleteParticleJob='true';button.textContent='Delete';
        const clone=document.createElement('button');clone.type='button';clone.setAttribute('role','menuitem');clone.textContent='Clone';
        const clear=document.createElement('button');clear.type='button';clear.setAttribute('role','menuitem');clear.textContent='Clear';
        menu.append(clone,clear,button);document.body.appendChild(menu);
        const r=status.getBoundingClientRect(),width=menu.offsetWidth,height=menu.offsetHeight;
        menu.style.left=Math.max(8,Math.min(r.right-width,innerWidth-width-8))+'px';
        menu.style.top=(r.bottom+height+8<innerHeight?r.bottom+6:Math.max(8,r.top-height-6))+'px';
        status.setAttribute('aria-expanded','true');
        const outside=event=>{if(!menu.contains(event.target) && event.target!==status)menu.closeMenu();};
        const escape=event=>{if(event.key==='Escape')menu.closeMenu();};
        menu.closeMenu=()=>{menu.remove();status.setAttribute('aria-expanded','false');document.removeEventListener('pointerdown',outside);document.removeEventListener('keydown',escape);window.removeEventListener('resize',menu.closeMenu);window.removeEventListener('scroll',menu.closeMenu,true);};
        document.addEventListener('pointerdown',outside);document.addEventListener('keydown',escape);window.addEventListener('resize',menu.closeMenu);window.addEventListener('scroll',menu.closeMenu,true);
        button.onclick=()=>{menu.closeMenu();confirmParticleJobDeletion(assembly.id,job);};
        clone.onclick=()=>{menu.closeMenu();reuseParticleJob(assembly.id,job,false);};
        clear.onclick=()=>{menu.closeMenu();reuseParticleJob(assembly.id,job,true);};
        clone.focus();
      };
      status.onclick=toggle;status.onkeydown=event=>{if(event.key==='Enter' || event.key===' '){event.preventDefault();toggle();}};
    });
  });
  root.querySelectorAll('[data-volume-assembly]').forEach(button=>button.onclick=()=>openParticleVolume(button.dataset.volumeAssembly,button.dataset.volumeJob,button.dataset.volumeKind,Number(button.dataset.volumeClass)));
  root.querySelectorAll('.assembly-task-list').forEach(list=>{
    const plus=list.querySelector('.assembly-new-job');
    const cards=Array.from(list.children).filter(card=>card.matches('[data-job-number]'));
    const sorted=cards.slice().sort((a,b)=>Number(a.dataset.jobNumber)-Number(b.dataset.jobNumber));
    if(cards.some((card,i)=>card!==sorted[i]))sorted.forEach(card=>list.insertBefore(card,plus));
  });
  // Append provisional submissions only after numbered cards are sorted.
  // Otherwise sorting moves every numbered card behind the provisional one.
  renderPendingParticleSubmissions(root);
  root.querySelectorAll('[data-new-assembly-job]').forEach(button=>button.onclick=()=>openAssemblyJobChooser(button.dataset.newAssemblyJob));
  root.querySelectorAll('[data-extract-assembly]').forEach(button=>button.onclick=async()=>{
    const projectId=state.project.id;button.disabled=true;
    try {const project=await api('/api/submit-particle-extraction',{project_id:projectId,assembly_id:button.dataset.extractAssembly});if(state.project?.id===projectId)saveProject(project);}
    catch(error){notify(error.message);if(button.isConnected)button.disabled=false;}
  });
  panels.forEach(loadAssemblyPreviews);
}

function renderPendingParticleSubmissions(root=document) {
  root.querySelectorAll('[data-pending-particle-submission]').forEach(card=>card.remove());
  for(const [token,pending] of state.staPendingSubmissions || []){
    if(pending.projectId!==state.project?.id)continue;
    const list=$('assembly-job-'+pending.assemblyId)?.querySelector('.assembly-task-list');
    if(!list || !root.contains(list))continue;
    const card=document.createElement('article');card.className='assembly-task-card';
    card.dataset.pendingParticleSubmission=token;
    const title={extraction:'Extract',classification:'Class3D',refinement3d:'Refine3D',polishing:'Polishing',ctf_refinement:'CTFRefine',postprocessing:'PostProcess'}[pending.kind];
    card.innerHTML=`<header class="assembly-card-header"><h3>${esc(title)}</h3><span class="badge pending">${pending.state==='failed'?'Submission failed':'Submitting…'}</span></header><div class="assembly-card-content">${pending.error?`<p class="form-error" role="alert">${esc(pending.error)}</p><button type="button" class="button secondary">Dismiss</button>`:'<p class="hint" role="status">Sending to Slurm…</p>'}</div>`;
    const dismiss=card.querySelector('button');if(dismiss)dismiss.onclick=()=>{state.staPendingSubmissions.delete(token);card.remove();};
    list.insertBefore(card,list.querySelector('.assembly-new-job'));
  }
}

async function reuseParticleJob(assemblyId,job,clear) {
  const projectId=state.project.id,name=(job.work || '').split('/').pop();
  try{
    const preset=await api('/api/particle-job-preset?'+new URLSearchParams({project_id:projectId,assembly_id:assemblyId,job:name}));
    if(state.project?.id!==projectId)return;
    if(clear)confirmParticleJobDeletion(assemblyId,job,preset);
    else openAssemblyJobChooser(assemblyId,preset);
  }catch(error){notify(error.message);}
}

function confirmParticleJobDeletion(assemblyId,job,preset=null) {
  let dialog=$('particle-job-delete-dialog');
  if(dialog?.deleting){notify('A job deletion is already in progress.');return;}
  if(!dialog){dialog=document.createElement('dialog');dialog.id='particle-job-delete-dialog';dialog.className='assembly-job-dialog';document.body.appendChild(dialog);}
  const projectId=state.project.id,name=(job.work || '').split('/').pop();
  const active=['submitted','running','submitting','unknown'].includes(job.state);
  dialog.innerHTML=`<form><h3>Delete ${esc(particleJobName(job,'job'))}?</h3><p>${active?'This will cancel the job in Slurm, wait for it to stop, and then permanently delete its folder and results.':'This permanently deletes the job folder and its results.'} Tomograms and other jobs are preserved.</p><label><input type="checkbox" required> I confirm ${active?'cancellation and ':''}permanent deletion</label><p class="form-error" role="alert"></p><div class="submit-actions"><button type="button" class="button secondary">Cancel</button><button type="submit" class="button primary" disabled>${active?'Cancel job & delete':'Delete'}</button></div></form>`;
  const check=dialog.querySelector('input'),submit=dialog.querySelector('[type=submit]');
  if(preset){dialog.querySelector('h3').textContent='Clear '+particleJobName(job,'job')+'?';submit.textContent=active?'Cancel job & clear':'Clear';}
  check.onchange=()=>{submit.disabled=!!dialog.deleting || !check.checked;};
  dialog.querySelector('[type=button]').onclick=()=>dialog.close();
  dialog.querySelector('form').onsubmit=async event=>{
    event.preventDefault();if(!check.checked || dialog.deleting)return;submit.disabled=true;dialog.deleting=true;check.disabled=true;
    dialog.querySelector('[type=button]').textContent='Close';
    try{
      let project;
      do{
        project=await api('/api/particle-job-delete',{project_id:projectId,assembly_id:assemblyId,job:name,confirmed:true});
        if(project.deletion_pending){
          dialog.querySelector('.form-error').textContent='Cancellation requested. Waiting for Slurm to stop the job before deleting files…';
          await new Promise(resolve=>setTimeout(resolve,1500));
        }
      }while(project.deletion_pending);
      dialog.close();if(state.project?.id===projectId)saveProject(project);
      notify('Job folder and results permanently deleted.');
      if(preset && state.project?.id===projectId)openAssemblyJobChooser(assemblyId,preset);
    }catch(error){dialog.querySelector('.form-error').textContent=error.message;}
    finally{dialog.deleting=false;check.disabled=false;submit.disabled=!check.checked;dialog.querySelector('[type=button]').textContent='Cancel';}
  };
  dialog.showModal();
}

