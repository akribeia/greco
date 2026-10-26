"use strict";

let grecoUpdateInfo=null,grecoUpdateTimer=null,grecoUpdateBusy=false,grecoUpdateTarget=null,grecoUpdateRestarting=false;
function renderGrecoUpdates(){
  const info=grecoUpdateInfo,button=$('updates-open');
  const installing=info?.operation?.state==='running';
  const available=!!(info?.available && info.latest?.archive);
  button.textContent=grecoUpdateBusy || installing?'Updating…':available?'Update to '+info.latest.version:'Version '+(info?.current || button.dataset.runningVersion || '…');
  button.disabled=grecoUpdateBusy || installing;
  button.classList.toggle('update-available',available);
  if(available)button.removeAttribute('aria-haspopup');else button.setAttribute('aria-haspopup','menu');
  button.title=info?.error || (installing?info.operation.message:'');
  if(info?.current)document.querySelector('.sidebar-footer-right').textContent='Greco '+info.current;
}
async function restartGrecoUpdate(version){
  if(grecoUpdateRestarting)return;
  grecoUpdateRestarting=true;
  await api('/api/update-restart',{version});
  $('updates-open').textContent='Restarting…';
  clearTimeout(grecoUpdateTimer);
  let attempts=0;
  const reconnect=async()=>{
    try{
      const response=await fetch('/',{cache:'no-store',credentials:'same-origin'});
      if(!response.ok)throw Error('Server not ready');
      const info=await api('/api/update-status');
      if(info.current===version){location.reload();return;}
    }catch{}
    if(++attempts<30)grecoUpdateTimer=setTimeout(reconnect,2000);
    else{
      grecoUpdateBusy=false;grecoUpdateTarget=null;grecoUpdateRestarting=false;renderGrecoUpdates();
      notify('Could not reconnect. Run greco status in the terminal.');
    }
  };
  grecoUpdateTimer=setTimeout(reconnect,2000);
}
async function checkGrecoUpdates(){
  if(grecoUpdateRestarting)return;
  clearTimeout(grecoUpdateTimer);
  try{
    grecoUpdateInfo=await api('/api/update-status');
    renderGrecoUpdates();
    if(grecoUpdateTarget){
      if(grecoUpdateInfo.installed?.includes(grecoUpdateTarget)){
        await restartGrecoUpdate(grecoUpdateTarget);return;
      }
      const operation=grecoUpdateInfo.operation;
      if(operation?.state==='failed' && operation.version===grecoUpdateTarget)throw Error(operation.message);
    }
  }catch(error){
    if(grecoUpdateTarget){
      grecoUpdateBusy=false;grecoUpdateTarget=null;grecoUpdateRestarting=false;renderGrecoUpdates();notify(error.message);
    }
  }
  grecoUpdateTimer=setTimeout(checkGrecoUpdates,grecoUpdateTarget || grecoUpdateInfo?.operation?.state==='running'?1500:86400000);
}
function closeGrecoVersions(){
  const menu=$('greco-versions');
  if(menu.hidePopover && menu.matches(':popover-open'))menu.hidePopover();
  menu.removeAttribute('data-open');$('updates-open').setAttribute('aria-expanded','false');
}
function showGrecoVersions(){
  const menu=$('greco-versions'),button=$('updates-open'),info=grecoUpdateInfo;
  const versions=[...new Set(info?.installed || [])];
  menu.innerHTML=`<div class="greco-version-heading" role="presentation">Greco versions</div>
    <div class="greco-version-list" role="group" aria-label="Installed versions">${versions.length?versions.map(version=>`<button type="button" role="menuitem" data-greco-version="${esc(version)}" ${version===info.current?'disabled aria-current="true"':''}><span>Greco ${esc(version)}</span>${version===info.current?'<small class="greco-version-running">Running</small>':'<span class="greco-version-arrow" aria-hidden="true">→</span>'}</button>`).join(''):'<span class="greco-version-empty">No installed versions available</span>'}</div>
    <div class="greco-version-footer" role="presentation">Selecting another version restarts Greco.</div>`;
  if(menu.showPopover)menu.showPopover();else menu.dataset.open='true';
  const rect=button.getBoundingClientRect();
  menu.style.left=Math.max(8,Math.min(rect.right-menu.offsetWidth,innerWidth-menu.offsetWidth-8))+'px';
  menu.style.top=Math.max(8,Math.min(rect.bottom+8,innerHeight-menu.offsetHeight-8))+'px';
  button.setAttribute('aria-expanded','true');
  (menu.querySelector('button:not(:disabled)') || menu).focus();
}
function startGrecoUpdates(){
  $('updates-open').addEventListener('click',async()=>{
    if(grecoUpdateBusy || grecoUpdateInfo?.operation?.state==='running')return;
    if(!grecoUpdateInfo?.available || !grecoUpdateInfo.latest?.archive){
      if($('updates-open').getAttribute('aria-expanded')==='true'){closeGrecoVersions();return;}
      if(!grecoUpdateInfo)await checkGrecoUpdates();
      showGrecoVersions();return;
    }
    closeGrecoVersions();
    grecoUpdateTarget=grecoUpdateInfo.latest.version;grecoUpdateBusy=true;renderGrecoUpdates();
    try{
      if(!grecoUpdateInfo.installed?.includes(grecoUpdateTarget)){
        await api('/api/update-install',{version:grecoUpdateTarget});
      }
      await checkGrecoUpdates();
    }catch(error){
      grecoUpdateBusy=false;grecoUpdateTarget=null;grecoUpdateRestarting=false;renderGrecoUpdates();notify(error.message);
    }
  });
  const menu=$('greco-versions');
  menu.addEventListener('toggle',event=>$('updates-open').setAttribute('aria-expanded',String(event.newState==='open')));
  menu.addEventListener('click',async event=>{
    const version=event.target.closest('[data-greco-version]')?.dataset.grecoVersion;
    if(!version || version===grecoUpdateInfo?.current || grecoUpdateBusy)return;
    closeGrecoVersions();grecoUpdateBusy=true;renderGrecoUpdates();
    try{await restartGrecoUpdate(version);}
    catch(error){grecoUpdateBusy=false;grecoUpdateRestarting=false;renderGrecoUpdates();notify(error.message);}
  });
  document.addEventListener('click',event=>{if(!menu.contains(event.target) && !$('updates-open').contains(event.target))closeGrecoVersions();});
  document.addEventListener('keydown',event=>{if(event.key==='Escape')closeGrecoVersions();});
  window.addEventListener('resize',closeGrecoVersions);
  renderGrecoUpdates();
  grecoUpdateTimer=setTimeout(checkGrecoUpdates,2000);
}
