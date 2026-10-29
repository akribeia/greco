"use strict";

function sameCTFRotation(a,b) {return Number.isFinite(Number(a)) && Number.isFinite(Number(b)) && Math.abs(Number(a)-Number(b))<1e-6;}
function ctfPanelOptions() {
  const options={output_binning:8};
  document.querySelectorAll('[data-preprocess]').forEach(input=>options[input.dataset.preprocess]=input.type==='checkbox'?input.checked:(input.value.trim()===''?null:Number(input.value)));
  const config=state.ctfOrientation;
  if(config?.projectId===state.project?.id && config.seriesId===state.selected) {
    const rotation=Number($('orientation-rotation').value);
    options.ctf_spatial_axis_deg=state.orientationFlipY?-rotation:rotation;
  } else options.ctf_spatial_axis_deg=state.preprocessOptions.ctf_spatial_axis_deg ?? null;
  return options;
}
async function saveCTFPanelSettings() {
  const options=ctfPanelOptions(),config=state.ctfOrientation;
  state.preprocessOptions={...state.preprocessOptions,...options};
  if(config?.projectId!==state.project?.id || config.seriesId!==state.selected)return options;
  config.calibrated=true;
  const rotation=Number($('orientation-rotation').value),hand=Number(options.df_hand);
  if(!Number.isFinite(rotation) || Math.abs(rotation)>360 || ![-1,1].includes(hand))throw new Error('Choose a valid rotation and defocus sign.');
  const tested=config.handTested && hand===config.testedHand && sameCTFRotation(rotation,config.testedRotation) && state.orientationFlipY===config.testedFlip;
  const orientation={rotation_deg:rotation,ctf_axis_deg:options.ctf_spatial_axis_deg,
    df_hand:hand,flip_y:state.orientationFlipY,df_hand_tested:!!tested};
  if(tested)orientation.df_hand_convention='physical_local_defocus_v2';
  const projectId=state.project.id;
  const project=await api('/api/orientation',{project_id:projectId,orientation});
  if(state.project?.id!==projectId)throw new Error('Project changed before submission.');
  state.project=project;
  config.rotation_deg=rotation;config.flip_y=orientation.flip_y;
  return options;
}
function mountOrientation(row) {
  state.preprocessStatus=null;state.axisVisible=false;state.ctfView=null;
  const previous=state.ctfOrientation;
  const same=previous?.projectId===state.project.id && previous.seriesId===row.id;
  const saved=state.project.orientation;
  state.orientationFlipY=same?previous.flip_y:!!saved?.flip_y;
  state.orientationPreview=same?previous.rotation_deg:(saved?.rotation_deg ?? row.mdoc_tilt_axis_deg ?? row.metadata.tilt_axis_deg ?? 0);
  state.orientationPreview=Math.round(state.orientationPreview*1e6)/1e6;
  const config=same?previous:{projectId:state.project.id,seriesId:row.id,
    rotation_deg:state.orientationPreview,flip_y:state.orientationFlipY,calibrated:!!saved,
    handTested:!!saved?.df_hand_tested && saved?.df_hand_convention==='physical_local_defocus_v2',
    testedHand:saved?.df_hand,testedRotation:saved?.rotation_deg,testedFlip:!!saved?.flip_y,handEditVersion:0};
  state.ctfOrientation=config;
  const handInput=document.querySelector('[data-preprocess="df_hand"]');
  state.orientationHandPreview=Number(handInput.value);state.orientationHandUnsaved=false;
  const controls=document.createElement('div');controls.className='orientation-controls';
  controls.innerHTML='<button type="button" class="button secondary" id="axis-toggle" data-tilt-diagnostic="axis" aria-label="Show tilt-axis guide" aria-pressed="false" title="Show vertical tilt-axis guide">[|]</button>';
  $('tilt-image-area').append(controls);
  const orientationFields=$('ctf-orientation-fields');orientationFields.className='ctf-orientation';
  orientationFields.innerHTML=`<div class="metadata-field"><label for="orientation-rotation">Tilt axis angle β (°)</label><input id="orientation-rotation" type="number" min="-360" max="360" step="any" value="${esc(state.orientationPreview)}"></div><div class="orientation-turns"><button type="button" class="button secondary" id="orientation-left">↶ −90°</button><button type="button" class="button secondary" id="orientation-right">↷ +90°</button><button type="button" class="button secondary" id="orientation-flip" aria-pressed="${state.orientationFlipY}">Flip Y</button><button type="button" class="button secondary" id="orientation-mdoc">Reset MDOC</button></div><p class="hint">MDOC: ${esc(row.mdoc_tilt_axis_deg ?? 'unknown')}° · Align the tilt axis with the vertical guide. Changes apply to project jobs.</p><p class="hint">Test handedness selects a sign.<br>Submit and Redo apply it.</p>${defocusHandHelp()}`;
  const handButton=$('orientation-hand-test'),redoButton=$('orientation-redo-ctf'),redoResult=$('ctf-redo-result');
  redoButton.title='Refit CTF using the settings above and saved motion shifts.';
  const busy={hand:false,redo:false};
  const refreshButtons=()=>{
    if(!redoButton.isConnected || !handButton.isConnected)return;
    const disabled=busy.hand || busy.redo || state.preprocessStatus?.state!=='completed';
    handButton.disabled=disabled;redoButton.disabled=disabled;
  };
  state.refreshCTFButtons=refreshButtons;refreshButtons();
  async function watchRedoCTF(submitNew=false) {
    const projectId=state.project.id,seriesId=state.selected;
    const current=()=>state.project?.id===projectId && state.selected===seriesId && redoButton.isConnected;
    busy.redo=true;refreshButtons();
    try {
      const options=submitNew?await saveCTFPanelSettings():null;
      if(!current())return;
      let job=await api('/api/ctf-redo-status?'+new URLSearchParams({project_id:projectId,series_id:seriesId}));
      if(!current())return;
      if(submitNew && !['submitted','running','unknown'].includes(job.state)) {
        await requireCTFCoordinateProtocol();
        if(!current())return;
        redoResult.textContent='Submitting CTF recalculation…';
        job=await api('/api/submit-ctf-redo',{project_id:projectId,series_id:seriesId,options});
        if(!current())return;
      }
      if(job.state==='pending')return;
      selectedSeries().ctf_redo=job;
      while(['submitted','running','unknown'].includes(job.state)) {
        if(!current())return;
        selectedSeries().ctf_redo=job;
        state.preprocessStatus=preprocessingState(state.preprocessStatus,selectedSeries());
        if($('prep-status')){$('prep-status').textContent=`${job.state} · Redo CTF`;$('prep-status').className='badge pending';}
        renderStages();refreshButtons();
        const progress=job.progress;
        redoResult.textContent=`Redo CTF: ${job.state}${job.job_id?' · Slurm '+job.job_id:''}${progress?' · '+progress.phase+(progress.total?' '+progress.tilt+'/'+progress.total:''):''}`;
        await new Promise(resolve=>setTimeout(resolve,3000));
        if(!current())return;
        job=await api('/api/ctf-redo-status?'+new URLSearchParams({project_id:projectId,series_id:seriesId}));
      }
      if(!current())return;
      selectedSeries().ctf_redo=job;
      if(job.state!=='completed')throw new Error(job.error || 'CTF recalculation failed. Previous preprocessing is unchanged.');
      const project=await api('/api/project?id='+encodeURIComponent(projectId));
      if(!current())return;
      state.project=project;
      const index=$('tilt-slider')?.value;
      await pollPreprocessing(projectId,seriesId);
      if(!current())return;
      if(index!=null)$('tilt-slider').value=index;
      showTiltPlane();
      config.handTested=false;
      watchHandedness();
      redoResult.textContent=`Redo CTF: completed${job.job_id?' · Slurm '+job.job_id:''}. ΔZ and spectra updated; existing reconstructions unchanged.`;
    } catch(error){if(current())redoResult.textContent=error.message==='Unknown route.'?'Restart the Greco server and reload this page to enable Redo CTF.':error.message;}
    finally{busy.redo=false;refreshButtons();}
  }

  redoButton.addEventListener('click',()=>watchRedoCTF(true));watchRedoCTF();
  for(const [mode,label,title] of [['comparison','◐','Compare CTF model (left) and measured spectrum (right)'],['landscape','ΔZ','Show saved defocus landscape']]) {
    const button=document.createElement('button');button.type='button';button.className='button secondary ctf-view-toggle';button.textContent=label;button.title=title;button.setAttribute('aria-label',title);button.setAttribute('aria-pressed','false');
    button.dataset.tiltDiagnostic=mode;controls.insertBefore(button,$('axis-toggle'));
    button.addEventListener('click',()=>{selectTiltDiagnostic(mode,controls);showTiltPlane();});
  }
  const preview=value=>{
    value=Math.round((((value+360)%720+720)%720-360)*1e6)/1e6;
    $('orientation-rotation').value=value;state.orientationPreview=value;
    config.rotation_deg=value;config.flip_y=state.orientationFlipY;config.calibrated=true;
    showTiltPlane();
  };
  $('axis-toggle').addEventListener('click',()=>{selectTiltDiagnostic('axis',controls);$('tilt-image-area').classList.toggle('show-axis',state.axisVisible);});
  $('orientation-rotation').addEventListener('input',()=>{const value=Number($('orientation-rotation').value);if(Number.isFinite(value)&&Math.abs(value)<=360)preview(value);});
  $('orientation-mdoc').disabled=row.mdoc_tilt_axis_deg==null;
  $('orientation-mdoc').addEventListener('click',()=>preview((state.orientationFlipY?-1:1)*Number(row.mdoc_tilt_axis_deg)));
  $('orientation-left').addEventListener('click',()=>preview(Number($('orientation-rotation').value)-90));
  $('orientation-right').addEventListener('click',()=>preview(Number($('orientation-rotation').value)+90));
  $('orientation-flip').addEventListener('click',()=>{state.orientationFlipY=!state.orientationFlipY;$('orientation-flip').setAttribute('aria-pressed',String(state.orientationFlipY));preview(-Number($('orientation-rotation').value));});
  handInput.addEventListener('change',()=>{
    state.preprocessOptions.df_hand=Number(handInput.value);state.orientationHandPreview=Number(handInput.value);
    config.handEditVersion++;config.handTested=false;
    state.orientationHandUnsaved=state.preprocessStatus?.ctf_local_geometry?.handedness!==Number(handInput.value);
    showTiltPlane();
  });
  async function watchHandedness(submitNew=false) {
    const projectId=state.project.id,seriesId=state.selected;
    const rotation=Number($('orientation-rotation').value),flipY=state.orientationFlipY;
    const button=$('orientation-hand-test'),result=$('orientation-hand-result');
    const editVersion=config.handEditVersion;
    let wasActive=false;
    busy.hand=true;refreshButtons();
    try {
      if(submitNew)await saveCTFPanelSettings();
      if(state.project?.id!==projectId || state.selected!==seriesId || !button.isConnected)return;
      let job=await api('/api/handedness-status?'+new URLSearchParams({project_id:projectId,series_id:seriesId}));
      if(state.project?.id!==projectId || state.selected!==seriesId || !button.isConnected)return;
      wasActive=['submitted','running','unknown'].includes(job.state);
      if(submitNew && !wasActive) {
        config.calibrated=true;
        job=await api('/api/submit-handedness-test',{project_id:projectId,series_id:seriesId,rotation_deg:rotation,flip_y:flipY});
      }
      if(job.state==='pending')return;
      while(['submitted','running','unknown'].includes(job.state)) {
        if(state.project?.id!==projectId || state.selected!==seriesId || !button.isConnected)return;
        result.textContent=`Handedness test: ${job.state}${job.job_id?' · Slurm '+job.job_id:''}`;
        await new Promise(resolve=>setTimeout(resolve,3000));
        job=await api('/api/handedness-status?'+new URLSearchParams({project_id:projectId,series_id:seriesId}));
      }
      if(state.project?.id!==projectId || state.selected!==seriesId || !button.isConnected)return;
      if(job.state!=='completed')throw new Error(job.error || 'Handedness test failed.');
      const scores=`+1: ${job.hand_scores['+1'].toFixed(5)} / −1: ${job.hand_scores['-1'].toFixed(5)}`;
      if(!sameCTFRotation(Number($('orientation-rotation').value),job.orientation.rotation_deg) || state.orientationFlipY!==job.orientation.flip_y)throw new Error('Orientation changed during test. Run the test again.');
      if(job.conclusive) {
        const resultId=String(job.job_id ?? job.work ?? JSON.stringify(job.orientation));
        const resultKey=`greco.ctfHandResult:${projectId}:${seriesId}`;
        let acknowledged=config.handResultId===resultId;
        try{acknowledged ||= localStorage.getItem(resultKey)===resultId;}catch{}
        // A test may finish while the tab is closed. Adopt that unseen result
        // on reopening, but never reapply it over a later manual selection.
        const apply=(submitNew || wasActive || (!acknowledged && config.handEditVersion===0)) && editVersion===config.handEditVersion;
        if(apply) {
          handInput.value=String(job.handedness);
          state.preprocessOptions.df_hand=job.handedness;
          state.orientationHandPreview=job.handedness;
          state.orientationHandUnsaved=state.preprocessStatus?.ctf_local_geometry?.handedness!==job.handedness;
          config.handTested=true;config.testedHand=job.handedness;
          config.testedRotation=rotation;config.testedFlip=flipY;
          await saveCTFPanelSettings();
          if(state.project?.id!==projectId || state.selected!==seriesId || !button.isConnected)return;
          showTiltPlane();
        }
        config.handResultId=resultId;
        try{localStorage.setItem(resultKey,resultId);}catch{}
        result.textContent=`Detected sign: ${job.handedness>0?'+1':'−1'} · ${scores} · ${job.n_views} tilts.${apply?' Selected and saved; Redo CTF applies it.':' Current selection unchanged.'}`;
      } else {
        result.textContent=`Inconclusive · ${scores} · tested axis ${Number(job.axis_deg ?? job.orientation.rotation_deg).toFixed(2)}°. Current sign unchanged. Align the visible tilt axis vertically before testing again.`;
      }
    } catch(error) {if(result.isConnected)result.textContent=error.message;}
    finally {busy.hand=false;refreshButtons();}
  }

  handButton.addEventListener('click',()=>watchHandedness(true));watchHandedness();
}
