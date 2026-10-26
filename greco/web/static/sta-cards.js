"use strict";

function particleJobName(job, fallback) {
  const path=(job.work || '').split('/');
  if (/^\d+$/.test(path.at(-1) || '')) {
    const labels={extraction:'Extract',classification:'Class3D',refinement:'Refine3D',polishing:'Polishing',ctf_refinement:'CTFRefine',postprocessing:'PostProcess'};
    return `${labels[path.at(-2)] || job.kind || 'Job'} ${path.at(-1)}`;
  }
  return job.name || fallback;
}

function particleJobNumber(job, fallback) {
  const directory=(job.work || '').split('/').pop() || '';
  const number=Number(job.number ?? (/^\d+$/.test(directory)?directory:undefined));
  return Number.isFinite(number) && number>0 ? number : fallback;
}

function particleArchivedJobs(assembly) {
  const current=new Set(['extraction','classification','refinement3d','polishing','ctf_refinement','postprocessing'].map(k=>assembly[k]?.work));
  const jobs=new Map();
  for(const job of (assembly.job_history || [])){
    if(job.work && !current.has(job.work))jobs.set(job.work,job);
  }
  return Array.from(jobs.values());
}

function particleArchivedCards(assembly) {
  return particleArchivedJobs(assembly).map(job=>particleJobCard(assembly,job,job.kind)).join('');
}

function particleClassificationCard(assembly) {
  return assembly.classification?particleJobCard(assembly,assembly.classification,'classification'):'';
}

function particleRefinementCard(assembly) {
  return assembly.refinement3d?particleJobCard(assembly,assembly.refinement3d,'refinement3d'):'';
}

function refinementMetrics(job) {
  const m=job.refinement_metrics;if(!m)return '';
  const number=v=>particleMetricNumber(v);
  const local=['local_search','trust_region'].includes(m.alignment_mode);
  const trust=m.alignment_mode==='trust_region';
  const optimizer=local && m.local_optimizer_converged_particles!=null
    ? `<p class="hint">Local optimizer · converged ${esc(m.local_optimizer_converged_particles)} · plateau ${esc(m.local_optimizer_plateau_particles ?? 0)}${trust && m.local_optimizer_numerical_plateau_particles>0?' · numerical plateau '+esc(m.local_optimizer_numerical_plateau_particles):''} · iteration limit ${esc(m.local_optimizer_limited_particles ?? 0)}${trust?' · stalled '+esc(m.local_optimizer_stalled_particles ?? 0):''} · failed ${esc(m.local_optimizer_failed_particles ?? 0)}</p>` : '';
  const posterior=m.local_marginalization?`<p class="hint">Local pose posterior · effective poses ${number(m.posterior_effective_poses)} · ${m.noise_updated?'residual noise updated':'fixed noise'}</p>`:'';
  const acceptance=(trust?`<p class="hint">Accepted steps ${esc(m.trust_accepted_steps ?? 0)} · rejected ${esc(m.trust_rejected_steps ?? 0)}</p>`:'');
  return `<p class="hint sta-result">${Object.hasOwn(m,'diagnostic_resolution_A')?(m.fsc_mode==='phase_randomization_corrected'?'Corrected FSC 0.143':'Diagnostic FSC 0.143')+' · <strong>'+(m.diagnostic_resolution_A!=null && Number.isFinite(Number(m.diagnostic_resolution_A)) && Number(m.diagnostic_resolution_A)>0?number(m.diagnostic_resolution_A)+' Å':'no crossing')+'</strong>':''}</p><details class="sta-job-details"><summary>Refinement details</summary>`+optimizer+posterior+acceptance+`<p class="hint">Alignment ${number(m.alignment_resolution_A)} Å · ${trust?'Local trust region':local?'Local pose increments':`angular range ±${esc(job.options?.angular_range_deg ?? '—')}°`}</p><p class="hint">Δ angles RMS ${number(m.angle_rms_change_deg)}° · Δ shifts RMS ${number(m.shift_rms_change_A)} Å</p><p class="hint">${local?'Outside extraction support: '+esc(m.support_rejected_particles ?? 0):'At bounds: '+esc(m.bound_particles ?? '—')} particles · stable passes ${esc(m.auto_refine?.stable_iterations ?? '—')} / 3</p>${job.convergence_state?`<p class="hint">${esc(job.convergence_state==='converged'?'Converged':job.convergence_state==='iteration_limit'?'Iteration limit reached — not converged':job.convergence_state==='alternating_limit'?'Alternating cycle limit reached — not converged':job.convergence_state==='fixed_iterations'?'Fixed iterations':'Not converged')}</p>`:''}</details>`;
}

function particleAssemblyCards(assembly) {
  return `<div class="assembly-task-list" aria-label="Assembly jobs">
    ${particleArchivedCards(assembly)}
    ${assembly.extraction?particleJobCard(assembly,assembly.extraction,'extraction'):''}
    ${particleClassificationCard(assembly)}
    ${particleRefinementCard(assembly)}
    ${particleObservationCards(assembly)}
    ${particlePostprocessCard(assembly)}
    <button class="assembly-new-job" type="button" data-new-assembly-job="${esc(assembly.id)}" aria-label="Create a new job in ${esc(assembly.name)}"><span aria-hidden="true">+</span></button>
  </div>`;
}

function particleObservationCards(assembly) {
  return ['polishing','ctf_refinement'].map(kind=>assembly[kind]?particleJobCard(assembly,assembly[kind],kind):'').join('');
}

function postprocessDetails(assembly,job) {
  const result=job.postprocess;
  if(!result)return '';
  const url=file=>'/api/particle-postprocess-file?'+new URLSearchParams({project_id:state.project.id,assembly_id:assembly.id,job:(job.work || '').split('/').pop(),file});
  return `<p class="hint">Final FSC 0.143: <strong>${result.resolution_A?Number(result.resolution_A).toFixed(3)+' Å':'no crossing'}</strong>${result.interpolated_resolution_A?' · interpolated '+Number(result.interpolated_resolution_A).toFixed(3)+' Å':''}</p><p class="hint">${result.mask_corrected?'Mask corrected':'Unmasked'} · B-factor ${particleMetricNumber(result.bfactor_A2,1)} Å²</p><p class="hint"><a href="${esc(url('map'))}" download>Map</a> · ${result.mask_corrected?`<a href="${esc(url('masked_map'))}" download>Masked map</a> · `:''}<a href="${esc(url('report'))}" download>PDF report</a></p>`;
}

function particlePostprocessCard(assembly) {
  return assembly.postprocessing?particleJobCard(assembly,assembly.postprocessing,'postprocessing'):'';
}

function particlePixelSize(value) {
  const pixel=Number(value);
  return Number.isFinite(pixel) && pixel>0 ? String(Number(pixel.toFixed(4))) : '—';
}

function particleObservationReport(assembly,job,kind) {
  if(!['polishing','ctf_refinement'].includes(kind) || job.state!=='completed')return '';
  const query=new URLSearchParams({project_id:state.project.id,assembly_id:assembly.id,kind,job:(job.work || '').split('/').pop()});
  return `<p class="hint"><a href="/api/particle-observation-report?${esc(query)}" download>PDF report</a></p>`;
}


function particleMetricNumber(value, digits=3) {
  return value!==null && value!==undefined && value!=='' && Number.isFinite(Number(value)) ? String(Number(Number(value).toFixed(digits))) : '—';
}

function particleJobCard(assembly,job,kind) {
  const active=['submitted','running','submitting','unknown'].includes(job.state);
  const p=job.progress || {},d=job.diagnostics || {};
  const classes=kind==='classification';
  const particles=job.used_particles ?? job.particle_count;
  const metadata=[];
  const coordinateSource=kind==='extraction' && job.coordinate_source?.kind==='classification' ? job.coordinate_source : null;
  const extractionSource=coordinateSource
    ? `<p class="hint sta-extraction-source">From · ${esc(particleJobName({work:coordinateSource.work,kind:'classification'},'Class3D'))} · Classes ${esc((coordinateSource.classes || []).join(', '))}</p>` : '';
  if(job.pixel_size_A!=null)metadata.push(`${esc(particlePixelSize(job.pixel_size_A))} Å / pixel`);
  if(job.box_size!=null)metadata.push(`${esc(job.box_size)}${kind==='extraction'?'²':'³'} px`);
  if(classes)metadata.push(`${esc(job.options?.classes ?? job.classes?.length ?? '—')} classes`);
  const iteration=active ? (p.iteration ?? job.latest_iteration) : (job.latest_iteration ?? p.iteration);
  const iterationSummary=["classification","refinement3d"].includes(kind) && iteration!=null
    ? `<p class="hint sta-job-iterations">Iterations · ${esc(iteration)}</p>` : "";
  const progress=[];
  if(active){
    if(p.phase)progress.push(esc(p.phase));
    if(p.iteration!=null)progress.push(`Iteration ${esc(p.iteration)}${p.iterations!=null?' / '+esc(p.iterations):''}`);
    else if(p.processed!=null && p.total!=null)progress.push(`${esc(p.processed)} / ${esc(p.total)} particles`);
    if(p.optimizer_iteration!=null)progress.push(`Optimizer iteration ${esc(p.optimizer_iteration)}`);
  }
  const fits=d.fits || [];
  const observation=['polishing','ctf_refinement'].includes(kind)
    ? `${fits.length?`<p class="hint">Fits converged · ${fits.filter(f=>f.converged).length} / ${fits.length}</p>`:''}${d.median_shift_A!=null?`<p class="hint">Median shift · ${particleMetricNumber(d.median_shift_A)} Å</p>`:''}${d.boundary_groups?.length?`<p class="hint">${esc(d.boundary_groups.length)} fits at bounds · original CTF retained</p>`:''}${particleObservationReport(assembly,job,kind)}` : '';
  const previewKind=kind;
  const preview=(job.state==='completed' && (job.volumes || job.classes || ['refinement3d','postprocessing'].includes(kind))) || (kind==='refinement3d' && job.latest_iteration);
  const retry=kind==='extraction' && !active && job.state!=='completed' && job.work===assembly.extraction?.work;
  return `<article class="assembly-task-card" data-sta-kind="${esc(kind)}" ${classes?'data-classification-card="true"':''} ${kind==='extraction'?'data-extraction-card="true"':''} style="--sta-preview-count:${classes?(job.classes?.length || job.options?.classes || 1):1}" data-job-number="${particleJobNumber(job,0)}">
    <header class="assembly-card-header"><h3>${esc(particleJobName(job,'Job'))}</h3>${job.job_id?`<span class="assembly-job-id">Slurm ${esc(job.job_id)}</span>`:''}<span class="${particleTaskBadgeClass(job.state)}">${jobStateLabel(job.state || 'pending')}</span></header>
    <div class="assembly-card-content">
      ${particles!=null?`<p class="assembly-card-metric">${esc(particles)} <small>particles</small></p>`:''}
      ${metadata.length?`<p class="hint sta-job-metadata">${metadata.join(' · ')}</p>`:''}
      ${extractionSource}
      ${iterationSummary}
      ${progress.length?`<p class="hint sta-job-progress" role="status">${progress.join(' · ')}</p>`:''}
      ${kind==='refinement3d'?refinementMetrics(job):kind==='postprocessing'?postprocessDetails(assembly,job):observation}
      ${job.error?`<p class="form-error" role="alert">${esc(job.error)}</p>`:''}
      ${preview?particleVolumeThumbnails(assembly,job,previewKind):''}
    </div>
    ${retry?`<footer class="assembly-card-actions"><button class="button secondary" type="button" data-extract-assembly="${esc(assembly.id)}">${job.state==='failed'?'Retry extraction':'Extract particles'}</button></footer>`:''}
  </article>`;
}
