import {INITIAL_DESIGN,normalizeDesign,SCENARIOS,designKey} from '../scenario/model.js';
export const KEY='buildmind-v02-north-state';
export function freshState(){return {lang:'zh',page:'plan',scenario:'mixed',stage:2,view:'overview',time:'golden',life:true,selected:'A02',applied:{...INITIAL_DESIGN},draft:{...INITIAL_DESIGN}};}
export function restoreState(raw){const s=freshState();if(!raw||typeof raw!=='object')return s;if(raw.lang==='en')s.lang='en';if(SCENARIOS.some(x=>x.id===raw.scenario))s.scenario=raw.scenario;if([0,1,2].includes(raw.stage))s.stage=raw.stage;if(['day','golden','night'].includes(raw.time))s.time=raw.time;s.applied=normalizeDesign(raw.applied);s.draft={...s.applied};if(typeof raw.life==='boolean')s.life=raw.life;return s;}
export function readState(storage){try{return restoreState(JSON.parse(storage.getItem(KEY)));}catch{return freshState();}}
export function persistState(storage,s){try{storage.setItem(KEY,JSON.stringify(s));}catch{}}
export function hasDraft(s){return designKey(s.draft)!==designKey(s.applied);}
export function applyDraft(s){s.applied=normalizeDesign(s.draft);s.draft={...s.applied};s.page='plan';s.stage=2;s.selected='A02';}
export function exportProject(s){return {schema:'buildmind/project@0.2',title:'Northern Waterfront / NM-02',coordinateSystem:{unit:'scene unit',sourceUnit:'millimetre',unitsPerMm:1.5,reference:'original 150 mm Northern Metropolis print assembly; not surveyed geography'},scenario:s.scenario,stage:s.stage,landmark:{id:'A02',design:normalizeDesign(s.applied)},presentation:{time:s.time,life:s.life}};}
export function importProject(raw){if(raw?.schema!=='buildmind/project@0.2'||raw.landmark?.id!=='A02'||!raw.landmark.design)throw Error('Unsupported project');const s=restoreState({scenario:raw.scenario,stage:raw.stage,applied:raw.landmark.design,time:raw.presentation?.time,life:raw.presentation?.life});return s;}
