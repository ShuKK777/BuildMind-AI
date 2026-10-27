import * as THREE from 'three';
import {createMaterials} from '../../third_party/valdrada/src/materials.js';
import {Batcher} from '../../third_party/valdrada/src/batcher.js';
import {makeKit} from '../../third_party/valdrada/src/kit.js';
import {generateBuilding} from '../../third_party/valdrada/src/buildings/index.js';
import {makeRng} from '../../third_party/valdrada/src/rng.js';
import {normalizeDesign} from '../scenario/model.js';
export const UPSTREAM='mkturkcan/valdrada@5dacc4ae0a34ee54fe70d4706cdf3f6662d596d2';
const basic=(color,extra={})=>new THREE.MeshStandardMaterial({color,roughness:.55,...extra});
function box(g,x,y,z,w,h,d,m){const o=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),m);o.position.set(x,y+h/2,z);o.castShadow=true;o.receiveShadow=true;g.add(o);return o;}
// Force the upstream's first massing choice without modifying its source files.
export function familyRng(seed,family){const r=makeRng(seed),weighted=r.weighted;let forced=false;r.weighted=pairs=>{if(!forced&&pairs.some(p=>p[0]==='slab')&&pairs.some(p=>p[0]==='twin')){forced=true;weighted(pairs);return family==='gardens'?'stepped':'twin';}return weighted(pairs);};return r;}
// Generated buildings keep their native architectural metres on all axes.
// Centre the model on its plot; never shrink storey heights to fit a print part.
function place(group){group.updateMatrixWorld(true);const b=new THREE.Box3().setFromObject(group);group.position.set(-(b.min.x+b.max.x)/2,-b.min.y,-(b.min.z+b.max.z)/2);const container=new THREE.Group();container.add(group);container.updateMatrixWorld(true);return {group:container,scale:1};}
function curveBuilding(d,M){const g=new THREE.Group();const glass=basic(d.palette==='warm'?0x93aaa5:0x71a3b1,{metalness:.38,roughness:.18}),white=M.get('limestone'),green=basic(0x76966b);box(g,0,0,0,d.width,3.5,d.depth,white);const floorH=3.1;
 for(const [side,count]of [[-1,d.floors],[1,d.floors-5]])for(let i=0;i<count;i++){const t=i/count,rx=d.width*(.18+.028*Math.sin(t*Math.PI)),rz=d.depth*(.39-.10*t);const shape=new THREE.CylinderGeometry(1,1,floorH,28);const shell=new THREE.Mesh(shape,glass);shell.scale.set(rx,1,rz);shell.position.set(side*d.width*.205+side*2.1*Math.sin(t*Math.PI),4+i*floorH,0);shell.rotation.y=side*t*.68;shell.castShadow=true;shell.receiveShadow=true;g.add(shell);const rib=new THREE.Mesh(new THREE.CylinderGeometry(1,1,.25,28),white);rib.position.copy(shell.position);rib.position.y+=floorH/2;rib.scale.set(rx+.35,1,rz+.35);rib.rotation.y=shell.rotation.y;g.add(rib);}
 if(d.green)for(let i=0;i<10;i++){const a=i/10*Math.PI*2;box(g,Math.cos(a)*d.width*.38,3.6,Math.sin(a)*d.depth*.35,2,.65,2,green);}return {group:g,bodyHeight:4+d.floors*floorH};}
export function makeBuildingFactory(){const {M,extra}=createMaterials();const cache=new Map(),nightMaterials=new Set();let nightAmount=0;
 function build(input,{id='A02',lotWidth=43,lotDepth=44,backgroundType=null}={}){const d=normalizeDesign(input),key=JSON.stringify([d,backgroundType,backgroundType?input.floors:null,lotWidth,lotDepth]);let template=cache.get(key);
  if(!template){let raw,bodyHeight;if(d.family==='flow'&&!backgroundType){const c=curveBuilding(d,M);raw=c.group;bodyHeight=c.bodyHeight;}else{const batcher=new Batcher(M),kit=makeKit(batcher,extra);const result=generateBuilding(backgroundType||'glasstower',{batcher,kit,extra},{frame:new THREE.Matrix4(),width:d.width,depth:d.depth,stories:backgroundType?input.floors:d.floors,commercial:true,corner:0,mirror:false,district:'buildmind-concept'},backgroundType?makeRng(d.seed):familyRng(d.seed,d.family));bodyHeight=result.height;raw=batcher.build();
   if(d.palette==='warm')raw.traverse(o=>{if(o.isMesh){o.material=o.material.clone();if(o.material.color)o.material.color.multiply(new THREE.Color(1,.94,.82));o.userData.ownedMaterial=true;}});
   if(!d.green)raw.traverse(o=>{if(o.isMesh&&/planter|foliage|leaf|bush/i.test(o.material?.name||''))o.visible=false;});
   if(d.family==='gardens'&&d.green&&!backgroundType){const green=basic(0x6b926b);for(let i=0;i<14;i++){const h=8+i*(bodyHeight-16)/14;const w=d.width*(1-.42*i/14);box(raw,0,h,-d.depth*.05,w,.65,2.6,green);for(let j=0;j<3;j++){const tree=new THREE.Mesh(new THREE.IcosahedronGeometry(.8,1),green);tree.position.set((j-1)*w*.32,h+1.35,-d.depth*.05);tree.scale.y=1.4;raw.add(tree);}}}
  }
  const fitted=place(raw);const group=fitted.group;const b=new THREE.Box3().setFromObject(group),size=b.getSize(new THREE.Vector3());template={group,metadata:{source:d.family==='flow'&&!backgroundType?'BuildMind curve generator + Valdrada material library':UPSTREAM,design:{...d,...(backgroundType?{floors:input.floors}:{})},bodyHeight:bodyHeight*fitted.scale,envelopeHeight:size.y,envelopeWidth:size.x,envelopeDepth:size.z,fitScale:fitted.scale,unit:'m',dimensionStatus:'design-geometry',typicalStoreyHeight:d.family==='flow'?3.1:3.05,fitsPlot:size.x<=lotWidth&&size.z<=lotDepth,frame:'local architectural design frame',bounds:{min:b.min.toArray(),max:b.max.toArray()}}};template.group.traverse(o=>{if(!o.isMesh)return;for(const m of Array.isArray(o.material)?o.material:[o.material])if(m.name==='litWindow'||m.name==='lampGlow'){nightMaterials.add(m);m.visible=nightAmount>.15;}});cache.set(key,template);
 }
 const group=template.group.clone(true);group.userData={id,...template.metadata};group.traverse(o=>{if(o.isMesh){o.userData={...o.userData,id};o.castShadow=true;o.receiveShadow=true;}});return {group,metadata:structuredClone(template.metadata)};
 }
 function setNight(amount){nightAmount=amount;for(const m of nightMaterials)m.visible=amount>.15;}
 return {build,setNight,materials:M,cache,dispose(){const geometries=new Set(),materials=new Set(M.values());for(const t of cache.values())t.group.traverse(o=>{if(o.isMesh){geometries.add(o.geometry);if(Array.isArray(o.material))o.material.forEach(m=>materials.add(m));else materials.add(o.material);}});geometries.forEach(g=>g.dispose());materials.forEach(m=>{for(const v of Object.values(m))if(v?.isTexture)v.dispose();m.dispose();});cache.clear();}};
}
