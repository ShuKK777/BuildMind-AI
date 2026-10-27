import {createRequire} from 'node:module';
import {writeFile} from 'node:fs/promises';
import * as THREE from 'three';
const require=createRequire(import.meta.url),{createCanvas}=require('@napi-rs/canvas');
globalThis.document={createElement(tag){if(tag!=='canvas')throw Error(tag);const c=createCanvas(1,1);c.style={};return c;}};
const {makeBuildingFactory}=await import('../../packages/building-kit/adapter.js');
const {SoftwareRenderer}=await import('../../packages/scene-core/software-renderer.js');
const {FAMILIES,INITIAL_DESIGN}=await import('../../packages/scenario/model.js');
const factory=makeBuildingFactory(),renderer=new SoftwareRenderer();renderer.setSize(360,238,false);
for(const f of FAMILIES){const r=factory.build({...INITIAL_DESIGN,family:f.id,floors:f.floors,seed:f.seed}),s=new THREE.Scene();s.background=new THREE.Color(0xeaf0ed);s.add(r.group);const h=r.metadata.envelopeHeight,span=Math.max(h*.7,42),c=new THREE.OrthographicCamera(-span,span,span*.66,-span*.66,.1,600);c.position.set(100,h*.65+80,120);c.lookAt(0,h*.48,0);renderer.render(s,c);await writeFile(new URL('../assets/landmarks/'+f.id+'.png',import.meta.url),renderer.domElement.toBuffer('image/png'));console.log(f.id,r.metadata.envelopeHeight);}
