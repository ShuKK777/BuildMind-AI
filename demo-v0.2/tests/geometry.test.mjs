import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import * as THREE from '../vendor/three.module.min.js';
const require=createRequire(import.meta.url);
const {createCanvas}=require('../tools/node_modules/@napi-rs/canvas');
globalThis.document={createElement(tag){assert.equal(tag,'canvas');return createCanvas(1,1);}};
const {makeBuildingFactory,UPSTREAM}=await import('../../packages/building-kit/adapter.js');
const {INITIAL_DESIGN}=await import('../../packages/scenario/model.js');
const factory=makeBuildingFactory();
const fingerprint=g=>{const hash=createHash('sha256');g.traverse(o=>{if(o.isMesh){const a=o.geometry.attributes.position.array;hash.update(new Uint8Array(a.buffer,a.byteOffset,a.byteLength));if(o.isInstancedMesh)hash.update(new Uint8Array(o.instanceMatrix.array.buffer));}});return hash.digest('hex');};
test('real upstream generation fits every landmark family to the lot with actual envelope height',()=>{for(const family of ['flow','twins','gardens']){const {group,metadata}=factory.build({...INITIAL_DESIGN,family});group.updateMatrixWorld(true);const b=new THREE.Box3().setFromObject(group),size=b.getSize(new THREE.Vector3());assert.ok(size.x<=41.501&&size.z<=42.501);assert.ok(Math.abs(size.y-metadata.envelopeHeight)<.0001);assert.ok(size.y>40);assert.ok(Number.isFinite(metadata.bodyHeight));group.traverse(o=>{if(o.isMesh){assert.equal(o.userData.id,'A02');for(const n of o.geometry.attributes.position.array)assert.ok(Number.isFinite(n));}});if(family!=='flow')assert.equal(metadata.source,UPSTREAM);}});
test('seeded candidates are reproducible and floor changes affect actual geometry',()=>{const a=factory.build(INITIAL_DESIGN),b=factory.build(INITIAL_DESIGN),c=factory.build({...INITIAL_DESIGN,seed:50}),d=factory.build({...INITIAL_DESIGN,floors:34});assert.equal(fingerprint(a.group),fingerprint(b.group));assert.notEqual(fingerprint(a.group),fingerprint(c.group));assert.ok(d.metadata.envelopeHeight>a.metadata.envelopeHeight);assert.notStrictEqual(a.group,b.group);});
test('background floor count is part of the real generator cache key',()=>{const a=factory.build({...INITIAL_DESIGN,floors:5},{id:'A01',backgroundType:'mixeduse'}),b=factory.build({...INITIAL_DESIGN,floors:9},{id:'A03',backgroundType:'mixeduse'});assert.ok(b.metadata.envelopeHeight>a.metadata.envelopeHeight);assert.equal(a.metadata.design.floors,5);assert.equal(b.metadata.design.floors,9);});

test('original Northern Metropolis material upgrade preserves every source mesh position',async()=>{const {printParts}=await import('../../demo/assets/print-assembly.js');const {buildOriginalNorth}=await import('../../packages/scene-core/north-scene.js');const city={world:new THREE.Group(),mats:{},metadata:new Map(),lotGroups:new Map(),cpu:false};buildOriginalNorth(city);assert.equal(city.sourceGroups.size,19);for(const [id,g]of city.sourceGroups){const p=printParts.find(p=>p.id===id),v=Float32Array.from(new Float32Array(Uint8Array.from(atob(p.v),c=>c.charCodeAt(0)).buffer));assert.deepEqual(g.children[0].geometry.attributes.position.array,v,id);assert.ok(Array.isArray(g.children[0].material));}assert.equal(city.before.children.length,8);});
