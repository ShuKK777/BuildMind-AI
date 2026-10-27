import * as THREE from '../../demo-v0.2/vendor/three.module.min.js';
import {SoftwareRenderer} from './software-renderer.js';
globalThis.document={createElement:()=>new OffscreenCanvas(1,1)};
const renderer=new SoftwareRenderer(),scene=new THREE.Scene(),camera=new THREE.Camera(),geometries=new Map(),materials=new Map(),textures=new Map(),meshes=new Map();
camera.matrixAutoUpdate=false;
self.onmessage=({data})=>{try{const p=data.packet;if(!p)return;
 for(const t of p.textures){const map=new THREE.DataTexture(t.data,t.w,t.h,THREE.RGBAFormat);map.image={data:t.data,width:t.w,height:t.h};textures.set(t.id,map);}
 for(const d of p.geometry){const g=new THREE.BufferGeometry();for(const [name,a]of Object.entries(d.attributes))g.setAttribute(name,new THREE.BufferAttribute(a.array,a.itemSize));if(d.index)g.setIndex(new THREE.BufferAttribute(d.index,1));for(const group of d.groups)g.addGroup(group.start,group.count,group.materialIndex);geometries.set(d.id,g);}
 for(const d of p.materials){let m=materials.get(d.id);if(!m){m=new THREE.MeshStandardMaterial();materials.set(d.id,m);}m.name=d.name;m.color.fromArray(d.color);Object.assign(m,{opacity:d.opacity,transparent:d.transparent,depthWrite:d.depthWrite,metalness:d.metalness,side:d.side,visible:d.visible,map:textures.get(d.map)||null});}
 for(const m of meshes.values())m.visible=false;
 for(const d of p.draws){let m=meshes.get(d.id);if(!m){const material=Array.isArray(d.material)?d.material.map(id=>materials.get(id)):materials.get(d.material);m=d.count?new THREE.InstancedMesh(geometries.get(d.geometry),material,d.count):new THREE.Mesh(geometries.get(d.geometry),material);m.matrixAutoUpdate=false;scene.add(m);meshes.set(d.id,m);}m.visible=true;m.matrix.fromArray(d.matrix);m.matrixWorldNeedsUpdate=true;if(d.instances)m.instanceMatrix.array.set(d.instances);if(d.colors)m.instanceColor=new THREE.InstancedBufferAttribute(d.colors,3);}
 scene.background=new THREE.Color().fromArray(p.background);scene.fog=p.fog?new THREE.Fog(new THREE.Color().fromArray(p.fog.color),p.fog.near,p.fog.far):null;scene.userData.daylight=p.daylight;camera.matrix.fromArray(p.camera.world);camera.matrix.decompose(camera.position,camera.quaternion,camera.scale);camera.matrixWorldNeedsUpdate=true;camera.projectionMatrix.fromArray(p.camera.projection);camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
 if(renderer.width!==p.width||renderer.height!==p.height)renderer.setSize(p.width,p.height,false);
 if(data.type==='prepare'){for(const g of p.geometry)renderer.triangles(geometries.get(g.id));self.postMessage({type:'prepared',id:data.id});return;}
 const start=performance.now();renderer.render(scene,camera);const bitmap=renderer.domElement.transferToImageBitmap();self.postMessage({type:'frame',id:data.id,width:p.width,height:p.height,bitmap,milliseconds:performance.now()-start},[bitmap]);
 }catch(error){self.postMessage({type:'error',message:error.message,stack:error.stack});}};
