import * as THREE from 'three';
import {printParts} from './assets/print-assembly.js';
const unitBox=new THREE.BoxGeometry(1,1,1),unitTree=new THREE.IcosahedronGeometry(1,1);
const mat=c=>new THREE.MeshStandardMaterial({color:c,roughness:.8});
const decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
export function makePrintWorld(city){
 const w=city.world,m=city.mats;city.before=new THREE.Group();w.add(city.before);city.originalTower=new THREE.Group();w.add(city.originalTower);city.afterGround=new THREE.Group();w.add(city.afterGround);city.beforeGround=new THREE.Group();w.add(city.beforeGround);
 const materials={earth:mat(0xb7aa83),grass:mat(0x8dac72),road:mat(0xa9b1b3),path:mat(0xddd8c9),water:mat(0x75b9c7),fieldgold:mat(0xcdbb70),fieldgreen:mat(0xa2ac68),base:mat(0xd6d4cc)};
 const box=(parent,cx,cy,width,depth,z,h,material)=>{const geo=(width>40||depth>40)?new THREE.BoxGeometry(1,1,1,Math.ceil(width/8),1,Math.ceil(depth/8)):unitBox;const mesh=new THREE.Mesh(geo,material);mesh.position.set((cx-75)*1.5,(z+h/2)*1.5,(75-cy)*1.5);mesh.scale.set(width*1.5,h*1.5,depth*1.5);mesh.castShadow=true;mesh.receiveShadow=true;parent.add(mesh);return mesh;};
 box(w,75,75,150,150,0,4.5,materials.base);
 // Lower-resolution reconstruction of the exact height-field formula supplied
 // with the print pack. Buildings retain their original STL geometry and colors.
 const n=76,positions=[],colors=[],indices=[];
 for(let j=0;j<n;j++)for(let i=0;i<n;i++){const x=i*150/(n-1),y=j*150/(n-1);let z=6;let gate=THREE.MathUtils.clamp((y-101)/13,0,1);gate=gate*gate*(3-2*gate);let hills=0;for(const [cx,cy,h,sx,sy] of [[25,129,21,17,17],[69,137,29,19,19],[117,131,24,19,16]])hills+=h*Math.exp(-(((x-cx)/sx)**2+((y-cy)/sy)**2));hills*=1+.09*Math.sin(x*.34+y*.22)*Math.sin(y*.37);z+=gate*hills;
  let water=false;for(const [cx,cy,rx,ry] of [[24,14,20,9],[73,13,19,9],[123,14,21,9]]){const d=((x-cx)/rx)**4+((y-cy)/ry)**4;if(d<1){z-=1.3*THREE.MathUtils.clamp((1-d)*5,0,1);water=true;}}
  const river=87+5*Math.sin((y-40)*.095);if((Math.abs(x-river)<3.4+1.4*Math.sin(y*.12)&&y>38&&y<105)||((x-88)/9)**2+((y-75)/12)**2<1){z=4.7;water=true;}
  // Color and shape of water are shared by both development states.
  positions.push((x-75)*1.5,z*1.5,(75-y)*1.5);const c=new THREE.Color(water?0x75b9c7:y>108?(z>23?0x7d986f:0x64845d):0xaeb996);colors.push(c.r,c.g,c.b);
 }
 for(let j=0;j<n-1;j++)for(let i=0;i<n-1;i++){const a=j*n+i,b=a+1,c=a+n,d=c+1;indices.push(a,b,d,a,d,c);}
 const terrainGeo=new THREE.BufferGeometry();terrainGeo.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));terrainGeo.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));terrainGeo.setIndex(indices);terrainGeo.computeVertexNormals();const terrain=new THREE.Mesh(terrainGeo,new THREE.MeshStandardMaterial({vertexColors:true,roughness:1}));terrain.receiveShadow=true;w.add(terrain);
 const perimeter=[...Array.from({length:n},(_,i)=>i),...Array.from({length:n-1},(_,j)=>(j+1)*n+n-1),...Array.from({length:n-1},(_,i)=>n*n-2-i),...Array.from({length:n-2},(_,j)=>(n-2-j)*n)];
 const skirt=[];for(let k=0;k<perimeter.length;k++){const a=positions.slice(perimeter[k]*3,perimeter[k]*3+3),b=positions.slice(perimeter[(k+1)%perimeter.length]*3,perimeter[(k+1)%perimeter.length]*3+3),ab=[a[0],6.75,a[2]],bb=[b[0],6.75,b[2]];skirt.push(...a,...ab,...b,...b,...ab,...bb);}
 const skirtGeo=new THREE.BufferGeometry();skirtGeo.setAttribute('position',new THREE.Float32BufferAttribute(skirt,3));skirtGeo.computeVertexNormals();w.add(new THREE.Mesh(skirtGeo,materials.grass));

 for(const args of [[75,32,150,6],[53,69,3.2,74],[122,73,3.2,82]])box(w,...args,5.6,1.1,materials.road);
 box(w,87,46,22,3,4.5,2.2,materials.path);
 for(const args of [[75,62,142,3],[75,89,142,3],[75,109,140,3]])box(city.afterGround,...args,5.6,1.1,materials.road);
 for(const [xi,x] of [15,36,68,110,138].entries())for(const [yi,y] of [51,74,95].entries()){if(x===110&&y===95)continue;const col=(xi+yi)%2?materials.fieldgold:materials.fieldgreen;box(city.beforeGround,x,y,13,15,5.7,.9,col);for(const off of [-4,0,4])box(city.beforeGround,x+off,y,.65,13,6.3,.4,col);}
 // Decode immutable per-part geometry; source Z-up maps to Three.js Y-up.
 for(const p of printParts){const bytes=decode(p.v),vertices=new Float32Array(bytes.buffer),colorBytes=decode(p.c),rgb=new Float32Array(vertices.length),color=new THREE.Color();for(let i=0;i<colorBytes.length/3;i++){color.setRGB(colorBytes[i*3]/255,colorBytes[i*3+1]/255,colorBytes[i*3+2]/255,THREE.SRGBColorSpace);for(let j=0;j<3;j++)rgb.set([color.r,color.g,color.b],i*9+j*3);}
  const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(vertices,3));geo.setAttribute('color',new THREE.BufferAttribute(rgb,3));geo.computeVertexNormals();const material=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.65,metalness:.08});const mesh=new THREE.Mesh(geo,material);mesh.castShadow=true;mesh.receiveShadow=true;mesh.userData.id=p.id;
  const g=new THREE.Group();g.add(mesh);g.userData={id:p.id,name:p.name,type:p.id[0]==='B'?'village':['A04','A13','A16','A19'].includes(p.id)?'housing':p.id==='A01'?'culture':'office',h:p.heightMm,phase:p.id==='A02'?2:Number(p.id.slice(1))%2?1:2,x:p.x,z:p.z};
  if(p.id==='A02'){city.originalTower.add(g);city.originalPart=g;}else if(p.id[0]==='B'){city.before.add(g);city.buildings.push(g);}else{city.future.add(g);city.buildings.push(g);}
 }
 const treeM=mat(0x4e805d);for(const [x,y] of [[7,39],[40,39],[59,39],[100,39],[142,39],[98,58],[98,67],[98,85],[80,88],[79,81],[77,55],[80,106],[95,105],[117,54],[117,85],[7,111],[40,112],[132,112]]){const mesh=new THREE.Mesh(unitTree,treeM);mesh.position.set((x-75)*1.5,12.3,(75-y)*1.5);mesh.scale.set(2.5,4,2.5);mesh.castShadow=true;city.greenGroup.add(mesh);}
 city.landmarks.position.set(-15,0,1.5);
}
