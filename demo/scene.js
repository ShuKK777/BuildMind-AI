import * as THREE from 'three';
import {makePrintWorld} from './print-scene.js';
import {SoftwareRenderer} from './software-renderer.js';
import {OrbitControls} from './vendor/OrbitControls.js';
import {easeMotion} from './motion.js';

const palette={glass:0x668e9e,cream:0xe9e9df,white:0xf2f2ed,land:0xd8e2cf,road:0xa8b5b8,water:0x79b8c7,green:0x628a70};
const mat=(color,extra={})=>new THREE.MeshStandardMaterial({color,roughness:.68,...extra});
const boxGeo=new THREE.BoxGeometry(1,1,1),sphereGeo=new THREE.IcosahedronGeometry(1,1);
function block(parent,x,y,z,w,h,d,material){const geometry=(w>50||d>50)&&h<10?new THREE.BoxGeometry(1,1,1,Math.ceil(w/9),1,Math.ceil(d/9)):boxGeo;const mesh=new THREE.Mesh(geometry,material);mesh.position.set(x,y+h/2,z);mesh.scale.set(w,h,d);mesh.castShadow=true;mesh.receiveShadow=true;parent.add(mesh);return mesh;}
function oval(parent,x,y,z,rx,rz,h,material,angle=0){const m=new THREE.Mesh(new THREE.CylinderGeometry(1,1,h,48),material);m.scale.set(rx,1,rz);m.position.set(x,y+h/2,z);m.rotation.y=angle;m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
function facadeTexture(){const c=document.createElement('canvas');c.width=128;c.height=128;const ctx=c.getContext('2d');ctx.fillStyle='#adc4cd';ctx.fillRect(0,0,128,128);for(let row=0;row<8;row++)for(let col=0;col<4;col++){ctx.fillStyle=['#6f929f','#88a7b3','#b3cbd1','#769ba9'][(row*7+col*3)%4];ctx.fillRect(col*32+3,row*16+2,26,11);}const t=new THREE.CanvasTexture(c);t.colorSpace=THREE.SRGBColorSpace;return t;}

export class CityScene {
 constructor(host,onSelect){
  this.host=host;this.onSelect=onSelect;this.state={};this.buildings=[];this.autoOrbit=false;this.active=true;this.pointerStart=null;this.reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');this.cameraMotion=null;this.motionPaused=false;this.lastFrame=performance.now();this.daylight=1;this.targetDaylight=1;this.dayColor=new THREE.Color(0xe8f0f3);this.nightColor=new THREE.Color(0x152b3f);
  try {const probe=document.createElement('canvas');const gl=probe.getContext('webgl2');if(!gl)throw new Error('WebGL unavailable');gl.getExtension('WEBGL_lose_context')?.loseContext();this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});} catch {this.renderer=new SoftwareRenderer();}
  this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.25;
  this.renderer.domElement.setAttribute('aria-label','Interactive 3D district');this.renderer.domElement.setAttribute('tabindex','0');host.appendChild(this.renderer.domElement);
  this.scene=new THREE.Scene();this.scene.background=new THREE.Color(0xe8f0f3);this.scene.fog=new THREE.Fog(0xe8f0f3,450,1000);
  this.camera=new THREE.OrthographicCamera(-170,170,120,-120,.1,1500);this.camera.position.set(185,240,320);
  this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.target.set(0,12,0);this.controls.enableDamping=true;this.controls.dampingFactor=.07;this.controls.minZoom=.65;this.controls.maxZoom=3.5;this.controls.maxPolarAngle=Math.PI/2.18;this.controls.minPolarAngle=.18;this.controls.enablePan=true;
  this.hemi=new THREE.HemisphereLight(0xe8f5ff,0x8d9d89,2.5);this.scene.add(this.hemi);
  this.sun=new THREE.DirectionalLight(0xfff7df,3.1);this.sun.position.set(-120,230,100);this.sun.castShadow=true;this.sun.shadow.mapSize.set(2048,2048);Object.assign(this.sun.shadow.camera,{left:-180,right:180,top:180,bottom:-180,near:1,far:600});this.sun.shadow.normalBias=.6;this.scene.add(this.sun);
  this.mats={cream:mat(palette.cream),white:mat(palette.white),grass:mat(palette.land),road:mat(palette.road),path:mat(0xeeeade),glass:mat(palette.glass,{metalness:.36,roughness:.3}),water:mat(palette.water,{metalness:.25,roughness:.26}),facade:mat(0xffffff,{map:facadeTexture(),metalness:.22,roughness:.4}),dark:mat(0x526c77),tree:mat(palette.green),trunk:mat(0x9b8b72),edge:mat(0x788b96)};
  this.world=new THREE.Group();this.scene.add(this.world);this.landmarks=new THREE.Group();this.landmarks.position.set(4,0,-5);this.world.add(this.landmarks);
  this.greenGroup=new THREE.Group();this.world.add(this.greenGroup);this.future=new THREE.Group();this.world.add(this.future);
  makePrintWorld(this);
  const ringGeo=new THREE.RingGeometry(22,23.2,80);this.selection=new THREE.Mesh(ringGeo,new THREE.MeshBasicMaterial({color:0x2e9fb1,transparent:true,opacity:.8,side:THREE.DoubleSide}));this.selection.rotation.x=-Math.PI/2;this.selection.position.set(4,1.5,-5);this.world.add(this.selection);
  this.raycaster=new THREE.Raycaster();this.pointer=new THREE.Vector2();
  this.renderer.domElement.addEventListener('pointerdown',e=>{this.pointerStart=[e.clientX,e.clientY];this.cameraMotion=null;this.controls.enableDamping=true;});
  this.renderer.domElement.addEventListener('wheel',()=>{this.cameraMotion=null;this.controls.enableDamping=true;},{passive:true});
  this.renderer.domElement.addEventListener('pointerup',e=>this.pick(e));
  this.renderer.domElement.addEventListener('keydown',e=>{if(e.key==='Home')this.resetCamera();if(e.key==='+'||e.key==='='){e.preventDefault();this.zoom(1.15);}if(e.key==='-'){e.preventDefault();this.zoom(.87);}});
  this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(host);this.resize();
  this.renderer.setAnimationLoop(now=>this.frame(now));
 }
 createLandmark(variant,height,material){const g=new THREE.Group();const factor=[.82,1,1.18][height];const glass=mat(material?0x8ba29c:0x709dad,{metalness:.48,roughness:.24,emissive:0x162c32,emissiveIntensity:this.state.night?.7:0});const white=mat(material?0xd6c9ae:0xf0f0e9,{roughness:.45});const planting=mat(0x6d9473);
  g.userData.id='A02';block(g,0,1.7,0,37,3,32,white);
  if(variant===0){
   // Two continuously twisting elliptical shells, with pronounced white ribs.
   for(let side=0;side<2;side++){const x=side?9:-9,levels=side?22:27;
    for(let i=0;i<levels;i++){const t=i/levels,angle=(side?-1:1)*t*.65;const rx=(side?7.2:8.3)*(1-.22*t)+Math.sin(t*Math.PI)*2;const rz=10*(1-.25*t);const xx=x+Math.sin(t*Math.PI)* (side?3:-3);oval(g,xx,4.7+i*2.35*factor,0,rx,rz,2.35*factor,glass,angle);oval(g,xx,4.7+i*2.35*factor,0,rx+.65,rz+.65,.45,white,angle);}
   }
   oval(g,0,4.8,0,24,18,1,white);oval(g,0,6,0,16,12,.25,planting);
  }else if(variant===1){
   for(const [x,levels] of [[-10,30],[10,25]]){block(g,x,4.7,0,13,levels*2.6*factor,17,glass);
    for(let i=0;i<=levels;i++)block(g,x,4.7+i*2.6*factor,0,13.6,.38,17.6,white);
    for(let j=-2;j<=2;j++){block(g,x+j*2.5,4.7,8.65,.18,levels*2.6*factor,.2,white);block(g,x+j*2.5,4.7,-8.65,.18,levels*2.6*factor,.2,white);}
    block(g,x,4.7+levels*2.6*factor,0,14,1,18,white);
   }
   block(g,0,46*factor,0,10,3.2,8,glass);block(g,0,46*factor+3.2,0,11,.5,9,white);
  }else{
   for(let i=0;i<10;i++){const y=4.7+i*5.5*factor,w=34-i*1.8,d=28-i*.9,x=Math.sin(i*.55)*3;
    block(g,x,y,0,w,5.1*factor,d,glass);block(g,x,y+5.1*factor,0,w+1.6,.55,d+1.6,white);
    block(g,x,y+5.65*factor,d/2-1,w-2,.3,2.2,planting);
    if(this.state.green!==false)for(let j=0;j<4;j++){const tree=new THREE.Mesh(sphereGeo,planting);tree.position.set(x-w/2+3+j*(w-6)/3,y+6.7*factor,d/2-1);tree.scale.set(1.3,1.9,1.3);g.add(tree);}
   }
  }
  g.traverse(o=>{if(o.isMesh){o.userData.id='A02';o.castShadow=true;o.receiveShadow=true;}});return g;
 }
 clearLandmark(){const materials=new Set(),geometries=new Set();for(const child of this.landmarks.children)child.traverse(o=>{if(o.isMesh){if(o.geometry!==boxGeo&&o.geometry!==sphereGeo)geometries.add(o.geometry);materials.add(o.material);}});geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());this.landmarks.clear();}
 apply(state){
  this.needsRender=true;const prev=this.state;
  const changed=['variant','height','material','green','stage','original'].some(k=>state[k]!==prev[k]);
  if(prev.stage!==undefined&&changed&&this.active)this.captureTransition();
  this.state={...state};if(['variant','height','material','green'].some(k=>state[k]!==prev[k])||!this.landmarks.children.length){this.clearLandmark();const g=this.createLandmark(state.variant,state.height,state.material);const bounds=new THREE.Box3().setFromObject(g),size=new THREE.Vector3();bounds.getSize(size);const targetHeight=([72,86,66][state.variant]-6)*1.5*[.82,1,1.18][state.height];g.scale.set(22/size.x,targetHeight/size.y,22/size.z);g.position.y=9-bounds.min.y*g.scale.y;this.landmarks.add(g);}
  this.landmarks.visible=state.stage>0&&!state.original;this.originalTower.visible=state.stage>0&&state.original;this.landmarks.scale.setScalar(1);this.before.visible=state.stage===0;this.beforeGround.visible=state.stage===0;this.afterGround.visible=state.stage>0;this.future.children.forEach(g=>g.visible=state.stage>0&&g.userData.phase<=state.stage);this.greenGroup.visible=state.green;
  this.targetDaylight=state.night?0:1;
  if(prev.night===undefined||this.reducedMotion.matches)this.daylight=this.targetDaylight;
  this.updateAtmosphere();
  this.select(state.selected);this.autoOrbit=state.autoOrbit||false;
 }
 select(id){if(id==='A02'){this.selection.position.set(-15,9.2,1.5);this.selection.scale.setScalar(.62);this.selection.visible=this.state.stage>0;return;}
  const b=this.buildings.find(b=>b.userData.id===id);if(b){this.selection.position.set(b.userData.x,9.2,b.userData.z);this.selection.scale.setScalar(.48);this.selection.visible=b.visible&&(b.parent!==this.before||this.before.visible);}else this.selection.visible=false;
 }
 pick(e){if(!this.pointerStart||Math.hypot(e.clientX-this.pointerStart[0],e.clientY-this.pointerStart[1])>6)return;const r=this.renderer.domElement.getBoundingClientRect();this.pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);this.raycaster.setFromCamera(this.pointer,this.camera);const hits=this.raycaster.intersectObjects([this.landmarks,this.originalTower,this.future,this.before],true);const h=hits.find(h=>{let a=h.object;while(a){if(!a.visible)return false;a=a.parent;}return h.object.userData.id;});if(h)this.onSelect(h.object.userData.id);}
 captureTransition(){
  if(this.reducedMotion.matches||this.host.clientWidth<1)return;
  this.crossfadeAnimation?.cancel();this.overlay?.remove();
  // Capture immediately after rendering so WebGL's default cleared drawing buffer is safe.
  this.renderer.render(this.scene,this.camera);
  const overlay=document.createElement('canvas');overlay.className='scene-crossfade';overlay.setAttribute('aria-hidden','true');
  overlay.width=this.renderer.domElement.width;overlay.height=this.renderer.domElement.height;
  overlay.getContext('2d').drawImage(this.renderer.domElement,0,0);
  this.host.appendChild(overlay);this.overlay=overlay;
  const animation=overlay.animate([{opacity:1},{opacity:0}],{duration:720,easing:'cubic-bezier(.22,.61,.36,1)',fill:'forwards'});
  this.crossfadeAnimation=animation;
  if(this.motionPaused)animation.pause();
  animation.finished.then(()=>overlay.remove(),()=>overlay.remove());
 }
 updateAtmosphere(){
  const light=this.daylight;
  this.scene.userData.daylight=light;
  this.scene.background.copy(this.nightColor).lerp(this.dayColor,light);this.scene.fog.color.copy(this.scene.background);
  this.hemi.intensity=.8+1.7*light;this.sun.intensity=.65+2.45*light;this.renderer.toneMappingExposure=1.05+.2*light;
  this.mats.facade.emissive.set(0xe6bc76);this.mats.facade.emissiveIntensity=.32*(1-light);
  this.landmarks.traverse(o=>{if(o.isMesh&&o.material.metalness>.4){o.material.emissive.set(0x92b9c4);o.material.emissiveIntensity=.35*(1-light);}});
 }
 moveCamera(position,target,zoom,duration=1200){
  this.controls.autoRotate=false;this.controls.enableDamping=false;this.controls.update(0);
  const toPosition=new THREE.Vector3(...position),toTarget=new THREE.Vector3(...target);
  if(this.reducedMotion.matches){this.camera.position.copy(toPosition);this.controls.target.copy(toTarget);this.camera.zoom=zoom;this.camera.updateProjectionMatrix();this.controls.update(0);this.controls.enableDamping=true;this.needsRender=true;return;}
  this.cameraMotion={fromPosition:this.camera.position.clone(),fromTarget:this.controls.target.clone(),fromZoom:this.camera.zoom,toPosition,toTarget,toZoom:zoom,elapsed:0,duration};
  this.needsRender=true;
 }
 resetCamera(){this.moveCamera([185,240,320],[0,12,0],1);}
 focus(){this.moveCamera([138,150,220],[-15,30,1.5],1.18);}
 zoom(factor){this.moveCamera(this.camera.position.toArray(),this.controls.target.toArray(),THREE.MathUtils.clamp(this.camera.zoom*factor,.65,3.5),420);}
 setPlaybackPaused(paused){
  this.motionPaused=paused;this.lastFrame=performance.now();
  if(this.crossfadeAnimation?.playState==='running'&&paused)this.crossfadeAnimation.pause();
  else if(this.crossfadeAnimation?.playState==='paused'&&!paused)this.crossfadeAnimation.play();
 }
 resize(){const w=this.host.clientWidth,h=this.host.clientHeight;if(w<1||h<1)return;const aspect=w/h;this.camera.left=-145*aspect;this.camera.right=145*aspect;this.camera.top=145;this.camera.bottom=-145;this.camera.updateProjectionMatrix();const current=this.renderer.getSize(new THREE.Vector2());if(current.x!==w||current.y!==h)this.renderer.setSize(w,h);this.needsRender=true;this.frame();}
 frame(now=performance.now()){
  const delta=Math.min(.05,Math.max(0,(now-this.lastFrame)/1000));this.lastFrame=now;
  if(!this.active||document.hidden)return;
  const motion=this.cameraMotion;
  if(motion&&!this.motionPaused){
   motion.elapsed+=delta*1000;const progress=easeMotion(motion.elapsed/motion.duration);
   this.camera.position.lerpVectors(motion.fromPosition,motion.toPosition,progress);
   this.controls.target.lerpVectors(motion.fromTarget,motion.toTarget,progress);
   this.camera.zoom=THREE.MathUtils.lerp(motion.fromZoom,motion.toZoom,progress);this.camera.updateProjectionMatrix();this.needsRender=true;
   if(motion.elapsed>=motion.duration){this.cameraMotion=null;this.controls.enableDamping=true;}
  }
  if(!this.motionPaused&&Math.abs(this.daylight-this.targetDaylight)>.001){
   this.daylight=THREE.MathUtils.damp(this.daylight,this.targetDaylight,4.8,delta);
   if(Math.abs(this.daylight-this.targetDaylight)<=.001)this.daylight=this.targetDaylight;
   this.updateAtmosphere();this.needsRender=true;
  }
  this.controls.autoRotate=this.autoOrbit&&!this.cameraMotion&&!this.motionPaused&&!this.reducedMotion.matches;this.controls.autoRotateSpeed=.65;
  const moved=this.controls.update(delta);
  if(moved||this.needsRender||this.controls.autoRotate){this.renderer.render(this.scene,this.camera);this.needsRender=false;}
 }
 thumbnail(variant){const scene=new THREE.Scene();scene.background=new THREE.Color(0xeaf0ee);scene.add(new THREE.HemisphereLight(0xffffff,0xadc5b7,3));const light=new THREE.DirectionalLight(0xfff5dd,3);light.position.set(-50,100,70);scene.add(light);const original=variant===-1;const g=original?this.originalPart.clone(true):this.createLandmark(variant,1,0);if(original)g.position.set(15,-9,-1.5);scene.add(g);const camera=new THREE.OrthographicCamera(-90,90,63,-63,.1,500);camera.position.set(95,105,130);camera.lookAt(0,43,0);const oldSize=new THREE.Vector2();this.renderer.getSize(oldSize);const pixel=this.renderer.getPixelRatio();this.renderer.setPixelRatio(1);this.renderer.setSize(300,210,false);this.renderer.render(scene,camera);const image=this.renderer.domElement.toDataURL('image/png');this.renderer.setPixelRatio(pixel);this.renderer.setSize(oldSize.x,oldSize.y,false);const materials=new Set(),geometries=new Set();g.traverse(o=>{if(o.isMesh){materials.add(o.material);if(o.geometry!==boxGeo&&o.geometry!==sphereGeo)geometries.add(o.geometry);}});if(!original){materials.forEach(m=>m.dispose());geometries.forEach(g=>g.dispose());}return image;}
 dispose(){this.crossfadeAnimation?.cancel();this.overlay?.remove();this.renderer.setAnimationLoop(null);this.controls.dispose();this.resizeObserver.disconnect();this.renderer.dispose();}
}
