// Original-print-coordinate camera adaptation of Valdrada core/sim/pathcam.mjs (MIT).
// Copyright (c) 2026 Mehmet Kerem Turkcan; adaptations by BuildMind AI.
import * as THREE from 'three';
export class CameraRig {
 constructor(camera,controls){this.camera=camera;this.controls=controls;this.motion=null;}
 static easeEnds(x,f=.2){if(f<=.001)return x;const norm=1-f;return (x<f?x*x/(2*f):x>1-f?norm-(1-x)*(1-x)/(2*f):x-f/2)/norm;}
 to(position,target,duration=1.8){const from=this.camera.position.clone(),a=this.controls.target.clone(),to=new THREE.Vector3(...position),b=new THREE.Vector3(...target);this.controls.autoRotate=false;this.controls.enableDamping=false;this.controls.update(0);this.motion={t:0,duration,positions:new THREE.CatmullRomCurve3([from,from.clone().lerp(to,.35).add(new THREE.Vector3(0,6,0)),to],false,'catmullrom',.25),targets:new THREE.CatmullRomCurve3([a,a.clone().lerp(b,.5),b],false,'catmullrom',.25)};}
 update(dt){if(!this.motion)return false;const m=this.motion;m.t+=dt;const u=CameraRig.easeEnds(Math.min(1,m.t/m.duration));this.camera.position.copy(m.positions.getPointAt(u));this.controls.target.copy(m.targets.getPointAt(u));this.controls.update(0);if(m.t>=m.duration){this.motion=null;this.controls.enableDamping=true;}return true;}
 cancel(){this.motion=null;this.controls.enableDamping=true;}
}
export const VIEWS={overview:{position:[188,109,222],target:[0,35,-5]},waterfront:{position:[155,95,205],target:[-4,32,-6]},street:{position:[-32,13,64.5],target:[-15,53,1.5]},landmark:{position:[80,90,128],target:[-15,53,1.5]}};
