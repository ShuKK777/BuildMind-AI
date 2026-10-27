// Camera easing adapted from Valdrada core/sim/pathcam.mjs (MIT).
// Copyright (c) 2026 Mehmet Kerem Turkcan; adaptations by BuildMind AI.
import * as THREE from 'three';

export const VIEWS={
 overview:{position:[188,109,222],target:[0,35,-5]},
 waterfront:{position:[155,95,205],target:[-4,32,-6]},
 street:{position:[-32,13,64.5],target:[-15,53,1.5]},
 landmark:{position:[86,104,158],target:[-15,57,1.5]}
};

// Time-weighted Hermite interpolation preserves velocity across uneven shot lengths.
export const SHOWCASE_SHOTS=[
 {at:0,...VIEWS.overview},
 {at:5,position:[174,105,215],target:[-2,35,-2]},
 {at:10,position:[145,96,200],target:[-4,39,0]},
 {at:15,position:[100,101,173],target:[-15,52,1.5]},
 {at:20,position:[77,105,158],target:[-15,59,1.5]},
 {at:25,position:[35,109,170],target:[-15,57,1.5]},
 {at:30,position:[-39,104,180],target:[-15,54,1.5]},
 {at:35,position:[-50,111,195],target:[-15,54,1.5]},
 {at:41,position:[150,111,217],target:[0,36,-5]},
 {at:45,...VIEWS.overview}
];
function tangent(i,key,axis){if(i===0||i===SHOWCASE_SHOTS.length-1)return 0;const a=SHOWCASE_SHOTS[i-1],b=SHOWCASE_SHOTS[i+1];return (b[key][axis]-a[key][axis])/(b.at-a.at);}
export function showcasePose(seconds){const t=THREE.MathUtils.clamp(seconds,0,45);let i=SHOWCASE_SHOTS.findLastIndex(s=>s.at<=t);i=Math.min(i,SHOWCASE_SHOTS.length-2);const a=SHOWCASE_SHOTS[i],b=SHOWCASE_SHOTS[i+1],duration=b.at-a.at,u=(t-a.at)/duration,h00=2*u*u*u-3*u*u+1,h10=u*u*u-2*u*u+u,h01=-2*u*u*u+3*u*u,h11=u*u*u-u*u;const pose={};for(const key of ['position','target'])pose[key]=a[key].map((v,k)=>h00*v+h10*duration*tangent(i,key,k)+h01*b[key][k]+h11*duration*tangent(i+1,key,k));return pose;}

export class CameraRig {
 constructor(camera,controls){this.camera=camera;this.controls=controls;this.motion=null;this.velocity=new THREE.Vector3();this.targetVelocity=new THREE.Vector3();}
 static easeEnds(x){x=THREE.MathUtils.clamp(x,0,1);return x*x*x*(x*(x*6-15)+10);}
 to(position,target,duration=1.8){const end=new THREE.Vector3(...position),look=new THREE.Vector3(...target);if(this.camera.position.distanceTo(end)<.001&&this.controls.target.distanceTo(look)<.001){this.cancel();return;}const v=this.motion?this.velocity.clone():new THREE.Vector3(),tv=this.motion?this.targetVelocity.clone():new THREE.Vector3();this.controls.autoRotate=false;this.controls.enableDamping=false;this.controls.update(0);this.motion={t:0,duration,start:this.camera.position.clone(),end,look:this.controls.target.clone(),target:look,v,tv};}
 update(dt){if(!this.motion)return false;const m=this.motion;m.t+=Math.max(0,dt);const u=Math.min(1,m.t/m.duration),s=CameraRig.easeEnds(u),ds=30*u*u*(u-1)*(u-1)/m.duration;const carry=u*(1-u)*(1-u),dc=(1-4*u+3*u*u);this.camera.position.copy(m.start).lerp(m.end,s).addScaledVector(m.v,m.duration*carry);this.controls.target.copy(m.look).lerp(m.target,s).addScaledVector(m.tv,m.duration*carry);this.velocity.subVectors(m.end,m.start).multiplyScalar(ds).addScaledVector(m.v,dc);this.targetVelocity.subVectors(m.target,m.look).multiplyScalar(ds).addScaledVector(m.tv,dc);this.controls.update(0);if(u===1){this.motion=null;this.controls.enableDamping=true;this.velocity.setScalar(0);this.targetVelocity.setScalar(0);}return true;}
 cancel(){this.motion=null;this.velocity.setScalar(0);this.targetVelocity.setScalar(0);this.controls.enableDamping=true;}
}
