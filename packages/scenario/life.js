import {signalState} from '../../third_party/valdrada/client/src/sim/signals.js';
// A small concept-road simulation. Signal timing is upstream; routes and actors are ours.
export function loopPose(distance,halfWidth=100,halfDepth=63){const w=halfWidth*2,d=halfDepth*2,length=2*(w+d);let s=((distance%length)+length)%length;if(s<w)return {x:-halfWidth+s,z:halfDepth,angle:Math.PI/2,ew:true,toCorner:w-s};s-=w;if(s<d)return {x:halfWidth,z:halfDepth-s,angle:Math.PI,ew:false,toCorner:d-s};s-=d;if(s<w)return {x:halfWidth-s,z:-halfDepth,angle:-Math.PI/2,ew:true,toCorner:w-s};s-=w;return {x:-halfWidth,z:-halfDepth+s,angle:0,ew:false,toCorner:d-s};}
export function advanceCar(car,dt,time){const pose=loopPose(car.distance,car.halfWidth,car.halfDepth??63);const red=signalState(time,pose.ew)!=='G';const stop=pose.toCorner>5&&pose.toCorner<12&&red;return {...car,distance:car.distance+(stop?0:car.speed*dt)};}
export function signalFor(time,ew){return signalState(time,ew);}
