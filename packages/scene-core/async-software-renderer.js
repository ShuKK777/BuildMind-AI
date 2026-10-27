import {SoftwareRenderer} from './software-renderer.js';
import {ScenePackets} from './software-protocol.js';
// CPU rasterisation stays off the UI thread. Only the newest requested pose is
// queued, so a slow compatibility renderer never builds a backlog of old shots.
export class AsyncSoftwareRenderer {
 constructor(){this.domElement=document.createElement('canvas');this.ctx=this.domElement.getContext('2d',{alpha:false});this.shadowMap={};this.isSoftwareRenderer=true;this.width=1;this.height=1;this.pixelRatio=1;this.serial=0;this.preparations=[];this.packets=new ScenePackets();this.worker=new Worker(new URL('./software-worker.js',import.meta.url),{type:'module'});this.worker.onmessage=({data})=>{if(data.type==='error'){this.fallback(new Error(data.message));return;}this.busy=false;if(data.type==='prepared'){this.preparing?.();this.preparing=null;}else{if(data.width===this.width&&data.height===this.height){this.ctx.drawImage(data.bitmap,0,0);this.domElement.dataset.frameMs=String(Math.round(data.milliseconds));this.domElement.dataset.frameId=String(data.id);this.onFrame?.(data.id);}data.bitmap.close();}if(this.pending||this.preparations.length)this.flush();};this.worker.onerror=e=>this.fallback(e);}
 fallback(error){if(this.sync)return;console.warn('Compatibility worker unavailable; using the main thread.',error.message);this.worker.terminate();this.sync=new SoftwareRenderer({canvas:this.domElement});this.sync.setSize(this.width,this.height);this.busy=false;this.pending||=this.latest;this.preparing?.();this.preparing=null;for(const p of this.preparations)p.resolve();this.preparations=[];if(this.pending)this.flush();}
 setPixelRatio(){}getPixelRatio(){return 1;}
 setSize(w,h,style=true){this.width=w;this.height=h;const scale=Math.min(1,960/w,720/h);this.domElement.width=Math.round(w*scale);this.domElement.height=Math.round(h*scale);if(style){this.domElement.style.width=w+'px';this.domElement.style.height=h+'px';}this.sync?.setSize(w,h,style);}
 getSize(target){return target.set(this.width,this.height);}
 setAnimationLoop(fn){if(this.frame)cancelAnimationFrame(this.frame);if(!fn)return;const tick=now=>{this.frame=requestAnimationFrame(tick);fn(now);};this.frame=requestAnimationFrame(tick);}
 render(scene,camera){this.pending={scene,camera};if(!this.busy)this.flush();}
 prepare(scene,camera){if(this.sync)return Promise.resolve();return new Promise(resolve=>{this.preparations.push({scene,camera,resolve,prepare:true});if(!this.busy)this.flush();});}
 flush(){const request=this.preparations.shift()||this.pending;if(!request)return;if(!request.prepare)this.pending=null;else this.preparing=request.resolve;if(this.sync){this.sync.render(request.scene,request.camera);this.onFrame?.(++this.serial);return;}this.busy=true;this.latest=request;const {packet,transfers}=this.packets.encode(request.scene,request.camera,this.width,this.height);this.worker.postMessage({id:++this.serial,type:request.prepare?'prepare':'render',packet},transfers);}
 dispose(){this.worker.terminate();if(this.frame)cancelAnimationFrame(this.frame);this.sync?.dispose();this.pending=null;}
}
