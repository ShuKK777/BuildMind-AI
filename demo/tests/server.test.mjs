import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';

test('loopback server serves offline modules and refuses writes / outside files',async t=>{
 const child=spawn(process.execPath,[fileURLToPath(new URL('../server.mjs',import.meta.url)),'--no-open'],{env:{...process.env,PORT:'0'},stdio:['ignore','pipe','pipe']});
 t.after(()=>child.kill());
 const base=await new Promise((resolve,reject)=>{let output='';const timeout=setTimeout(()=>reject(Error('Server did not start')),5000);child.on('error',reject);child.stdout.on('data',chunk=>{output+=chunk;const match=output.match(/http:\/\/127\.0\.0\.1:\d+/);if(match){clearTimeout(timeout);resolve(match[0]);}});});
 const home=await fetch(base);assert.equal(home.status,200);assert.match(await home.text(),/BuildMind AI/);
 const worker=await fetch(base+'/vendor/pdf.worker.min.mjs',{method:'HEAD'});assert.match(worker.headers.get('content-type'),/javascript/);assert.equal(await worker.text(),'');
 assert.equal((await fetch(base+'/index.html',{method:'POST',body:'no writes'})).status,405);
 assert.equal((await fetch(base+'/%2e%2e%2fREADME.md')).status,403);
 assert.equal((await fetch(base+'/not-found')).status,404);
});
