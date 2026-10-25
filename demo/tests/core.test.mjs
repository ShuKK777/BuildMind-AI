import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,readdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {defaults,validateState,metrics,tour} from '../state.js';
import {sample,validateFile} from '../documents.js';
const root=path.dirname(fileURLToPath(new URL('../index.html',import.meta.url)));
test('persisted state rejects invalid settings without corrupting defaults',()=>{const s=validateState({variant:99,height:-1,page:'bad',lang:'xx',stage:Infinity,night:'false',selected:'<script>'});assert.deepEqual(s,defaults);assert.equal(validateState({variant:2,height:2,lang:'en',night:true}).variant,2);});
test('height choices update displayed parameters monotonically for all variants',()=>{for(let v=0;v<3;v++){const m=[0,1,2].map(h=>metrics({...defaults,variant:v,height:h,original:false}));assert.ok(m[0].height<m[1].height&&m[1].height<m[2].height);assert.ok(m[0].area<m[1].area&&m[1].area<m[2].area);}});
test('tour has 9 ordered stages and includes every module within 90 seconds',()=>{assert.equal(tour.length,9);assert.equal(tour[0].at,0);assert.ok(tour.at(-1).at<90);for(let i=1;i<tour.length;i++)assert.ok(tour[i].at>tour[i-1].at);assert.deepEqual(new Set(tour.map(x=>x.page)),new Set(['plan','design','review']));});
test('all preset findings map to unique source paragraphs and bilingual evidence',()=>{assert.equal(new Set(sample.issues.map(i=>i.id)).size,3);for(const issue of sample.issues){assert.ok(sample.paragraphs.some(p=>p.id===issue.id));assert.equal(issue.quote.length,2);assert.match(issue.rule,/^BM-NM-01 §/);}for(const p of sample.paragraphs)assert.equal(p.text.length,2);});
test('file intake only permits bounded DOCX/PDF files',()=>{assert.equal(validateFile({name:'test.DOCX',size:1}),null);assert.equal(validateFile({name:'test.pdf',size:20*1024*1024}),null);assert.equal(validateFile({name:'test.doc',size:1}),'type');assert.equal(validateFile({name:'test.pdf',size:20*1024*1024+1}),'size');});
test('all browser entry assets are local and shipped, including recursive Three.js imports',async()=>{const html=await readFile(path.join(root,'index.html'),'utf8');assert.doesNotMatch(html,/(?:src|href)=["']https?:/);const files=['app.js','state.js','scene.js','software-renderer.js','print-scene.js','assets/print-assembly.js','documents.js','styles.css','vendor/three.module.min.js','vendor/three.core.min.js','vendor/OrbitControls.js','vendor/mammoth.browser.min.js'];for(const f of files){const text=await readFile(path.join(root,f),'utf8');assert.ok(text.length>0);if(!f.startsWith('vendor/'))assert.doesNotMatch(text,/\bfetch\s*\(|\bXMLHttpRequest\b|\bWebSocket\b/);}const module=await readFile(path.join(root,'vendor/three.module.min.js'),'utf8');assert.match(module,/three\.core\.min\.js/);});

test('print assembly has the source 8 + 19 buildings; A02 is the highest',async()=>{
 const rows=JSON.parse(await readFile(path.join(root,'assets/part-manifest.json'),'utf8'));
 const {printParts}=await import('../assets/print-assembly.js');
 assert.equal(rows.filter(r=>r.id.startsWith('B')).length,8);assert.equal(rows.filter(r=>r.id.startsWith('A')).length,19);
 assert.deepEqual(printParts.map(p=>p.id),rows.map(p=>p.id));
 const highest=[...rows].sort((a,b)=>(b.bottom_z_mm+b.height_mm)-(a.bottom_z_mm+a.height_mm))[0];
 assert.equal(highest.id,'A02');assert.equal(highest.height_mm+highest.bottom_z_mm,72);
 for(const row of rows){const part=printParts.find(p=>p.id===row.id);assert.equal(part.x,(row.center_x_mm-75)*1.5);assert.equal(part.z,(75-row.center_y_mm)*1.5);assert.equal(part.heightMm,row.height_mm);
  const bytes=Buffer.from(part.v,'base64');assert.equal(bytes.length,part.triangles*9*4);assert.equal(Buffer.from(part.c,'base64').length,part.triangles*3);
  const y=[];for(let i=4;i<bytes.length;i+=12)y.push(bytes.readFloatLE(i));assert.ok(Math.abs(Math.max(...y)/1.5-(row.height_mm+row.bottom_z_mm))<.02,`${row.id} highest mesh vertex matches assembly height`);
 }
});

test('original tower dimensions do not change until a variant is selected',()=>{
 for(let variant=0;variant<3;variant++)for(let height=0;height<3;height++)assert.equal(metrics({...defaults,variant,height,original:true}).height,72);
 assert.equal(tour.find(step=>step.page==='design').original,false);
});

test('bundled vendor bytes match the recorded SHA-256 manifest',async()=>{
 const {createHash}=await import('node:crypto'),manifest=JSON.parse(await readFile(path.join(root,'vendor/SHA256.json'),'utf8'));
 assert.ok(manifest['pdf.worker.min.mjs']);assert.ok(Object.keys(manifest).some(p=>p.startsWith('cmaps/')));assert.ok(Object.keys(manifest).some(p=>p.startsWith('wasm/')));
 for(const [file,expected]of Object.entries(manifest)){const bytes=await readFile(path.join(root,'vendor',file));assert.equal(createHash('sha256').update(bytes).digest('hex'),expected,file);}
});
