// Developer-only: npm ci --omit=optional --ignore-scripts, then node vendor.mjs.
import {cp,readFile,writeFile,readdir,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {createHash} from 'node:crypto';
const here=path.dirname(fileURLToPath(import.meta.url)),modules=path.join(here,'node_modules'),out=path.resolve(here,'../vendor');
await mkdir(out,{recursive:true});
const files=[['three/build/three.module.min.js','three.module.min.js'],['three/build/three.core.min.js','three.core.min.js'],['three/examples/jsm/controls/OrbitControls.js','OrbitControls.js'],['three/LICENSE','THREE-LICENSE.txt'],['mammoth/mammoth.browser.min.js','mammoth.browser.min.js'],['mammoth/LICENSE','MAMMOTH-LICENSE.txt'],['pdfjs-dist/build/pdf.min.mjs','pdf.min.mjs'],['pdfjs-dist/build/pdf.worker.min.mjs','pdf.worker.min.mjs'],['pdfjs-dist/LICENSE','PDFJS-LICENSE.txt'],['pdfjs-dist/cmaps','cmaps'],['pdfjs-dist/standard_fonts','standard_fonts'],['pdfjs-dist/wasm','wasm']];
for(const [source,target] of files)await cp(path.join(modules,source),path.join(out,target),{recursive:true});
let notices='# Dependency license notices\n\nVersions are pinned in tools/package-lock.json. PDF.js font/CMap/WASM licenses also reside beside those assets.\n';
const seen=new Set();async function licenses(name){if(seen.has(name))return;seen.add(name);const dir=path.join(modules,name),pkg=JSON.parse(await readFile(path.join(dir,'package.json'),'utf8'));notices+=`\n## ${pkg.name} ${pkg.version}\n`;
for(const f of await readdir(dir))if(/^(licen[sc]e|copying|notice)(\.|$)/i.test(f)){try{notices+='\n'+await readFile(path.join(dir,f),'utf8')+'\n';}catch{}}
for(const dep of Object.keys(pkg.dependencies||{}))await licenses(dep);}
for(const name of ['three','mammoth','pdfjs-dist'])await licenses(name);
await writeFile(path.join(out,'THIRD_PARTY_NOTICES.txt'),notices);
const hashes={};async function walk(dir){for(const item of await readdir(dir,{withFileTypes:true})){const file=path.join(dir,item.name);if(item.isDirectory())await walk(file);else if(item.name!=='SHA256.json')hashes[path.relative(out,file).split(path.sep).join('/')]=createHash('sha256').update(await readFile(file)).digest('hex');}}
await walk(out);await writeFile(path.join(out,'SHA256.json'),JSON.stringify(hashes,null,2)+'\n');console.log(`Bundled ${Object.keys(hashes).length} local vendor files.`);
