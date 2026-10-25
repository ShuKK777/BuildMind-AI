"""Convert user-supplied print STL parts into a browser-local colored assembly.
Usage: python tools/convert-print-pack.py /path/to/print-pack.zip
Requires numpy, only at asset-authoring time. Original STL files are never modified.
"""
from pathlib import Path
from zipfile import ZipFile
import json,sys,base64
import numpy as np
out=Path(__file__).resolve().parents[1]/'assets'
palette={'ivory':'E8E4DA','glass':'7DABB8','darkglass':'456F80','garden':'86A179','clay':'C69B7D','roof':'DDD9CB','bronze':'BCA37E','plate':'D4D3C7'}
with ZipFile(sys.argv[1]) as z:
 rows=json.loads(z.read('source/part_manifest.json'));paint=json.loads(z.read('source/style_paint.json'))
 parts=[]
 for row in rows:
  name=next(n for n in z.namelist() if n.startswith('02_树脂建筑/') and Path(n).name.startswith(row['id']+'_') and n.endswith('.stl'))
  raw=z.read(name);count=int.from_bytes(raw[80:84],'little');tri=np.frombuffer(raw, dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]),count=count,offset=84)
  verts=tri['v'].astype(np.float32).copy();verts+=np.array([row['center_x_mm'],row['center_y_mm'],row['bottom_z_mm']],dtype=np.float32)
  centers=verts.mean(axis=1);normals=tri['normal'];colors=np.tile([232,228,218],(count,1)).astype(np.uint8)
  if row['id'][0]=='A':
   for reg in paint[row['id']]:
    lo,hi=np.array(reg['bounds']);inside=np.all((centers>=lo-.003)&(centers<=hi+.003),axis=1)
    if reg['up']:inside&=normals[:,2]>.65
    col=palette[reg['color']];colors[inside]=[int(col[k:k+2],16) for k in (0,2,4)]
   colors[centers[:,2]<8.001]=[212,211,199]
  else:
   colors[:]=[233,223,199];colors[centers[:,2]>9.05]=[174,123,96];colors[centers[:,2]<5.7]=[183,170,131]
  # Omit print-only locating tabs and underside faces hidden below the terrain.
  keep=np.max(verts[:,:,2],axis=1)>5.9
  verts=verts[keep];colors=colors[keep]
  web=np.empty_like(verts);web[:,:,0]=(verts[:,:,0]-75)*1.5;web[:,:,1]=verts[:,:,2]*1.5;web[:,:,2]=(75-verts[:,:,1])*1.5
  parts.append({'id':row['id'],'name':row['name'],'heightMm':row['height_mm'],'x':(row['center_x_mm']-75)*1.5,'z':(75-row['center_y_mm'])*1.5,'v':base64.b64encode(web.astype('<f4').tobytes()).decode(),'c':base64.b64encode(colors.tobytes()).decode(),'triangles':len(web)})
 out.mkdir(exist_ok=True)
 (out/'print-assembly.js').write_text('// Derived from user-supplied 150 mm print pack. A02 is the replaceable tallest tower.\nexport const printParts='+json.dumps(parts,separators=(',',':'),ensure_ascii=False)+';\n')
 (out/'part-manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 print('parts',len(parts),'triangles',sum(p['triangles'] for p in parts),'bytes',(out/'print-assembly.js').stat().st_size)
