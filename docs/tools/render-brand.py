"""Build the repository's SVG identity assets using the existing outlined glyphs.
Run from any directory: python3 docs/tools/render-brand.py
The SVG files are self-contained and use no web fonts or remote images.
"""
from pathlib import Path
import json, html
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'docs/assets'
NS = '{http://www.w3.org/2000/svg}'
existing = ET.parse(ASSETS / 'project-card.svg').getroot()
GLYPHS = {p.attrib['id']: p.attrib['d'] for p in existing.iter(NS+'path') if p.attrib.get('id','').startswith('glyph-')}

def text(x,y,value,size=20,color='#d8e4e7',weight=400,spacing=0,anchor='start'):
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" letter-spacing="{spacing}" text-anchor="{anchor}" font-family="Arial,Helvetica,sans-serif">{html.escape(value)}</text>'

def zh(x,y,value,size=30,color='#eaf0eb'):
    out=[f'<g fill="{color}" role="img" aria-label="{html.escape(value)}"><title>{html.escape(value)}</title>']
    for ch in value:
        if ch==' ': x+=size*.33;continue
        key='glyph-'+str(ord(ch))
        if key not in GLYPHS: raise ValueError('Missing existing brand glyph: '+ch)
        out.append(f'<use href="#{key}" transform="translate({x:.3f} {y:.3f}) scale({size/1000:.4f} {-size/1000:.4f})"/>')
        x+=size
    return ''.join(out)+'</g>'

def line(x1,y1,x2,y2,color='#9ebdbf',width=1,opacity=1,dash=None):
    return f'<path d="M{x1:.2f},{y1:.2f}L{x2:.2f},{y2:.2f}" stroke="{color}" stroke-width="{width}" opacity="{opacity}" fill="none"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'

def poly(points,fill,stroke='#8eb9bc',opacity=1,width=.8):
    return '<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+f'" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'

def rect(x,y,w,h,fill,stroke='none',rx=0,opacity=1):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'

def svg_start(w,h,title,description):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(description)}</desc><defs>'+''.join(f'<path id="{k}" d="{d}"/>' for k,d in GLYPHS.items())+'''<linearGradient id="background" x2="1" y2="1"><stop stop-color="#172b33"/><stop offset="1" stop-color="#0b1920"/></linearGradient><linearGradient id="glass" x2="1" y2="1"><stop stop-color="#426b76"/><stop offset=".48" stop-color="#213e4c"/><stop offset="1" stop-color="#162b38"/></linearGradient><linearGradient id="roof" x2="1" y2="1"><stop stop-color="#b0c2b4"/><stop offset="1" stop-color="#728f86"/></linearGradient><radialGradient id="glow"><stop stop-color="#b3d4b4" stop-opacity=".12"/><stop offset="1" stop-color="#b3d4b4" stop-opacity="0"/></radialGradient><pattern id="grid" width="44" height="44" patternUnits="userSpaceOnUse"><path d="M44 0H0V44" fill="none" stroke="#849fa6" stroke-opacity=".045"/></pattern></defs>'''

def project(x,z,height=0):return (1340+(x-z)*2.55,558+(x+z)*1.14-height*3)

def building(cx,cz,w,d,h,accent=False):
    a,b,c,e=[project(cx+dx,cz+dz) for dx,dz in [(-w/2,-d/2),(w/2,-d/2),(w/2,d/2),(-w/2,d/2)]]
    at,bt,ct,et=[(x,y-h*3) for x,y in [a,b,c,e]]
    edge='#c9dfb7' if accent else '#779ca6'
    out=poly([b,c,ct,bt],'#1a3541',edge,.95)+poly([e,c,ct,et],'url(#glass)',edge,.97)+poly([at,bt,ct,et],'url(#roof)',edge,.96)
    for v in range(1,max(2,int(h/4))):
        y=v*12
        out+=line(e[0],e[1]-y,c[0],c[1]-y,'#b6d4d4',.7,.5)+line(b[0],b[1]-y,c[0],c[1]-y,'#8ab2bb',.7,.4)
    for n in range(1,5):
        t=n/5
        for p,q in [(e,c),(b,c)]:
            x=p[0]+(q[0]-p[0])*t;y=p[1]+(q[1]-p[1])*t
            out+=line(x,y,x,y-h*3,'#9ec0c7',.55,.35)
    return out

def city():
    out='<ellipse cx="1350" cy="470" rx="420" ry="340" fill="url(#glow)"/>'
    out+=poly([project(-78,-78),project(78,-78),project(78,78),project(-78,78)],'#142b32','#729597',.8)
    # Contour lines and a waterfront datum lend the graphic a spatial scale.
    for n in range(7):
        y=678+n*12
        out+=f'<path d="M980 {y}C1115 {y-78} 1330 {y+22} 1733 {y-94}" stroke="#759a9b" stroke-opacity="{.04+n*.014}" fill="none"/>'
    for z in [-55,-18,22,64]:out+=line(*project(-75,z),*project(75,z),'#91aaa6',2,.35)
    for x in [-54,-18,26,62]:out+=line(*project(x,-70),*project(x,68),'#91aaa6',2,.25)
    rows=json.loads((ROOT/'demo/assets/part-manifest.json').read_text())
    rows=sorted([r for r in rows if r['id'].startswith('A')],key=lambda r:r['center_x_mm']-r['center_y_mm'])
    for r in rows:
        x=r['center_x_mm']-75;z=75-r['center_y_mm'];w=r['width_mm'];d=r['depth_mm'];h=r['height_mm']
        if r['id']=='A02':
            out+=building(x-3.8,z,7.2,d,h,True)+building(x+4.2,z,6.8,d,h*.68,True)
        elif r['id'] in ['A06','A07','A08','A09']:
            out+=building(x,z,w,d,h*.5)+building(x+.7,z-.8,w*.72,d*.7,h*.8)+building(x+1.4,z-1.6,w*.48,d*.46,h)
        else:out+=building(x,z,w,d,h)
    x,y=project(-10,1,66)
    out+=f'<circle cx="{x}" cy="{y}" r="4" fill="#d2e7ac"/>'
    out+=f'<path d="M{x} {y}L{x+70} {y-72}H1575" stroke="#b7cea7" stroke-width="1" fill="none" opacity=".7"/>'
    out+=text(1575,y-83,'LANDMARK / A02',15,'#b6c8b8',400,2,'end')
    out+=text(1338,785,'SPATIAL VISION / NORTHERN METROPOLIS',13,'#718b94',400,2,'middle')
    return out

out=[svg_start(1800,1040,'BuildMind AI · 智筑未来','让每一个建设决策，都更有远见。连接城市规划、建筑设计与工程协同。'),rect(0,0,1800,1040,'url(#background)'),rect(0,0,1800,1040,'url(#grid)')]
out += [rect(96,88,9,9,'#d4e6b0',rx=1),text(123,99,'BUILDMIND AI / STUDIO FOR THE BUILT WORLD',16,'#9eb4b8',500,2.3),text(1697,99,'PRODUCT VISION  /  V0.2',13,'#7b969d',400,1.8,'end')]
out += [text(89,256,'BuildMind',116,'#f0f4eb',600,-5),text(674,256,'AI',110,'#d2e5a7',400,-3),zh(97,321,'智筑未来',29,'#91b5b8'),zh(94,437,'让每一个建设决策，',62,'#e8efeb'),zh(94,520,'都更有远见。',72,'#d2e5a7')]
out += [zh(97,592,'面向建筑全生命周期的多智能体',23,'#8ea9b1'),text(97,630,'AI',24,'#8ea9b1'),zh(138,630,'决策引擎',23,'#8ea9b1'),text(97,690,'Better decisions. Better places.',23,'#d1dedb',400,.1)]
for x,w,label in [(97,174,'ONE PROJECT'),(284,237,'THREE PERSPECTIVES'),(534,191,'45s SHOWCASE')]:
    out += [rect(x,736,w,39,'#1b333b','#40575c',4),text(x+w/2,761,label,12,'#b9cccb',400,1.4,'middle')]
out += [city(),line(96,835,1700,835,'#536970',1,.6)]
for x,num,title,en,caption in [(96,'01','智策','URBAN INTELLIGENCE','理解城市 · 推演未来'),(665,'02','智绘','DESIGN INTELLIGENCE','生成方案 · 共创空间'),(1234,'03','智管','PROJECT INTELLIGENCE','关联证据 · 协同决策')]:
    out += [text(x,890,num,13,'#6e8c96',400,2),zh(x+48,899,title,32,'#dce9dd'),text(x+140,895,en,12,'#a3b8bc',400,1.4),zh(x+48,947,caption,20,'#8ba7ad')]
out += [text(96,1006,'PLANNING / ARCHITECTURE / CONSTRUCTION',11,'#5e7a84',400,2),text(1700,1006,'FROM POSSIBILITY TO PLACE.',11,'#7a9297',400,1.8,'end'),'</svg>']
(ASSETS/'project-card.svg').write_text(''.join(out))

# Compact module cards are designed for the README's three-column layout.
for name,num,title,en,caption in [('plan','01','智策','URBAN INTELLIGENCE','理解城市 · 推演未来'),('design','02','智绘','DESIGN INTELLIGENCE','生成方案 · 共创空间'),('review','03','智管','PROJECT INTELLIGENCE','关联证据 · 协同决策')]:
    a=[svg_start(560,300,title,en),rect(0,0,560,300,'#152b33'),rect(0,0,560,300,'url(#grid)'),text(30,43,num,14,'#8fa6ad',400,2),text(530,43,en,11,'#78949c',400,1.2,'end')]
    a += [zh(29,113,title,42,'#d4e6b0'),zh(30,157,caption,19,'#afc5c6')]
    if name=='plan':
        for i in range(7):a.append(f'<path d="M282 {112+i*14}C335 {60+i*13} 365 {170+i*7} 534 {110+i*15}" fill="none" stroke="#83a7a8" stroke-width="1" opacity="{.25+i*.05}"/>')
        for x,y,h in [(337,153,39),(377,161,66),(419,172,48),(459,155,31)]:
            a += [rect(x,y-h,17,h,'#284d5b','#98bfc1'),line(x,y,x+17,y-7,'#d0e3b3',1.4)]
    elif name=='design':
        for x,h in [(312,64),(354,98),(399,121),(444,82)]:
            a += [poly([(x,210),(x+22,218),(x+22,218-h),(x,210-h)],'url(#glass)','#789ba7'),poly([(x+22,218),(x+38,209),(x+38,209-h),(x+22,218-h)],'#25414e','#789ba7')]
            for yy in range(211-h+10,209,12):a.append(line(x,yy,x+22,yy+8,'#b3cace',.6,.65))
    else:
        a += [rect(324,80,124,145,'#1b3741','#92b3bb',4),line(343,105,427,105,'#b8cfcf',1.2)]
        for i in range(4):a += [line(344,130+i*21,407-(i%2)*20,130+i*21,'#708e99',1)]
        a += [rect(335,140,102,28,'#b5cf9720','#abc58d',2),f'<circle cx="479" cy="154" r="17" fill="#d2e5a7"/>','<path d="M471 154l6 6 11-13" fill="none" stroke="#263d33" stroke-width="2"/>',line(449,154,462,154,'#a9c59b',1)]
    a += [line(30,250,530,250,'#425d67',1,.6),text(30,279,{'plan':'EXPLORE THE CITY.','design':'SHAPE THE POSSIBILITIES.','review':'FOLLOW THE EVIDENCE.'}[name],11,'#78979d',400,2),'</svg>']
    (ASSETS/f'module-{name}.svg').write_text(''.join(a))
print('Rendered project-card.svg and 3 module cards')
