export const defaults = {lang:'zh',page:'plan',variant:0,height:1,material:0,night:false,green:true,stage:2,selected:'A02',original:true,layers:true};
export const variants = [
 {zh:'潮汐之环',en:'Tidal Loop',style:['流线曲面 · 未来地标','Fluid forms · Future landmark'],height:72,area:68400,color:'#b1d4de'},
 {zh:'湾区双子',en:'Harbour Twins',style:['玻璃双塔 · 商务科技','Glass towers · Business district'],height:86,area:82600,color:'#83b3c6'},
 {zh:'云上花园',en:'Sky Gardens',style:['层叠退台 · 垂直绿洲','Terraced gardens · Vertical oasis'],height:66,area:59200,color:'#a6ba9b'}
];
export function validateState(raw={}){const s={...defaults}; if(raw.lang==='en')s.lang='en';if(['plan','design','review'].includes(raw.page))s.page=raw.page;for(const k of ['variant','height','stage'])if(Number.isInteger(raw[k])&&raw[k]>=0&&raw[k]<=2)s[k]=raw[k];if(raw.material===1)s.material=1;for(const k of ['night','green','layers','original'])if(typeof raw[k]==='boolean')s[k]=raw[k];if(typeof raw.selected==='string'&&/^([A-B]\d{2})$/.test(raw.selected))s.selected=raw.selected;return s;}
export function loadState(){try{return validateState(JSON.parse(localStorage.getItem('buildmind-demo-v1'))||{});}catch{return {...defaults};}}
export function saveState(s){try{localStorage.setItem('buildmind-demo-v1',JSON.stringify(s));}catch{/* Private browsing can disable persistence. */}}
export function metrics(s){const factor=[.82,1,1.18][s.height];return {height:s.original?72:Math.round(6+(variants[s.variant].height-6)*factor),area:Math.round(variants[s.variant].area*factor),population:[0,4200,8600][s.stage],green:[62,44,38][s.stage]};}
export const tour = [
 {at:0,page:'plan',stage:0,selected:'B01',text:['从一片滨水土地，开始构想城市。','A waterfront site. A new urban possibility.']},
 {at:5,page:'plan',stage:1,selected:'A02',text:['回放开发阶段，让规划过程清晰可见。','Explore a staged development scenario.']},
 {at:10,page:'plan',stage:2,text:['住宅、办公与公共空间，在同一个街区协同。','Homes, workplaces and public space, together.']},
 {at:15,page:'design',original:false,variant:0,text:['方案一：流动曲面塑造滨水天际线。','Option 01: a fluid silhouette on the waterfront.']},
 {at:20,page:'design',original:false,variant:1,text:['方案二：双塔与空中连廊连接商务空间。','Option 02: twin towers connected by a sky bridge.']},
 {at:25,page:'design',original:false,variant:2,text:['方案三：层叠花园将绿意带入建筑。','Option 03: terraced gardens bring nature upwards.']},
 {at:30,page:'design',original:false,variant:2,night:true,text:['保留选定方案，继续探索夜间场景。','Keep your chosen design. Explore it after dark.']},
 {at:35,page:'review',text:['审查示范文档，每个问题都能定位与追溯。','Review the sample. Trace every finding to its source.']},
 {at:41,page:'plan',stage:2,text:['让规划、设计与审查，汇聚到同一个项目。','One project, from planning to design and review.']}
];
