export const DURATION=45;
export const CUES=[
 {at:0,page:'plan',stage:0,view:'overview',time:'day',text:['从一片滨水土地，开始构想城市。','A waterfront site. A new possibility.']},
 {at:5,page:'plan',stage:1,view:'overview',text:['让开发过程可见，让空间逐步生长。','Make development visible, step by step.']},
 {at:10,page:'plan',stage:2,view:'waterfront',text:['走进街区，观察建筑与人的关系。','Enter the district. See how places connect.']},
 {at:15,page:'design',family:'flow',view:'landmark',text:['潮汐之环：流动曲面回应滨水天际线。','Tidal Loop: a fluid waterfront silhouette.']},
 {at:20,page:'design',family:'twins',text:['湾区双子：比较双塔与连廊的空间表达。','Harbour Twins: explore a connected skyline.']},
 {at:25,page:'design',family:'gardens',text:['云上花园：让退台成为垂直绿洲。','Sky Gardens: bring nature upwards.']},
 {at:30,page:'design',time:'night',text:['选定方向，在夜色中继续打磨。','Keep your direction. Refine it after dark.']},
 {at:35,page:'review',text:['从方案走向工程，让问题回到原文与依据。','From design to construction: trace every finding.']},
 {at:41,page:'plan',stage:2,view:'overview',text:['规划、设计与工程，共享同一个项目。','Planning, design and engineering. One project.']}
];
export class Timeline {constructor(){this.running=false;this.paused=false;this.elapsed=0;this.step=-1;this.last=0;}start(now){this.running=true;this.paused=false;this.elapsed=0;this.step=-1;this.last=now;}advance(now){if(this.running&&!this.paused)this.elapsed=Math.min(DURATION,this.elapsed+Math.max(0,now-this.last)/1000);this.last=now;return this.elapsed;}pause(value,now){this.advance(now);this.paused=value;}stop(){this.running=false;this.paused=false;}get cue(){return CUES.findLastIndex(c=>c.at<=this.elapsed);}get finished(){return this.elapsed>=DURATION;}}
export const ease=t=>{t=Math.max(0,Math.min(1,t));return t*t*t*(t*(t*6-15)+10);};
