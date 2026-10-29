(()=>{
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const $=id=>document.getElementById(id);
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
let W=0,H=0,DPR=1,S=1;
const rnd=(a,b)=>a+Math.random()*(b-a);
const clamp=(v,a,b)=>v<a?a:v>b?b:v;
const TAU=Math.PI*2;

/* ---------- storage ---------- */
function loadBest(){try{return +localStorage.getItem('swi-best')||0}catch(e){return 0}}
function saveBest(v){try{localStorage.setItem('swi-best',String(v))}catch(e){}}

/* ---------- audio ---------- */
let AC=null,muted=false,noiseBuf=null;
function ensureAudio(){
  if(!AC){try{AC=new (window.AudioContext||window.webkitAudioContext)();
    noiseBuf=AC.createBuffer(1,AC.sampleRate*0.6,AC.sampleRate);const d=noiseBuf.getChannelData(0);for(let i=0;i<d.length;i++)d[i]=Math.random()*2-1;}catch(e){AC=null}}
  if(AC&&AC.state==='suspended')AC.resume().catch(()=>{});
}
function tone(f,dur,type,vol,slide){
  if(!AC||muted)return;const t=AC.currentTime,o=AC.createOscillator(),g=AC.createGain();
  o.type=type;o.frequency.setValueAtTime(f,t);if(slide)o.frequency.exponentialRampToValueAtTime(slide,t+dur);
  g.gain.setValueAtTime(vol,t);g.gain.exponentialRampToValueAtTime(0.0001,t+dur);o.connect(g).connect(AC.destination);o.start(t);o.stop(t+dur+0.02);
}
function noise(dur,vol,freq){
  if(!AC||muted||!noiseBuf)return;const t=AC.currentTime,s=AC.createBufferSource(),f=AC.createBiquadFilter(),g=AC.createGain();
  s.buffer=noiseBuf;f.type='lowpass';f.frequency.setValueAtTime(freq,t);f.frequency.exponentialRampToValueAtTime(80,t+dur);
  g.gain.setValueAtTime(vol,t);g.gain.exponentialRampToValueAtTime(0.0001,t+dur);s.connect(f).connect(g).connect(AC.destination);s.start(t);s.stop(t+dur);
}
const sfx={
  ion:()=>tone(880,0.07,'square',0.025,320),
  plasma:()=>tone(1400,0.05,'sawtooth',0.018,700),
  hit:()=>tone(300,0.05,'triangle',0.04,180),
  boom:()=>noise(0.35,0.22,1800),
  big:()=>{noise(0.9,0.35,900);tone(90,0.8,'sine',0.25,30)},
  pick:()=>{tone(1200,0.06,'sine',0.06);setTimeout(()=>tone(1800,0.08,'sine',0.05),50)},
  gift:()=>[523,659,784,1046].forEach((f,i)=>setTimeout(()=>tone(f,0.12,'triangle',0.08),i*60)),
  over:()=>tone(220,0.4,'sawtooth',0.06,90),
  hurt:()=>{noise(0.6,0.35,2400);tone(160,0.5,'square',0.1,40)},
  nova:()=>{tone(80,1.2,'sine',0.35,400);noise(1,0.3,3000)},
  wave:()=>[392,523,659].forEach((f,i)=>setTimeout(()=>tone(f,0.16,'triangle',0.07),i*90)),
};
function toggleMute(){muted=!muted;['muteBtn1','muteBtn2'].forEach(id=>$(id).textContent='Sound: '+(muted?'off':'on'))}

/* ---------- background ---------- */
let stars=[],bg=null;
function initStars(){
  stars=[];const n=Math.round(W*H/2600);
  for(let i=0;i<n;i++)stars.push({x:Math.random()*W,y:Math.random()*H,z:Math.random()});
  bg=document.createElement('canvas');bg.width=cv.width;bg.height=cv.height;const b=bg.getContext('2d');b.setTransform(DPR,0,0,DPR,0,0);
  b.fillStyle='#060918';b.fillRect(0,0,W,H);
  const neb=(x,y,r,c)=>{const g=b.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,c);g.addColorStop(1,'rgba(0,0,0,0)');b.fillStyle=g;b.fillRect(0,0,W,H)};
  neb(W*0.15,H*0.25,W*0.6,'rgba(90,40,160,0.22)');
  neb(W*0.9,H*0.1,W*0.5,'rgba(30,90,170,0.2)');
  neb(W*0.7,H*0.55,W*0.45,'rgba(160,50,90,0.12)');
  // planet limb at the bottom
  const pr=Math.max(W,H)*1.1,py=H+pr-H*0.09;
  const pg=b.createRadialGradient(W*0.35,py-pr*0.9,pr*0.2,W/2,py,pr);
  pg.addColorStop(0,'#2b5aa8');pg.addColorStop(0.85,'#14284f');pg.addColorStop(1,'#0b1530');
  b.fillStyle=pg;b.beginPath();b.arc(W/2,py,pr,0,TAU);b.fill();
  b.strokeStyle='rgba(120,200,255,0.55)';b.lineWidth=2;b.beginPath();b.arc(W/2,py,pr,Math.PI*1.15,Math.PI*1.85);b.stroke();
  const at=b.createRadialGradient(W/2,py,pr,W/2,py,pr+40);at.addColorStop(0,'rgba(90,170,255,0.25)');at.addColorStop(1,'rgba(90,170,255,0)');
  b.fillStyle=at;b.beginPath();b.arc(W/2,py,pr+40,0,TAU);b.fill();
}
function measureHUD(){
  const t=$('top'),b=$('bottom');
  G.topH=t.hidden?56:t.offsetHeight;G.botH=b.hidden?60:b.offsetHeight;
}
function resize(){
  measureHUD();
  DPR=Math.min(2,window.devicePixelRatio||1);W=cv.clientWidth;H=cv.clientHeight;
  cv.width=Math.round(W*DPR);cv.height=Math.round(H*DPR);ctx.setTransform(DPR,0,0,DPR,0,0);
  S=clamp(Math.min(W/520,H/760),0.68,1.25);initStars();
  if(P){P.r=14*S;P.x=clamp(P.x,P.r,W-P.r);P.y=clamp(P.y,Math.max(H*0.3,G.topH+40*S),H-G.botH-P.r-14*S)}
}

/* ---------- state ---------- */
const G={topH:56,botH:60,state:'title',score:0,lives:3,wave:0,dust:0,novas:1,best:loadBest(),t:0,shake:0,flash:0,
  banner:'',sub:'',bannerT:0,between:false,nextT:0,spawnQ:[],waveT:0,nextLife:20000,overT:0,ambientT:0};
let P=null,bullets=[],enemies=[],ebul=[],pickups=[],parts=[],texts=[];

function newPlayer(){return{x:W/2,y:H-110*S,r:14*S,inv:2.2,heat:0,over:false,wt:'ion',lv:1,cd:0,dead:false,shield:0,
  up:{armor:0,thr:0,cool:0,rate:0,mag:0}}}

/* ---------- hangar upgrades ---------- */
const UPGRADES=[
  {id:'armor',name:'Shield generator',desc:'Each level adds a shield charge that absorbs one hit. One charge refills every wave.',costs:[40,90,160]},
  {id:'rate',name:'Rapid cannons',desc:'Fires 7% faster per level and mounts small wing guns on the hull.',costs:[45,95,170]},
  {id:'cool',name:'Coolant system',desc:'A little less heat per shot and a little faster cooldown.',costs:[30,70,120]},
  {id:'thr',name:'Thrusters',desc:'Moves 8% faster per level. Adds side engines.',costs:[25,55,100]},
  {id:'mag',name:'Stardust magnet',desc:'Pulls stars and capsules in from further away.',costs:[20,50]},
  {id:'power',name:'Weapon power',desc:'Raises your current weapon one level (max 5). Lost levels can be bought back.',repeat:true},
  {id:'nova',name:'Nova bomb',desc:'One screen-clearing bomb. Hold up to 5.',repeat:true},
  {id:'life',name:'Spare ship',desc:'One extra life. Hold up to 6.',repeat:true},
];
function upgCost(u){
  if(u.id==='power')return P.lv>=5?null:40+P.lv*30;
  if(u.id==='nova')return G.novas>=5?null:50;
  if(u.id==='life')return G.lives>=6?null:150;
  const l=P.up[u.id];return l>=u.costs.length?null:u.costs[l];
}
function anyAffordable(){return P&&UPGRADES.some(u=>{const c=upgCost(u);return c!=null&&c<=G.dust})}
function buildShop(){
  $('shopBal').textContent='★ '+G.dust;
  const list=$('upgList');list.innerHTML='';
  for(const u of UPGRADES){
    const c=upgCost(u),el=document.createElement('div');el.className='upg';
    let pips='';
    if(!u.repeat){pips='<div class="pips">'+u.costs.map((_,i)=>`<span class="pip${i<P.up[u.id]?' on':''}"></span>`).join('')+'</div>'}
    else{const cur=u.id==='power'?'Level '+P.lv+'/5':u.id==='nova'?'Holding '+G.novas+'/5':'Ships '+G.lives+'/6';pips=`<span class="lbl">${cur}</span>`}
    el.innerHTML=`<h3>${u.name}</h3><p>${u.desc}</p>${pips}<div class="buy"><span class="lbl">${c==null?'Maxed':'Cost ★ '+c}</span><button id="buy-${u.id}" ${c==null||c>G.dust?'disabled':''}>${c==null?'Max':'Buy'}</button></div>`;
    list.appendChild(el);
    el.querySelector('button').onclick=()=>buy(u);
  }
}
function buy(u){
  const c=upgCost(u);if(c==null||c>G.dust)return;
  G.dust-=c;
  if(u.id==='power')P.lv++;
  else if(u.id==='nova')G.novas++;
  else if(u.id==='life')G.lives++;
  else{P.up[u.id]++;if(u.id==='armor')P.shield++}
  sfx.gift();buildShop();updHUD(true);
  const b=$('buy-'+u.id);if(b&&!b.disabled)b.focus();
}
function openShop(){if(G.state!=='play'||!P||P.dead)return;G.state='shop';buildShop();show('shopScr',true);$('shopScr').scrollTop=0;$('shopClose').focus({preventScroll:true})}
function closeShop(){if(G.state!=='shop')return;G.state='play';show('shopScr',false)}

/* ---------- input ---------- */
const keys={};let ctrl='key';const mouse={x:0,y:0,has:false};const ptr={active:false,type:'',lx:0,ly:0};
addEventListener('keydown',e=>{
  if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Space'].includes(e.code))e.preventDefault();
  keys[e.code]=true;ensureAudio();
  if(e.repeat)return;
  if((e.code==='KeyP'||e.code==='Escape')&&(G.state==='play'||G.state==='pause'))togglePause();
  if((e.code==='KeyB'||e.code==='KeyX')&&G.state==='play')nova();
  if(e.code==='KeyU'||e.code==='KeyH'){if(G.state==='play')openShop();else if(G.state==='shop')closeShop()}
  if(e.code==='Escape'&&G.state==='shop')closeShop();
  if(e.code==='KeyM')toggleMute();
  if((e.code==='Enter'||e.code==='Space')&&(G.state==='title'||(G.state==='over'&&G.overT>1))&&!(e.target&&e.target.tagName==='BUTTON'))start(!!G.check);
});
addEventListener('keyup',e=>{keys[e.code]=false});
function rel(e){const r=cv.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top}}
cv.addEventListener('pointerdown',e=>{
  if(ptr.active&&e.pointerId!==ptr.id)return;ptr.id=e.pointerId;
  ensureAudio();ptr.active=true;ptr.type=e.pointerType;ptr.lx=e.clientX;ptr.ly=e.clientY;
  if(e.pointerType==='mouse'){ctrl='mouse';const p=rel(e);mouse.x=p.x;mouse.y=p.y;mouse.has=true}else ctrl='touch';
  try{cv.setPointerCapture(e.pointerId)}catch(_){}
});
cv.addEventListener('pointermove',e=>{
  if(e.pointerType==='mouse'){const p=rel(e);mouse.x=p.x;mouse.y=p.y;mouse.has=true;ctrl='mouse';return}
  if(ptr.active&&e.pointerId===ptr.id&&P&&G.state==='play'){const k=1.5;P.x+=(e.clientX-ptr.lx)*k;P.y+=(e.clientY-ptr.ly)*k;ptr.lx=e.clientX;ptr.ly=e.clientY}
});
const endPtr=e=>{if(e.pointerId===ptr.id)ptr.active=false};
cv.addEventListener('pointerup',endPtr);cv.addEventListener('pointercancel',endPtr);
cv.addEventListener('contextmenu',e=>e.preventDefault());

/* ---------- helpers ---------- */
function rr(c,x,y,w,h,r){c.beginPath();c.moveTo(x+r,y);c.arcTo(x+w,y,x+w,y+h,r);c.arcTo(x+w,y+h,x,y+h,r);c.arcTo(x,y+h,x,y,r);c.arcTo(x,y,x+w,y,r);c.closePath()}
function burst(x,y,col,n,spd,size){
  for(let i=0;i<n&&parts.length<500;i++){const a=Math.random()*TAU,v=rnd(0.2,1)*spd*S;
    parts.push({x,y,vx:Math.cos(a)*v,vy:Math.sin(a)*v,life:rnd(0.4,0.9),max:0.9,c:col,s:rnd(1,size||3)*S})}
}
function floatText(x,y,t,c){texts.push({x,y,t,c:c||'#e6eaff',life:1})}
function addScore(v,x,y){
  G.score+=v;if(x!=null)floatText(x,y,'+'+v);
  if(G.score>=G.nextLife){G.nextLife+=25000;G.lives=Math.min(G.lives+1,6);floatText(W/2,H*0.5,'Extra ship','#49e3ff');sfx.gift()}
}

/* ---------- waves ---------- */
const TIER_COL=['#ff7a3d','#49e3ff','#ff4d9d','#a6ff5c'];
const LAST_WAVE=10;
const WAVES=[null,
  {name:'Lost crew',k:'grid',lvl:0},
  {name:'Comet shower',k:'comets',lvl:0},
  {name:'Spacewalk loop',k:'swoop',lvl:0},
  {name:'Crew and comets',k:'mixed',lvl:0},
  {name:'The Black Hole',k:'boss',boss:'hole'},
  {name:'Meteor storm',k:'comets',lvl:1},
  {name:'Loop the loop',k:'swoop',lvl:1},
  {name:'Veteran crew',k:'grid',lvl:1},
  {name:'Last stand',k:'mixed',lvl:1},
  {name:'Zorgath the Alien',k:'boss',boss:'alien'},
];
function makeAstro(o){const tier=o.tier||0;const hp=[2,3,5,7][tier]+Math.floor(G.wave/6);
  return Object.assign({type:'astro',r:17*S,hp,mhp:hp,tier,t:Math.random()*10,flash:0,delay:0},o)}
function makeComet(x,y,vx,vy,size){const r=[10,17,27][size]*S;const hp=[1,3,7][size]+Math.floor(G.wave/8);
  const verts=[];for(let i=0;i<9;i++)verts.push(rnd(0.75,1.05));
  return{type:'comet',x,y,vx,vy,r,size,hp,mhp:hp,rot:Math.random()*TAU,vr:rnd(-2,2),verts,flash:0}}
function nextWave(){
  G.wave++;G.between=false;G.waveT=0;G.spawnQ=[];
  if(P)makeCheck();
  const w=WAVES[G.wave],lvl=w.lvl||0;
  G.banner=w.k==='boss'?'Boss':'Wave '+G.wave;G.sub=w.name;G.bannerT=2.6;sfx.wave();
  if(P&&P.shield<P.up.armor){P.shield++}
  $('waveLbl').textContent='Wave '+G.wave+'/'+LAST_WAVE;$('waveName').textContent=w.name;
  const grid=rows=>{
    const cols=clamp(Math.floor((W-60*S)/(54*S)),4,9),gx=(W-(cols-1)*54*S)/2;
    for(let r=0;r<rows;r++)for(let c=0;c<cols;c++)
      enemies.push(makeAstro({mode:'grid',hx:gx+c*54*S,hy:G.topH+28*S+r*50*S,x:gx+c*54*S,y:-60-r*60-c*14,delay:(r*cols+c)*0.04,tier:Math.min(3,lvl+(r===0?1:0))}));
  };
  const comets=(n,gap,start)=>{
    for(let i=0;i<n;i++)G.spawnQ.push({at:start+i*gap,fn:()=>{
      const size=Math.random()<0.22+lvl*0.12?2:Math.random()<0.5?1:0;
      const x=rnd(-W*0.1,W*1.1),vx=(W/2-x)*rnd(0.1,0.35)+rnd(-40,40)*S;
      enemies.push(makeComet(x,-40*S,vx,rnd(110,170)*S*(1+lvl*0.2),size))}});
  };
  if(w.k==='grid')grid(3+lvl);
  else if(w.k==='comets')comets(16+lvl*8,0.75-lvl*0.2,0.6);
  else if(w.k==='swoop'){
    const groups=3+lvl,per=6;
    for(let g=0;g<groups;g++){const dir=g%2?-1:1,base=rnd(70,H*0.4),amp=rnd(30,80)*S,fr=rnd(0.008,0.014)/S,ph=Math.random()*TAU;
      for(let i=0;i<per;i++)G.spawnQ.push({at:0.5+g*2.4,fn:()=>enemies.push(makeAstro({mode:'swoop',dir,s:-i*48*S,base,amp,fr,ph,spd:(150+lvl*25)*S,x:-100,y:-100,tier:Math.min(3,lvl+(g%2))}))});
    }
  }
  else if(w.k==='mixed'){grid(2+lvl);comets(8+lvl*4,1.6-lvl*0.3,3)}
  else if(w.boss==='hole'){
    enemies.push({type:'boss',kind:'hole',name:'The Black Hole',x:W/2,y:-120*S,r:44*S,hp:180,mhp:180,t:0,flash:0,
      laser:{state:'idle',T:2.5,ang:Math.PI/2},meteorT:2});
  }else{
    enemies.push({type:'boss',kind:'alien',name:'Zorgath the Alien',x:W/2,y:-120*S,r:48*S,hp:260,mhp:260,t:0,flash:0,
      slimeT:1.6,fireT:3.2,look:Math.PI/2});
  }
}

/* ---------- bosses ---------- */
function rockVerts(){const v=[];for(let i=0;i<7;i++)v.push(rnd(0.7,1.05));return v}
function updHole(e,dt,live){
  e.t+=dt;const tx=W/2+Math.sin(e.t*0.35)*(W/2-90*S),ty=140*S+Math.sin(e.t*0.9)*18*S;
  e.x+=(tx-e.x)*Math.min(1,dt*1.2);e.y+=(ty-e.y)*Math.min(1,dt*1.2);
  if(!live||!P||P.dead)return;
  const rage=e.hp<e.mhp*0.5;
  // gentle gravity pull on the ship
  {const dx=e.x-P.x,dy=e.y-P.y,l=Math.hypot(dx,dy)||1,pull=(rage?28:18)*S;P.x+=dx/l*pull*dt;P.y+=dy/l*pull*dt}
  // dark laser: aim, then fire
  const L=e.laser;L.T-=dt;const want=Math.atan2(P.y-e.y,P.x-e.x);
  if(L.state==='idle'&&L.T<=0){L.state='aim';L.T=1.4;L.ang=want;tone(120,1.1,'sawtooth',0.05,420)}
  else if(L.state==='aim'){const da=((want-L.ang+Math.PI*3)%TAU)-Math.PI;L.ang+=da*Math.min(1,dt*1.4);
    if(L.T<=0){L.state='fire';L.T=0.6;noise(0.7,0.25,600);tone(60,0.7,'sawtooth',0.12,40);G.shake=Math.max(G.shake,reduce?0:5)}}
  else if(L.state==='fire'){
    const vx=Math.cos(L.ang),vy=Math.sin(L.ang),px=P.x-e.x,py=P.y-e.y,along=px*vx+py*vy;
    if(along>0&&Math.abs(px*vy-py*vx)<8*S+P.r*0.45)hurtPlayer();
    if(L.T<=0){L.state='idle';L.T=rage?3.2:4.3}}
  // meteor bursts flung from the disk
  e.meteorT-=dt;
  if(e.meteorT<=0){e.meteorT=rage?2.6:3.4;const n=rage?5:4,spin=e.t*1.7;
    for(let i=0;i<n;i++){const a=Math.PI/2+(i-(n-1)/2)*0.32+Math.sin(spin)*0.2,v=(135+(rage?15:0))*S;
      ebul.push({x:e.x+Math.cos(a)*e.r,y:e.y+Math.sin(a)*e.r,vx:Math.cos(a)*v,vy:Math.sin(a)*v,r:6.5*S,k:'rock',rot:Math.random()*TAU,verts:rockVerts()})}
    tone(180,0.2,'triangle',0.06,80)}
  if(Math.hypot(e.x-P.x,e.y-P.y)<e.r*0.9+P.r*0.7)hurtPlayer();
}
function updAlien(e,dt,live){
  e.t+=dt;const tx=W/2+Math.sin(e.t*0.5)*(W/2-90*S),ty=150*S+Math.sin(e.t*1.0)*30*S;
  e.x+=(tx-e.x)*Math.min(1,dt*1.4);e.y+=(ty-e.y)*Math.min(1,dt*1.2);
  if(P)e.look=Math.atan2(P.y-e.y,P.x-e.x);
  if(!live||!P||P.dead)return;
  const rage=e.hp<e.mhp*0.5;
  // green slime globs, lobbed in arcs
  e.slimeT-=dt;
  if(e.slimeT<=0){e.slimeT=rage?1.9:2.5;const n=rage?3:2,T=1.5;
    for(let i=0;i<n;i++){const tx2=P.x+(i-(n-1)/2)*55*S,g=240*S,vx=(tx2-e.x)/T,vy=(P.y-e.y)/T-0.5*g*T;
      ebul.push({x:e.x,y:e.y+20*S,vx,vy:Math.max(vy,-220*S),g,r:9*S,k:'slime',t:Math.random()*6})}
    tone(140,0.25,'sine',0.1,60)}
  // fire meteors, aimed
  e.fireT-=dt;
  if(e.fireT<=0){e.fireT=rage?2.8:3.6;const n=rage?3:2,base=Math.atan2(P.y-e.y,P.x-e.x);
    for(let i=0;i<n;i++){const a=base+(i-(n-1)/2)*0.22,v=175*S;
      ebul.push({x:e.x,y:e.y+20*S,vx:Math.cos(a)*v,vy:Math.sin(a)*v,r:7*S,k:'rock',fire:true,rot:Math.random()*TAU,verts:rockVerts()})}
    noise(0.3,0.15,2500)}
  if(Math.hypot(e.x-P.x,e.y-P.y)<e.r*0.85+P.r*0.7)hurtPlayer();
}
function win(){
  G.state='over';G.overT=0;show('top',false);show('bottom',false);
  const bonus=G.lives*5000+G.dust*20;G.score+=bonus;
  const nb=G.score>G.best;if(nb){G.best=G.score;saveBest(G.best)}
  $('overTitle').textContent='Mission complete';
  $('finalScore').textContent=G.score.toLocaleString('en-US');
  $('overMsg').textContent=`All ${LAST_WAVE} waves cleared. Bonus ${bonus.toLocaleString('en-US')} for ships and stardust left.`+(nb?' New best!':' Best: '+G.best.toLocaleString('en-US')+'.');
  bullets=[];ebul=[];pickups=[];
  G.check=null;saveCheck(null);setOverButtons(false);
  show('overScr',true);sfx.gift();
}

/* ---------- player actions ---------- */
function shoot(){
  const lv=P.lv,rf=Math.pow(0.93,P.up.rate),hm=1-0.1*P.up.cool;
  // wing guns from Rapid cannons
  if(P.up.rate>=1)for(const s of[-1,1])bullets.push({x:P.x+s*16*S,y:P.y,vx:0,vy:-780*S,dmg:0.35,c:P.wt,r:3*S});
  if(P.up.rate>=3)for(const s of[-1,1])bullets.push({x:P.x+s*16*S,y:P.y,vx:s*110*S,vy:-760*S,dmg:0.35,c:P.wt,r:3*S});
  const heat0=P.heat;
  if(P.wt==='ion'){
    P.cd=0.26*rf;const n=lv,sp=lv>3?0.1:0.13;
    for(let i=0;i<n;i++){const a=(i-(n-1)/2)*sp;bullets.push({x:P.x,y:P.y-P.r,vx:Math.sin(a)*640*S,vy:-Math.cos(a)*640*S,dmg:0.9,c:'ion',r:4.5*S})}
    P.heat+=0.04+0.007*lv;sfx.ion();
  }else{
    P.cd=0.14*rf;const lanes=Math.min(lv,3);
    for(let i=0;i<lanes;i++)bullets.push({x:P.x+(i-(lanes-1)/2)*9*S,y:P.y-P.r,vx:0,vy:-900*S,dmg:0.6,c:'plasma',r:3.5*S});
    if(lv>=4)for(const s of[-1,1])bullets.push({x:P.x+s*10*S,y:P.y-P.r,vx:s*140*S,vy:-880*S,dmg:0.6,c:'plasma',r:3.5*S});
    if(lv>=5)for(const s of[-1,1])bullets.push({x:P.x+s*14*S,y:P.y,vx:s*300*S,vy:-820*S,dmg:0.6,c:'plasma',r:3.5*S});
    P.heat+=0.026+0.004*lv;sfx.plasma();
  }
  P.heat=heat0+(P.heat-heat0)*hm;
  if(P.heat>=1){P.heat=1;P.over=true;sfx.over();floatText(P.x,P.y-30*S,'Overheated','#ff4d6d')}
}
function nova(){
  if(G.state!=='play'||G.novas<=0||!P||P.dead)return;
  G.novas--;G.flash=0.5;G.shake=reduce?0:14;sfx.nova();
  ebul=[];
  for(let i=0;i<60;i++){const a=i/60*TAU;parts.push({x:P.x,y:P.y,vx:Math.cos(a)*700*S,vy:Math.sin(a)*700*S,life:0.8,max:0.8,c:'#ffd65a',s:4*S})}
  for(const e of enemies){if(e.y<-20||e.y>H+20)continue;damage(e,e.type==='boss'?15:6)}
  updHUD(true);
}
function hurtPlayer(){
  if(P.inv>0||P.dead)return;
  if(P.shield>0){P.shield--;P.inv=1.2;sfx.over();burst(P.x,P.y,'#49e3ff',30,240,3);floatText(P.x,P.y-36*S,'Shield hit','#49e3ff');G.shake=reduce?0:6;return}
  sfx.hurt();burst(P.x,P.y,'#ffb35a',60,320,4);burst(P.x,P.y,'#49e3ff',30,200,3);
  G.shake=reduce?0:16;G.lives--;P.lv=Math.max(1,P.lv-1);P.heat=0;P.over=false;
  if(G.lives<=0){P.dead=true;G.state='dying';G.overT=0;}
  else{P.inv=2.6;P.x=W/2;P.y=H-110*S}
  updHUD(true);
}

/* ---------- damage & drops ---------- */
function drop(x,y,kind,wt){pickups.push({x,y,vx:rnd(-40,40)*S,vy:rnd(-80,-20)*S,kind,wt,t:Math.random()*6})}
function damage(e,d){
  e.hp-=d;e.flash=0.08;
  if(e.hp>0){return}
  e.dead=true;
  if(e.type==='astro'){
    sfx.boom();burst(e.x,e.y,'#ffffff',18,220,3);burst(e.x,e.y,TIER_COL[e.tier],14,180,3);
    addScore(100*(e.tier+1),e.x,e.y);G.shake=Math.max(G.shake,reduce?0:3);
    drop(e.x,e.y,'dust');if(Math.random()<0.3)drop(e.x+8,e.y,'dust');
    if(Math.random()<0.05)drop(e.x,e.y,'gift',Math.random()<0.5?'ion':'plasma');
  }else if(e.type==='comet'){
    sfx.boom();burst(e.x,e.y,'#9fdcff',12+e.size*8,200+e.size*40,3);burst(e.x,e.y,'#8a93b8',8,140,4);
    addScore([40,80,150][e.size],e.x,e.y);
    if(e.size>0)for(const s of[-1,1])enemies.push(makeComet(e.x,e.y,e.vx*0.6+s*rnd(60,120)*S,e.vy*0.8,e.size-1));
    if(Math.random()<0.45)drop(e.x,e.y,'dust');
    if(e.size===2&&Math.random()<0.08)drop(e.x,e.y,'gift',Math.random()<0.5?'ion':'plasma');
  }else if(e.type==='boss'){
    sfx.big();G.shake=reduce?0:24;G.flash=0.4;
    for(let i=0;i<5;i++)setTimeout(()=>burst(e.x+rnd(-50,50)*S,e.y+rnd(-50,50)*S,i%2?'#ff4d9d':'#ffd65a',50,380,5),i*120);
    addScore(e.kind==='hole'?6000:10000,e.x,e.y);
    for(let i=0;i<14;i++)drop(e.x+rnd(-40,40)*S,e.y+rnd(-30,30)*S,'dust');
    drop(e.x-20*S,e.y,'gift','ion');drop(e.x+20*S,e.y,'gift','plasma');
  }
}

/* ---------- update ---------- */
function update(dt){
  G.t+=dt;
  for(const s of stars){s.y+=(10+s.z*s.z*70)*dt*S;if(s.y>H){s.y=-2;s.x=Math.random()*W}}
  if(G.bannerT>0)G.bannerT-=dt;
  if(G.flash>0)G.flash-=dt;
  G.shake*=Math.pow(0.02,dt);if(G.shake<0.3)G.shake=0;

  if(G.state==='title'||G.state==='over'){
    G.overT+=dt;G.ambientT-=dt;
    if(G.ambientT<=0){G.ambientT=rnd(0.8,1.8);const x=rnd(0,W);enemies.push(makeComet(x,-40,(W/2-x)*0.15,rnd(60,110)*S,Math.random()<0.3?2:Math.random()<0.5?1:0))}
    for(const e of enemies){e.x+=e.vx*dt;e.y+=e.vy*dt;e.rot+=e.vr*dt;if(e.y>H+80)e.dead=true}
    enemies=enemies.filter(e=>!e.dead);
    stepParts(dt);return;
  }
  if(G.state==='pause'||G.state==='shop')return;

  if(G.state==='dying'){G.overT+=dt;stepWorld(dt,false);if(G.overT>1.6)gameOver();return}

  // player
  const sp=380*S*(1+0.08*P.up.thr);
  const ax=((keys.ArrowRight||keys.KeyD)?1:0)-((keys.ArrowLeft||keys.KeyA)?1:0);
  const ay=((keys.ArrowDown||keys.KeyS)?1:0)-((keys.ArrowUp||keys.KeyW)?1:0);
  if(ax||ay){ctrl='key';const m=ax&&ay?0.7071:1;P.x+=ax*sp*m*dt;P.y+=ay*sp*m*dt}
  else if(ctrl==='mouse'&&mouse.has){const k=Math.min(1,dt*14);P.x+=(mouse.x-P.x)*k;P.y+=(mouse.y-P.y)*k}
  P.x=clamp(P.x,P.r,W-P.r);P.y=clamp(P.y,Math.max(H*0.3,G.topH+40*S),H-G.botH-P.r-14*S);
  if(P.inv>0)P.inv-=dt;
  P.cd-=dt;P.heat=Math.max(0,P.heat-dt*(P.over?0.5:0.32)*(1+0.15*P.up.cool));if(P.over&&P.heat<=0)P.over=false;
  const firing=keys.Space||ptr.active;
  if(firing&&!P.over&&P.cd<=0)shoot();

  // spawns
  G.waveT+=dt;
  for(let i=G.spawnQ.length-1;i>=0;i--)if(G.spawnQ[i].at<=G.waveT){G.spawnQ[i].fn();G.spawnQ.splice(i,1)}

  stepWorld(dt,true);

  if(!G.between&&enemies.length===0&&G.spawnQ.length===0){G.between=true;G.nextT=2;if(G.wave>0)floatText(W/2,H*0.42,'Wave clear','#a6ff5c')}
  if(G.between){G.nextT-=dt;if(G.nextT<=0){if(G.wave>=LAST_WAVE)win();else nextWave()}}
  updHUD(false);
}

function stepWorld(dt,live){
  const d=G.wave;
  // bullets
  for(const b of bullets){b.x+=b.vx*dt;b.y+=b.vy*dt;if(b.y<-20||b.x<-20||b.x>W+20)b.dead=true}
  // enemies
  const sway=Math.sin(G.t*0.7)*Math.min(W*0.08,60*S);
  for(const e of enemies){
    if(e.flash>0)e.flash-=dt;
    if(e.type==='astro'){
      e.t+=dt;if(e.throwT>0)e.throwT-=dt;
      if(e.delay>0){e.delay-=dt;continue}
      let tx,ty;
      if(e.mode==='grid'){tx=e.hx+sway;ty=e.hy+Math.sin(e.t*2)*4*S+Math.min(G.waveT*1.2,40)*S;const k=Math.min(1,dt*2.6);e.x+=(tx-e.x)*k;e.y+=(ty-e.y)*k}
      else{e.s+=e.spd*dt;if(e.s<0){e.x=-200;e.y=-200;continue}
        e.x=e.dir>0?-40*S+e.s:W+40*S-e.s;e.y=e.base+e.amp*Math.sin(e.s*e.fr+e.ph);
        if(e.s>W+80*S){e.s=0;e.base+=40*S;if(e.base>H*0.6)e.base=70*S}}
      // fire
      if(live&&e.y>0&&e.y<H*0.75&&ebul.length<10+d*1.5&&Math.random()<(0.07+0.012*d)*dt*(e.mode==='swoop'?1.5:1)){
        let vx=0,vy=(150+d*7)*S;
        if(e.tier>=2&&P){const dx=P.x-e.x,dy=P.y-e.y,l=Math.hypot(dx,dy)||1;vx=dx/l*vy;vy=dy/l*vy}
        const verts=[];for(let i=0;i<7;i++)verts.push(rnd(0.7,1.05));
        ebul.push({x:e.x,y:e.y+10*S,vx,vy,r:6.5*S,k:'rock',rot:Math.random()*TAU,verts});e.throwT=0.35}
      if(live&&P&&!P.dead&&Math.hypot(e.x-P.x,e.y-P.y)<e.r+P.r*0.8){hurtPlayer();damage(e,99)}
    }else if(e.type==='comet'){
      e.x+=e.vx*dt;e.y+=e.vy*dt;e.rot+=e.vr*dt;if(e.y>H+60||e.x<-120||e.x>W+120)e.dead=true;
      if(live&&P&&!P.dead&&Math.hypot(e.x-P.x,e.y-P.y)<e.r*0.85+P.r*0.8){hurtPlayer();damage(e,99)}
    }else if(e.type==='boss'){
      if(e.kind==='hole')updHole(e,dt,live);else updAlien(e,dt,live);
    }
  }
  // bullet collisions
  for(const b of bullets){if(b.dead)continue;
    for(const e of enemies){if(e.dead||e.delay>0||e.y<-30)continue;
      if(Math.hypot(b.x-e.x,b.y-e.y)<e.r+b.r){b.dead=true;damage(e,b.dmg);sfx.hit();burst(b.x,b.y,b.c==='ion'?'#ff9a5a':'#7fefff',3,120,2);break}}}
  // enemy bullets
  for(const b of ebul){if(b.g)b.vy+=b.g*dt;b.x+=b.vx*dt;b.y+=b.vy*dt;if(b.rot!=null)b.rot+=dt*4;if(b.t!=null)b.t+=dt;
    if(b.y>H+20||b.y<-140||b.x<-40||b.x>W+40)b.dead=true;
    else if(live&&P&&!P.dead&&Math.hypot(b.x-P.x,b.y-P.y)<b.r+P.r*0.55){b.dead=true;if(b.k==='slime')burst(b.x,b.y,'#7dff5a',16,160,4);hurtPlayer()}}
  // pickups
  for(const p of pickups){p.t+=dt;p.vy=Math.min(p.vy+200*S*dt,130*S);p.vx*=Math.pow(0.3,dt);
    if(P&&!P.dead){const dx=P.x-p.x,dy=P.y-p.y,l=Math.hypot(dx,dy);
      if(l<90*S*(1+0.5*P.up.mag)){p.x+=dx/l*360*S*dt;p.y+=dy/l*360*S*dt}
      if(l<P.r+14*S){p.dead=true;collect(p);continue}}
    p.x+=p.vx*dt;p.y+=p.vy*dt;if(p.y>H+20)p.dead=true}
  bullets=bullets.filter(b=>!b.dead);enemies=enemies.filter(e=>!e.dead);ebul=ebul.filter(b=>!b.dead);pickups=pickups.filter(p=>!p.dead);
  stepParts(dt);
}
function stepParts(dt){
  for(const p of parts){p.x+=p.vx*dt;p.y+=p.vy*dt;p.vx*=Math.pow(0.15,dt);p.vy*=Math.pow(0.15,dt);p.life-=dt}
  parts=parts.filter(p=>p.life>0);
  for(const t of texts){t.y-=40*dt;t.life-=dt*0.9}
  texts=texts.filter(t=>t.life>0);
}
function collect(p){
  if(p.kind==='dust'){G.dust++;addScore(25);sfx.pick()}
  else{sfx.gift();
    if(p.wt===P.wt){if(P.lv<5){P.lv++;floatText(P.x,P.y-36*S,'Power up','#a6ff5c')}else addScore(1000,P.x,P.y-36*S)}
    else{P.wt=p.wt;floatText(P.x,P.y-36*S,p.wt==='ion'?'Ion blaster':'Plasma lance',p.wt==='ion'?'#ff7a3d':'#49e3ff')}}
}

/* ---------- drawing ---------- */
function drawAstro(e,boss){
  const s=e.r/17,t=e.t;ctx.save();ctx.translate(e.x,e.y);ctx.rotate(Math.sin(t*1.6)*0.2);ctx.scale(s,s);
  const suit=e.flash>0?'#ffffff':'#e9edfb',shade='#b8c0dc',col=boss?'#ff4d6d':TIER_COL[e.tier];
  if(boss){// jetpack flames
    const fl=8+Math.random()*8;ctx.globalCompositeOperation='lighter';
    for(const sx of[-8,8]){const g=ctx.createLinearGradient(0,10,0,24+fl);g.addColorStop(0,'rgba(255,200,90,0.9)');g.addColorStop(1,'rgba(255,80,40,0)');
      ctx.fillStyle=g;ctx.beginPath();ctx.moveTo(sx-4,12);ctx.lineTo(sx+4,12);ctx.lineTo(sx,24+fl);ctx.fill()}
    ctx.globalCompositeOperation='source-over'}
  ctx.fillStyle='#8790b4';rr(ctx,-12,-9,24,21,4);ctx.fill();               // backpack
  ctx.fillStyle=shade;                                                        // legs
  ctx.save();ctx.translate(-5,10);ctx.rotate(Math.sin(t*3)*0.25);rr(ctx,-4,0,8,14,3);ctx.fill();ctx.restore();
  ctx.save();ctx.translate(5,10);ctx.rotate(-Math.sin(t*3)*0.25);rr(ctx,-4,0,8,14,3);ctx.fill();ctx.restore();
  ctx.fillStyle=suit;rr(ctx,-11,-6,22,20,7);ctx.fill();                       // torso
  ctx.fillStyle=suit;                                                         // arms
  ctx.save();ctx.translate(-10,-2);ctx.rotate(0.9+Math.sin(t*4)*0.6);rr(ctx,-3,0,6,14,3);ctx.fill();ctx.fillStyle=shade;ctx.beginPath();ctx.arc(0,14,3.4,0,TAU);ctx.fill();ctx.restore();
  ctx.fillStyle=suit;
  ctx.save();ctx.translate(10,-2);ctx.rotate(e.throwT>0?-0.4+e.throwT*6:-0.9-Math.sin(t*4+1)*0.6);rr(ctx,-3,0,6,14,3);ctx.fill();ctx.fillStyle=shade;ctx.beginPath();ctx.arc(0,14,3.4,0,TAU);ctx.fill();ctx.restore();
  ctx.fillStyle=col;rr(ctx,-5,1,10,5,1.5);ctx.fill();                         // chest patch
  ctx.fillStyle='#2a3150';ctx.fillRect(-2,8,4,3);
  ctx.fillStyle=suit;ctx.beginPath();ctx.arc(0,-13,11,0,TAU);ctx.fill();      // helmet
  ctx.strokeStyle=shade;ctx.lineWidth=1.2;ctx.stroke();
  const vg=ctx.createLinearGradient(-7,-19,7,-6);vg.addColorStop(0,boss?'#ff8fa8':'#ffe08a');vg.addColorStop(0.5,boss?'#d0204a':'#e59a2b');vg.addColorStop(1,boss?'#5a0a20':'#6b3a10');
  ctx.fillStyle=vg;ctx.beginPath();ctx.ellipse(0,-12.5,8,6.6,0,0,TAU);ctx.fill();
  ctx.fillStyle='rgba(255,255,255,0.75)';ctx.beginPath();ctx.ellipse(-3.5,-15,2.4,1.3,-0.5,0,TAU);ctx.fill();
  ctx.restore();
}
function drawComet(e){
  const sp=Math.hypot(e.vx,e.vy)||1,nx=-e.vx/sp,ny=-e.vy/sp,len=e.r*(4+e.size);
  ctx.save();ctx.globalCompositeOperation='lighter';
  const g=ctx.createLinearGradient(e.x,e.y,e.x+nx*len,e.y+ny*len);
  g.addColorStop(0,'rgba(140,220,255,0.55)');g.addColorStop(0.4,'rgba(90,140,255,0.22)');g.addColorStop(1,'rgba(90,140,255,0)');
  ctx.fillStyle=g;ctx.beginPath();ctx.moveTo(e.x-ny*e.r*0.95,e.y+nx*e.r*0.95);ctx.lineTo(e.x+nx*len,e.y+ny*len);ctx.lineTo(e.x+ny*e.r*0.95,e.y-nx*e.r*0.95);ctx.fill();
  const gl=ctx.createRadialGradient(e.x,e.y,e.r*0.5,e.x,e.y,e.r*1.8);gl.addColorStop(0,'rgba(160,230,255,0.35)');gl.addColorStop(1,'rgba(160,230,255,0)');
  ctx.fillStyle=gl;ctx.beginPath();ctx.arc(e.x,e.y,e.r*1.8,0,TAU);ctx.fill();
  ctx.restore();
  ctx.save();ctx.translate(e.x,e.y);ctx.rotate(e.rot);
  ctx.fillStyle=e.flash>0?'#ffffff':'#5d6890';ctx.strokeStyle='#b9e6ff';ctx.lineWidth=1.5*S;ctx.beginPath();
  e.verts.forEach((v,i)=>{const a=i/e.verts.length*TAU;const px=Math.cos(a)*e.r*v,py=Math.sin(a)*e.r*v;i?ctx.lineTo(px,py):ctx.moveTo(px,py)});
  ctx.closePath();ctx.fill();ctx.stroke();
  ctx.fillStyle='rgba(20,26,50,0.5)';ctx.beginPath();ctx.arc(e.r*0.25,-e.r*0.2,e.r*0.22,0,TAU);ctx.arc(-e.r*0.3,e.r*0.3,e.r*0.15,0,TAU);ctx.fill();
  ctx.restore();
}
function drawHole(e){
  const r=e.r,t=e.t;
  ctx.save();ctx.translate(e.x,e.y);
  ctx.globalCompositeOperation='lighter';
  const gl=ctx.createRadialGradient(0,0,r*0.8,0,0,r*2.8);gl.addColorStop(0,'rgba(150,70,255,0.4)');gl.addColorStop(1,'rgba(150,70,255,0)');
  ctx.fillStyle=gl;ctx.beginPath();ctx.arc(0,0,r*2.8,0,TAU);ctx.fill();
  const disk=front=>{
    for(let j=0;j<70;j++){const a=j/70*TAU+t*(0.9+(j%3)*0.35),rad=r*(1.15+(j%6)*0.13);
      const sy=Math.sin(a);if((sy>0)!==front)continue;
      const x=Math.cos(a)*rad,y=sy*rad*0.32-r*0.05;
      ctx.fillStyle=j%4===0?'rgba(255,220,140,0.9)':j%2?'rgba(255,120,60,0.75)':'rgba(220,90,255,0.75)';
      const s=(1.6+(j%3))*S;ctx.fillRect(x-s/2,y-s/2,s,s)}
    ctx.strokeStyle=front?'rgba(255,150,80,0.55)':'rgba(200,100,255,0.35)';ctx.lineWidth=3*S;
    ctx.beginPath();ctx.ellipse(0,-r*0.05,r*1.5,r*0.48,0,front?0:Math.PI,front?Math.PI:TAU);ctx.stroke();
  };
  disk(false);
  ctx.globalCompositeOperation='source-over';
  ctx.fillStyle='#000';ctx.beginPath();ctx.arc(0,0,r,0,TAU);ctx.fill();
  ctx.strokeStyle=e.flash>0?'#ffffff':'rgba(255,190,120,0.9)';ctx.lineWidth=2.5*S;ctx.beginPath();ctx.arc(0,0,r*1.02,0,TAU);ctx.stroke();
  ctx.globalCompositeOperation='lighter';
  disk(true);
  ctx.restore();
  // dark laser
  const L=e.laser;if(L.state==='aim'||L.state==='fire'){
    ctx.save();ctx.translate(e.x,e.y);ctx.rotate(L.ang);const len=Math.hypot(W,H)*1.2;
    if(L.state==='aim'){const p=1-L.T/1.4;ctx.strokeStyle=`rgba(190,90,255,${0.25+0.5*p})`;ctx.lineWidth=(1+p*2)*S;ctx.setLineDash([10*S,8*S]);
      ctx.beginPath();ctx.moveTo(r,0);ctx.lineTo(len,0);ctx.stroke();ctx.setLineDash([])}
    else{const w=18*S*Math.min(1,(0.6-L.T)*8,L.T*6+0.2);
      ctx.globalCompositeOperation='lighter';const g=ctx.createLinearGradient(0,-w,0,w);
      g.addColorStop(0,'rgba(160,60,255,0)');g.addColorStop(0.3,'rgba(170,70,255,0.8)');g.addColorStop(0.7,'rgba(170,70,255,0.8)');g.addColorStop(1,'rgba(160,60,255,0)');
      ctx.fillStyle=g;ctx.fillRect(r*0.8,-w,len,w*2);
      ctx.globalCompositeOperation='source-over';ctx.fillStyle='#07020f';ctx.fillRect(r*0.8,-w*0.38,len,w*0.76);
      if(Math.random()<0.6)burst(e.x+Math.cos(L.ang)*rnd(r,H),e.y+Math.sin(L.ang)*rnd(r,H),'#b46bff',1,80,3)}
    ctx.restore()}
}
function drawAlien(e){
  const s=e.r/48,t=e.t;
  ctx.save();ctx.translate(e.x,e.y);
  ctx.globalCompositeOperation='lighter';
  const gl=ctx.createRadialGradient(0,0,e.r*0.5,0,0,e.r*2);gl.addColorStop(0,'rgba(120,255,90,0.25)');gl.addColorStop(1,'rgba(120,255,90,0)');
  ctx.fillStyle=gl;ctx.beginPath();ctx.arc(0,0,e.r*2,0,TAU);ctx.fill();
  ctx.globalCompositeOperation='source-over';
  ctx.scale(s,s);
  const fl=e.flash>0;
  // tentacles
  ctx.lineCap='round';
  for(let i=0;i<6;i++){const x0=-25+i*10,w1=Math.sin(t*3+i)*12,w2=Math.sin(t*3+i+1.5)*14;
    ctx.strokeStyle=fl?'#fff':'#2f9a3e';ctx.lineWidth=8-Math.abs(i-2.5)*0.8;
    ctx.beginPath();ctx.moveTo(x0,12);ctx.bezierCurveTo(x0+w1,30,x0-w2,42,x0+w2*0.6,58);ctx.stroke();
    ctx.fillStyle='#b9ff7a';ctx.beginPath();ctx.arc(x0+w2*0.6,58,2.5,0,TAU);ctx.fill()}
  // body
  ctx.fillStyle=fl?'#fff':'#47c24f';ctx.beginPath();ctx.ellipse(0,8,28,20,0,0,TAU);ctx.fill();
  // antennae
  for(const sx of[-1,1]){const tipx=sx*(22+Math.sin(t*2+sx)*4),tipy=-66+Math.cos(t*2.4+sx)*3;
    ctx.strokeStyle='#3caa45';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(sx*12,-44);ctx.quadraticCurveTo(sx*16,-58,tipx,tipy);ctx.stroke();
    ctx.fillStyle='#ff5ad1';ctx.beginPath();ctx.arc(tipx,tipy,4.5,0,TAU);ctx.fill();
    ctx.fillStyle='rgba(255,90,209,0.3)';ctx.beginPath();ctx.arc(tipx,tipy,9,0,TAU);ctx.fill()}
  // head
  const hg=ctx.createRadialGradient(-10,-34,4,0,-22,40);hg.addColorStop(0,fl?'#fff':'#a6ff8a');hg.addColorStop(1,fl?'#ddd':'#2e8c3a');
  ctx.fillStyle=hg;ctx.beginPath();ctx.ellipse(0,-22,34,27,0,0,TAU);ctx.fill();
  ctx.fillStyle='rgba(20,60,25,0.35)';ctx.beginPath();ctx.arc(18,-36,4,0,TAU);ctx.arc(-22,-14,3,0,TAU);ctx.fill();
  // eyes that follow the ship
  const lx=Math.cos(e.look)*2.6,ly=Math.sin(e.look)*2.6;
  for(const [ex,ey,er] of [[-14,-22,8],[14,-22,8],[0,-38,5.5]]){
    ctx.fillStyle='#10140f';ctx.beginPath();ctx.ellipse(ex,ey,er,er*1.15,0,0,TAU);ctx.fill();
    ctx.fillStyle='#ffe95a';ctx.beginPath();ctx.arc(ex+lx,ey+ly,er*0.45,0,TAU);ctx.fill();
    ctx.fillStyle='#10140f';ctx.beginPath();ctx.ellipse(ex+lx,ey+ly,er*0.15,er*0.38,0,0,TAU);ctx.fill()}
  // mouth with dripping slime
  ctx.fillStyle='#1a3d1c';ctx.beginPath();ctx.ellipse(0,-6,10,4+Math.sin(t*4)*1.5,0,0,Math.PI);ctx.fill();
  const drip=(t*0.8)%1;ctx.fillStyle='#7dff5a';ctx.beginPath();ctx.ellipse(4,-2+drip*14,2.2,3+drip*2,0,0,TAU);ctx.fill();
  ctx.restore();
}
function drawShip(){
  if(!P||P.dead)return;if(P.inv>0&&Math.floor(P.inv*12)%2)return;
  const s=P.r/14,acc=P.wt==='ion'?'#ff7a3d':'#49e3ff';
  ctx.save();ctx.translate(P.x,P.y);ctx.scale(s,s);
  ctx.globalCompositeOperation='lighter';const fl=12+Math.random()*10;
  const g=ctx.createLinearGradient(0,13,0,13+fl);g.addColorStop(0,'rgba(255,255,255,0.95)');g.addColorStop(0.3,P.wt==='ion'?'rgba(255,140,60,0.9)':'rgba(80,220,255,0.9)');g.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=g;ctx.beginPath();ctx.moveTo(-5,13);ctx.lineTo(5,13);ctx.lineTo(0,13+fl);ctx.fill();
  if(P.up.thr>0){const f2=6+P.up.thr*3+Math.random()*6;for(const sx of[-11,11]){ctx.beginPath();ctx.moveTo(sx-2.5,12);ctx.lineTo(sx+2.5,12);ctx.lineTo(sx,12+f2);ctx.fill()}}
  ctx.globalCompositeOperation='source-over';
  if(P.up.thr>0){ctx.fillStyle='#9aa4cc';for(const sx of[-11,11])rr(ctx,sx-3,4,6,9,2),ctx.fill()}
  if(P.up.rate>0){ctx.fillStyle='#b7c0e6';for(const sx of[-16,16]){rr(ctx,sx-2,-4-P.up.rate*2,4,12+P.up.rate*2,1.5);ctx.fill()}}
  ctx.fillStyle='#dfe5ff';ctx.strokeStyle='#7d88b8';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(0,-21);ctx.lineTo(6,-6);ctx.lineTo(17,7);ctx.lineTo(17,13);ctx.lineTo(6,10);ctx.lineTo(4,15);ctx.lineTo(-4,15);ctx.lineTo(-6,10);ctx.lineTo(-17,13);ctx.lineTo(-17,7);ctx.lineTo(-6,-6);ctx.closePath();ctx.fill();ctx.stroke();
  ctx.fillStyle=acc;ctx.beginPath();ctx.moveTo(-6,-2);ctx.lineTo(-15,8);ctx.lineTo(-15,10);ctx.lineTo(-6,6);ctx.fill();ctx.beginPath();ctx.moveTo(6,-2);ctx.lineTo(15,8);ctx.lineTo(15,10);ctx.lineTo(6,6);ctx.fill();
  const cg=ctx.createLinearGradient(0,-14,0,2);cg.addColorStop(0,'#c9f6ff');cg.addColorStop(1,'#1b6f9a');ctx.fillStyle=cg;ctx.beginPath();ctx.ellipse(0,-6,3.6,7.5,0,0,TAU);ctx.fill();
  ctx.restore();
  if(P.up.mag>0){ctx.strokeStyle='rgba(255,214,90,0.18)';ctx.lineWidth=1;ctx.setLineDash([3,6]);ctx.beginPath();ctx.arc(P.x,P.y,P.r*1.9,G.t,G.t+TAU);ctx.stroke();ctx.setLineDash([])}
  if(P.shield>0){ctx.save();ctx.globalCompositeOperation='lighter';
    const sg=ctx.createRadialGradient(P.x,P.y,P.r*1.2,P.x,P.y,P.r*1.75);sg.addColorStop(0,'rgba(73,227,255,0)');sg.addColorStop(1,`rgba(73,227,255,${0.18+0.1*P.shield})`);
    ctx.fillStyle=sg;ctx.beginPath();ctx.arc(P.x,P.y,P.r*1.75,0,TAU);ctx.fill();
    ctx.strokeStyle='rgba(140,240,255,0.6)';ctx.lineWidth=1.5;
    for(let i=0;i<P.shield;i++){const a0=-Math.PI/2+G.t*0.8+i*TAU/P.shield;ctx.beginPath();ctx.arc(P.x,P.y,P.r*1.75,a0,a0+TAU/P.shield-0.3);ctx.stroke()}
    ctx.restore()}
  if(P.over){ctx.fillStyle='rgba(255,77,109,0.3)';ctx.beginPath();ctx.arc(P.x,P.y,P.r*1.6,0,TAU);ctx.fill()}
}
function drawPickup(p){
  ctx.save();ctx.translate(p.x,p.y);
  if(p.kind==='dust'){ctx.rotate(p.t*2);const s=6*S;ctx.globalCompositeOperation='lighter';
    ctx.fillStyle='rgba(255,214,90,0.25)';ctx.beginPath();ctx.arc(0,0,s*1.8,0,TAU);ctx.fill();
    ctx.fillStyle='#ffd65a';ctx.beginPath();for(let i=0;i<8;i++){const a=i/8*TAU,r=i%2?s*0.45:s;ctx.lineTo(Math.cos(a)*r,Math.sin(a)*r)}ctx.fill()}
  else{const c=p.wt==='ion'?'#ff7a3d':'#49e3ff',s=11*S;ctx.rotate(Math.sin(p.t*3)*0.3);
    ctx.globalCompositeOperation='lighter';ctx.fillStyle=p.wt==='ion'?'rgba(255,122,61,0.3)':'rgba(73,227,255,0.3)';ctx.beginPath();ctx.arc(0,0,s*1.7,0,TAU);ctx.fill();
    ctx.globalCompositeOperation='source-over';ctx.fillStyle=c;rr(ctx,-s,-s*0.7,s*2,s*1.4,s*0.6);ctx.fill();
    ctx.fillStyle='#fff';rr(ctx,-s*0.15,-s*0.7,s*0.3,s*1.4,1);ctx.fill();ctx.fillStyle='rgba(255,255,255,0.5)';rr(ctx,-s*0.8,-s*0.5,s*0.5,s*0.25,1);ctx.fill()}
  ctx.restore();
}
function drawEbul(b){
  ctx.save();ctx.translate(b.x,b.y);
  if(b.k==='rock'){
    const sp=Math.hypot(b.vx,b.vy)||1,nx=-b.vx/sp,ny=-b.vy/sp,len=b.r*4.5;
    ctx.globalCompositeOperation='lighter';
    const g=ctx.createLinearGradient(0,0,nx*len*(b.fire?1.5:1),ny*len*(b.fire?1.5:1));
    if(b.fire){g.addColorStop(0,'rgba(255,240,140,0.95)');g.addColorStop(0.35,'rgba(255,120,30,0.8)');g.addColorStop(1,'rgba(255,40,20,0)')}
    else{g.addColorStop(0,'rgba(255,150,60,0.75)');g.addColorStop(1,'rgba(255,60,40,0)')}
    ctx.fillStyle=g;ctx.beginPath();ctx.moveTo(-ny*b.r*0.9,nx*b.r*0.9);ctx.lineTo(nx*len,ny*len);ctx.lineTo(ny*b.r*0.9,-nx*b.r*0.9);ctx.fill();
    const gl=ctx.createRadialGradient(0,0,b.r*0.4,0,0,b.r*1.9);gl.addColorStop(0,'rgba(255,120,50,0.45)');gl.addColorStop(1,'rgba(255,120,50,0)');
    ctx.fillStyle=gl;ctx.beginPath();ctx.arc(0,0,b.r*1.9,0,TAU);ctx.fill();
    ctx.globalCompositeOperation='source-over';ctx.rotate(b.rot);
    ctx.fillStyle='#6a5a55';ctx.strokeStyle='#ffb067';ctx.lineWidth=1.3*S;ctx.beginPath();
    b.verts.forEach((v,i)=>{const a=i/b.verts.length*TAU,px=Math.cos(a)*b.r*v,py=Math.sin(a)*b.r*v;i?ctx.lineTo(px,py):ctx.moveTo(px,py)});
    ctx.closePath();ctx.fill();ctx.stroke();
    ctx.fillStyle='rgba(30,20,20,0.55)';ctx.beginPath();ctx.arc(b.r*0.25,-b.r*0.15,b.r*0.25,0,TAU);ctx.fill();
    if(b.fire){ctx.globalCompositeOperation='lighter';ctx.fillStyle=`rgba(255,${120+Math.random()*80|0},40,0.55)`;ctx.beginPath();ctx.arc(0,0,b.r*(0.9+Math.random()*0.3),0,TAU);ctx.fill();
      if(Math.random()<0.4)burst(b.x,b.y,'#ff9a3a',1,40,2.5)}}
  else if(b.k==='slime'){
    const w=1+Math.sin(b.t*10)*0.12;ctx.globalCompositeOperation='lighter';
    ctx.fillStyle='rgba(125,255,90,0.25)';ctx.beginPath();ctx.arc(0,0,b.r*1.7,0,TAU);ctx.fill();
    ctx.globalCompositeOperation='source-over';ctx.rotate(Math.atan2(b.vy,b.vx)-Math.PI/2);
    ctx.fillStyle='#5fe03c';ctx.beginPath();ctx.ellipse(0,0,b.r*w,b.r*(2-w)*1.1,0,0,TAU);ctx.fill();
    ctx.fillStyle='#2f9a2a';ctx.beginPath();ctx.ellipse(0,-b.r*0.9,b.r*0.35,b.r*0.6,0,0,TAU);ctx.fill();
    ctx.fillStyle='rgba(230,255,210,0.8)';ctx.beginPath();ctx.arc(-b.r*0.35,b.r*0.3,b.r*0.25,0,TAU);ctx.fill()}
  else{ctx.globalCompositeOperation='lighter';const g=ctx.createRadialGradient(0,0,0,0,0,b.r*2);g.addColorStop(0,'#ffffff');g.addColorStop(0.35,'#ff4d9d');g.addColorStop(1,'rgba(255,77,157,0)');
    ctx.fillStyle=g;ctx.beginPath();ctx.arc(0,0,b.r*2,0,TAU);ctx.fill()}
  ctx.restore();
}
function render(){
  ctx.save();
  if(G.shake){ctx.translate(rnd(-G.shake,G.shake),rnd(-G.shake,G.shake))}
  ctx.drawImage(bg,0,0,W,H);
  for(const s of stars){ctx.fillStyle=`rgba(220,230,255,${0.25+s.z*0.7})`;const r=(0.5+s.z*1.3)*S;ctx.fillRect(s.x,s.y,r,r*(1+s.z*2))}
  for(const p of pickups)drawPickup(p);
  for(const e of enemies){if(e.delay>0)continue;if(e.type==='comet')drawComet(e);else if(e.type==='astro')drawAstro(e,false);else if(e.kind==='hole')drawHole(e);else drawAlien(e)}
  ctx.save();ctx.globalCompositeOperation='lighter';
  for(const b of bullets){if(b.c==='ion'){ctx.fillStyle='rgba(255,122,61,0.35)';ctx.beginPath();ctx.arc(b.x,b.y,b.r*2,0,TAU);ctx.fill();ctx.fillStyle='#ffd2a8';ctx.beginPath();ctx.ellipse(b.x,b.y,b.r*0.7,b.r*1.5,Math.atan2(b.vx,-b.vy),0,TAU);ctx.fill()}
    else{ctx.fillStyle='rgba(73,227,255,0.4)';ctx.fillRect(b.x-b.r,b.y-b.r*4,b.r*2,b.r*8);ctx.fillStyle='#e6fdff';ctx.fillRect(b.x-b.r*0.4,b.y-b.r*3.5,b.r*0.8,b.r*7)}}
  ctx.restore();
  for(const b of ebul)drawEbul(b);
  drawShip();
  ctx.save();ctx.globalCompositeOperation='lighter';
  for(const p of parts){ctx.globalAlpha=Math.max(0,p.life/p.max);ctx.fillStyle=p.c;ctx.fillRect(p.x-p.s/2,p.y-p.s/2,p.s,p.s)}
  ctx.restore();
  ctx.textAlign='center';
  for(const t of texts){ctx.globalAlpha=Math.max(0,Math.min(1,t.life*1.5));ctx.fillStyle=t.c;ctx.font=`700 ${Math.round(13*S+2)}px "Chakra Petch",sans-serif`;ctx.fillText(t.t,t.x,t.y)}
  ctx.globalAlpha=1;
  // boss bar
  const boss=enemies.find(e=>e.type==='boss');
  if(boss&&G.state==='play'){const bw=Math.min(W-32,420),bx=(W-bw)/2,by=G.topH+6;
    ctx.fillStyle='rgba(255,255,255,0.1)';ctx.fillRect(bx,by,bw,6);ctx.fillStyle='#ff4d6d';ctx.fillRect(bx,by,bw*Math.max(0,boss.hp/boss.mhp),6);
    ctx.fillStyle='#e6eaff';ctx.font='600 11px "Chakra Petch",sans-serif';ctx.fillText(boss.name.toUpperCase(),W/2,by+20)}
  // banner
  if(G.bannerT>0&&G.state==='play'){const a=Math.min(1,G.bannerT,(2.6-G.bannerT)*3);ctx.globalAlpha=a;
    ctx.fillStyle='#ffd65a';ctx.font=`${Math.round(34*S+6)}px Bungee,Impact,sans-serif`;ctx.fillText(G.banner,W/2,H*0.42);
    ctx.fillStyle='#e6eaff';ctx.font=`600 ${Math.round(15*S+3)}px "Chakra Petch",sans-serif`;ctx.fillText(G.sub.toUpperCase(),W/2,H*0.42+30*S);ctx.globalAlpha=1}
  ctx.restore();
  if(G.flash>0){ctx.fillStyle=`rgba(255,240,200,${Math.min(0.6,G.flash)})`;ctx.fillRect(0,0,W,H)}
}

/* ---------- HUD ---------- */
const shipSvg='<svg viewBox="0 0 34 36"><path d="M17 0 23 15 34 28 34 34 23 31 21 36 13 36 11 31 0 34 0 28 11 15Z" fill="#dfe5ff"/></svg>';
const hud={};
function updHUD(force){
  const set=(k,v,fn)=>{if(force||hud[k]!==v){hud[k]=v;fn(v)}};
  set('score',G.score,v=>$('score').textContent=v.toLocaleString('en-US'));
  set('lives',G.lives,v=>$('lives').innerHTML=shipSvg.repeat(Math.max(0,v)));
  set('dust',G.dust,v=>$('dustVal').textContent='★ '+v);
  set('ready',anyAffordable(),v=>$('hangarBtn').classList.toggle('ready',v));
  set('novas',G.novas,v=>{$('novaBtn').textContent='NOVA ×'+v;$('novaBtn').disabled=v<=0});
  if(P){const h=Math.round(P.heat*100);set('heat',h,v=>$('heatFill').style.width=v+'%');
    const nm=W<480?(P.wt==='ion'?'Ion':'Plasma'):(P.wt==='ion'?'Ion blaster':'Plasma lance');const l=(P.over?'Overheated':nm)+' · Lv '+P.lv;
    set('hl',l,v=>{$('heatLbl').textContent=v;$('heatLbl').classList.toggle('over',P.over)})}
}

/* ---------- flow ---------- */
function show(id,on){$(id).hidden=!on}
/* checkpoint: saved at the start of every wave, so a lost game can resume there */
function loadCheck(){try{return JSON.parse(localStorage.getItem('swi-check')||'null')}catch(e){return null}}
function saveCheck(c){try{c?localStorage.setItem('swi-check',JSON.stringify(c)):localStorage.removeItem('swi-check')}catch(e){}}
G.check=loadCheck();
function makeCheck(){G.check={wave:G.wave,score:G.score,dust:G.dust,novas:G.novas,nextLife:G.nextLife,wt:P.wt,lv:P.lv,up:Object.assign({},P.up)};saveCheck(G.check)}
function start(fromCheck){
  ensureAudio();
  const c=fromCheck===true?G.check:null;
  if(!c){G.check=null;saveCheck(null)}
  Object.assign(G,{state:'play',score:0,lives:3,wave:0,dust:0,novas:1,t:0,shake:0,flash:0,between:true,nextT:0.4,spawnQ:[],waveT:0,nextLife:20000,bannerT:0});
  bullets=[];enemies=[];ebul=[];pickups=[];parts=[];texts=[];P=newPlayer();
  if(c){Object.assign(G,{wave:c.wave-1,score:c.score,dust:c.dust,novas:Math.max(1,c.novas),nextLife:c.nextLife});
    P.wt=c.wt;P.lv=c.lv;P.up=Object.assign(P.up,c.up);P.shield=Math.max(0,P.up.armor-1)}
  show('titleScr',false);show('overScr',false);show('pauseScr',false);show('shopScr',false);show('top',true);show('bottom',true);updHUD(true);measureHUD();
  cv.focus?.();
}
function gameOver(){
  $('overTitle').textContent='Ship lost';
  G.state='over';G.overT=0;show('top',false);show('bottom',false);
  const nb=G.score>G.best;if(nb){G.best=G.score;saveBest(G.best)}
  $('finalScore').textContent=G.score.toLocaleString('en-US');
  $('overMsg').textContent=(nb&&G.score>0?'New best! ':'Best: '+G.best.toLocaleString('en-US')+'. ')+'You lost on wave '+G.wave+'. Continue from there with 3 fresh ships, keeping your upgrades and the score and stardust you had when the wave began.';
  bullets=[];ebul=[];pickups=[];enemies=enemies.filter(e=>e.type==='comet');
  setOverButtons(!!G.check);
  show('overScr',true);$(G.check?'contBtn':'againBtn').focus();
}
function togglePause(){
  if(G.state==='play'){G.state='pause';show('pauseScr',true)}
  else if(G.state==='pause'){G.state='play';show('pauseScr',false)}
}
function setOverButtons(canCont){
  $('contBtn').hidden=!canCont;
  if(canCont)$('contBtn').textContent='Continue from wave '+G.check.wave;
  $('againBtn').textContent=canCont?'Start over':'Fly again';
  $('againBtn').className=canCont?'':'cta';
}
$('startBtn').onclick=()=>start(false);$('againBtn').onclick=()=>start(false);
$('contBtn').onclick=()=>start(true);$('contTitleBtn').onclick=()=>start(true);
$('pauseBtn').onclick=togglePause;$('resumeBtn').onclick=togglePause;
$('quitBtn').onclick=()=>{G.state='title';show('pauseScr',false);show('top',false);show('bottom',false);enemies=[];bullets=[];ebul=[];pickups=[];P=null;refreshTitle();show('titleScr',true)};
$('muteBtn1').onclick=toggleMute;$('muteBtn2').onclick=toggleMute;
$('novaBtn').onclick=()=>{ensureAudio();nova()};
$('hangarBtn').onclick=()=>{ensureAudio();openShop()};
$('shopClose').onclick=closeShop;
function refreshTitle(){$('bestTitle').textContent=G.best?'Best score on this device: '+G.best.toLocaleString('en-US'):'';
  $('contTitleBtn').hidden=!G.check;if(G.check)$('contTitleBtn').textContent='Continue wave '+G.check.wave;
  $('startBtn').textContent=G.check?'New game':'Launch'}
document.addEventListener('visibilitychange',()=>{if(document.hidden&&G.state==='play')togglePause()});
addEventListener('blur',()=>{for(const k in keys)keys[k]=false;ptr.active=false});

addEventListener('resize',resize);
resize();refreshTitle();
let last=performance.now();
function frame(now){const dt=Math.min(0.033,(now-last)/1000);last=now;update(dt);render();requestAnimationFrame(frame)}
requestAnimationFrame(frame);
})();
