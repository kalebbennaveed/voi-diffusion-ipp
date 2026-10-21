"""Editable scientific architecture schematic (SVG); data panels are schematic."""
from pathlib import Path
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/presentation/wind_aware_mbd/assets'
N='#00274c';G='#008653';B='#2474bf';P='#9b4682';O='#d77717';M='#ffcb05';GRAY='#667889'
s=[]
def add(x):s.append(x)
def rect(x,y,w,h,fill='white',stroke='#bdcad5',rx=0,dash=''):
 add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
def line(x1,y1,x2,y2,c=N,w=3,dash=''):
 add(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{c}" stroke-width="{w}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
def arrow(points,c=N,w=3):
 add(f'<polyline points="{points}" fill="none" stroke="{c}" stroke-width="{w}" marker-end="url(#{c[1:]})"/>')
def text(x,y,t,sz=25,c=N,b=False,anchor='start'):
 sub={'ₜ':'t','ₖ':'k','ᵢ':'i','₁':'1','₂':'2','₋':'−'}
 rendered=''.join(f'<tspan baseline-shift="sub" font-size="70%">{sub[ch]}</tspan>' if ch in sub else escape(ch) for ch in t)
 add(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{sz}" fill="{c}" font-weight="{700 if b else 400}" text-anchor="{anchor}">{rendered}</text>')
def path(d,c,w=3,dash='',opacity=1):
 add(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}" opacity="{opacity}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
def circle(x,y,r,c,opacity=1):add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" opacity="{opacity}"/>')
def world(x,y,w,h,mode='single',direction=1):
 rect(x,y,w,h,'#fafbfd','#c3ced7',5)
 # Shared geometry, two monitored regions and two obstacles.
 for a,b,label in [(.77,.25,'A'),(.23,.75,'B')]:
  rect(x+(a-.14)*w,y+(b-.17)*h,.28*w,.34*h,'#e8f0f4','#879daf');text(x+a*w,y+b*h+5,label,20,N,True,'middle')
 for a,b in [(.43,.33),(.57,.67)]:circle(x+a*w,y+b*h,.12*w,'#6c7d88')
 for t in [.18,.38,.58,.78]:
  xx=x+(1-t)*w;yy=y+t*h
  arrow(f'{xx},{yy} {xx-direction*.055*w},{yy+direction*.035*h}',GRAY,1.6)
 if mode=='belief':
  for i in range(7):
   for j in range(5):
    opacity=.10+.35*(1-abs(i-3)/4)*(1-abs(j-2)/3)
    rect(x+6+i*(w-12)/7,y+6+j*(h-12)/5,(w-12)/7,(h-12)/5,f'rgba(36,116,191,{opacity})','none')
  for t in [.1,.3,.5,.7,.9]:circle(x+t*w,y+.86*h,3,O)
 elif mode=='proposals':
  for k in range(7):
   path(f'M{x+.8*w},{y+.22*h} C{x+(.85-.08*k)*w},{y+(.7+.015*k)*h} {x+(.08+.045*k)*w},{y+(.1+.025*k)*h} {x+.2*w},{y+.8*h}',[G,B,P,O,'#00818b','#7e6fc4','#b58916'][k],2.5,opacity=.8)
  circle(x+.8*w,y+.22*h,6,N)
 else:
  path(f'M{x+.78*w},{y+.23*h} L{x+.22*w},{y+.77*h}',O,2.8,'7 5')
  def curve(t):
   a=(1-t)**3;b=3*(1-t)**2*t;c=3*(1-t)*t*t;d=t**3
   return x+w*(a*.78+b*.99+c*.35+d*.22),y+h*(a*.23+b*.95+c*1.15+d*.77)
  end=1.0 if mode=='future' else .30
  points=[curve(i*end/35) for i in range(36)]
  d='M'+' L'.join(f'{a},{b}' for a,b in points)
  path(d,G,4)
  px,py=curve(.4 if mode=='future' else .30)
  circle(px,py,.085*w,'#e83b3b',.18);circle(px,py,5,G)
  if mode=='future':
   for t in [.08,.15,.22,.3]:
    a,b=curve(t);circle(a,b,3.3,P)

add('<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="1050" viewBox="0 0 2000 1050"><title>Wind-aware MBD with anticipated learning: full system architecture</title><desc>MBD samples nominal policy parameters. Candidate simulations pass through RPCBF, generate observations and update POAM causally. Executed monitoring costs weight policy parameters. The selected policy executes for two seconds, earns real measurements and replans. All miniature paths are schematic.</desc><defs>')
for c in [N,G,B,P,O,GRAY]:add(f'<marker id="{c[1:]}" markerWidth="9" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 Z" fill="{c}"/></marker>')
add('</defs><rect width="2000" height="1050" fill="white"/>')
text(40,43,'Wind-aware MBD with anticipated learning',38,N,True)
text(40,82,'Nominal policies are weighted by the monitoring cost of their RPCBF-filtered execution',25,GRAY)
# planning region
rect(25,112,1950,625,'#fbfcfe','#a9bdcd',15)
text(52,152,'SIMULATED PLANNING  ·  10 s forecast',27,N,True)
text(60,208,'Available information',26,N,True)
world(60,233,250,195,'belief')
text(185,465,'Robot xₜ  ·  clarity qₜ',23,N,False,'middle')
text(185,498,'POAM belief bₜ = (μₜ, Σₜ)',23,B,True,'middle')
text(185,545,'Shared starting state',22,GRAY,False,'middle')
arrow('320,330 386,330')
text(405,208,'MBD proposals',27,N,True)
world(405,233,310,195,'proposals')
text(560,465,'K policies θₖ from noisy yᵢ',23,N,False,'middle')
text(560,499,'18 nominal-policy parameters',22,GRAY,False,'middle')
text(560,545,'Gaussian proposal + support check',21,GRAY,False,'middle')
arrow('725,330 790,330')
# causal rollouts highlighted as a single scientific panel
rect(810,182,720,440,'#edf4fb','#5e87b4',10)
text(1170,219,'Causal rollout of each candidate',28,B,True,'middle')
text(1000,258,'0–2 s: current belief',23,N,True,'middle')
text(1350,258,'2–10 s: earned updates',23,N,True,'middle')
world(840,277,290,180,'single');world(1200,277,290,180,'future')
arrow('1138,369 1190,369',P,3)
text(1165,491,'Observe → POAM → rebuild RPCBF',23,P,True,'middle')
text(1170,534,'Nominal control → RPCBF → motion → sensing',24,B,True,'middle')
text(1170,574,'Coherent wind worlds; no access to future observations',21,GRAY,False,'middle')
# cost and weights
arrow('1540,330 1600,330')
text(1630,208,'Executed task cost',27,N,True)
for i,(val,c) in enumerate([(180,G),(125,B),(68,O)]):
 y=266+i*52;rect(1650,y,val,29,c,'none');text(1617,y+24,str(i+1),22,N,False,'middle')
text(1775,456,'Ĵ(θₖ) = mean world cost',24,N,True,'middle')
text(1775,502,'wₖ ∝ exp(−Ĵ(θₖ) / λ)',25,P,True,'middle')
text(1775,547,'Weight nominal θₖ',24,P,True,'middle')
text(1775,580,'then update yᵢ₋₁',24,P,True,'middle')
arrow('1775,600 1775,682 560,682 560,570',P,4)
rect(827,654,690,48,'#fbfcfe','none')
text(1172,686,'Reverse refinement: score estimate → next noisy iterate',23,P,True,'middle')
# legend
line(65,593,115,593,O,3,'7 5');text(125,602,'Nominal path',21)
line(65,630,115,630,G,4);text(125,639,'Filtered motion',21)
circle(89,672,12,'#e83b3b',.3);text(125,680,'Sensing footprint',21)
text(40,778,'REAL EXECUTION  ·  commit only the first 2 s, then replan',27,N,True)
# actual loop simple object+scene
text(66,850,'Selected policy θ*',27,G,True)
text(66,890,'After MBD refinement',22,GRAY)
arrow('320,865 380,865',G,4)
rect(403,817,270,100,'#edf5f0',G,8);text(538,855,'RPCBF + dynamics',25,G,True,'middle');text(538,893,'20 Hz actual execution',21,N,False,'middle')
arrow('685,865 744,865',G,4)
world(768,797,255,155,'single')
text(895,983,'Actual unknown wind',22,GRAY,False,'middle')
arrow('1035,865 1110,865',G,4)
text(1140,841,'Executed sensing',26,N,True);text(1140,881,'Wind observations',23,N);text(1140,919,'Updated clarity + state',23,N)
arrow('1445,865 1510,865',B,4)
rect(1540,817,350,100,'#edf4fb',B,8);text(1715,855,'Online POAM update',26,B,True,'middle');text(1715,894,'New belief bₜ₊₂',25,N,False,'middle')
arrow('1905,865 1950,865 1950,1023 35,1023 35,453 53,453',B,3)
rect(1050,1002,320,36,'white','none');text(1210,1029,'Next planning cycle',22,B,True,'middle')
add('</svg>');(OUT/'architecture.svg').write_text('\n'.join(s))
print(OUT/'architecture.svg')
