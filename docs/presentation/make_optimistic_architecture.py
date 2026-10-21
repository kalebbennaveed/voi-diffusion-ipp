"""Editable scientific workflow diagram for the method-review deck."""
from pathlib import Path
from html import escape
D=Path(__file__).resolve().parents[2]/'output/presentation/wind_aware_mbd/assets'
a=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1530 690"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#00274c"/></marker></defs>']
def text(x,y,s,size=23,color='#00274c',bold=False):a.append(f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}">{escape(s)}</text>')
def box(id,x,y,w,h,title,lines,color='#2474bf'):
 a.append(f'<g id="{id}" data-panel="true" data-bounds="{x},{y},{w},{h}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="#f5f8fb" stroke="{color}" stroke-width="2"/>')
 text(x+18,y+35,title,26,color,True)
 for i,s in enumerate(lines):text(x+18,y+78+i*35,s)
 a.append('</g>')
def arrow(src,dst,pts):a.append(f'<polyline data-connector="{src}:{dst}" points="{pts}" stroke="#00274c" stroke-width="2.5" fill="none" marker-end="url(#a)"/>')
text(25,28,'PLANNING  ·  current disturbance posterior stays fixed',25,'#2474bf',True)
text(1130,28,'REAL EXECUTION',25,'#008653',True)
box('input',25,65,275,205,'1  Current inputs',['Measured robot state','Current clarity','Disturbance posterior','Geometry and targets'])
box('search',360,65,315,205,'2  MBD policy search',['Draw nominal parameters','Predict filtered execution','Score optimistic monitoring','Correct weights and refine'])
box('select',735,65,335,205,'3  Final reevaluation',['After all diffusion levels','Use fresh shared fields','Check all tested executions','Select by optimistic cost'])
box('execute',1130,65,365,205,'4  Execute one interval',['Actual tracking and filter','Actual disturbance','Actual motion and sensing','Collect wind observations'],'#008653')
arrow('input','search','300,165 360,165');arrow('search','select','675,165 735,165');arrow('select','execute','1070,165 1130,165')
# Explicit feedback within numerical search, separate from physical learning loop.
arrow('search','search','445,270 445,320 600,320 600,270')
text(25,330,'Within MBD:',23,'#2474bf',True)
text(25,365,'repeat only while',22);text(25,397,'diffusion levels remain',22)
box('rollout',360,390,710,175,'Inside every policy evaluation',['Performance fields: disturbed motion and clarity costs','Construction fields: state-dependent design-policy maxima','The filter is rebuilt at each predicted state'])
arrow('search','rollout','675,210 705,210 705,390')
box('belief',1130,390,365,175,'5  POAM update',['Assimilate real measurements','Retain attained state and clarity','Start the next planning call'],'#008653')
arrow('execute','belief','1310,270 1310,390')
arrow('belief','input','1310,565 1310,620 8,620 8,165 25,165')
text(390,657,'Real execution advances time and updates the posterior',25,'#008653',True)
a.append('</svg>');(D/'optimistic_architecture.svg').write_text('\n'.join(a))
