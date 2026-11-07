"""Design screen: successful, substantially slowed returns and bypass impacts."""
import argparse,json
from pathlib import Path
import robot_review as r

def screen(site,scenarios=(1,2)):
    rows=[]
    for scenario in scenarios:
        field=r.load_wind(scenario,site)
        for route,direction,filtered in [('central','B-A',True),('central','B-A',False),
                ('lower','B-A',False),('upper','B-A',False),('upper','A-B',False)]:
            result=r.simulate(r.Reference(route,direction,site),field,filtered)
            row=dict(scenario=scenario,route=route,direction=direction,filtered=filtered,**result['summary'])
            rows.append(row);print(json.dumps(row),flush=True)
            out=site/'screen';out.mkdir(exist_ok=True)
            (out/f's{scenario}-{route}-{direction}-{int(filtered)}.json').write_text(json.dumps(result))
    (site/'strength-screen.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--scenarios',type=int,nargs='+',default=[1,2])
    args=p.parse_args();screen(args.site,args.scenarios)
