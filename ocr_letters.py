"""Offline Tesseract single-glyph baseline; does not move the robot."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import cv2
from track_letters import mask

def recognize(image,candidates):
    binary=mask(cv2.imread(str(image)))
    results={}
    env={**os.environ,'OMP_THREAD_LIMIT':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    for identifier,item in candidates.items():
        x,y,w,h=item['bbox']
        glyph=255-binary[y:y+h,x:x+w]
        glyph=cv2.resize(glyph,None,fx=5,fy=5,interpolation=cv2.INTER_NEAREST)
        glyph=cv2.copyMakeBorder(glyph,25,25,25,25,cv2.BORDER_CONSTANT,value=255)
        _,encoded=cv2.imencode('.png',glyph)
        proc=subprocess.run(['tesseract','stdin','stdout','--psm','10','--oem','1','-l','eng',
            '-c','tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ','tsv'],input=encoded.tobytes(),
            capture_output=True,check=True,timeout=10,env=env)
        rows=list(csv.DictReader(io.StringIO(proc.stdout.decode()),delimiter='\t'))
        words=[r for r in rows if r.get('text','').strip()]
        results[identifier]={'text':''.join(r['text'] for r in words),
            'confidence':min((float(r['conf']) for r in words),default=-1),
            'pixel':item['pixel'],'bbox':item['bbox']}
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image',type=Path)
    p.add_argument('candidate_map',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=recognize(args.image,json.loads(args.candidate_map.read_text()))
    args.output.write_text(json.dumps(r,indent=2));print(json.dumps(r))
