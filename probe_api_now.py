"""Independent text/image API checks; never connects to the simulator."""
import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from PIL import Image
from openai_connection import OfficialOpenAI

ROOT=Path(__file__).resolve().parent

def request(label,messages,out,timeout):
    client=OfficialOpenAI()
    key=client.key_file.read_text().strip()
    endpoint=client.base_url+'/chat/completions'
    body={'model':client.model,'reasoning_effort':'low','messages':messages}
    req=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    result={'test':label,'endpoint':endpoint,'model':body['model'],'ok':False,'timeout_seconds':timeout}
    start=time.monotonic()
    try:
        with client.opener.open(req,timeout=timeout) as response:
            result.update(http_status=response.status,time_to_headers_seconds=round(time.monotonic()-start,3))
            raw=response.read().decode()
            (out/f'{label}.response.json').write_text(raw.replace(key,'<redacted>'))
            parsed=json.loads(raw)
            result['content']=parsed['choices'][0]['message']['content']
            result['usage']=parsed.get('usage')
            result['ok']=True
    except urllib.error.HTTPError as exc:
        result.update(http_status=exc.code,error_type=type(exc).__name__,error_body=exc.read(8192).decode(errors='replace').replace(key,'<redacted>'))
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc).replace(key,'<redacted>'))
    result['elapsed_seconds']=round(time.monotonic()-start,3)
    (out/f'{label}.result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

async def main(args):
    out=ROOT/'runs'/time.strftime('api_only_%Y%m%d_%H%M%S');out.mkdir()
    print('RUN_DIR',out,flush=True)
    im=Image.open(ROOT/'runs/hi_grasp_final.jpg').convert('RGB');im.thumbnail((960,540))
    data=io.BytesIO();im.save(data,format='JPEG',quality=80)
    text=[{'role':'user','content':'Calculate 17 + 26. Reply with only the number.'}]
    vision=[{'role':'user','content':[{'type':'text','text':'Read the two separate letter cubes on the front part of the table, closest to the robot. Reply with their letters from left to right only.'},
            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data.getvalue()).decode()}}]}]
    (out/'tests.json').write_text(json.dumps({'text_prompt':text[0]['content'],'vision_prompt':vision[0]['content'][0]['text'],
        'image_source':'runs/hi_grasp_final.jpg','sent_image_size':im.size,'simulator_connected':False},indent=2))
    results=await asyncio.gather(*(asyncio.to_thread(request,label,messages,out,args.timeout) for label,messages in [('text',text),('vision',vision)]))
    (out/'report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--timeout',type=float,default=180)
    args=p.parse_args()
    if args.timeout<=0:p.error('Timeout must be positive')
    asyncio.run(main(args))
