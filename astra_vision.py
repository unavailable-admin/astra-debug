"""Audited official-OpenAI image decisions for the simulated grasp controller."""
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import time
import urllib.request
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
from openai_connection import OfficialOpenAI

ROOT=Path(__file__).resolve().parent

def parse_json(content):
    text=content.strip()
    if text.startswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    value=json.loads(text)
    if not isinstance(value,dict):raise ValueError('Expected a JSON object')
    return value

class AstraVision:
    def __init__(self,directory,timeout=120):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.timeout=timeout
        self.client=OfficialOpenAI()

    def _call(self,stage,prompt,images):
        stem=self.directory/f'{time.time_ns()}_{stage}'
        content=[{'type':'text','text':prompt}]
        for path in images:
            im=Image.open(path).convert('RGB');im.thumbnail((960,540))
            data=io.BytesIO();im.save(data,format='JPEG',quality=80)
            content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data.getvalue()).decode()}})
        body={'model':self.client.model,'reasoning_effort':'low','max_completion_tokens':2048,
              'messages':[{'role':'system','content':'Inspect robot simulation images carefully. Return only the requested JSON object. Report uncertainty explicitly. A commanded robot motion does not prove an object moved. Never infer success from the task instruction.'},
                          {'role':'user','content':content}]}
        stem.with_suffix('.request.json').write_text(json.dumps({**self.client.metadata(),'stage':stage,'prompt':prompt,'images':list(map(str,images))},indent=2))
        start=time.monotonic()
        try:
            raw=self.client.chat(body,timeout=self.timeout)
            stem.with_suffix('.response.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2))
            if raw['choices'][0].get('finish_reason')!='stop':raise ValueError('Incomplete model response')
            result=parse_json(raw['choices'][0]['message']['content'])
            stem.with_suffix('.decision.json').write_text(json.dumps({'elapsed_seconds':time.monotonic()-start,'decision':result},ensure_ascii=False,indent=2))
            return result
        except Exception as exc:
            stem.with_suffix('.error.json').write_text(json.dumps({'error_type':type(exc).__name__,'message':str(exc),'elapsed_seconds':time.monotonic()-start},indent=2))
            raise

    async def scene(self,image):
        return await asyncio.to_thread(self._call,'scene',
            'Find the H and I letter cubes in this image. Return JSON: '
            '{"letters":[{"letter":"H","u":0.0,"v":0.0,"upright":true,"confidence":0.0}],'
            '"row_complete":false,"next_action":{"action":"pick_place|finish|stop","letter":"H|I|null"},'
            '"evidence":"brief visible evidence"}. '
            'u,v are the glyph center normalized by image width/height, each 0..1. '
            'Return only visible H/I; upright means resting with the readable letter on top, not on its side. '
            'Confidence is 0..1. row_complete means both cubes are separate from the hands, side by side H then I '
            'on the empty front part of the table, with both letters readable. Modest in-plane rotation is allowed. '
            'Do not count a letter held by a hand as placed. '
            'Choose pick_place for a visible upright H or I that still needs to move from the letter grid '
            'to the front row. Choose finish if the readable HI row is complete, otherwise stop when uncertain. '
            'The script supplies fixed front-row placement targets and validates measured coordinates.',[image])

    async def lift(self,letter,before,after,focus_pixel=None):
        images=[before,after]
        crop_note=''
        if focus_pixel is not None:
            u,v=focus_pixel
            for label,path in [('before',before),('after',after)]:
                im=Image.open(path).convert('RGB')
                crop=im.crop((max(0,int(u)-120),max(0,int(v)-120),min(im.width,int(u)+120),min(im.height,int(v)+120)))
                dest=self.directory/f'{time.time_ns()}_{label}_lift_crop.jpg';crop.save(dest);images.append(dest)
            crop_note=' Images 3 and 4 are close-up crops of the same before/after frames, in that order. '
        return await asyncio.to_thread(self._call,'lift',
            f'Two chronological frames: first after closing on {letter} but before lifting, second after the wrist lift. '
            +crop_note+
            'Determine whether the target cube actually rose with the hand and remains pinched between thumb and index. '
            'Return {"state":"held|empty|uncertain","confidence":0.0,"evidence":"specific visual change"}. '
            'If occluded or the images do not establish the cube moved with the hand, use uncertain. '
            'A cube visible on the table outside the pinch means empty.',images)

    async def scene_candidates(self,image,word='ACE'):
        """Ask for measured component IDs instead of ambiguous normalized axes."""
        from track_letters import mask
        original=Image.open(image).convert('RGB')
        binary=mask(cv2.imread(str(image)))
        count,_,stats,_=cv2.connectedComponentsWithStats(binary)
        components=[tuple(map(int,stats[i])) for i in range(1,count)
                    if stats[i,cv2.CC_STAT_AREA]>=20 and max(stats[i,2:4])<=80]
        components.sort(key=lambda item:(item[1]//25,item[0]))
        candidates={};marked=original.copy();draw=ImageDraw.Draw(marked)
        sheet=Image.new('RGB',(6*100,max(1,(len(components)+5)//6)*100),'white');sd=ImageDraw.Draw(sheet)
        for index,(x,y,w,h,area) in enumerate(components,1):
            candidates[str(index)]={'pixel':[x+w/2,y+h/2],'bbox':[x,y,w,h]}
            draw.rectangle((x-2,y-2,x+w+2,y+h+2),outline='red',width=1)
            draw.text((x,y-13),str(index),fill='red')
            crop=original.crop((max(0,x-7),max(0,y-7),min(original.width,x+w+7),min(original.height,y+h+7)))
            crop.thumbnail((74,74));gx=((index-1)%6)*100;gy=((index-1)//6)*100
            sheet.paste(crop,(gx+12,gy+20));sd.text((gx+8,gy+4),str(index),fill='black')
        stem=self.directory/f'{time.time_ns()}_candidates'
        marked_path=stem.with_suffix('.scene.jpg');sheet_path=stem.with_suffix('.glyphs.jpg')
        marked.save(marked_path);sheet.save(sheet_path)
        stem.with_suffix('.map.json').write_text(json.dumps(candidates,indent=2))
        result=await asyncio.to_thread(self._call,'scene_candidates',
            f'Task: arrange the letter cubes to spell {word} left to right on the empty front row of the table. '
            'Image 1 is the current scene with numbered cyan glyph components. Image 2 enlarges the SAME numbered '
            'components for identity recognition; its layout is NOT the physical table. '
            f'Identify visible letters from {list(word)} by candidate_id. Use the full scene to assess location and upright pose. '
            'Return JSON {"letters":[{"letter":"A","candidate_id":1,"upright":true,"confidence":0.99}],'
            '"row_complete":false,"next_action":{"action":"pick_place|finish|stop","letter":"A"},'
            '"evidence":"what is visible"}. Do not output u/v or invent world coordinates. '
            'upright means its glyph faces up. Omit missing/occluded letters. Choose pick_place for one visible upright '
            'target still in the original letter grid, finish only when all target letters are placed and separate from '
            'the hands in the front row in correct order, otherwise stop if uncertain. '
            'A robot skill will use the measured candidate center and fixed row targets.',[marked_path,sheet_path])
        result['coordinate_mode']='candidate_id';result['candidate_map']=candidates
        return result

def accept_lift(decision,gap):
    return (decision.get('state')=='held' and type(decision.get('confidence')) in (int,float)
            and .9<=decision['confidence']<=1 and .04<=gap<=.075
            and isinstance(decision.get('evidence'),str) and len(decision['evidence'])>=10)

def identified_pixels(decision,shape,word='HI'):
    h,w=shape[:2];result={}
    for item in decision.get('letters',[]):
        if not isinstance(item,dict):continue
        letter=item.get('letter')
        if not isinstance(letter,str) or letter not in word or item.get('upright') is not True:continue
        if decision.get('coordinate_mode')=='candidate_id':
            candidate=item.get('candidate_id');confidence=item.get('confidence')
            if type(candidate) is not int or type(confidence) not in (int,float) or not .9<=confidence<=1:continue
            measured=decision.get('candidate_map',{}).get(str(candidate))
            if measured:result[letter]=measured['pixel']
            continue
        values=[item.get(k) for k in ('u','v','confidence')]
        if any(type(x) not in (int,float) or not 0<=x<=1 for x in values):continue
        u,v,confidence=values
        if confidence>=.9:result[letter]=[u*w,v*h]
    return result
