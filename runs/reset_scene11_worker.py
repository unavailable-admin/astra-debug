"""Explicit cold reset of idle 8081; restores full36 before returning."""
import asyncio,json,websockets
from pathlib import Path
URI='ws://10.19.4.253:8081'
async def recv(ws,kind):
 while True:
  r=json.loads(await asyncio.wait_for(ws.recv(),15))
  if r.get('type')==kind:return r
async def change(layout):
 async with websockets.connect(URI,proxy=None,ping_interval=None,open_timeout=8,close_timeout=2) as ws:
  await ws.send(json.dumps({'type':'status'}));r=await recv(ws,'status_response')
  if r.get('step_result_subscribed') or r.get('is_executing') or r.get('queue_length'):raise RuntimeError('Worker occupied; stopping')
  if r.get('scene_id') not in ('showroom_scene_11','showroom_scene_11_stereo'):raise RuntimeError('Unexpected scene')
  await ws.send(json.dumps({'type':'switch_action_layout','action_layout':layout}))
  ack=await recv(ws,'switch_action_layout_response');print('SWITCH',ack,flush=True)
  if not ack.get('ok'):raise RuntimeError(str(ack))
 for i in range(60):
  await asyncio.sleep(5)
  try:
   async with websockets.connect(URI,proxy=None,ping_interval=None,open_timeout=3,close_timeout=1) as ws:
    await ws.send(json.dumps({'type':'status'}));r=await recv(ws,'status_response')
    if r.get('action_layout')==layout:
     print('READY',layout,'step',r.get('step'),flush=True);return r
  except (OSError,TimeoutError,websockets.exceptions.WebSocketException):pass
  if i%6==0:print('WAIT',layout,flush=True)
 raise TimeoutError('Worker restart timed out')
async def main():
 await change('arm26')
 r=await change('full36')
 print('COLD_RESET_COMPLETE',flush=True)
 Path(__file__).with_name('trajectory_reset_status.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':asyncio.run(main())
