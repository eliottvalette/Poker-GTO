/** Real static UI -> Supabase Storage -> ONNX inference and refresh recovery. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const cdp=process.env.POKER_CDP_ORIGIN ?? 'http://127.0.0.1:9335';
const origin=process.env.POKER_UI_ORIGIN ?? 'http://127.0.0.1:3118';
const storage=process.env.POKER_STORAGE_ORIGIN ?? 'https://vcyctuntcwrrcizvadwh.supabase.co/storage/v1/object/public/poker-policies';
const versions=await Promise.all(['hu','3max'].map(async track=>{
  const response=await fetch(`${storage}/${track}/current.json?smoke=${Date.now()}`);
  assert.ok(response.ok);return (await response.json()).iteration;
}));
const versionText=`Policies · HU ${versions[0]} · 3-max ${versions[1]}`;
const page=await (await fetch(`${cdp}/json/new?about:blank`,{method:'PUT'})).json();
const socket=new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
let sequence=0;const pending=new Map();
socket.onmessage=event=>{const msg=JSON.parse(event.data);const task=pending.get(msg.id);if(!task)return;pending.delete(msg.id);clearTimeout(task.timer);msg.error?task.reject(new Error(JSON.stringify(msg.error))):task.resolve(msg.result);};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;const timer=setTimeout(()=>{pending.delete(id);reject(new Error(`CDP timeout ${method}`));},45000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));});
async function evaluate(expression){const value=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(value.exceptionDetails)throw new Error(JSON.stringify(value.exceptionDetails));return value.result.value;}
async function until(expression){const deadline=Date.now()+45000;while(Date.now()<deadline){if(await evaluate(expression))return;await new Promise(resolve=>setTimeout(resolve,200));}throw new Error(`Timeout: ${expression}; ${await evaluate('document.body.innerText')}`);}
try {
  await call('Page.enable');
  await call('Page.addScriptToEvaluateOnNewDocument',{source:`
    window.__requests=[];window.__errors=[];window.__offline=false;
    const original=window.fetch;
    window.fetch=(...args)=>{const url=String(args[0]);window.__requests.push(url);if(window.__offline&&url.includes('/current.json'))return Promise.reject(Error('Simulated publication outage'));return original(...args);};
    window.addEventListener('error',e=>window.__errors.push(e.message));
    window.addEventListener('unhandledrejection',e=>window.__errors.push(String(e.reason)));
    window.__workerUrls=[];window.__workerOptions=[];const WorkerClass=window.Worker;window.Worker=class extends WorkerClass{constructor(url,options){window.__workerUrls.push(String(url));window.__workerOptions.push(options);super(url,options);}};
  `});
  await call('Page.navigate',{url:origin});
  await until(`document.body.innerText.includes(${JSON.stringify(versionText)})`);
  await until("!!document.querySelector('[aria-label=\"Poker actions\"] button:not(:disabled)')");
  assert.equal(await evaluate("document.querySelector('[role=alert]')?.textContent??null"),null);
  const urls=await evaluate('window.__requests');
  assert.equal(urls.filter(url=>url.endsWith('.onnx')&&url.includes('.supabase.co/')).length,2);
  assert.equal(urls.some(url=>url.includes('/policy/index.json')),false);
  await until('window.__workerUrls.length>0');
  const fixtures=JSON.parse(await readFile('tests/fixtures/hybrid_parity.json','utf8'));
  for (const fixture of fixtures.filter(row=>row.street==='RIVER')) {
    const message={requestId:1,state:fixture.state,ranges:fixture.ranges,budget:fixture.budget,
      observerSeat:0,updateOnly:true,likelihoodProfiles:Object.fromEntries(Object.keys(fixture.ranges).map(seat=>[seat,'published']))};
    const response=await evaluate(`new Promise((resolve,reject)=>{
      const worker=new Worker(window.__workerUrls[0],window.__workerOptions[0]);
      const timeout=setTimeout(()=>{worker.terminate();reject(Error('Worker timed out'));},30000);
      worker.onmessage=e=>{clearTimeout(timeout);worker.terminate();resolve(e.data);};
      worker.onerror=e=>{clearTimeout(timeout);worker.terminate();reject(Error(e.message));};
      worker.postMessage(${JSON.stringify(message)});
    })`);
    assert.equal(response.error,undefined);assert.ok(response.display);assert.ok(response.marginals);
  }
  await evaluate("window.dispatchEvent(new Event('focus'))");
  await new Promise(resolve=>setTimeout(resolve,1500));
  assert.equal(await evaluate("window.__requests.filter(url=>url.endsWith('.onnx')).length"),2,'Unchanged models must not be downloaded again');
  await evaluate("window.__offline=true;window.dispatchEvent(new Event('focus'))");
  await until("document.querySelector('[role=alert]')?.innerText.includes('Simulated publication outage')");
  assert.ok(await evaluate(`document.body.innerText.includes(${JSON.stringify(versionText)})`));
  await evaluate("window.__offline=false;window.dispatchEvent(new Event('focus'))");
  await until("!document.querySelector('[role=alert]')");
  const result=await evaluate('({requests:window.__requests,errors:window.__errors,workers:window.__workerUrls.length})');
  assert.deepEqual(result.errors,[]);
  console.log(JSON.stringify({status:'passed',...result}));
} finally {socket.close();await fetch(`${cdp}/json/close/${page.id}`);}
