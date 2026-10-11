/** Exercise all street policies through the actual browser Bayesian worker. */
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
const origin=process.env.POKER_UI_ORIGIN ?? 'http://127.0.0.1:3136';
const cdp=process.env.POKER_CDP_ORIGIN ?? 'http://127.0.0.1:9436';
const page=await (await fetch(`${cdp}/json/new?about:blank`,{method:'PUT'})).json();
const socket=new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
let sequence=0;const pending=new Map();
socket.onmessage=event=>{const msg=JSON.parse(event.data);const task=pending.get(msg.id);if(!task)return;pending.delete(msg.id);clearTimeout(task.timer);msg.error?task.reject(new Error(JSON.stringify(msg.error))):task.resolve(msg.result);};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;const timer=setTimeout(()=>reject(Error(`CDP timeout ${method}`)),45000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));});
async function evaluate(expression){const value=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(value.exceptionDetails)throw Error(JSON.stringify(value.exceptionDetails));return value.result.value;}
async function until(expression){const deadline=Date.now()+45000;while(Date.now()<deadline){if(await evaluate(expression))return;await new Promise(resolve=>setTimeout(resolve,200));}throw Error(`Timeout ${expression}: ${await evaluate('document.body.innerText')}`);}
try {
 await call('Page.enable');
 await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
 await call('Page.addScriptToEvaluateOnNewDocument',{source:`window.__workerURLs=[];window.__workerOptions=[];const OriginalWorker=window.Worker;window.Worker=class extends OriginalWorker{constructor(url,options){window.__workerURLs.push(String(url));window.__workerOptions.push(options);super(url,options);}};`});
 await call('Page.navigate',{url:origin});
 await until("document.body.innerText.includes('Policies · HU 5 · 3-max 5')");
 await until('window.__workerURLs.length>0');
 const fixtures=await (await fetch(`${origin}/__street-fixtures.json`)).json();
 const results=await evaluate(`(async()=>{
   const fixtures=${JSON.stringify(fixtures)};const worker=new Worker(window.__workerURLs[0],window.__workerOptions[0]);const results=[];
   try {for(let i=0;i<fixtures.length;i++) {const row=fixtures[i];
     const result=await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('Worker timed out')),30000);worker.onmessage=e=>{clearTimeout(timer);resolve(e.data);};worker.onerror=e=>{clearTimeout(timer);reject(Error(e.message));};worker.postMessage({...row,requestId:i,updateOnly:true,likelihoodProfiles:Object.fromEntries(Object.keys(row.state.players).map(p=>[p,'published']))});});
     if(result.error)throw Error(result.error);results.push({count:row.count,street:row.street,expected:row.expected,actual:result.ranges[row.actor].map(r=>r.probability)});
   }}finally{worker.terminate();}return results;
 })()`);
 let error=0;for(const row of results){assert.equal(row.actual.length,row.expected.length);row.actual.forEach((p,i)=>{error=Math.max(error,Math.abs(p-row.expected[i]));});}
 assert.ok(error<1e-6,`Browser/Python posterior error ${error}`);
 assert.equal(new Set(results.map(r=>`${r.count}:${r.street}`)).size,8);
 const pages=[];
 const click=text=>evaluate(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()===${JSON.stringify(text)}).click()`);
 await click('Overview');
 for(const track of ['3-max','HU']) {
   await click(track);
   await until(`document.querySelector('[aria-label="Preflop action range"]')!==null`);
   assert.equal(await evaluate(`document.querySelectorAll('[aria-label$="action frequencies"]').length`),169);
   pages.push(`Overview/${track}`);
 }
 await click('Specific spot');
 for(const track of ['3-Max','HU']) {
   await click(track);
   for(const street of ['PREFLOP','FLOP','TURN','RIVER']) {
     await evaluate(`(()=>{const s=document.querySelector('select[aria-label="Street"]');s.value='${street}';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
     await click('Apply state');
     await until(`document.querySelector('[aria-label="Action policy"]')?.innerText.includes('%')`);
     assert.equal(await evaluate(`document.querySelectorAll('[role="alert"]').length`),0);
     pages.push(`Specific spot/${track}/${street}`);
   }
 }
 await click('Test Live');
 await until('window.__workerURLs.length>0');
 pages.push('Test Live');
 const report={pages,status:'passed',routes:8,transitions:results.length,max_absolute_posterior_error:error,results};
 if(process.env.POKER_STREET_REPORT)await writeFile(process.env.POKER_STREET_REPORT,JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({status:report.status,routes:8,transitions:results.length,max_absolute_posterior_error:error}));
} finally {socket.close();await fetch(`${cdp}/json/close/${page.id}`);}
