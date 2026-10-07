const manifest=JSON.parse(document.querySelector('#manifest').textContent);
const labels={'real-marker-d2':'Real · Marker D2','real-square-d2':'Real · Square D2','real-routing-d2':'Real · Routing D2','sim-square-narrow':'Sim · Square-Narrow','sim-square-broad':'Sim · Square-Broad'};
const order=Object.keys(labels);
const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
for(const task of order){const s=manifest.summary[task],card=el('button',undefined,'card');card.type='button';card.append(el('b',labels[task]),el('strong',String(s.repositories)),el('span','released repositories'),el('span',s.original_episodes.toLocaleString()+' original recorded episodes'));card.addEventListener('click',()=>{document.querySelector('#task').value=task;render();document.querySelector('.toolbar').scrollIntoView({behavior:'smooth',block:'start'});});document.querySelector('#cards').append(card);document.querySelector('#task').append(new Option(labels[task],task));}
const bytes=Object.values(manifest.summary).reduce((a,s)=>a+s.copy_tree_bytes_including_views,0);
document.querySelector('#totals').textContent=manifest.datasets.length+' datasets · '+(bytes/1e9).toFixed(2)+' GB, including views';
for(const role of [...new Set(manifest.datasets.map(r=>r.role))].sort())document.querySelector('#role').append(new Option(role,role));
for(let r=0;r<=5;r++)document.querySelector('#round').append(new Option('R'+r,String(r)));
function render(){const search=document.querySelector('#search').value.toLowerCase(),task=document.querySelector('#task').value,role=document.querySelector('#role').value,tier=document.querySelector('#tier').value,round=document.querySelector('#round').value;
const rows=manifest.datasets.filter(r=>(!task||r.task===task)&&(!role||r.role===role)&&(!tier||r.tier===tier)&&(round===''||r.model_rounds.includes(Number(round)))&&JSON.stringify(r).toLowerCase().includes(search));
document.querySelector('#count').textContent=rows.length+' repositories shown. Episode totals include derived views only at row level.';
const body=document.querySelector('#rows');body.replaceChildren();
for(const r of rows){const tr=el('tr'),name=el('td');const dest=el('a',r.destination_repo,'name');dest.href='https://huggingface.co/datasets/'+r.destination_repo+'/tree/'+r.release_revision;name.append(dest);
const details=el('details');details.append(el('summary','Details'));if(r.parent_destination_repo){const p=el('p','Parent: '),a=el('a',r.parent_destination_repo);a.href='https://huggingface.co/datasets/'+r.parent_destination_repo;p.append(a);details.append(p);}if(r.variant)details.append(el('p','Source policy / variant: '+r.variant));if(r.cameras.length)details.append(el('p','Recorded cameras: '+r.cameras.join(', ')));name.append(details);
const kind=el('td');kind.append(el('div',r.role),el('span',r.tier,'pill tag-'+r.tier));let rounds=r.model_rounds.map(x=>'R'+x).join(', ');if(r.source_round!==null)rounds+=(rounds?' · ':'')+'source '+r.source_round;tr.append(name,kind,el('td',rounds||'See source'),el('td',r.episodes===null?'No metadata':r.episodes.toLocaleString()));body.append(tr);}}
for(const id of ['search','task','role','tier','round'])document.querySelector('#'+id).addEventListener(id==='search'?'input':'change',render);
render();
