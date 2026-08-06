import {
  formatAgentState,
  menuItems,
  mutationDisabled,
  nextTakeover,
  reconcilePlan,
  shouldApplyAgentSnapshot,
} from 'chat-model';

const config=JSON.parse(document.getElementById('chat-config').textContent);
const messagesDiv=document.getElementById('messages');
const modeChip=document.getElementById('agent-mode');
const banner=document.getElementById('status-banner');
const takeoverText=document.getElementById('takeover-text');
const ring=document.querySelector('.ring-progress');
const menu=document.getElementById('context-menu');
const form=document.getElementById('form');
const input=document.getElementById('input');
const attach=document.getElementById('attach');
const send=document.getElementById('send');
const fileInput=document.getElementById('file-input');
const attachmentDiv=document.getElementById('attachments');
const uploadError=document.getElementById('upload-error');

const allowedTypes=new Set(['image/png','image/jpeg','image/webp','image/gif']);
const messageNodes=new Map();
let messagesById=new Map();
let orderedMessages=[];
let agentState={mode:'offline',detail:'',locks_input:false};
let agentServerId=null;
let offline=true;
let chatRevision=null;
let selectedFiles=[];
let sending=false;
let editingId=null;
let menuId=null;
let takeoverTimer=null;
let takeoverFrame=null;
let fallbackTimer=null;
let refreshPromise=null;

function isAgentLocked(){
  return !offline&&Boolean(agentState.locks_input);
}

function atBottom(){
  return messagesDiv.scrollHeight-messagesDiv.scrollTop-messagesDiv.clientHeight<60;
}

function firstVisibleAnchor(survivingIds){
  const top=messagesDiv.getBoundingClientRect().top;
  for(const child of messagesDiv.children){
    if(!survivingIds.has(child.dataset.messageId))continue;
    const rect=child.getBoundingClientRect();
    if(rect.bottom>=top)return {id:child.dataset.messageId,top:rect.top};
  }
  return null;
}

function createMessageNode(message){
  const wrap=document.createElement('article');
  wrap.dataset.messageId=message.id;

  const trigger=document.createElement('button');
  trigger.type='button';
  trigger.className='menu-trigger';
  trigger.textContent='⋯';
  trigger.title='Message actions';
  trigger.setAttribute('aria-label','Message actions');
  trigger.setAttribute('aria-expanded','false');
  trigger.addEventListener('click',event=>{
    event.stopPropagation();
    openMenu(message.id,trigger);
  });

  const role=document.createElement('div');
  role.className='role';
  const body=document.createElement('div');
  body.className='msg';
  const text=document.createElement('div');
  text.className='msg-text';
  const media=document.createElement('div');
  media.className='media-grid';
  body.append(trigger,text,media);
  wrap.append(role,body);
  return wrap;
}

function updateMessageNode(node,message){
  const signature=JSON.stringify([
    message.role,message.content,message.images,message.time,message.queued,
  ]);
  if(node.dataset.signature===signature)return;
  node.dataset.signature=signature;
  node.className=`msg-wrap ${message.role}${message.queued?' queued':''}`;
  node.setAttribute('aria-label',message.role==='user'?'You':'AI Friend');
  node.querySelector('.role').textContent=message.time||'';

  if(editingId!==message.id){
    node.querySelector('.msg-text').textContent=message.content||'';
  }
  const media=node.querySelector('.media-grid');
  media.replaceChildren();
  for(const image of message.images||[]){
    const link=document.createElement('a');
    link.href=image.url;
    link.target='_blank';
    link.rel='noopener';
    const img=document.createElement('img');
    img.className='chat-media';
    img.src=image.url;
    img.alt=image.name||'Uploaded image';
    img.loading='lazy';
    link.append(img);
    media.append(link);
  }
}

function reconcileMessages(messages){
  const wasAtBottom=atBottom();
  const survivingIds=new Set(messages.map(message=>message.id));
  const anchor=wasAtBottom?null:firstVisibleAnchor(survivingIds);
  const plan=reconcilePlan([...messageNodes.keys()],messages);
  const nextById=new Map(messages.map(message=>[message.id,message]));

  for(const message of plan.upsert){
    let node=messageNodes.get(message.id);
    if(!node){
      node=createMessageNode(message);
      messageNodes.set(message.id,node);
    }
    updateMessageNode(node,message);
  }

  plan.order.forEach((id,index)=>{
    const node=messageNodes.get(id);
    const current=messagesDiv.children[index];
    if(current!==node)messagesDiv.insertBefore(node,current||null);
  });

  for(const id of plan.remove){
    if(editingId===id){
      editingId=null;
      showTakeover('That message was already sent and cannot be edited.');
    }
    if(menuId===id)closeMenu();
    messageNodes.get(id)?.remove();
    messageNodes.delete(id);
  }

  orderedMessages=messages.slice();
  messagesById=nextById;
  if(wasAtBottom){
    messagesDiv.scrollTop=messagesDiv.scrollHeight;
  }else if(anchor&&messageNodes.has(anchor.id)){
    const nextTop=messageNodes.get(anchor.id).getBoundingClientRect().top;
    messagesDiv.scrollTop+=nextTop-anchor.top;
  }
}

function applyAgentState(agent,{isOffline=false,serverId=null}={}){
  if(agent&&shouldApplyAgentSnapshot(
    agentState.revision,agentServerId,agent.revision,serverId,
  )){
    agentState=agent;
    if(serverId)agentServerId=serverId;
  }
  offline=isOffline;
  modeChip.textContent=formatAgentState(agentState,offline);
  modeChip.className=offline?'offline':agentState.mode;
  const locked=isAgentLocked();
  input.disabled=locked;
  attach.disabled=locked;
  send.disabled=locked||sending;
  document.querySelectorAll('.attachment button').forEach(button=>{
    button.disabled=locked||sending;
  });
  document.querySelectorAll('.edit-save').forEach(button=>{
    button.disabled=locked;
  });
  menu.querySelectorAll('button[data-action]').forEach(button=>{
    button.disabled=mutationDisabled(button.dataset.action,locked);
  });
}

function closeMenu(){
  menu.classList.remove('show');
  menu.replaceChildren();
  document.querySelectorAll('.menu-trigger[aria-expanded="true"]').forEach(button=>{
    button.setAttribute('aria-expanded','false');
  });
  menuId=null;
}

function openMenu(id,trigger){
  const message=messagesById.get(id);
  if(!message)return;
  closeMenu();
  menuId=id;
  trigger.setAttribute('aria-expanded','true');
  for(const action of menuItems(message)){
    const button=document.createElement('button');
    button.type='button';
    button.role='menuitem';
    button.dataset.action=action;
    button.textContent={undo:'Undo send',edit:'Edit',copy:'Copy'}[action];
    button.disabled=mutationDisabled(action,isAgentLocked());
    button.addEventListener('click',()=>{
      closeMenu();
      if(action==='copy')copyMessage(message.content||'');
      if(action==='undo')undoMessage(id);
      if(action==='edit')startEdit(id);
    });
    menu.append(button);
  }
  menu.classList.add('show');
  const rect=trigger.getBoundingClientRect();
  const menuRect=menu.getBoundingClientRect();
  const left=Math.max(8,Math.min(rect.left,window.innerWidth-menuRect.width-8));
  const top=Math.max(8,Math.min(rect.bottom+5,window.innerHeight-menuRect.height-8));
  menu.style.left=`${left}px`;
  menu.style.top=`${top}px`;
}

async function copyMessage(rawText){
  try{
    if(navigator.clipboard&&window.isSecureContext){
      await navigator.clipboard.writeText(rawText);
      return;
    }
    throw new Error('Clipboard API unavailable');
  }catch(error){
    const textarea=document.createElement('textarea');
    textarea.value=rawText;
    textarea.setAttribute('readonly','');
    textarea.style.position='fixed';
    textarea.style.opacity='0';
    document.body.append(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
  }
}

function dismissTakeover(){
  if(takeoverTimer)clearTimeout(takeoverTimer);
  if(takeoverFrame)cancelAnimationFrame(takeoverFrame);
  takeoverTimer=null;
  takeoverFrame=null;
  banner.classList.remove('show');
  ring.style.strokeDashoffset='0';
}

function showTakeover(text){
  dismissTakeover();
  const durationMs=config.takeoverSeconds*1000;
  const state=nextTakeover(text,performance.now(),durationMs);
  takeoverText.textContent=state.text;
  banner.classList.add('show');
  const circumference=2*Math.PI*8;
  const tick=now=>{
    const progress=Math.min(1,(now-state.startedAt)/durationMs);
    ring.style.strokeDashoffset=String(circumference*progress);
    if(progress<1)takeoverFrame=requestAnimationFrame(tick);
  };
  takeoverFrame=requestAnimationFrame(tick);
  takeoverTimer=setTimeout(dismissTakeover,durationMs);
}

banner.addEventListener('click',dismissTakeover);

async function responseJSON(response){
  let data={};
  try{data=await response.json();}catch(error){}
  if(!response.ok){
    const failure=new Error(data.error||`Request failed (${response.status}).`);
    failure.status=response.status;
    throw failure;
  }
  return data;
}

async function undoMessage(id){
  try{
    const response=await fetch(`/chat/queue/${encodeURIComponent(id)}`,{method:'DELETE'});
    const data=await responseJSON(response);
    restoreComposer(data.restored);
    await refresh();
  }catch(error){
    showTakeover(error.message||'Could not undo that message.');
    await refresh().catch(()=>{});
  }
}

function cancelEdit(){
  if(!editingId)return;
  const cancelledId=editingId;
  const node=messageNodes.get(cancelledId);
  node?.querySelector('.inline-editor')?.remove();
  const text=node?.querySelector('.msg-text');
  if(text){
    text.textContent=messagesById.get(cancelledId)?.content||'';
    text.hidden=false;
  }
  if(node)node.dataset.signature='';
  editingId=null;
}

function startEdit(id){
  const message=messagesById.get(id);
  const node=messageNodes.get(id);
  if(!message?.queued||!node||isAgentLocked())return;
  cancelEdit();
  editingId=id;
  const text=node.querySelector('.msg-text');
  text.hidden=true;
  const editor=document.createElement('div');
  editor.className='inline-editor';
  const textarea=document.createElement('textarea');
  textarea.value=message.content||'';
  const error=document.createElement('div');
  error.className='edit-error';
  const actions=document.createElement('div');
  actions.className='edit-actions';
  const cancel=document.createElement('button');
  cancel.type='button';
  cancel.textContent='Cancel';
  cancel.addEventListener('click',cancelEdit);
  const save=document.createElement('button');
  save.type='button';
  save.className='edit-save';
  save.textContent='Save';
  save.disabled=isAgentLocked();
  save.addEventListener('click',()=>saveEdit(id,textarea.value,error,save));
  actions.append(cancel,save);
  editor.append(textarea,error,actions);
  node.querySelector('.msg').insertBefore(editor,node.querySelector('.media-grid'));
  textarea.addEventListener('keydown',event=>{
    if(event.key==='Escape'){
      event.preventDefault();
      cancelEdit();
    }else if(event.key==='Enter'&&!event.shiftKey){
      event.preventDefault();
      save.click();
    }
  });
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length,textarea.value.length);
}

async function saveEdit(id,value,errorElement,saveButton){
  if(isAgentLocked())return;
  saveButton.disabled=true;
  errorElement.textContent='';
  try{
    const response=await fetch(`/chat/queue/${encodeURIComponent(id)}`,{
      method:'PATCH',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:value}),
    });
    const data=await responseJSON(response);
    cancelEdit();
    const next=orderedMessages.map(message=>message.id===id?data.message:message);
    reconcileMessages(next);
  }catch(error){
    if(error.status===400){
      errorElement.textContent=error.message;
      saveButton.disabled=false;
      return;
    }
    cancelEdit();
    showTakeover(error.message||'Could not edit that message.');
    await refresh().catch(()=>{});
  }
}

function showUploadError(message=''){
  uploadError.textContent=message;
  uploadError.classList.toggle('show',Boolean(message));
}

function dataURLBytes(value){
  const comma=value.indexOf(',');
  if(comma<0)return 0;
  const encoded=value.slice(comma+1);
  return Math.max(0,Math.floor(encoded.length*3/4)-(encoded.endsWith('==')?2:encoded.endsWith('=')?1:0));
}

function revokeAttachment(item){
  if(item.previewUrl?.startsWith('blob:'))URL.revokeObjectURL(item.previewUrl);
}

function renderAttachments(){
  attachmentDiv.replaceChildren();
  selectedFiles.forEach((item,index)=>{
    const wrap=document.createElement('div');
    wrap.className='attachment';
    const image=document.createElement('img');
    image.src=item.previewUrl;
    image.alt=item.name;
    const remove=document.createElement('button');
    remove.type='button';
    remove.textContent='×';
    remove.title='Remove attachment';
    remove.disabled=isAgentLocked()||sending;
    remove.addEventListener('click',()=>{
      const [removed]=selectedFiles.splice(index,1);
      if(removed)revokeAttachment(removed);
      renderAttachments();
    });
    wrap.append(image,remove);
    attachmentDiv.append(wrap);
  });
  attachmentDiv.classList.toggle('show',selectedFiles.length>0);
}

function clearAttachments(){
  selectedFiles.forEach(revokeAttachment);
  selectedFiles=[];
  fileInput.value='';
  renderAttachments();
}

function restoreComposer(payload){
  clearAttachments();
  input.value=payload?.text||'';
  selectedFiles=(payload?.images||[]).map(image=>({
    name:image.name||'image',
    type:image.type,
    size:dataURLBytes(image.data_url),
    dataUrl:image.data_url,
    previewUrl:image.data_url,
    file:null,
  }));
  renderAttachments();
  input.focus();
}

function addFiles(files){
  showUploadError();
  for(const file of files){
    if(!allowedTypes.has(file.type)){
      showUploadError(`${file.name} is not a supported image or GIF.`);
      continue;
    }
    if(selectedFiles.length>=config.maxImages){
      showUploadError(`You can attach up to ${config.maxImages} files.`);
      break;
    }
    const duplicate=selectedFiles.some(item=>
      item.name===file.name&&item.size===file.size&&item.lastModified===file.lastModified
    );
    const total=selectedFiles.reduce((sum,item)=>sum+item.size,0);
    if(!duplicate&&total+file.size>config.maxMediaBytes){
      showUploadError(`Attachments can total at most ${Math.floor(config.maxMediaBytes/1024/1024)} MB.`);
      continue;
    }
    if(!duplicate){
      selectedFiles.push({
        file,
        name:file.name,
        type:file.type,
        size:file.size,
        lastModified:file.lastModified,
        dataUrl:null,
        previewUrl:URL.createObjectURL(file),
      });
    }
  }
  renderAttachments();
}

function readFile(file){
  return new Promise((resolve,reject)=>{
    const reader=new FileReader();
    reader.onload=()=>resolve(reader.result);
    reader.onerror=()=>reject(new Error(`Could not read ${file.name}.`));
    reader.readAsDataURL(file);
  });
}

async function filePayload(item){
  return {
    name:item.name,
    type:item.type,
    data_url:item.dataUrl||await readFile(item.file),
  };
}

attach.addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',()=>addFiles(fileInput.files));

let dragDepth=0;
document.addEventListener('dragenter',event=>{
  if(Array.from(event.dataTransfer?.items||[]).some(item=>item.kind==='file')){
    dragDepth++;
    document.body.classList.add('dragging');
  }
});
document.addEventListener('dragleave',()=>{
  dragDepth=Math.max(0,dragDepth-1);
  if(!dragDepth)document.body.classList.remove('dragging');
});
document.addEventListener('dragover',event=>event.preventDefault());
document.addEventListener('drop',event=>{
  event.preventDefault();
  dragDepth=0;
  document.body.classList.remove('dragging');
  addFiles(event.dataTransfer?.files||[]);
});
document.addEventListener('paste',event=>{
  const files=Array.from(event.clipboardData?.files||[]).filter(file=>allowedTypes.has(file.type));
  if(files.length){
    event.preventDefault();
    addFiles(files);
  }
});

form.addEventListener('submit',async event=>{
  event.preventDefault();
  const message=input.value.trim();
  if((!message&&!selectedFiles.length)||sending||isAgentLocked())return;
  const submittedFiles=selectedFiles.slice();
  sending=true;
  showUploadError();
  applyAgentState(agentState,{isOffline:offline});
  renderAttachments();
  try{
    const images=await Promise.all(submittedFiles.map(filePayload));
    const response=await fetch('/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message,images}),
    });
    const data=await responseJSON(response);
    if(input.value.trim()===message)input.value='';
    if(selectedFiles.length===submittedFiles.length&&selectedFiles.every((item,index)=>item===submittedFiles[index])){
      clearAttachments();
    }
    const next=orderedMessages.slice();
    const existingIndex=next.findIndex(item=>item.id===data.id);
    if(existingIndex>=0)next[existingIndex]=data.message;
    else next.push(data.message);
    reconcileMessages(next);
    await refresh();
  }catch(error){
    showUploadError(error.message||'Could not send that message.');
  }finally{
    sending=false;
    applyAgentState(agentState,{isOffline:offline});
    renderAttachments();
    if(!isAgentLocked())input.focus();
  }
});

async function refresh(){
  if(refreshPromise)return refreshPromise;
  refreshPromise=(async()=>{
    try{
      const response=await fetch('/chat');
      const data=await responseJSON(response);
      chatRevision=data.chat_revision;
      applyAgentState(data.agent,{isOffline:false,serverId:data.server_id});
      reconcileMessages(data.messages||[]);
      return data;
    }catch(error){
      applyAgentState(agentState,{isOffline:true});
      throw error;
    }finally{
      refreshPromise=null;
    }
  })();
  return refreshPromise;
}

function startFallback(){
  if(fallbackTimer)return;
  fallbackTimer=setInterval(()=>refresh().catch(()=>{}),2000);
}

function stopFallback(){
  if(fallbackTimer)clearInterval(fallbackTimer);
  fallbackTimer=null;
}

function connectEvents(){
  const events=new EventSource('/chat/events');
  events.addEventListener('open',()=>{
    stopFallback();
  });
  events.addEventListener('snapshot',event=>{
    try{
      const snapshot=JSON.parse(event.data);
      applyAgentState(snapshot.agent,{
        isOffline:false,
        serverId:snapshot.server_id,
      });
      if(snapshot.chat_revision!==chatRevision)refresh().catch(()=>{});
    }catch(error){}
  });
  events.addEventListener('error',()=>{
    startFallback();
    refresh().catch(()=>{});
  });
}

document.addEventListener('click',event=>{
  if(!menu.contains(event.target)&&!event.target.closest('.menu-trigger'))closeMenu();
});
document.addEventListener('keydown',event=>{
  if(event.key==='Escape'&&menuId)closeMenu();
});
messagesDiv.addEventListener('scroll',closeMenu,{passive:true});
window.addEventListener('resize',closeMenu);

refresh().catch(()=>{});
connectEvents();
