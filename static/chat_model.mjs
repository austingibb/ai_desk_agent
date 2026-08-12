export function formatAgentState(agent, offline=false){
  if(offline)return 'offline';
  const mode=agent?.mode||'offline';
  const detail=agent?.detail||'';
  if(mode==='acting')return `acting (${detail})`;
  if(mode==='acting_long')return `acting (long) (${detail})`;
  if(mode==='blocked')return `blocked (${detail})`;
  return mode;
}

export function menuItems(message){
  if(message?.role==='user'&&message?.queued)return ['undo','edit','copy'];
  return ['copy'];
}

export function mutationDisabled(action,locksInput){
  return action!=='copy'&&Boolean(locksInput);
}

export function shouldApplyAgentSnapshot(
  currentRevision,currentServerId,nextRevision,nextServerId,
){
  if(currentServerId&&nextServerId&&currentServerId!==nextServerId)return true;
  return Number(nextRevision||0)>=Number(currentRevision||0);
}

export function reconcilePlan(currentIds,messages){
  const nextIds=messages.map(message=>message.id);
  const nextSet=new Set(nextIds);
  return {
    remove:currentIds.filter(id=>!nextSet.has(id)),
    order:nextIds,
    upsert:messages,
  };
}

export function nextTakeover(text,now,durationMs){
  return {text,startedAt:now,expiresAt:now+durationMs};
}

export const CHAT_COLOR_PALETTES=Object.freeze({
  background:Object.freeze([
    {name:'Midnight',value:'#111214'},
    {name:'Graphite',value:'#1C1C1E'},
    {name:'Soft black',value:'#242326'},
    {name:'Mushroom',value:'#302C2A'},
    {name:'Cocoa',value:'#342A27'},
    {name:'Deep olive',value:'#2C2D25'},
    {name:'Forest smoke',value:'#24302B'},
    {name:'Blue slate',value:'#242C35'},
    {name:'Night violet',value:'#2D2937'},
    {name:'Dusty plum',value:'#342A33'},
    {name:'Pewter',value:'#55575A'},
    {name:'Warm gray',value:'#77726C'},
    {name:'Stone',value:'#A39D94'},
    {name:'Oat',value:'#D5CFC4'},
    {name:'Porcelain',value:'#F1EDE5'},
  ]),
  assistant:Object.freeze([
    {name:'Charcoal',value:'#3A3A3C'},
    {name:'Ink',value:'#242529'},
    {name:'DeepSeek blue',value:'#4D6BFE'},
    {name:'Storm blue',value:'#35445A'},
    {name:'Denim smoke',value:'#3F5068'},
    {name:'Slate teal',value:'#344A4C'},
    {name:'Evergreen',value:'#35483F'},
    {name:'Dark sage',value:'#485347'},
    {name:'Olive gray',value:'#535344'},
    {name:'Espresso',value:'#4B4037'},
    {name:'Clay smoke',value:'#59433C'},
    {name:'Muted berry',value:'#59404B'},
    {name:'Aubergine',value:'#493C55'},
    {name:'Soft violet',value:'#514864'},
    {name:'Steel',value:'#555C66'},
  ]),
  user:Object.freeze([
    {name:'Electric blue',value:'#0A84FF'},
    {name:'Sky',value:'#64D2FF'},
    {name:'Lagoon',value:'#40C8E0'},
    {name:'Mint',value:'#66D4CF'},
    {name:'Fresh green',value:'#30D158'},
    {name:'Limeade',value:'#A8E063'},
    {name:'Sunshine',value:'#FFD60A'},
    {name:'Tangerine',value:'#FF9F0A'},
    {name:'Coral',value:'#FF6B6B'},
    {name:'Hot pink',value:'#FF375F'},
    {name:'Facade pink',value:'#FACADE'},
    {name:'Orchid',value:'#BF5AF2'},
    {name:'Periwinkle',value:'#8E8CFF'},
    {name:'Lavender',value:'#C6A8FF'},
    {name:'Pearl',value:'#E7E5DF'},
  ]),
});

export const DEFAULT_CHAT_THEME=Object.freeze({
  background:'#1C1C1E',
  assistant:'#3A3A3C',
  user:'#0A84FF',
});

export function contrastingTextColor(hexColor){
  const value=String(hexColor||'').replace(/^#/,'');
  if(!/^[0-9a-f]{6}$/i.test(value))return '#FFFFFF';
  const channels=[0,2,4].map(index=>{
    const channel=parseInt(value.slice(index,index+2),16)/255;
    return channel<=.04045?channel/12.92:((channel+.055)/1.055)**2.4;
  });
  const luminance=.2126*channels[0]+.7152*channels[1]+.0722*channels[2];
  return luminance>.179?'#000000':'#FFFFFF';
}

export function resolveChatTheme(candidate={}){
  return Object.fromEntries(Object.entries(CHAT_COLOR_PALETTES).map(([key,options])=>{
    const requested=String(candidate?.[key]||'').toUpperCase();
    const match=options.find(option=>option.value===requested);
    return [key,match?.value||DEFAULT_CHAT_THEME[key]];
  }));
}
