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
