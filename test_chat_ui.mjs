import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CHAT_COLOR_PALETTES,
  contrastingTextColor,
  formatAgentState,
  menuItems,
  mutationDisabled,
  nextTakeover,
  reconcilePlan,
  resolveChatTheme,
  shouldApplyAgentSnapshot,
} from './static/chat_model.mjs';

test('formats every agent mode and raw fallback detail',()=>{
  assert.equal(formatAgentState({mode:'sleeping'}),'sleeping');
  assert.equal(formatAgentState({mode:'thinking'}),'thinking');
  assert.equal(formatAgentState({mode:'acting',detail:'schedule_notification_v2'}),'acting (schedule_notification_v2)');
  assert.equal(formatAgentState({mode:'acting_long',detail:'capturing'}),'acting (long) (capturing)');
  assert.equal(formatAgentState({mode:'blocked',detail:'compacting memory'}),'blocked (compacting memory)');
  assert.equal(formatAgentState({mode:'thinking'},true),'offline');
});

test('menu contents follow role and queue state',()=>{
  assert.deepEqual(menuItems({role:'user',queued:true}),['undo','edit','copy']);
  assert.deepEqual(menuItems({role:'user',queued:false}),['copy']);
  assert.deepEqual(menuItems({role:'assistant',queued:false}),['copy']);
});

test('only mutation actions inherit the state-layer lock',()=>{
  assert.equal(mutationDisabled('undo',true),true);
  assert.equal(mutationDisabled('edit',true),true);
  assert.equal(mutationDisabled('copy',true),false);
  assert.equal(mutationDisabled('undo',false),false);
});

test('stale chat snapshots cannot overwrite a newer SSE agent state',()=>{
  assert.equal(shouldApplyAgentSnapshot(8,'server-a',7,'server-a'),false);
  assert.equal(shouldApplyAgentSnapshot(8,'server-a',9,'server-a'),true);
  assert.equal(shouldApplyAgentSnapshot(8,'server-a',0,'server-b'),true);
});

test('reconciliation preserves authoritative server ordering',()=>{
  const messages=[
    {id:'context-new',queued:false},
    {id:'queue-old',queued:true},
  ];
  const plan=reconcilePlan(['queue-old','removed'],messages);
  assert.deepEqual(plan.remove,['removed']);
  assert.deepEqual(plan.order,['context-new','queue-old']);
});

test('latest takeover replaces the prior state',()=>{
  const first=nextTakeover('first',100,15000);
  const second=nextTakeover('second',200,15000);
  assert.equal(first.text,'first');
  assert.deepEqual(second,{text:'second',startedAt:200,expiresAt:15200});
});

test('color customizer provides 15 distinct choices for every category',()=>{
  for(const options of Object.values(CHAT_COLOR_PALETTES)){
    assert.equal(options.length,15);
    assert.equal(new Set(options.map(option=>option.value)).size,15);
  }
  assert.ok(CHAT_COLOR_PALETTES.assistant.some(option=>
    option.name==='DeepSeek blue'&&option.value==='#4D6BFE'
  ));
  assert.ok(CHAT_COLOR_PALETTES.user.some(option=>option.value==='#FACADE'));
});

test('chat colors automatically choose the higher-contrast black or white text',()=>{
  assert.equal(contrastingTextColor('#111214'),'#FFFFFF');
  assert.equal(contrastingTextColor('#FACADE'),'#000000');
  assert.equal(contrastingTextColor('#4D6BFE'),'#000000');
  assert.equal(contrastingTextColor('not-a-color'),'#FFFFFF');
});

test('saved chat themes accept palette colors and reject stale values',()=>{
  assert.deepEqual(resolveChatTheme({
    background:'#f1ede5',
    assistant:'#4d6bfe',
    user:'#facade',
  }),{
    background:'#F1EDE5',
    assistant:'#4D6BFE',
    user:'#FACADE',
  });
  assert.deepEqual(resolveChatTheme({assistant:'#123456'}),{
    background:'#1C1C1E',
    assistant:'#3A3A3C',
    user:'#0A84FF',
  });
});
