import test from 'node:test';
import assert from 'node:assert/strict';

import {
  formatAgentState,
  menuItems,
  mutationDisabled,
  nextTakeover,
  reconcilePlan,
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
