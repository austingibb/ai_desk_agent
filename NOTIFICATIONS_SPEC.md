# Notification System — Implementation Spec

The AI agent can propose recurring notifications to the user. The user approves or rejects them. Approved notifications fire on schedule. The system learns which categories the user likes.

## Context: Existing System

Read `CLAUDE.md` for full architecture. Key points for this feature:

- **Agent loop** is in `main.py:Orchestrator._turn()`. Each turn: send messages to LLM, get response, execute tool calls, repeat.
- **`_tool_wait()`** (main.py:241) polls every 1s for button presses and chat messages. If either occurs, it returns early with `{"status": "interrupted", ...}`. This is where notification checks go.
- **Backoff**: wait duration starts at `BACKOFF_BASE` (10s), triples each peaceful wait, caps at `BACKOFF_MAX` (900s). Resets on any user interaction.
- **Buttons**: two physical buttons, but both mean the same thing — "user wants attention." There is no yes/no distinction. `_tool_wait` checks `/buttons/state` and injects a nudge message on press.
- **Chat**: web UI on :8080. Messages injected via `self.ctx.add_user()`, wake the agent via `self.chat_event`.
- **Context**: messages stored in OpenAI format in `context.py:Context`. Persisted to `context.json`.
- **Tools**: defined in `config.py:TOOL_DEFINITIONS`, executed in `main.py:_execute_tool()`.
- **Compaction**: summarizes old messages periodically, preserving recent ones.

## New Tool: `propose_notification`

Add to `TOOL_DEFINITIONS` in `config.py`:

```json
{
  "type": "function",
  "function": {
    "name": "propose_notification",
    "description": "Propose a recurring notification for the user. They will see it on the display and can approve (button press) or reject (via chat). Only propose when the review prompt suggests a pattern worth acting on.",
    "parameters": {
      "type": "object",
      "properties": {
        "message": {
          "type": "string",
          "description": "Notification text, max 100 chars, e-ink friendly (no emoji)."
        },
        "category": {
          "type": "string",
          "enum": ["health", "productivity", "time", "environment", "misc"],
          "description": "Category of the notification."
        },
        "trigger_type": {
          "type": "string",
          "enum": ["interval", "time_of_day"],
          "description": "How the notification fires: on a repeating interval, or at a specific time daily."
        },
        "trigger_value": {
          "type": "string",
          "description": "For interval: seconds between firings (e.g. '3600'). For time_of_day: 24h time (e.g. '14:30')."
        }
      },
      "required": ["message", "category", "trigger_type", "trigger_value"]
    }
  }
}
```

## Data Model

New file `notifications.json` (same directory as `context.json`):

```json
{
  "notifications": [
    {
      "id": "notif_1715000000",
      "status": "approved",
      "message": "Time to stretch!",
      "category": "health",
      "trigger_type": "interval",
      "trigger_value": "3600",
      "proposed_at": 1715000000,
      "decided_at": 1715000060,
      "last_fired": 1715003600,
      "fire_count": 3,
      "decay_score": 0.8
    }
  ],
  "category_scores": {
    "health": 0.0,
    "productivity": 0.0,
    "time": 0.0,
    "environment": 0.0,
    "misc": 0.0
  }
}
```

### Fields

- `id`: `"notif_{unix_timestamp}"` — unique identifier
- `status`: one of `proposed`, `approved`, `rejected`, `expired`
- `message`: the text shown to the user when the notification fires
- `category`: one of `health`, `productivity`, `time`, `environment`, `misc`
- `trigger_type`: `interval` or `time_of_day`
- `trigger_value`: seconds string for interval, `"HH:MM"` for time_of_day
- `proposed_at`: unix timestamp when proposed
- `decided_at`: unix timestamp when approved/rejected (null while proposed)
- `last_fired`: unix timestamp of last firing (null if never fired)
- `fire_count`: number of times this notification has fired
- `decay_score`: starts at 1.0, decays when user ignores firings (see Decay section)
- `category_scores`: global preference signal per category, range [-1, 1]

## New File: `notifications.py`

Create a `NotificationStore` class that handles:

### Load/Save
- Load from / save to `notifications.json`
- Called from `Orchestrator.__init__()` (load) and `Orchestrator.cleanup()` (save)
- Also save after any state change (proposal, approval, rejection, firing, decay)

### `create_proposal(message, category, trigger_type, trigger_value) -> dict`
- Create a new notification record with status `proposed`, `decay_score=1.0`, `fire_count=0`
- Return the created record

### `approve_pending() -> dict | None`
- Find the notification with status `proposed`, set status to `approved`, set `decided_at`
- Update `category_scores[category] += 0.2` (clamp to 1.0)
- Return the notification, or None if nothing pending

### `reject_pending() -> dict | None`
- Find the notification with status `proposed`, set status to `rejected`, set `decided_at`
- Update `category_scores[category] -= 0.3` (clamp to -1.0)
- Return the notification, or None if nothing pending

### `expire_pending() -> dict | None`
- Find the notification with status `proposed`, set status to `rejected` (treated as soft rejection)
- Update `category_scores[category] -= 0.1` (clamp to -1.0)
- Return the notification, or None if nothing pending

### `get_due_notification() -> dict | None`
- Check all `approved` notifications
- For `interval` type: due if `time.time() - last_fired >= trigger_value` (or `last_fired is None`)
- For `time_of_day` type: due if current time matches `HH:MM` and hasn't fired today
- If multiple are due, return the one with the highest `decay_score`
- Return None if nothing is due

### `record_firing(notification_id)`
- Set `last_fired = time.time()`, increment `fire_count`

### `record_acknowledgment(notification_id)`
- `decay_score = min(1.0, decay_score + 0.1)`

### `decay_unacknowledged(notification_id)`
- `decay_score -= 0.3`
- If `decay_score < 0.2`, set status to `expired`

### `has_pending_proposal() -> bool`
- Return True if any notification has status `proposed`

### `get_review_summary() -> str`
- Return a formatted string for injection into the conversation (see Review Prompt section)

## Integration into `main.py`

### Orchestrator.__init__
- Create `self.notification_store = NotificationStore()`
- Add `self.last_review_time = time.time()`
- Add `self.last_fired_notification_id = None` (tracks which notification was last fired, for acknowledgment tracking)

### _execute_tool — add handler for `propose_notification`
```
elif name == "propose_notification":
    # Rate limit: max 1 proposal per 2 hours
    # Check notification_store for recent proposals
    # If rate limited, return error
    # Otherwise: create proposal, save, return result telling agent to display it
    return {
        "status": "ok",
        "message": "Proposal saved. Now show it to the user with update_display: '{message} — press button to approve!'"
    }
```

### _tool_wait — add to the polling loop (alongside button and chat checks)

Two new checks in the existing `while time.monotonic() - start < seconds` loop:

**Check 1: Review prompt injection**
```
if time.time() - self.last_review_time > REVIEW_INTERVAL:
    if self.backoff < 270:  # suppress when user is absent
        # Expire any pending proposal that was never responded to
        self.notification_store.expire_pending()
        # Inject review prompt
        self.ctx.add_user(self.notification_store.get_review_summary())
        self.last_review_time = time.time()
        return {"status": "interrupted", "reason": "notification_review"}
```

**Check 2: Due notification firing**
```
due = self.notification_store.get_due_notification()
if due:
    # If there was a previously fired notification that wasn't acknowledged, decay it
    if self.last_fired_notification_id:
        self.notification_store.decay_unacknowledged(self.last_fired_notification_id)
    self.notification_store.record_firing(due["id"])
    self.last_fired_notification_id = due["id"]
    self.ctx.add_user(f'[Notification] Time to show: "{due["message"]}"')
    self._reset_backoff()
    return {"status": "interrupted", "reason": "notification_due"}
```

### Button press handling — route based on pending proposal

In `_tool_wait`, the existing button press code currently always injects "The user pressed a button — they want you to say something!" Change this:

```
if result.get("button"):
    self._reset_backoff()
    http_post("/buttons/reset", {}, timeout=5)

    if self.notification_store.has_pending_proposal():
        approved = self.notification_store.approve_pending()
        self.ctx.add_user(f'The user approved your notification: "{approved["message"]}"')
    else:
        # If last fired notification is pending acknowledgment, record it
        if self.last_fired_notification_id:
            self.notification_store.record_acknowledgment(self.last_fired_notification_id)
            self.last_fired_notification_id = None
        self.ctx.add_user("The user pressed a button — they want you to say something!")

    return {"status": "interrupted", ...}
```

Do the same in `_idle_wait` for the button press handling there.

### Chat message handling — check for rejection

In `ChatHandler._post_message`, after adding the user message to context:

```
REJECTION_KEYWORDS = ["no", "nah", "don't", "stop", "cancel", "never", "quit", "not that"]

if orch.notification_store.has_pending_proposal():
    if any(kw in message.lower() for kw in REJECTION_KEYWORDS):
        rejected = orch.notification_store.reject_pending()
        # The rejection message is already in context as a user message,
        # so the agent will see it naturally
```

Also: if a chat message arrives while `last_fired_notification_id` is set, treat it as acknowledgment:
```
if orch.last_fired_notification_id:
    orch.notification_store.record_acknowledgment(orch.last_fired_notification_id)
    orch.last_fired_notification_id = None
```

## Review Prompt

Generated by `NotificationStore.get_review_summary()`. Format:

```
[Notification review] Time: 2:30pm Wed.
Active notifications (1): "stretch reminder" (health, every 60min, last fired 1:30pm).
Category scores: health=0.6, productivity=0.0, time=0.0, environment=-0.2, misc=0.0.
Patterns detected: user at desk for 2+ hours, 3 photos with no scene change.
-> If you see a pattern worth a notification, call propose_notification. Otherwise continue your rhythm.
```

The "Patterns detected" line is populated by simple heuristics (see Pattern Detection section). If no patterns are detected, omit the line.

## Pattern Detection

Computed by the orchestrator based on tool call history. Keep this simple — just count events and check timestamps. Implemented as a method on Orchestrator, not NotificationStore.

| Pattern | Heuristic | Text injected |
|---------|-----------|---------------|
| Seated long | 3+ `take_photo` calls over 90+ minutes with no chat/button interaction between them | "user at desk for {N} hours" |
| Late night | Current time is after 23:00 | "it's late ({time})" |
| Long absence | No button/chat event for 4+ hours and it's between 8:00-22:00 | "no user interaction for {N} hours" |

Don't do image analysis or CV. Just use timestamps and event counts from the context message history.

## Config Constants

Add to `config.py`:

```python
REVIEW_INTERVAL = int(os.environ.get("REVIEW_INTERVAL", "1800"))  # 30 minutes
MAX_PROPOSAL_INTERVAL = 7200  # 2 hours — min time between proposals
MAX_FIRINGS_PER_HOUR = 1
CATEGORY_COOLDOWN_REVIEWS = 3  # after proposing in a category, skip it for N reviews
```

## System Prompt Addition

Append to `SYSTEM_PROMPT` in `config.py`:

```
NOTIFICATIONS:
You can propose recurring notifications with propose_notification.
- Only propose when the review prompt suggests a real pattern.
- The user approves by pressing a button. They reject via chat ("no", "stop", etc).
- Check category scores in the review prompt — negative means stop proposing that type.
- Max 100 chars for notification messages. Keep them friendly and casual.
- NEVER propose about: hygiene, weight, appearance, diet, relationships, or anything judgmental.
- Good proposals: stretch reminders, break nudges, "it's getting late", weather alerts.
- It's completely fine to never propose anything. Only propose genuinely useful things.
```

## Rate Limits Summary

| Limit | Value | Enforced by |
|-------|-------|-------------|
| Max 1 proposal per 2 hours | `MAX_PROPOSAL_INTERVAL` | `_execute_tool` returns error |
| Max 1 notification firing per hour | `MAX_FIRINGS_PER_HOUR` | `get_due_notification()` returns None |
| Category cooldown after proposal | 3 review cycles | `get_review_summary()` excludes category |
| Suppress reviews when user absent | backoff >= 270 | `_tool_wait` skips review check |

## Decay

Approved notifications decay when the user ignores their firings:

- After a notification fires and the agent displays it, `last_fired_notification_id` is set
- If the user presses a button or sends a chat message before the next firing: `decay_score += 0.1` (acknowledged)
- If the next notification fires or a review occurs without acknowledgment: `decay_score -= 0.3`
- If `decay_score < 0.2`: status set to `expired`, inform agent in next review prompt

This handles "user approved it once but doesn't care anymore" without requiring explicit cancellation.

## What NOT to Build

- No UI for managing notifications (the chat and buttons are the UI)
- No image analysis or computer vision for pattern detection
- No separate thread — everything runs in the existing `_tool_wait` polling loop
- No database — `notifications.json` is sufficient
- No undo/edit — user rejects via chat, agent can propose a modified version later
