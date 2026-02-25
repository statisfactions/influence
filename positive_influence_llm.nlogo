; Positive Influence Model with LLM Conversations
; Based on Smaldino's positive influence opinion dynamics model
; Agents converse via a local Ollama LLM instead of using numeric opinion updates
;
; Requirements:
;   - NetLogo 6.x+ with the Python extension (py)
;   - Python 3.7+ with standard library
;   - Ollama running locally (http://localhost:11434)
;   - A pulled model, e.g.: ollama pull phi3:mini
;
; Alternative small models: tinyllama, qwen2:1.5b, gemma2:2b

extensions [py]

globals [
  last-snippet     ; snippet of the most recent conversation
]

turtles-own [
  opinion          ; float in [-1, 1]
  agent-id         ; integer ID matching the Python memory file
]

;; ── Setup ────────────────────────────────────────────────────────────────────

to setup
  clear-all
  reset-ticks

  ;; Initialize Python and import the helper module
  py:setup py:python
  ;; Add the model's directory to Python's path so it can find llm_helper.py
  ;; We write a temp file to detect the NetLogo working directory from Python
  py:run "import sys, os"
  py:run "import llm_helper"

  ;; Set the topic and model in Python, then initialize agents
  ;; Use py:set to safely pass string values (avoids quoting issues)
  py:set "nl_topic" discussion-topic
  py:set "nl_model" ollama-model
  py:set "nl_num" num-agents
  py:set "nl_memlen" memory-length
  let init-opinions py:runresult "llm_helper.setup_agents(nl_num, nl_topic, nl_model, nl_memlen)"

  ;; Create agents on a grid
  let grid-size ceiling sqrt num-agents
  let agent-counter 0

  create-turtles num-agents [
    set agent-id agent-counter
    set opinion item agent-counter init-opinions

    ;; Place on grid
    let col (agent-counter mod grid-size)
    let row (floor (agent-counter / grid-size))
    let spacing-x (max-pxcor - min-pxcor) / (grid-size + 1)
    let spacing-y (max-pycor - min-pycor) / (grid-size + 1)
    setxy (min-pxcor + spacing-x * (col + 1))
          (min-pycor + spacing-y * (row + 1))

    set shape "circle"
    set size 1.5
    recolor

    set agent-counter agent-counter + 1
  ]

  set last-snippet "(no conversation yet)"
end

;; ── Step (one conversation per tick) ─────────────────────────────────────────

to step
  if count turtles < 2 [ stop ]

  ;; Pick a random agent-A
  let agent-a one-of turtles
  ;; Pick a random partner agent-B (different from A)
  let agent-b one-of other turtles

  let id-a [agent-id] of agent-a
  let id-b [agent-id] of agent-b

  ;; Run the LLM conversation via Python
  ;; Store result in a Python global to avoid dict serialization issues
  py:set "conv_id_a" id-a
  py:set "conv_id_b" id-b
  py:set "conv_tick" ticks
  py:set "conv_memlen" memory-length
  py:run "conv_result = llm_helper.run_conversation(conv_id_a, conv_id_b, conv_tick, conv_memlen)"

  ;; Fetch individual values from the Python result dict
  let new-opinion-a py:runresult "conv_result['opinion_a']"
  let new-opinion-b py:runresult "conv_result['opinion_b']"
  let snippet py:runresult "conv_result['snippet']"

  ;; Update opinions and recolor
  ask agent-a [
    set opinion new-opinion-a
    recolor
  ]
  ask agent-b [
    set opinion new-opinion-b
    recolor
  ]

  ;; Update display
  set last-snippet snippet

  tick
end

;; ── Helpers ──────────────────────────────────────────────────────────────────

to recolor  ;; turtle procedure
  ;; Map opinion [-1, 1] to color on a grayscale: 0 = black, 9.9 = white
  let shade (opinion + 1) / 2 * 9.9
  set color shade
end
@#$#@#$#@
GRAPHICS-WINDOW
210
10
620
420
-1
-1
12.0
1
10
1
1
1
0
0
0
1
-16
16
-16
16
1
1
1
ticks
30.0

BUTTON
10
10
90
43
setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
95
10
190
43
go-once
step
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
10
48
190
81
go
step
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
10
90
190
123
num-agents
num-agents
4
100
25.0
1
1
NIL
HORIZONTAL

SLIDER
10
128
190
161
memory-length
memory-length
1
20
5.0
1
1
NIL
HORIZONTAL

INPUTBOX
10
170
410
230
discussion-topic
whether AI should be heavily regulated or allowed to develop freely
1
0
String

INPUTBOX
10
235
200
295
ollama-model
phi3:mini
1
0
String

PLOT
210
430
620
600
Opinion Distribution
Opinion
Count
-1.0
1.0
0.0
10.0
true
false
"set-plot-x-range -1 1\nset-histogram-num-bars 10" ""
PENS
"default" 0.2 1 -16777216 true "" "histogram [opinion] of turtles"

MONITOR
420
10
620
55
Last Conversation
last-snippet
0
1
11

MONITOR
420
60
520
105
Tick
ticks
0
1
11

MONITOR
525
60
620
105
Num Agents
count turtles
0
1
11

@#$#@#$#@
## WHAT IS IT?

This model recreates Smaldino's positive influence opinion dynamics model, replacing the simple numeric opinion update rule with actual natural language conversations between agents powered by a local Ollama LLM.

Each agent maintains a memory file of past conversations and all interactions are logged to a master transcript. After each conversation, the LLM extracts a numeric opinion score in [-1, 1] for visualization.

## HOW IT WORKS

**Setup:** Creates N agents, each assigned a random initial opinion stance on the configured topic. Agent memories are stored as text files in `agent_memories/`.

**Each tick:** One random agent initiates a 3-turn conversation with a random partner via the Ollama LLM. The conversation is informed by both agents' current stances and recent memory. After the conversation, a separate LLM call extracts updated opinion scores for both agents.

**Visualization:** Agents are colored on a black-to-white scale based on their opinion (-1 = black/against, +1 = white/in favor).

## HOW TO USE IT

### Prerequisites
1. Install [Ollama](https://ollama.ai)
2. Pull a small model: `ollama pull phi3:mini`
3. Start Ollama: `ollama serve`
4. NetLogo 6.x+ with the Python extension

### Running
1. Set the **discussion-topic** (the issue agents will debate)
2. Set the **ollama-model** (default: phi3:mini)
3. Set **num-agents** (keep low, e.g. 25, since each tick requires LLM calls)
4. Click **setup** to initialize agents
5. Click **go-once** for a single conversation, or **go** for continuous

### Parameters
- **num-agents**: Number of agents (4-100, default 25)
- **memory-length**: How many past conversations to include in prompts (default 5)
- **discussion-topic**: The topic agents discuss
- **ollama-model**: Which Ollama model to use (phi3:mini, tinyllama, qwen2:1.5b, gemma2:2b)

## THINGS TO NOTICE

- Opinions tend to converge over time, matching the original positive influence model
- The histogram shows the distribution of opinions shifting
- Check `transcript.txt` for full conversation logs
- Check `agent_memories/` for individual agent histories

## CREDITS AND REFERENCES

Based on Smaldino's positive influence model. LLM integration by Claude.
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

circle
false
0
Circle -7500403 true true 0 0 300

@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
1
@#$#@#$#@
