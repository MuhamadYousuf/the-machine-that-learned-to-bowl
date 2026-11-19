# "The Machine That Learned to Bowl" — documentary scripts (locked 2026-11-17)

Two cuts from one story. **Short** (~3:40) for LinkedIn and Reels, and **Long** (~10:30) for YouTube and
your portfolio. Narration in English (Kokoro voice). Every number below is checked against the
training logs and replays. ⟦NEW⟧ marks footage we still have to make. Everything else already exists.

**Tone:** a cricket documentary, not a tech demo. Treat the batters as characters and the AI as a
young bowler learning its trade. The cricket lover should feel the duel. The tech lover should
get the "how".

**Visual language**
- **3D players** ⟦NEW⟧: a realistic bowler (Unity animation pack) and batter, used in the cold open,
  the batter introductions, the duel and the wickets.
- **Hawk-Eye world** (exists): trajectories, pitch maps, the swarm of 192 balls.
- **Brain world** (exists): network, brain cam, curves, family tree.
- **Batter HUD** ⟦NEW⟧: a confidence meter, "reads the pattern" indicator, shot label and runs,
  driven by the simulator's real batter state for each ball.

---

## SHORT CUT (~3:40)

| # | Time | Visual | Narration | On screen |
|---|---|---|---|---|
| 1 | 0:00 | ⟦NEW⟧ 3D night stadium, slow push-in on the batter taking guard. Bowler at the top of his mark. Silence, crowd murmur. | Every great fast bowler has spent years learning one thing. | |
| 2 | 0:06 | ⟦NEW⟧ Bowler runs in, delivers, slow-motion stumps cartwheel. Cut to black. | This one learned it in six hours. And nobody taught it a thing. | **THE MACHINE THAT LEARNED TO BOWL** |
| 3 | 0:14 | Gen 0 chaos (broadcast + batter end) | At the start, it's hopeless. Random speeds, random lengths. It gets hit on more than half its balls. | GENERATION 0 |
| 4 | 0:24 | ⟦NEW⟧ Three batters walk out one by one, each with a name card | Its opponents are three batters, each with a hidden weakness. | |
| 5 | 0:29 | ⟦NEW⟧ The Slogger, swinging hard + his scoring zones lighting up | The Slogger. He hits anything full or short, but a yorker on the stumps, or a slower ball, can get him. | THE SLOGGER |
| 6 | 0:38 | ⟦NEW⟧ The Technician leaving outside off + weakness zone | The Technician. He leaves anything wide and blocks the straight ball. His only flaw: a good-length ball that moves away late. | THE TECHNICIAN |
| 7 | 0:49 | ⟦NEW⟧ The Tailender hopping at a short ball | And the Tailender. Brave, but not much of a batter. | THE TAILENDER |
| 8 | 0:54 | ⟦NEW⟧ Batter HUD over a replay: confidence rises with each repeated ball | And all three learn. Bowl the same ball twice, and they're ready for it. | CONFIDENCE ▲ |
| 9 | 1:02 | Brain morph (network) | The bowler is a tiny brain: eighteen numbers in about the batter and the over, four out: speed, length, line and swing. | 18 IN · 4 OUT |
| 10 | 1:12 | Swarm of 192 balls → family tree | A hundred and ninety-two brains bowl. The best survive and breed the next generation. Fifteen hundred generations. About eighty-three million deliveries. | 192 BRAINS · 1,500 GENERATIONS |
| 11 | 1:26 | v1 learning curve, red grade | And then, it got good. Suspiciously good. | VERSION 1 |
| 12 | 1:32 | v1 pitch map: bouncer, yorker, bouncer, yorker. v1 brain cam with pinned dials | It had found a trick. Bouncer, yorker, bouncer, yorker. Flat out, then dead slow. Three bouncers an over, which is illegal. | REWARD HACKING |
| 13 | 1:44 | Black, then a single line of text | It hadn't learned to bowl. It had learned to fool my batters. | |
| 14 | 1:50 | ⟦NEW⟧ Batter HUD: "pattern read" lights up by ball 4 | So I made the batters smarter. Now they spot patterns. Alternate bouncer and yorker, and they've read you by the fourth ball. And only two bouncers an over, like the real laws. | PATTERN READ · BALL 4 |
| 15 | 2:04 | Rewind → Gen 1 swarm | Then I deleted everything, and started again. | GENERATION 1 |
| 16 | 2:10 | Gen 1 vs Gen 1500 orbit | Fifteen hundred generations later, every ball lands where it means to. | GENERATION 1,500 |
| 17 | 2:18 | ⟦NEW⟧ 3D duel vs Technician. Pitch map fills the good-length band | Against the Technician, it bowls good length on fourth stump, seventy-three percent of the time, and gives away barely one run an over. Then, the one that moves away. | 73% GOOD LENGTH |
| 18 | 2:32 | Caught-behind hero (stump cam / broadcast) + ⟦NEW⟧ 3D edge | Edged. Gone. | WICKET · CAUGHT BEHIND |
| 19 | 2:36 | ⟦NEW⟧ 3D duel vs Slogger: he hits a boundary first (HUD confidence up) | The Slogger fights back. He still scores almost six an over. But the bowler keeps it full and straight... | 4 · CONFIDENCE ▲ |
| 20 | 2:47 | Bowled hero, stumps flying | ...until one yorker is too good. | WICKET · BOWLED |
| 21 | 2:52 | LBW hero vs Tailender | And the Tailender? It skips the bouncer and goes full and straight. Trapped in front. | WICKET · LBW |
| 22 | 3:02 | Brain cam (live activations) | You can watch it decide. Every neuron, every ball. | BRAIN CAM |
| 23 | 3:08 | Result bars | Scored against the hand-written expert plan for each batter, the final brain beats every one. And the cheating version, retested under the new rules? Not even close. | EXPERT PLAN = 1.0 |
| 24 | 3:22 | ⟦NEW⟧ 3D bowler walking back to his mark, slow motion, stadium lights | No coaching. No examples. Just a reward, a few hard lessons, and evolution. | |
| 25 | 3:32 | End card | This is the machine that learned to bowl. | Full story + code: GitHub |

~520 words of narration.

---

## LONG CUT (~10:30) — chapters

Same opening and ending; the middle goes deep. Chapter cards on screen (01 THE PROBLEM, ...).

### Cold open (0:00–0:35)
Short cut scenes 1–2, plus: *"This is the story of how it learned, how it cheated, and how it
finally earned its wickets."*

### 01 · The problem (0:35–1:40)
- **Visual:** ⟦NEW⟧ 3D bowler in slow motion, labelled controls appearing: pace, length, line, swing.
  Then Hawk-Eye side view of one delivery.
- **Narration:** "Bowling looks simple. Run in, let go. But every ball is four decisions: how fast,
  how full, what line, and which way it swings. And you never quite hit your mark: a few
  centimetres of error at release become half a metre at the batter. My bowler has that problem
  too. Every ball it bowls gets a random wobble, so it can't rely on perfect aim. Then there's
  the batter."
- **On screen:** physics callouts: gravity, air drag, swing, bounce, seam.

### 02 · Meet the batters (1:40–3:20)
For each batter: ⟦NEW⟧ 3D walk-out with a player card, their weakness and scoring maps, and a
signature moment from the replays.
- **The Slogger:** "Aggression ninety-four out of a hundred. Anything full or short goes over the
  ropes. His weaknesses: the yorker on the stumps, and the slower ball he swings at too early."
- **The Technician:** "The opposite. Skill ninety-one. Leaves everything wide, blocks everything
  straight. One flaw: a good-length ball on fourth stump that moves away late, and he edges it."
- **The Tailender:** "Brave, not skilled. He'll hop around against the short ball, and a full
  straight ball can sneak through."
- **How batters think** ⟦NEW⟧ HUD over a replay: "None of them is a fixed target. Each one
  tracks what you've been bowling: length, line, movement, pace. Repeat yourself and their
  confidence climbs, and a confident batter is harder to get out. Surprise them and you're
  in the game. Real batters work the same way."
- **Note:** every batter is randomised by up to fifteen percent each over, so the bowler never
  faces exactly the same player twice.

### 03 · The brain (3:20–4:10)
- **Visual:** brain morph, then input labels lighting up node by node; weight heatmaps.
- **Narration:** "The bowler is a neural network with eighteen inputs: which batter, his
  confidence, the score in the over, the last two balls. Two layers of twenty-four neurons. Four
  outputs. That's 1,156 numbers in total. No rules of cricket written anywhere. Whatever it
  learns, it learns by playing."

### 04 · Evolution (4:10–5:20)
- **Visual:** gen 1 swarm, behaviour swarm tightening, family tree, learning curve, generation
  counter.
- **Narration:** "Instead of teaching it, I let it evolve. A hundred and ninety-two random brains
  each bowl forty-eight overs. The best eight go straight through. The rest of the next
  generation is bred from the winners, by mixing two parents and adding small mutations. Repeat
  that fifteen hundred times and you've bowled about eighty-three million deliveries."

### 05 · The cheat (5:20–6:40)
- **Visual:** v1 curve rising fast; v1 pitch map with two clusters at the extremes; brain cam v1
  with orange pinned dials; side view of the alternating trajectories.
- **Narration:** "Version one learned fast. Too fast. When I watched what it was bowling, it was
  doing one thing: bouncer, yorker, bouncer, yorker. A hundred and fifteen kilometres an hour,
  then a hundred and fifty. Nothing in between. Half of all its balls were bouncers, three an
  over, which is illegal in real cricket."
- "The reason was my batter model. My batters judged surprise only by how far a ball was from
  what they expected. Alternating between the two extremes was always maximally surprising. The
  AI didn't learn cricket. It found a bug in my world and farmed it."
- "This has a name in AI research: reward hacking. The system gets exactly what you asked for,
  not what you meant."
- **On screen:** REWARD HACKING stamp.

### 06 · Rewriting the rules (6:40–7:20)
- **Visual:** ⟦NEW⟧ HUD demo: alternating sequence, surprise bar falling 0.44 → 0.02, "pattern read" by ball 4.
- **Narration:** "So I fixed the world, not the bowler. The batters now remember the last two
  balls and spot alternating patterns. They read a bouncer-yorker routine by the fourth ball.
  Bouncers above the shoulder are capped at two an over, like the real laws. Then I threw the old
  brains away and started from nothing."

### 07 · The duel (7:20–9:10) — the heart of the film
One over per batter, told ball by ball from the batter's side. ⟦NEW⟧ 3D batter and bowler, with
the HUD showing confidence, shot and runs, cut against the Hawk-Eye replay of the same ball.
- **vs the Technician:** "Good length, fourth stump. He leaves. Again. He's getting comfortable,
  confidence rising. Again. Blocked. Then the same length, but this one moves late." Edge, caught
  behind.
- **vs the Slogger:** "He's dangerous early: a boundary, confidence up." (Use a real four from the
  replay.) "The bowler doesn't panic. Full and straight, full and straight. He's waiting for the
  full toss that never comes." Yorker, bowled.
- **vs the Tailender:** "The obvious plan is to bounce him. The bowler doesn't. Full, straight,
  fast." LBW, red wicket-zone pulse.
- **Stat overlay:** per batter, wickets per over and economy (0.44 / 5.7, 0.21 / 1.2, 0.34 / 1.5).
- **The honest line:** "It isn't unbeatable. The Slogger still scores almost six an over. But it
  takes a wicket nearly every other over against him."

### 08 · Inside its head (9:10–9:50)
- **Visual:** brain cam vs Technician; decision map sweeping batter confidence.
- **Narration:** "Watch the brain cam. Every ball, you can see which inputs it's reading. The
  decision map shows how its plan shifts as the batter's confidence changes. One habit it never
  outgrew: it always swings the ball away. Outswing is the classic fast bowler's weapon, so it's
  a good habit, but it's a habit, not a choice."

### 09 · The verdict (9:50–10:15)
- **Visual:** result bars (expert plan line at 1.0); v1 vs v2 split.
- **Narration:** "Measured against the hand-written expert plan for each batter, where random
  bowling is zero and the expert plan is one, the final brain scores 1.25, 1.16 and 1.18. It beats
  the expert against all three. The cheater, retested under the fair rules, scores around 0.6."

### Ending (10:15–10:30)
- ⟦NEW⟧ 3D bowler walking back to his mark.
- **Narration:** "No coaching. No examples. Just a reward, a world that wouldn't let it cheat, and
  evolution. This is the machine that learned to bowl."
- **Teaser card:** "Next: point it at any batter." (Hint at the open-source tool without promising
  dates.)

~1,450 words of narration.

---

## Decisions (locked)
- Title: **The Machine That Learned to Bowl**.
- No behind-the-scenes line; no tooling credit.
- End card: title + "Full story and code on GitHub" (add a handle later if you want one).
- "Expert plan" = the hand-written specialist plan per batter; v2 trained in about 6.4 hours.
