# Proposed Computational Mechanics Day

Recorded 2026-07-29 from the proposal sent to Xavier and collaborators.

## Schedule

- **10:00–10:30 — Lecture 1: Introduction.** Keep the existing overview, plus
  whatever changes Ben proposes.
- **10:30–12:00 — HMMs and MSPs, guided derivation.** Give a short introductory
  lecture (about 10 minutes), then have students derive the main results in an
  ARENA-style notebook.
- **12:00–12:30 — HMMs and MSPs, synthesis.** Give a longer lecture emphasizing
  geometry and the main "money-shot" result, while preserving room for
  questions, unfinished exercises, and schedule buffer.
- **12:30–13:30 — Lunch.**
- **13:30–15:00 — Transformers and belief geometry.** Give a short introduction,
  then use a minimally coded notebook as a "guided discovery fiction" for how
  one might have invented the belief-state-geometry paper: students do the
  mathematics, reveal a worked derivation, and fill the corresponding small
  blank in a supplied Colab analysis.
- **15:00–15:15 — Break and pacing check.** If the group is substantially behind,
  drop the Hankel/PSR block.
- **15:15–15:30 — PSR and Hankel introduction.**
- **15:30–16:45 — PSR and Hankel guided derivations.**
- **16:45–17:00 — Break.**
- **17:00–18:00 — Synthesis or buffer.** Summarize the day; if time permits,
  deepen PSR/Hankel material through WFA reconstruction, otherwise use the time
  for questions, exploration, or paper reading.

The original message listed the final block as 17:15–18:00 after a 15-minute
break; the normalized schedule above uses 17:00–18:00 so there is no unexplained
15-minute gap. This timing should be confirmed with the teaching team.

## Prepared artifacts

Six guided notebooks (exercise and worked-solution variants) now cover the three
derivational blocks, and six offline micro-decks frame, gate, and synthesize
them. The decks begin at 10:30, leaving the existing 10:00–10:30 overview intact
pending Ben's edits; presenter HTML, speaker notes, and matching PDF handouts are
under `slides/dist/`. The Hankel/PSR core remains droppable, and the final WFA
route remains an optional 8–10-minute instructor demonstration. This artifact
preparation does not resolve the 17:00 versus 17:15 final-block timing question
above.

## Pedagogical intent

- Mix lecture and exercises through worked, ARENA-like derivation notebooks;
  preserve the option to split those blocks if the teaching team prefers a
  conventional lecture/exercise separation.
- Adapt Xavier's existing notebooks while shifting the work from Python
  implementation to derivations, simple calculations, interpretation, and
  small numerical entries.
- Aim for approximately zero student-written code, except possibly a very small
  analysis step in the transformer notebook.
- Structure the day in roughly 90-minute blocks and let students encounter each
  major topic three times: a short introduction, an in-depth derivation, and a
  theory extension or recap.
- Make PSR, Hankel, and WFA coverage modular so that later pieces can be removed
  without breaking the earlier arc.
- Keep the appealing "cook your own HMM" idea, but scaffold it with a constrained
  design brief, examples, intermediate checks, and a supplied visualizer.

## Notebook design brief

Each notebook should:

1. begin with a substantial but visual explanation of the objects and notation;
2. start with hand-holding and progressively remove scaffolding;
3. use exercises for most derivation steps, with hints and worked solutions;
4. mix symbolic derivations with small numerical calculations;
5. keep diagrams, simplex plots, or supplied animations close to the mathematics
   they illustrate;
6. end with a harder application or constrained design problem;
7. remain fully usable as a paper worksheet even when executable visual cells
   are skipped.

## Original proposal text

<details>
<summary>Verbatim message supplied for the project record</summary>

> So I thought about it and wanted to run by you guys this modified version:
>
> 10-10h30 --- Lec1 - into: same / whatever changes Ben proposes  
> 10h30 - 12h00 --- short intro lecture (~10 min) on HMMs and MSPs, but mainly
> get them to do a guided derivation of the main results in an arena-style
> notebook  
> 12h00 - 12h30 --- longer lecture on HMMs and MSPs, with geometry + money-shot
> + questions / more time for exercises / buffer
>
> 12h30 - 13h30 lunch
>
> 13h30 - 15h00 : short lecture on transformers representing belief geometry,
> then a notebook that is a 'guided discovery fiction' of how you could've
> invented the belief state geometry paper, but with very minimal coding (e.g.
> have them do the math, show solution to math derivation, have them fill in
> corresponding blank in colab notebook)  
> (15 minute break + check speed and time - drop Hankel/PSR if they're
> significantly behind.)
>
> 15h15 - 15h30 : quick intro lecture on PSRs and Hankel  
> 15h30 - 16h45 : more guided derivation notebooks on PSR/Hankel  
> (15 minute break)  
> 17h15 - 18h00 : day summary + IF TIME longer lecture on PSR / Hankel , incl.
> WFA reconstruction, ELSE buffer for questions / exploratory / paper reading
>
> Small thoughts  
> I really like the idea of mixing lecture/exercises by having them do
> arena-like notebooks. But this might be an idiosyncratic-to-me thing, so please
> let me know if that's true. I'd make the notebooks from the ones already, just
> focus more on derivations and math than code? If we prefer lecture/exercise
> split, we can just split those blocks.  
> PSR/Hankel and WFA will act as a time buffer, since can cover each fairly
> modularly  
> I tried to organize it into 90-ish minute blocks, and to have them see each
> major topic three times as intro/in-depth/(theory or recap).  
> I'd feel sad to lose the 'cook your own HMM' - I was thinking to have a more
> guided version in the notebooks, since you said ppl had a hard time last time?
> This is sth I'd have to work through myself to see how hard it is
>
> I think per-topic I budgeted less time than you did because I was conditioning
> on getting to the Hankel lecture lol; presumably that's unwise. I'd ask around
> as we're going and slow down if need be, at expense of that last hankel
> lecture.
>
> I envision nearly 0 coding except perhaps in the transformers section -
> thoughts on that? Not sure if that's wise, but its what i'd enjoy more as a
> student lol

</details>
