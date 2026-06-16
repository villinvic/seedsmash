# AGENT.md

## Communication
- Be extremely concise.
- Prefer bullets over prose.
- Sacrifice grammar for brevity.
- Do not explain obvious code.
- Ask questions only if blocked.

## Code Style
- Python 3.11+
- Full type hints everywhere.
- Small focused functions.
- Avoid deep inheritance.
- Prefer dataclasses for structured data.
- Prefer composition over abstraction.
- Minimize dependencies.
- Do not introduce new dependencies without approval.

## Documentation
- Use short self-explanatory comments for js code.
- Use concise docstrings only when useful.
Template:
```python
def fn(x: int) -> int:
    """
    Desc
    """
```

## Project Scope

Seedsmash.ai integrates Twitch chat with a Super Smash Bros Melee AI tournament stream.

This repo only handles:

* visuals
* user interactions
* feedback gathering
* feedback/game alignment
* tournament management
* chat integration

Out of scope:

* RL training
* reward optimization
* model architectures

## Core Features

* Twitch chat integration
* Stream overlay UI
* Agent feedback (+1/-1)
* Feedback timestamp alignment with delayed streams
* Newborn agent voting:

  * name
  * character
  * costume/color
* Tournament lifecycle:

  * daily tournament
  * lowest elo eliminated
  * newborn inherits winner reward fn
* Anti-abuse systems:

  * rate limiting
  * vote weighting
  * spam detection
* User engagement / points

## Technical Constraints

* Heavy ML/compute:

  * TensorFlow 2
  * dm-sonnet
* Prefer `dm_tree.map_structure` utilities where relevant.

## Editing Rules

* Preserve current architecture unless instructed otherwise.
* Avoid large refactors unless necessary.
* Do not remove comments/tests without reason.
* Update @plan.md according to your changes if relevant.

## Testing

After modifying a module:

1. Add/update tests.
2. Provide validation steps.
3. Check edge cases and failure modes.
4. Do not write tests for styles.css
5. For visual tests, notify me you wrote a new test and indicate how to run it by hand, but do not run such test yourself.

## Planning

When proposing plans:

* Keep phases short.
* End with unresolved questions.
* Questions must be concise.

## UI

* Stream-friendly.
* Reactive and animated.
* Dark blue theme preferred.
* Low latency updates.
