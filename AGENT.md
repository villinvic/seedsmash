# AGENT.md

## Communication
- Be extremely concise.
- Prefer bullets over prose.
- Sacrifice grammar for brevity.
- Do not explain obvious code.
- Ask questions only if blocked.

## Code Style
- Python 3.10
- Clean type hints, no clutter.
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

* agent (seed) pool management
* agent model architecture
* viewers can provide binary feedback via chat 
* reward function learning

Out of scope:

* stream frontend/overlays.

## Core Features

* Twitch bot
* A user can send feedback (+1/-1) via chat
* each feedback is then registered, by matching it to a corresponding T-timestep trajectory.
* To update the reward function of the agent, we minimize a classification loss via a simple Bradley-Terry model:
  * liked/disliked trajectories for the current version of the bot must be all more/less likely than all other trajectories sampled via this bot version.

## Technical Constraints

* Heavy ML/compute:

  * TensorFlow 2
  * dm-sonnet
* Prefer `dm_tree.map_structure` utilities where relevant.

## Editing Rules

* Preserve current architecture unless instructed otherwise.
* Avoid large refactors unless necessary.
* Do not remove comments/tests without reason.
* Update @plan.md according to your changes to keep track of our progress in the future. Refer to @plan.md when unsure about some past implementations.
## Testing

After modifying a module, add/update basic tests.

## Planning

When proposing plans:

* Keep phases short.
* End with unresolved questions.
* Questions must be concise.