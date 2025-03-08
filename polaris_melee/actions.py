from melee import Action, PlayerState, Character, YoshiMoves

SHIELDING_ACTIONS = (
    Action.SHIELD,
    Action.SHIELD_REFLECT,
    Action.SHIELD_STUN,
    Action.SHIELD_RELEASE,
    Action.SHIELD_START
)

YOSHI_SHIELD_VALUES = (
    YoshiMoves.ShieldHold.value,
    YoshiMoves.ShieldRelease.value,
    YoshiMoves.ShieldDamage.value,
    YoshiMoves.ShieldStartup.value,
)

MOVEMENT_ACTIONS = (
    Action.DASHING,
    Action.KNEE_BEND,
)

ALL_MOVEMENT_ACTIONS = (
    Action.DASHING,
    Action.KNEE_BEND,
    Action.RUN_BRAKE,
    Action.RUNNING,
    Action.WALK_SLOW,
    Action.WALK_FAST,
    Action.WALK_MIDDLE,
    Action.STANDING,
    Action.LANDING,
    Action.LANDING_SPECIAL
)


JUMP_ACTIONS = (
    Action.JUMPING_FORWARD,
    Action.JUMPING_BACKWARD,
)
AERIAL_MOVEMENT_ACTIONS = (
    Action.JUMPING_ARIAL_FORWARD,
    Action.JUMPING_BACKWARD,
)
DODGE_ACTIONS = (
    Action.SPOTDODGE,
    Action.ROLL_FORWARD,
    Action.ROLL_BACKWARD
)
CROUCH_ACTIONS = (
    Action.CROUCH_START,
    Action.CROUCHING
)
NO_GROUND_TECH_ACTIONS = (
    Action.TECH_MISS_UP,
    Action.TECH_MISS_DOWN
)
LEDGE_ATTACK_ACTIONS = (
    Action.EDGE_ATTACK_SLOW,
    Action.EDGE_ATTACK_QUICK
)
LEDGE_NEUTRAL_ACTIONS = (
    Action.EDGE_GETUP_SLOW,
    Action.EDGE_GETUP_QUICK
)
LEDGE_JUMP_ACTIONS = (
    Action.EDGE_JUMP_1_SLOW,
    Action.EDGE_JUMP_2_SLOW,
    Action.EDGE_JUMP_1_QUICK,
    Action.EDGE_JUMP_2_QUICK
)
LEDGE_ROLL_ACTIONS = (
    Action.EDGE_ROLL_SLOW,
    Action.EDGE_ROLL_QUICK
)
FALLING_ACTIONS = (
    Action.FALLING,
    Action.FALLING_FORWARD,
    Action.FALLING_BACKWARD
)
NEUTRAL_GETUP_ACTIONS = (
    Action.GROUND_GETUP,
    Action.NEUTRAL_GETUP,
)
BACKWARD_GETUP_ACTIONS = (
    Action.GROUND_ROLL_BACKWARD_UP,
    Action.GROUND_ROLL_BACKWARD_DOWN
)
FORWARD_GETUP_ACTIONS = (
    Action.GROUND_ROLL_FORWARD_UP,
    Action.GROUND_ROLL_FORWARD_DOWN
)



def is_shield(player: PlayerState) -> bool:
    if player.character == Character.YOSHI:
        return player.action.value in YOSHI_SHIELD_VALUES
    return player.action in SHIELDING_ACTIONS
