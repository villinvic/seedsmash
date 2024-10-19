from collections import deque, namedtuple
from copy import copy, deepcopy
from enum import Enum, IntEnum
from typing import Sequence, List, Union, Deque, Dict, Tuple, Any

from gymnasium.spaces import Discrete
from melee import ControllerState, enums, Controller, Console, PlayerState, stages, GameState
from melee.enums import Button, Character, Action
from functools import partial
import itertools
import numpy as np


class ComboPad(Controller):

    def __init__(self, console, port, type=enums.ControllerType.STANDARD):
        super().__init__(console, port, type)
        self.previous_state = ControllerInput()


class DummyPad(ComboPad):
    def __init__(self, console, port, type=enums.ControllerType.STANDARD):
        self.previous_state = ControllerInput()


class ControllerStateCombo(dict):
    def __init__(self, buttons: tuple = None, stick=(0., 0.), c_stick=(0., 0.), duration=3, force_sticks=False,
                 test=None):
        dict.__init__(self)
        if buttons is not None:
            if not isinstance(buttons, tuple):
                buttons = (buttons,)

            for button in buttons:
                for item in enums.Button:
                    if button and item == button:
                        self[button] = True
                    else:
                        self[item] = False

        self.to_press = set(buttons) if buttons is not None else set()

        self['stick'] = stick
        self['c_stick'] = c_stick
        self['duration'] = duration
        self.force_sticks = force_sticks
        self.test = test

    def __str__(self):

        string = ""
        for item in enums.Button:
            string += "%s(%d) " % (item.name, self[item.name])
        string += "S(%s,%s) " % self['stick']
        string += "C(%s,%s)" % self['c_stick']

        return string

    def send_controller(self, pad: Controller, main_stick_input=None, c_stick_input=None):
        to_release = pad.previous_state.to_press - self.to_press
        press_new = self.to_press - pad.previous_state.to_press
        for button in to_release:
            pad.release_button(button)
        for button in press_new:
            pad.press_button(button)

        if self.force_sticks or (not main_stick_input and not c_stick_input):
            pad.tilt_analog_unit(enums.Button.BUTTON_MAIN, *self['stick'])
            pad.tilt_analog_unit(enums.Button.BUTTON_C, *self['c_stick'])
        else:
            pad.tilt_analog_unit(enums.Button.BUTTON_MAIN, *main_stick_input)
            pad.tilt_analog_unit(enums.Button.BUTTON_C, *c_stick_input)
        pad.previous_state = self


class SimpleActionSpace:

    def __getitem__(self, item):
        return self.controller_states[item]

    def __init__(self, short_hop=2):
        """
        1 directions with no button
        2 tap b, lazer, plumbers downb
        3 tilts
        4 smashes with c stick and cardinals
        5 jumps, normal, front, back, full hop, short hop
        6 upbs
        7 L with all directions (rolls, spot dodge, shield, air dodges...)
        8 wavedashes
        9 shield grab, jump grab
        10 tap down b, tap side b
        11 noop

        """
        # act every 3
        # TODO -> lasting effect on env side
        # choose tap b (b, noop, b)
        # step
        # choose tap b (noop, b, noop)

        # choose wavedash (x for number of frames for short_hop, noop if remaining frames)
        # step
        # keep on wavedashing (incoming actions do not take effect)

        all_states = ([ControllerStateCombo()] +
                      [ControllerStateCombo(c_stick=(-1, 0))] +
                      [ControllerStateCombo(c_stick=(1, 0))] +
                      [ControllerStateCombo(c_stick=(0, 1))] +
                      [ControllerStateCombo(c_stick=(0, - 1))] +
                      [ControllerStateCombo(stick=(0, - 1), buttons=Button.BUTTON_B)] +
                      [ControllerStateCombo(stick=(0, 1), buttons=Button.BUTTON_B)] +
                      [ControllerStateCombo(stick=(1, 0), buttons=Button.BUTTON_B)] +
                      [ControllerStateCombo(stick=(-1, 0), buttons=Button.BUTTON_B)] +
                      [ControllerStateCombo(stick=(0.707, -0.707))] +
                      [ControllerStateCombo(stick=(0.707, 0.707))] +
                      [ControllerStateCombo(stick=(-0.707, 0.707))] +
                      [ControllerStateCombo(stick=(-0.707, -0.707))] +
                      [ControllerStateCombo(stick=(0, - 1))] +
                      [ControllerStateCombo(stick=(0, 1))] +
                      [ControllerStateCombo(stick=(1, 0))] +
                      [ControllerStateCombo(stick=(-1, 0))] +
                      [[ControllerStateCombo(buttons=Button.BUTTON_X, duration=short_hop),
                        ControllerStateCombo(duration=1)]] +
                      [ControllerStateCombo(buttons=Button.BUTTON_B)] +
                      [ControllerStateCombo(buttons=Button.BUTTON_A)] +
                      [ControllerStateCombo(buttons=Button.BUTTON_L)] +
                      [ControllerStateCombo(buttons=Button.BUTTON_L, stick=(1, 0))] +  # CHECK for Link
                      [ControllerStateCombo(buttons=Button.BUTTON_L, stick=(-1, 0))] +
                      [ControllerStateCombo(buttons=Button.BUTTON_L, stick=(0, 1))] +
                      [ControllerStateCombo(buttons=Button.BUTTON_L, stick=(0, -1))] +
                      [ControllerStateCombo(buttons=Button.BUTTON_X)] +
                      [ControllerStateCombo(buttons=Button.BUTTON_X, stick=(-1, 0))] +
                      [ControllerStateCombo(buttons=Button.BUTTON_X, stick=(1, 0))] +
                      [ControllerStateCombo(buttons=Button.BUTTON_X)] +
                      [ControllerStateCombo(buttons=Button.BUTTON_Z)] +
                      [ControllerStateCombo(buttons=(Button.BUTTON_A, Button.BUTTON_L))] +
                      [[ControllerStateCombo(buttons=Button.BUTTON_X, duration=1),
                        ControllerStateCombo(buttons=Button.BUTTON_Z, duration=2)]] +
                      [ControllerStateCombo(stick=(-0.953, -0.294), buttons=Button.BUTTON_L)] +
                      [ControllerStateCombo(stick=(0.953, -0.294), buttons=Button.BUTTON_L)] +
                      [[ControllerStateCombo(buttons=Button.BUTTON_X, duration="CS", test=lambda s: s.on_ground),
                        ControllerStateCombo(stick=(-0.953, -0.294), buttons=Button.BUTTON_L, duration=1,
                                             test=lambda s: s.on_ground)]] +
                      [[ControllerStateCombo(buttons=Button.BUTTON_X, duration="CS", test=lambda s: s.on_ground),
                        ControllerStateCombo(stick=(0.953, -0.294), buttons=Button.BUTTON_L, duration=1,
                                             test=lambda s: s.on_ground)]] +
                      [[ControllerStateCombo(buttons=Button.BUTTON_L, duration=2),
                        ControllerStateCombo(buttons=Button.BUTTON_L, stick=(0., -0.675), duration=1)]] +
                      [ControllerStateCombo(buttons=Button.BUTTON_A, stick=(-0.4, 0.0))] +
                      [ControllerStateCombo(buttons=Button.BUTTON_A, stick=(0.4, 0.0))] +
                      [ControllerStateCombo(buttons=Button.BUTTON_A, stick=(0.0, 0.4))]
                      )

        self.controller_states = np.array(
            all_states
            , dtype=ControllerState)

        self.dim = len(self.controller_states)


class ActionSpace:

    def __getitem__(self, item):
        return self.controller_states[item]

    def __init__(self, short_hop=2):
        """
        1 directions with no button
        2 tap b, lazer, plumbers downb
        3 tilts
        4 smashes with c stick and cardinals
        5 jumps, normal, front, back, full hop, short hop
        6 upbs
        7 L with all directions (rolls, spot dodge, shield, air dodges...)
        8 wavedashes
        9 shield grab, jump grab
        10 tap down b, tap side b
        11 noop

        """

        all_states = [ControllerStateCombo()] + \
                     [[ControllerStateCombo(buttons=Button.BUTTON_B, duration=1), ControllerStateCombo(duration=2)]] + \
                     [[ControllerStateCombo(buttons=Button.BUTTON_X, duration=short_hop),
                       ControllerStateCombo(duration=1)]] + \
                     [ControllerStateCombo(buttons=Button.BUTTON_B)] + \
                     [ControllerStateCombo(buttons=Button.BUTTON_A)] + \
                     [ControllerStateCombo(buttons=Button.BUTTON_L)] + \
                     [ControllerStateCombo(buttons=Button.BUTTON_X)] + \
                     [ControllerStateCombo(buttons=Button.BUTTON_Z)] + \
                     [ControllerStateCombo(buttons=(Button.BUTTON_A, Button.BUTTON_L))] + \
                     [[ControllerStateCombo(buttons=Button.BUTTON_X, duration=1),
                       ControllerStateCombo(buttons=Button.BUTTON_Z, duration=2)]]

        self.controller_states = np.array(
            all_states
            , dtype=ControllerState)

        self.dim = len(self.controller_states)


class ActionSpaceStick:

    def __getitem__(self, item):
        return self.controller_states[item]

    def __init__(self, short_hop=2):
        """
        1 directions with no button
        2 tap b, lazer, plumbers downb
        3 tilts
        4 smashes with c stick and cardinals
        5 jumps, normal, front, back, full hop, short hop
        6 upbs
        7 L with all directions (rolls, spot dodge, shield, air dodges...)
        8 wavedashes
        9 shield grab, jump grab
        10 tap down b, tap side b
        11 noop

        """
        neutral = [
            (0., 0.)
        ]
        tilt_stick_states = [
            (-0.4, 0.0),
            (0.4, 0.0),
            (0.0, -0.4),
            (0.0, 0.4)
        ]

        wave_dash_sticks = [
            (-0.953, -0.294),
            (0.953, -0.294)
        ]

        shield_drop_sticks = [
            (0., -0.675),
        ]

        all_states = neutral + [
            (np.cos(x), np.sin(x)) for x in np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ] + tilt_stick_states + wave_dash_sticks + shield_drop_sticks

        self.controller_states = np.array(
            all_states
            , dtype=np.float32)

        self.dim = len(self.controller_states)


class ActionSpaceCStick:

    def __getitem__(self, item):
        return self.controller_states[item]

    def __init__(self):
        """
        1 directions with no button
        2 tap b, lazer, plumbers downb
        3 tilts
        4 smashes with c stick and cardinals
        5 jumps, normal, front, back, full hop, short hop
        6 upbs
        7 L with all directions (rolls, spot dodge, shield, air dodges...)
        8 wavedashes
        9 shield grab, jump grab
        10 tap down b, tap side b
        11 noop

        """

        all_states = [
            (np.cos(x), np.sin(x)) for x in np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ]  # + [(0., 0.)]

        self.controller_states = np.array(
            all_states
            , dtype=ControllerState)

        self.dim = len(self.controller_states)


char2kneebend = {}
for char in (
        Character.FOX,
        Character.POPO,
        Character.KIRBY,
        Character.SAMUS,
        Character.SHEIK,
        Character.PICHU,
        Character.PIKACHU
):
    char2kneebend[char] = 3

for char in (
        Character.MARIO,
        Character.DOC,
        Character.LUIGI,
        Character.YLINK,
        Character.CPTFALCON,
        Character.MARTH,
        Character.NESS,
        Character.GAMEANDWATCH
):
    char2kneebend[char] = 4

for char in (
        Character.FALCO,
        Character.JIGGLYPUFF,
        Character.PEACH,
        Character.YOSHI,
        Character.DK,
        Character.MEWTWO,
        Character.ROY,
):
    char2kneebend[char] = 5

for char in (
        Character.GANONDORF,
        Character.ZELDA,
        Character.LINK
):
    char2kneebend[char] = 6

char2kneebend[Character.BOWSER] = 8


class StickPosition(Enum):
    NEUTRAL = (0, 0)
    UP = (0, 1)
    DOWN = (0, -1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    UP_LEFT = (-0.707, 0.707)
    DOWN_LEFT = (-0.707, -0.707)
    DOWN_RIGHT = (0.707, -0.707)
    UP_RIGHT = (0.707, 0.707)

    UP_TILT = (0.0, 0.4)
    DOWN_TILT = (0.0, -0.4)
    RIGHT_TILT = (0.4, 0.0)
    LEFT_TILT = (-0.4, 0.0)

    WAVE_LEFT = (-0.953, -0.294)
    WAVE_RIGHT = (0.953, -0.294)

    # TODO
    # SHIELD_DROP1 = (0., -0.43)  # 0.675
    # SHIELD_DROP2 = (0., -0.44)  # 0.675


class ControllerInput:
    def __init__(self, buttons: Union[Button, Tuple[Button, Button]] = (), stick=StickPosition.NEUTRAL,
                 c_stick=StickPosition.NEUTRAL, analog_press=False, duration=3,
                 test_func=lambda game_state, char_state, curr_action: True,
                 energy_cost=1.):
        self.buttons = {buttons} if isinstance(buttons, Button) else set(buttons)
        self.stick = stick
        self.c_stick = c_stick
        self.analog_press = analog_press
        self.test_func = test_func

        self.duration = duration
        self.idx = 0
        self.energy_cost = energy_cost # cost per frame, so total cost is duration * energy_cost

        self.alternative_stick = None

    def reset(self):
        self.idx = 0
        self.alternative_stick = None

    def is_done(self):
        done = self.idx == self.duration
        if done:
            self.reset()

        return done

    def use(self):
        self.idx += 1

    def __repr__(self):
        return f"Action<{tuple(self.buttons)}{self.stick}*{self.c_stick}({self.idx}/{self.duration})>"


class NamedSequence:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Sequence[{self.name}]"

    def terminate(self):
        pass

    def is_done(self):
        return True

    def get_initial_stick(self, char_state: PlayerState):
        return StickPosition.NEUTRAL

    def advance(self, state):
        return ControllerInput()

    def allows_new_stick_inputs(self):
        return False


class InputSequence(NamedSequence):
    # TODO: different behavior if repeated
    # b mashing, first input the move, then you can input new sticks while in the mashing action
    def __init__(self, seq: Union[List[ControllerInput], ControllerInput], free_stick_at_frame=np.inf,
                 name: Any = "Undefined"):
        super().__init__(name)

        self.sequence = [seq] if isinstance(seq, ControllerInput) else seq
        self.sequence_length = seq.duration if isinstance(seq, ControllerInput) else sum([a.duration for a in seq])
        self.free_stick_at_frame = free_stick_at_frame

        self.idx = 0
        self.frame = 0

    def terminate(self):
        if self.idx < len(self.sequence):
            self.sequence[self.idx].reset()
        self.idx = len(self.sequence)
        self.frame = self.sequence_length

    def is_done(self):
        done = self.frame == self.sequence_length
        if done:
            self.idx = 0
            self.frame = 0

        return done

    def get_initial_stick(self, char_state: PlayerState):
        return self.sequence[0].stick

    def advance(self, char_state: PlayerState):
        action = self.sequence[self.idx]
        action.use()
        if action.is_done():
            self.idx += 1
        self.frame += 1
        return action

    def allows_new_stick_inputs(self):
        return self.frame >= self.free_stick_at_frame

    def __repr__(self):
        return f"Sequence[{self.name}({self.idx}/{len(self.sequence)})]"


class CharDependentInputSequence(NamedSequence):
    """
    mostly for char dependent actions
    """

    def __init__(self, char2input_sequence: Dict[Character, InputSequence], name="UnknownCharDependent"):
        super().__init__(name)
        self.table = char2input_sequence
        self.input_sequence: InputSequence = None

    def terminate(self):
        self.input_sequence.terminate()

    def is_done(self):
        done = self.input_sequence.is_done()
        if done:
            self.input_sequence = None
        return done

    def get_initial_stick(self, char_state: PlayerState):
        return self.table[char_state.character].get_initial_stick(char_state)

    def advance(self, char_state: PlayerState):
        if self.input_sequence is None:
            self.input_sequence = self.table[char_state.character]
        # return a copy so that we can mutate the object
        return self.input_sequence.advance(char_state)

    def allows_new_stick_inputs(self):
        return self.input_sequence.allows_new_stick_inputs()

    def __repr__(self):
        return f"Sequence[{self.name}({self.input_sequence})]"


class InputQueue:
    def __init__(self):
        self.queue: Deque[InputSequence] = deque()
        self.current_action: InputSequence = InputSequence([])
        self.current_action_stick_only: Union[tuple, None] = None
        self.waiting_for_next_input = False

    def push(self, input_sequence: InputSequence):
        self.queue.append(input_sequence)
        if len(self.queue) > 1:
            # TODO : no need of an actual queue right ?
            self.queue.popleft()

        assert len(self.queue) < 4, f"Queue is not getting emptied ? {self.queue}"

    def pull(self, should_deque: bool, char_state: PlayerState) -> Union[Tuple[ControllerInput, InputSequence], None]:

        if self.waiting_for_next_input or self.current_action.is_done():
            if should_deque:
                assert len(self.queue) > 0, "no more actions, not expected"
                self.current_action = self.queue.popleft()
                self.current_action_stick_only = None
                self.waiting_for_next_input = False
            else:
                # Wait for the next "every 3 frame timing"
                # TODO : we might have terminated an action (e.g. wavedash,
                #  but can't act for a while, should not matter)

                self.waiting_for_next_input = True
                return
        elif self.current_action.allows_new_stick_inputs():
            if ((self.current_action_stick_only and should_deque)
                    # We can move sticks even though we are not on the "every 3 frame timing", dequeue an action
                    # if we did not dequeue already in the 3 frame window.
                    or (not self.current_action_stick_only)
            ):
                self.current_action_stick_only = self.queue.popleft().get_initial_stick(char_state)

        else:
            # We are stepping on the executed action
            pass

        action = self.current_action.advance(char_state)
        if self.current_action_stick_only:
            action.alternative_stick = self.current_action_stick_only

        return action, self.current_action


class ActionControllerInterface:

    def __init__(self):
        pass

    @staticmethod
    def send_controller(action: ControllerInput, pad: ComboPad, game_state, char_state, current_action_sequence):

        if action.test_func(game_state, char_state, current_action_sequence):

            to_release = pad.previous_state.buttons - action.buttons
            press_new = action.buttons - pad.previous_state.buttons

            if action.analog_press:
                pad.press_shoulder(button=Button.BUTTON_L, amount=1.)
            elif pad.previous_state.analog_press and not action.analog_press:
                pad.press_shoulder(button=Button.BUTTON_L, amount=0.)

            for button in to_release:
                pad.release_button(button)
            for button in press_new:
                pad.press_button(button)

            relevant_stick: StickPosition = action.stick if action.alternative_stick is None \
                else action.alternative_stick

            pad.tilt_analog_unit(enums.Button.BUTTON_MAIN, *relevant_stick.value)
            pad.tilt_analog_unit(enums.Button.BUTTON_C, *action.c_stick.value)

            pad.previous_state = action


def disable_on_ground(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = not char_state.on_ground
    if not allow:
        curr_action.terminate()
    return allow


def disable_in_air(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = char_state.on_ground
    if not allow:
        curr_action.terminate()
    return allow

def disable_on_shield(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = not char_state.action in (Action.SHIELD_START, Action.SHIELD, Action.SHIELD_STUN) # TODO: shield release as well ?
    if not allow:
        curr_action.terminate()
    return allow


def disable_on_shield_air(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = (not char_state.action in (Action.SHIELD_START, Action.SHIELD, Action.SHIELD_STUN)) and char_state.on_ground # TODO: shield release as well ?
    if not allow:
        curr_action.terminate()
    return allow

def check_kneebend(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = char_state.action == Action.KNEE_BEND
    if not allow:
        curr_action.terminate()
    return allow

def allow_shield_drop1(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = char_state.on_ground and char_state.y > 16
    if not allow:
        curr_action.terminate()
    return allow

def allow_shield_drop(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = char_state.action in (Action.SHIELD_START, Action.SHIELD, Action.SHIELD_REFLECT, Action.SHIELD_STUN) and char_state.y > 8
    if not allow:
        curr_action.terminate()
    return allow

def allow_waveland(game_state: GameState, char_state: PlayerState, curr_action: InputSequence):

    allow = not (char_state.off_stage or char_state.on_ground)
    if allow:
        dy_ground = char_state.position.y
        allow = dy_ground < 3
        if not allow:
            for (y, x1, x2) in [stages.top_platform_position(game_state.stage),
                                stages.side_platform_position(False, game_state),
                                stages.side_platform_position(True, game_state)
            ]:
                dy = char_state.position.y - y
                # we are above a platform
                allow = (dy<3 and x1 < char_state.position.x < x2)
                if allow:
                    break
    if not allow:
        curr_action.terminate()
    return allow

def allow_wavedash(game_state, char_state: PlayerState, curr_action: InputSequence):
    dist_from_ledge = abs(abs(char_state.position.x)-stages.EDGE_GROUND_POSITION[game_state.stage])
    allow = char_state.action == Action.KNEE_BEND and (dist_from_ledge > 4)
    if not allow:
        curr_action.terminate()
    return allow

def allow_l_down(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = allow_waveland(game_state, char_state, curr_action)
    allow = allow or char_state.on_ground
    if not allow:
        curr_action.terminate()
    return allow



# SPECIFIC TO DOC AND MARIO, multishine for fox and falco too OP
# Tap B for other chars
TORNARDO_FRAMES = 37



# check if still in tornado, else terminate
# intuitively set to not be costly, because the action is long
# free stick pos after 3 frames
def allow_tornado(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = (char_state.action == Action.SWORD_DANCE_2_HIGH) and char_state.off_stage
    if not allow:
        curr_action.terminate()
    return allow

def allow_tornado_init(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = char_state.off_stage or (char_state.character == Character.LUIGI)
    if not allow:
        curr_action.terminate()
    return allow

def allow_jc_grab(game_state, char_state: PlayerState, curr_action: InputSequence):
    allow = char_state.action in (Action.RUNNING, Action.DASHING, Action.WALK_FAST, Action.WALK_MIDDLE)
    if not allow:
        curr_action.terminate()
    return allow



def debug(game_state, char_state: PlayerState, curr_action: InputSequence):
    print("debuging action", char_state.action, char_state.on_ground)
    return True


MARIO_TORNADO = []
while len(MARIO_TORNADO) < 41:  # actually 37
    if len(MARIO_TORNADO) > 0: # 6
        test_func = allow_tornado
    else:
        test_func = allow_tornado_init
    MARIO_TORNADO.extend(
        [ControllerInput(buttons=Button.BUTTON_B, stick=StickPosition.DOWN, duration=1,
                         test_func=test_func),
         ControllerInput(duration=1, stick=StickPosition.DOWN, test_func=test_func)])
    if len(MARIO_TORNADO) == 42:
        MARIO_TORNADO.pop()


class SSBMActionSpace:
    # TODO : Can remove simple down-b
    # check for wavedash, they look weird
    # check for mario action state in tornado, not same as doc ?

    RESET_CONTROLLER = lambda _: InputSequence(ControllerInput(energy_cost=0.))
    LEFT = lambda _: InputSequence(ControllerInput(stick=StickPosition.LEFT, energy_cost=0.))
    RIGHT = lambda _: InputSequence(ControllerInput(stick=StickPosition.RIGHT, energy_cost=0.))
    DOWN = lambda _: InputSequence(ControllerInput(stick=StickPosition.DOWN, energy_cost=0.))
    UP = lambda _: InputSequence(ControllerInput(stick=StickPosition.UP, energy_cost=0.))
    UP_LEFT = lambda _: InputSequence(ControllerInput(stick=StickPosition.UP_LEFT, energy_cost=0.))
    UP_RIGHT = lambda _: InputSequence(ControllerInput(stick=StickPosition.UP_RIGHT, energy_cost=0.))
    DOWN_LEFT = lambda _: InputSequence(ControllerInput(stick=StickPosition.DOWN_LEFT, energy_cost=0.))
    DOWN_RIGHT = lambda _: InputSequence(ControllerInput(stick=StickPosition.DOWN_RIGHT, energy_cost=0.))
    A_NEUTRAL = lambda _: InputSequence(ControllerInput(buttons=Button.BUTTON_A))
    #Disable this when in the air, its the same as c stick
    TILT_UP = lambda _: InputSequence(
        [ControllerInput(buttons=Button.BUTTON_A, stick=StickPosition.UP_TILT, duration=2, test_func=disable_in_air),
         ControllerInput(test_func=disable_in_air, duration=1),
         ]
    )
    TILT_DOWN = lambda _: InputSequence(
        [ControllerInput(buttons=Button.BUTTON_A, stick=StickPosition.DOWN_TILT, duration=2, test_func=disable_in_air),
         ControllerInput(test_func=disable_in_air, duration=1),
         ]
    )
    TILT_LEFT = lambda _: InputSequence(
        [ControllerInput(buttons=Button.BUTTON_A, stick=StickPosition.LEFT_TILT, duration=2, test_func=disable_in_air),
         ControllerInput(test_func=disable_in_air, duration=1),
         ]
    )
    TILT_RIGHT = lambda _: InputSequence(
        [ControllerInput(buttons=Button.BUTTON_A, stick=StickPosition.RIGHT_TILT, duration=2, test_func=disable_in_air),
         ControllerInput(test_func=disable_in_air, duration=1),
         ]
    )

    # TODO: cannot work if we are already shielding
    SHIELD_DROP_LEFT = lambda _: InputSequence([
        ControllerInput(stick=StickPosition.LEFT, test_func=disable_on_shield_air, duration=1),
        ControllerInput(stick=StickPosition.LEFT, buttons=Button.BUTTON_L, test_func=allow_shield_drop1, duration=1),
        ControllerInput(buttons=Button.BUTTON_L, stick=StickPosition.DOWN_LEFT, duration=1, test_func=allow_shield_drop),
    ])

    B_NEUTRAL = lambda _: InputSequence(ControllerInput(buttons=Button.BUTTON_B))

    # Do it depending on the char ?
    # B_UP = lambda _: InputSequence(ControllerInput(buttons=Button.BUTTON_B, stick=StickPosition.UP, duration=3))
    B_UP_LEFT = lambda _: InputSequence([
        ControllerInput(buttons=Button.BUTTON_B, stick=StickPosition.UP, duration=2),
        # allows reversed up-b
        ControllerInput(stick=StickPosition.LEFT, duration=1),
    ])
    B_UP_RIGHT = lambda _: InputSequence([
        ControllerInput(buttons=Button.BUTTON_B, stick=StickPosition.UP, duration=2),
        # allows reversed up-b
        ControllerInput(stick=StickPosition.RIGHT, duration=1),
    ])
    #########B_DOWN = lambda _: InputSequence(ControllerInput(buttons=Button.BUTTON_B, stick=StickPosition.DOWN))
    B_LEFT = lambda _: InputSequence(ControllerInput(buttons=Button.BUTTON_B, stick=StickPosition.LEFT))
    B_RIGHT = lambda _: InputSequence(ControllerInput(buttons=Button.BUTTON_B, stick=StickPosition.RIGHT))
    C_UP = lambda _: InputSequence(ControllerInput(c_stick=StickPosition.UP))
    C_RIGHT = lambda _: InputSequence(ControllerInput(c_stick=StickPosition.RIGHT))
    C_LEFT = lambda _: InputSequence(ControllerInput(c_stick=StickPosition.LEFT))
    C_DOWN = lambda _: InputSequence(ControllerInput(c_stick=StickPosition.DOWN))

    # using this on ground makes you jump
    L_UP = lambda _: InputSequence(
        ControllerInput(buttons=Button.BUTTON_L, stick=StickPosition.UP, test_func=disable_on_ground))

    # We can use this for puff, or to waveland with no angle
    L_DOWN = lambda _: InputSequence(ControllerInput(buttons=Button.BUTTON_L, stick=StickPosition.DOWN, test_func=allow_l_down))

    # Only allow this when on ground ? No for basic L cancel
    L_NEUTRAL = lambda _: InputSequence(
       ControllerInput(buttons=Button.BUTTON_L, test_func=disable_in_air, energy_cost=0.1))
    # For teching and l cancel without windows
    L_NEUTRAL_LIGHT = lambda _: InputSequence(
        ControllerInput(analog_press=True)) # was disabled in air
    L_RIGHT = lambda _: InputSequence(
        ControllerInput(buttons=Button.BUTTON_L, stick=StickPosition.RIGHT, test_func=disable_in_air))
    L_LEFT = lambda _: InputSequence(
        ControllerInput(buttons=Button.BUTTON_L, stick=StickPosition.LEFT, test_func=disable_in_air))

    # Do not use those actions on ground, this is the same as L_LEFT and L_RIGHT otherwise
    WAVELAND_LEFT = lambda _: InputSequence(ControllerInput(buttons=Button.BUTTON_L, stick=StickPosition.WAVE_LEFT,
                                                            test_func=allow_waveland, energy_cost=0.))
    WAVELAND_RIGHT = lambda _: InputSequence(ControllerInput(buttons=Button.BUTTON_L, stick=StickPosition.WAVE_RIGHT,
                                                             test_func=allow_waveland, energy_cost=0.))

    # Hook, Z-cancel, NAIR
    Z = lambda _: InputSequence(ControllerInput(buttons=Button.BUTTON_Z, test_func=disable_on_ground))

    JC_GRAB = lambda _: InputSequence([
        ControllerInput(buttons=Button.BUTTON_X, duration=2, test_func=allow_jc_grab),
        ControllerInput(buttons=Button.BUTTON_Z, duration=1, test_func=allow_jc_grab),
    ])
    SHIELD_GRAB = lambda _: InputSequence(
        ControllerInput(buttons=(Button.BUTTON_A, Button.BUTTON_L), test_func=disable_in_air),
    )
    FULL_HOP_NEUTRAL = lambda _: CharDependentInputSequence(
        {
            character: InputSequence(
                ControllerInput(buttons=Button.BUTTON_X, duration=short_hop_frames, energy_cost=0.), name=character)
            for character, short_hop_frames in char2kneebend.items()
        }
    )
    SHORT_HOP_NEUTRAL = lambda _: InputSequence(
        [ControllerInput(buttons=Button.BUTTON_X, duration=2, energy_cost=0.),
         ControllerInput(duration=1, energy_cost=0.)]
    )
    FULL_HOP_LEFT = lambda _: CharDependentInputSequence(
        {
            character: InputSequence(
                ControllerInput(buttons=Button.BUTTON_X, duration=short_hop_frames, stick=StickPosition.LEFT,
                                energy_cost=0.), name=character)
            for character, short_hop_frames in char2kneebend.items()
        }
    )
    SHORT_HOP_LEFT = lambda _: InputSequence(
        [ControllerInput(buttons=Button.BUTTON_X, duration=2, stick=StickPosition.LEFT, energy_cost=0.),
         ControllerInput(duration=1, stick=StickPosition.LEFT, energy_cost=0.)]
    )

    FULL_HOP_RIGHT = lambda _: CharDependentInputSequence(
        {
            character: InputSequence(
                ControllerInput(buttons=Button.BUTTON_X, duration=short_hop_frames, stick=StickPosition.RIGHT,
                                energy_cost=0.), name=character)
            for character, short_hop_frames in char2kneebend.items()
        }
    )
    SHORT_HOP_RIGHT = lambda _: InputSequence(
        [ControllerInput(buttons=Button.BUTTON_X, duration=2, stick=StickPosition.RIGHT, energy_cost=0.),
         ControllerInput(duration=1, stick=StickPosition.RIGHT, energy_cost=0.)]
    )

    # TODO: only allow when on ground ?
    # TODO: debug, short_hop_frames+1
    # Pressing for short_hop_frames seems to lead to short hop, do we need one more frame then ?
    WAVEDASH_LEFT = lambda _: CharDependentInputSequence(
        {
            character: InputSequence([
                ControllerInput(buttons=Button.BUTTON_X, duration=short_hop_frames+1, stick=StickPosition.WAVE_LEFT,
                                test_func=disable_on_shield_air, energy_cost=0.),
                ControllerInput(buttons=Button.BUTTON_L, duration=1, stick=StickPosition.WAVE_LEFT,
                                test_func=allow_wavedash, energy_cost=0.),
            ], free_stick_at_frame=short_hop_frames + 2, name=character)
            for character, short_hop_frames in char2kneebend.items()
        }
    )
    WAVEDASH_RIGHT = lambda _: CharDependentInputSequence(
        {
            character: InputSequence([
                ControllerInput(buttons=Button.BUTTON_X, duration=short_hop_frames+1, stick=StickPosition.WAVE_RIGHT,
                                test_func=disable_on_shield_air, energy_cost=0.),
                ControllerInput(buttons=Button.BUTTON_L, duration=1, stick=StickPosition.WAVE_RIGHT,
                                test_func=allow_wavedash, energy_cost=0.),
            ], free_stick_at_frame=short_hop_frames + 2, name=character)
            for character, short_hop_frames in char2kneebend.items()
        }
    )
    WAVEDASH_NEUTRAL = lambda _: CharDependentInputSequence(
        {
            character: InputSequence([
                ControllerInput(buttons=Button.BUTTON_X, duration=short_hop_frames+1, stick=StickPosition.DOWN,
                                test_func=disable_on_shield_air, energy_cost=0.),
                ControllerInput(buttons=Button.BUTTON_L, duration=1, stick=StickPosition.DOWN,
                                test_func=check_kneebend, energy_cost=0.),
            ], free_stick_at_frame=short_hop_frames + 2, name=character)
            for character, short_hop_frames in char2kneebend.items()
        }
    )
    WAVEDASH_SLIGHT_LEFT = lambda _: CharDependentInputSequence(
        {
            character: InputSequence([
                ControllerInput(buttons=Button.BUTTON_X, duration=short_hop_frames+1, stick=StickPosition.DOWN_LEFT,
                                test_func=disable_on_shield_air, energy_cost=0.),
                ControllerInput(buttons=Button.BUTTON_L, duration=1, stick=StickPosition.DOWN_LEFT,
                                test_func=allow_wavedash, energy_cost=0.),
            ], free_stick_at_frame=short_hop_frames + 2, name=character)
            for character, short_hop_frames in char2kneebend.items()
        }
    )
    WAVEDASH_SLIGHT_RIGHT = lambda _: CharDependentInputSequence(
        {
            character: InputSequence([
                ControllerInput(buttons=Button.BUTTON_X, duration=short_hop_frames+1, stick=StickPosition.DOWN_RIGHT,
                                test_func=disable_on_shield_air, energy_cost=0.),
                ControllerInput(buttons=Button.BUTTON_L, duration=1, stick=StickPosition.DOWN_RIGHT,
                                test_func=allow_wavedash, energy_cost=0.),
            ], free_stick_at_frame=short_hop_frames + 2, name=character)
            for character, short_hop_frames in char2kneebend.items()
        }
    )

    B_DOWN_MASH = lambda _: CharDependentInputSequence({

        character: InputSequence(MARIO_TORNADO, free_stick_at_frame=3, name=character) if character in (
            Character.MARIO, Character.LUIGI, Character.DOC)
        else
        # TODO: char specific move: gentleman, charged neutral b, etc.
        # for now, down b (no other action for down b)
        InputSequence([ControllerInput(buttons=Button.BUTTON_B, stick=StickPosition.DOWN, duration=2),
                       ControllerInput(duration=1),
                       ], name=character)

        for character in Character
    })

    def __init__(self):

        to_register = list()
        self._registered = set()
        for member in dir(self):
            actual_member = getattr(self, member)
            if callable(actual_member) and actual_member.__name__ == "<lambda>":
                inst = actual_member()
                inst.name = member
                to_register.append(inst)
                self._registered.add(member)
            else:
                pass

        self._by_idx = {
            i: action for i, action in enumerate(to_register)
        }
        self._by_name = {
            action.name: action for action in to_register
        }
        self.n = len(to_register)
        self.gym_spec = Discrete(self.n)

    def __getitem__(self, item):
        if isinstance(item, (np.int64, np.int32, int)):
            return self._by_idx.get(item, None)
        elif isinstance(item, str):
            return self._by_name.get(item, None)
        else:
            print(item)
            raise NotImplementedError

    def __getattribute__(self, item):
        if item == "_registered":
            return super().__getattribute__(item)
        if item in self._registered:
            return self._by_name[item]
        else:
            return super().__getattribute__(item)

    def __repr__(self):
        return self._by_idx.__repr__()


if __name__ == '__main__':
    action_space = SSBMActionSpace()
    action_space2 = SSBMActionSpace()

    print(action_space2)
    print(action_space)
