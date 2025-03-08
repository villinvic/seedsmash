from typing import Set

import numpy as np
from melee import Character, PlayerState, GameState

from melee.enums import LuigiMoves, DKMoves, SamusMoves, MewtwoMoves, MarioMoves, DocMoves, GameAndWatchMoves

class CharacterSpecificObservation:

    def __init__(self, delay: int):
        self.delay = delay

    def get(self):
        return 0.

    def update(
            self,
            player: PlayerState,
            gamestate: GameState
    ):
        return self.get()


class MarioObservations(CharacterSpecificObservation):

    def __init__(self, delay: int):
        super().__init__(delay)
        self.tornardo_charge = 1

    def get(self):
        return self.tornardo_charge

    def update(
            self,
            player: PlayerState,
            gamestate: GameState
    ):
        # We are on ground and performing cyclone, gets a charge
        if player.on_ground: # TODO: do we recover it when dying ?
            self.tornardo_charge = 1

        # airborne on frame 44 of cyclone, loses the charge
        elif player.action.value == MarioMoves.TornadoAir.value:
            self.tornardo_charge = 0

        return super().update(player, gamestate)


class DocObservations(CharacterSpecificObservation):

    def __init__(self, delay: int):
        super().__init__(delay)
        self.tornardo_charge = 1

    def get(self):
        return self.tornardo_charge

    def update(
            self,
            player: PlayerState,
            gamestate: GameState
    ):
        # We are on ground and performing cyclone, gets a charge
        if player.on_ground: # TODO: do we recover it when dying ?
            self.tornardo_charge = 1

        # airborne on frame 44 of cyclone, loses the charge
        elif player.action.value == DocMoves.TornadoAir.value:
            self.tornardo_charge = 0

        return super().update(player, gamestate)


class LuigiObservations(CharacterSpecificObservation):

    def __init__(self, delay: int):
        super().__init__(delay)
        self.cyclone_charge = 0

    def get(self):
        return self.cyclone_charge

    def update(
            self,
            player: PlayerState,
            gamestate: GameState
    ):
        # We are on ground and performing cyclone, gets a charge
        if player.on_ground and player.action.value == LuigiMoves.CycloneGround.value:
            self.cyclone_charge = 1 # does not lose charge when KOed

        # airborne on frame 44 of cyclone, loses the charge
        elif ((not player.on_ground) and player.action.value == LuigiMoves.CycloneAir.value
              and player.action_frame == 44):
            self.cyclone_charge = 0

        return super().update(player, gamestate)


class ChargeObservation(CharacterSpecificObservation):
    def __init__(
            self,
            delay: int,
            charge_frame: int,
            max_charge: int,
            charging_move_values: Set[int],
            discharging_moves: Set[int],
            canceling_moves: Set[int] | None= None,
    ):
        super().__init__(delay)
        self.charges = 0
        self.charge_frame = charge_frame
        self.max_charge = max_charge
        self.charging_moves = charging_move_values
        self.discharging_moves = discharging_moves
        self.canceling_moves = {} if canceling_moves is None else canceling_moves

        self.prev_action = 0

    def get(self):
        return self.charges / self.max_charge

    def update(
            self,
            player: PlayerState,
            gamestate: GameState
    ):
        # KO, used charges or canceled upb, remove charges
        if (player.action.value <= 0xa or player.action.value in self.discharging_moves or
            (self.prev_action in (self.canceling_moves | self.charging_moves) and player.hitstun_frames_left > 0)
        ):
            self.charges = 0

        elif player.action.value in self.charging_moves and player.action_frame == self.charge_frame:
            self.charges += 1
            # TODO, looks like if you cancel at the "charge frame exactly, you do not get the charge" ?
            if self.charges > self.max_charge:
                print("exceeded number of possible charges somehow", player.character, self.charges)
                self.charges = self.max_charge

        self.prev_action = player.action.value

        return super().update(player, gamestate)



class DKObservations(ChargeObservation):

    def __init__(self, delay: int):
        super().__init__(
            delay,
            charge_frame=11,
            max_charge=10,
            charging_move_values={
                DKMoves.GiantPunchAirChargeLoop.value,
                DKMoves.GiantPunchGroundChargeLoop.value
            },
            discharging_moves={
                DKMoves.GiantPunchAirEarlyPunch.value,
                DKMoves.GiantPunchGroundEarlyPunch.value,
                DKMoves.GiantPunchGroundFullChargePunch.value,
                DKMoves.GiantPunchAirFullChargePunch.value
            },
            canceling_moves={
                DKMoves.SpinningKongAir.value,
                DKMoves.SpinningKongGround.value
            }
        )


class SamusObservations(ChargeObservation):

    def __init__(self, delay: int):
        super().__init__(
            delay,
            charge_frame=10,
            max_charge=11,
            charging_move_values={
                SamusMoves.ChargeShotGroundLoop.value,
            },
            discharging_moves={
                SamusMoves.ChargeShotGroundFire.value,
                SamusMoves.ChargeShotAirFire.value,
            },
            canceling_moves={
                SamusMoves.ScrewAttackAir.value,
                SamusMoves.ScrewAttackGround.value
            }
        )


class MewTwoObservations(ChargeObservation):

    def __init__(self, delay: int):
        super().__init__(
            delay,
            charge_frame=8,
            max_charge=14,
            charging_move_values={
                MewtwoMoves.ShadowBallAirChargeLoop.value,
                MewtwoMoves.ShadowBallGroundChargeLoop.value
            },
            discharging_moves={
                MewtwoMoves.ShadowBallAirFire.value,
                MewtwoMoves.ShadowBallGroundFire.value,
            }
        )


class GameAndWatchObservations(CharacterSpecificObservation):

    JUDGEMENT_VALUES = {
        GameAndWatchMoves.Judgment1Air.value: 1,
        GameAndWatchMoves.Judgment1Ground.value: 1,
        GameAndWatchMoves.Judgment2Air.value: 2,
        GameAndWatchMoves.Judgment2Ground.value: 2,
        GameAndWatchMoves.Judgment3Air.value: 3,
        GameAndWatchMoves.Judgment3Ground.value: 3,
        GameAndWatchMoves.Judgment4Air.value: 4,
        GameAndWatchMoves.Judgment4Ground.value: 4,
        GameAndWatchMoves.Judgment5Air.value: 5,
        GameAndWatchMoves.Judgment5Ground.value: 5,
        GameAndWatchMoves.Judgment6Air.value: 6,
        GameAndWatchMoves.Judgment6Ground.value: 6,
        GameAndWatchMoves.Judgment7Air.value: 7,
        GameAndWatchMoves.Judgment7Ground.value: 7,
        GameAndWatchMoves.Judgment8Air.value: 8,
        GameAndWatchMoves.Judgment8Ground.value: 8,
        GameAndWatchMoves.Judgment9Air.value: 9,
        GameAndWatchMoves.Judgment9Ground.value: 9,
    }

    def __init__(self, delay: int):
        super().__init__(delay)
        self.judgment_store = []

    def get(self):
        if 9 in self.judgment_store:
            return 0
        return 7/(9 - len(self.judgment_store))

    def update(
            self,
            player: PlayerState,
            gamestate: GameState
    ):
        # reset store on death
        if player.action.value < 0xa:
            self.judgment_store = []

        if player.action_frame == 1:
            action_value = player.action.value
            if action_value in GameAndWatchObservations.JUDGEMENT_VALUES:
                self.judgment_store.append(GameAndWatchObservations.JUDGEMENT_VALUES[action_value])
                if len(self.judgment_store) > 2:
                    self.judgment_store.pop(0)



# TODO: sheik

specific_observations = {
    Character.MARIO: MarioObservations,
    Character.DOC: DocObservations,
    Character.LUIGI: LuigiObservations,
    Character.DK: DKObservations,
    Character.SAMUS: SamusObservations,
    Character.MEWTWO: MewTwoObservations,
    Character.GAMEANDWATCH: GameAndWatchObservations
}

def get_character_specific_observations(character: Character, delay: int):
    return specific_observations.get(character, CharacterSpecificObservation)(delay)
