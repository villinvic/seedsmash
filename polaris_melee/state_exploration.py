from typing import Set, Tuple

import numpy as np
from collections import defaultdict

import tree
from melee import Character, Action, MarioMoves, Moves, BowserMoves, CaptainFalconMoves, DKMoves, DocMoves, FalcoMoves, \
    FoxMoves, GameAndWatchMoves, GanonMoves, JigglypuffMoves, LinkMoves, LuigiMoves, MarthMoves, MewtwoMoves, NessMoves, \
    PeachMoves, PichuMoves, PikachuMoves, PopoMoves, RoyMoves, SamusMoves, SheikMoves, YoshiMoves, YoungLinkMoves, \
    ZeldaMoves, character_moves


def move_to_action(move):
    if isinstance(move, Action):
        return move
    return Action(move.value)

explorable_states = {
    #Action.DASHING,
    #Action.RUNNING,

    Action.ITEM_PICKUP_LIGHT,
    Action.ITEM_PICKUP_HEAVY,
    Action.SHIELD_STUN,
    Action.PLATFORM_DROP,

    # no running grabs
    Action.GRAB_PULLING,

    # tactical states
    Action.NEUTRAL_TECH,
    Action.FORWARD_TECH,
    Action.BACKWARD_TECH,
    Action.WALL_TECH,
    Action.WALL_TECH_JUMP,
    Action.CEILING_TECH,

    # normal rolls should be more easily learned, skipping

}
# item throw/ pickup, helps for item centric chars like Link
for action_id in range(0x5D, 0x78):
    explorable_states.add(Action(action_id))

def set2dict(s):
    return {
                e.name: e
                for e in s
            }

class ExplorableActions:

    def __init__(
            self,
            explorables: set | dict
    ):

        if isinstance(explorables, set):
            explorables = set2dict(explorables)

        explorables = tree.map_structure(
            move_to_action,
            explorables | set2dict(explorable_states)
        )
        self.explorables = {}

        for name, explorable in explorables.items():
            if isinstance(explorable, tuple):
                for action in explorable:
                    self.explorables[action] = name
            else:
                self.explorables[explorable] = name

        self.explorable_names = set(self.explorables.values())

    def __getitem__(self, item):
        return self.explorables.get(item)

    def __len__(self):
        return len(self.explorable_names)

    def get_counter(self):
        return ActionCounter(self)

    # def merge(self, other):
    #     new = ExplorableActions(explorables=set())
    #     new.merged = self.merged | other.merged
    #     new.explorables = self.explorables | other.explorables
    #     return new


class ActionCounter:
    def __init__(self, explorables: ExplorableActions):
        self.explorables = explorables

        self.counts = {
            explorable: 0
            for explorable in self.explorables.explorable_names
        }

    def count(self, action_name: str):
        if action_name is not None:
            self.counts[action_name] += 1

    def action_name(self, action: Action):
        return self.explorables[action]

    def get(self):
        return self.counts


base_moves = {
    "Jab": (Moves.Jab1, Moves.Jab2, Moves.Jab3, Moves.RapidJabs),
    "Dash Attack": Moves.DashAttack,
    "Ftilt": Moves.FrontTilt,
    "Utilt": Moves.UpTilt,
    "Dtilt": Moves.DownTilt,
    "Smashes": (Moves.FrontSmash, Moves.UpSmash, Moves.DownSmash),
    "Nair": Moves.Nair,
    "Fair": Moves.Fair,
    "Bair": Moves.Bair,
    "Uair": Moves.Uair,
    "Dair": Moves.Dair,
    "Pummel": Moves.Pummel,
    "Fthrow": Moves.ForwardThrow,
    "Bthrow": Moves.BackThrow,
    "Uthrow": Moves.UpThrow,
    "Dthrow": Moves.DownThrow,
}

bowser_explorable_moves = ExplorableActions(base_moves | {
    "Getup Attack (Below 100)": Moves.LedgeGetUpAttack,
    "Fire Breath": (BowserMoves.FireBreathGroundStartup, BowserMoves.FireBreathAirStartup,
     BowserMoves.FireBreathAirLoop, BowserMoves.FireBreathGroundLoop,
     BowserMoves.FireBreathGroundEnd, BowserMoves.FireBreathAirEnd),
    "KoopaKlaw Pummel": (BowserMoves.KoopaKlawGroundPummel, BowserMoves.KoopaKlawAirPummel),
    "KoopaKlaw Fthrow": (BowserMoves.KoopaKlawGroundThrowF, BowserMoves.KoopaKlawAirThrowF),
    "KoopaKlaw Bthrow": (BowserMoves.KoopaKlawGroundThrowB, BowserMoves.KoopaKlawAirThrowB),
    "Whirling Fortress (Ground)": (BowserMoves.WhirlingFortressGround, BowserMoves.WhirlingFortressAir)
})

captain_falcon_explorable_moves = ExplorableActions(base_moves | {
    "Falcon Punch": (CaptainFalconMoves.FalconPunchAir, CaptainFalconMoves.FalconPunchGround),
    "Raptor Boost (Air)": CaptainFalconMoves.RaptorBoostAirHit,
    "Raptor Boost (Ground)": CaptainFalconMoves.RaptorBoostGroundHit,
    "Falcon Dive": CaptainFalconMoves.FalconDiveCatch,
    "Flacon Kick (Ending in Air)": CaptainFalconMoves.FalconKickAirEndingInAir,
    "Falcon Kick (Ground)": CaptainFalconMoves.FalconKickGround
})

dk_explorable_moves = ExplorableActions(base_moves | {
    "Giant Punch (Early Punch)": (DKMoves.GiantPunchGroundEarlyPunch, DKMoves.GiantPunchAirEarlyPunch, DKMoves.GiantPunchAirFullChargePunch, DKMoves.GiantPunchGroundFullChargePunch),
    #"Giant Punch (Full Charge)": (DKMoves.GiantPunchAirFullChargePunch, DKMoves.GiantPunchGroundFullChargePunch),
    "Headbutt": (DKMoves.HeadbuttGround, DKMoves.HeadbuttGround),
    "Spinning Kong": DKMoves.SpinningKongGround,
    "KongKarry Fthrow": (DKMoves.KongKarryGroundThrowForward, DKMoves.KongKarryAirThrowForward),
    "KongKarry Bthrow": (DKMoves.KongKarryGroundThrowBackward, DKMoves.KongKarryAirThrowBackward),
    "KongKarry Uthrow": (DKMoves.KongKarryGroundThrowUp, DKMoves.KongKarryAirThrowUp),
    "KongKarry Dthrow": (DKMoves.KongKarryGroundThrowDown, DKMoves.KongKarryAirThrowDown),
})

doc_explorable_moves = ExplorableActions(base_moves | {
    "Megavitamin": (DocMoves.MegavitaminAir, DocMoves.MegavitaminGround),
    "Super Sheet": (DocMoves.SuperSheetAir, DocMoves.SuperSheetGround),
    "Super Jump Punch": (DocMoves.SuperJumpPunchGround, DocMoves.SuperJumpPunchAir),
    "Tornado": DocMoves.TornadoAir
})

falco_explorable_moves = ExplorableActions(base_moves | {
    "Blaster": (FalcoMoves.BlasterAirStartup, FalcoMoves.BlasterAirEnd, FalcoMoves.BlasterAirLoop),
    "Phantasm": (FalcoMoves.PhantasmGroundStartup, FalcoMoves.PhantasmGround, FalcoMoves.PhantasmGroundEnd,
                 FalcoMoves.PhantasmStartupAir, FalcoMoves.PhantasmAir, FalcoMoves.PhantasmAirEnd),
    "Reflector": (FalcoMoves.ReflectorGroundStartup, FalcoMoves.ReflectorGroundLoop, FalcoMoves.ReflectorGroundEnd,
                          FalcoMoves.ReflectorAirStartup, FalcoMoves.ReflectorAirLoop, FalcoMoves.ReflectorAirEnd),
})

fox_explorable_moves = ExplorableActions(base_moves | {
    "Blaster": (FoxMoves.BlasterAirStartup, FoxMoves.BlasterAirEnd, FoxMoves.BlasterAirLoop,
                FoxMoves.BlasterGroundEnd, FoxMoves.BlasterGroundStartup, FoxMoves.BlasterGroundLoop),
    "Illusion": (FoxMoves.IllusionGroundStartup, FoxMoves.IllusionGround, FoxMoves.IllusionGroundEnd, FoxMoves.IllusionStartupAir, FoxMoves.IllusionAir, FoxMoves.IllusionAirEnd),
    "Reflector": (FoxMoves.ReflectorGroundStartup, FoxMoves.ReflectorGroundLoop, FoxMoves.ReflectorGroundEnd,
                  FoxMoves.ReflectorAirStartup, FoxMoves.ReflectorAirLoop, FoxMoves.ReflectorAirEnd),
})

game_and_watch_explorable_moves = ExplorableActions({
    "Dash Attack": Moves.DashAttack,
    "Ftilt": Moves.FrontTilt,
    "Utilt": Moves.UpTilt,
    "Dair": Moves.Dair,
    "Pummel": Moves.Pummel,
    "Fthrow": Moves.ForwardThrow,
    "Dthrow": Moves.DownThrow,
    "Uthrow": Moves.UpThrow,
    "Bthrow": Moves.BackThrow,
    "Jab": (GameAndWatchMoves.Jab, GameAndWatchMoves.RapidJabsStart, GameAndWatchMoves.RapidJabsLoop, GameAndWatchMoves.RapidJabsEnd),
    "Dtilt": GameAndWatchMoves.DownTilt,
    "Nair": GameAndWatchMoves.Nair,
    "Bair": GameAndWatchMoves.Bair,
    "Uair": GameAndWatchMoves.Uair,
    "Chef": (GameAndWatchMoves.ChefAir, GameAndWatchMoves.ChefGround),
    "Judgement 1-8": (GameAndWatchMoves.Judgment1Air, GameAndWatchMoves.Judgment2Air,GameAndWatchMoves.Judgment3Air,GameAndWatchMoves.Judgment4Air,GameAndWatchMoves.Judgment5Air,GameAndWatchMoves.Judgment6Air,GameAndWatchMoves.Judgment7Air,GameAndWatchMoves.Judgment8Air,
     GameAndWatchMoves.Judgment1Ground, GameAndWatchMoves.Judgment2Ground,GameAndWatchMoves.Judgment3Ground,GameAndWatchMoves.Judgment4Ground,GameAndWatchMoves.Judgment5Ground,GameAndWatchMoves.Judgment6Ground,GameAndWatchMoves.Judgment7Ground,GameAndWatchMoves.Judgment8Ground),
    "Judgment 9": (GameAndWatchMoves.Judgment9Air, GameAndWatchMoves.Judgment9Ground),
    "Fire": (GameAndWatchMoves.FireAir, GameAndWatchMoves.FireGround),
    "Oil Panic (Absorb)": (GameAndWatchMoves.OilPanicAirAbsorb, GameAndWatchMoves.OilPanicGroundAbsorb),
    "Oil Panic (Spill)": (GameAndWatchMoves.OilPanicAirSpill, GameAndWatchMoves.OilPanicGroundSpill) # TODO has no way to track charges for that move right now
})

ganon_explorable_moves = ExplorableActions(base_moves | {
    "Warlock Punch": (GanonMoves.WarlockPunchAir, GanonMoves.WarlockPunchGround),
    "Gerudo Dragon (Air)": GanonMoves.GerudoDragonAirHit,
    "Gerudo Dragon (Ground)": GanonMoves.GerudoDragonGroundHit,
    "Dark Dive": GanonMoves.DarkDiveCatch,
    "Wizards Foot (Ground)": GanonMoves.WizardsFootGround,
    "Wizards Foot (Ending in Air)": GanonMoves.WizardsFootAirEndingInAir
})

jigglypuff_explorable_moves = ExplorableActions(base_moves | {
    "Rollout": JigglypuffMoves.RolloutHit,
    "Pound": (JigglypuffMoves.PoundAir, JigglypuffMoves.PoundGround),
    "Sing": (JigglypuffMoves.SingGroundLeft, JigglypuffMoves.SingAirLeft,
     JigglypuffMoves.SingGroundRight, JigglypuffMoves.SingAirRight),
    "Rest": (JigglypuffMoves.RestAirLeft, JigglypuffMoves.RestAirRight,
     #JigglypuffMoves.RestGroundLeft, JigglypuffMoves.RestGroundRight
             )
})

# todo
kirby_explorable_moves = ExplorableActions(base_moves)


link_explorable_moves = ExplorableActions(base_moves | {
    "Bow": (LinkMoves.BowGroundFire, LinkMoves.BowAirFire),
    "Boomerang": (LinkMoves.BoomerangGroundThrow, LinkMoves.BoomerangAirThrow),
    "Spin Attack": (LinkMoves.SpinAttackGround, LinkMoves.SpinAttackAir),
    "Bomb": (LinkMoves.BombAir, LinkMoves.BombGround),
    "Wall Hook": LinkMoves.ZairCatch,
})

luigi_moves = base_moves.copy()
luigi_moves.pop("Dash Attack")
luigi_explorable_moves = ExplorableActions(luigi_moves | {
    "Fire Ball": (LuigiMoves.FireballAir, LuigiMoves.FireballGround),
    "Green Missile": (LuigiMoves.GreenMissileGroundStartup,
    LuigiMoves.GreenMissileGroundCharge,
    LuigiMoves.GreenMissileGroundLanding,
    LuigiMoves.GreenMissileGroundTakeoff,
    LuigiMoves.GreenMissileGroundTakeoffMisfire,
    LuigiMoves.GreenMissileAirStartup,
    LuigiMoves.GreenMissileAirCharge,
    LuigiMoves.GreenMissileAir,
    LuigiMoves.GreenMissileAirEnd,
    LuigiMoves.GreenMissileAirTakeoff,
    LuigiMoves.GreenMissileAirTakeoffMisfire),
    "Super Jump Punch": LuigiMoves.SuperJumpPunchGround,
    "Cyclone": (LuigiMoves.CycloneAir, LuigiMoves.CycloneGround)
})

mario_explorable_moves = ExplorableActions(base_moves | {
    "Fire Ball": (MarioMoves.FireballAir, MarioMoves.FireballGround),
    "Cape": (MarioMoves.CapeAir, MarioMoves.CapeGround),
    "Super Jump Punch": (MarioMoves.SuperJumpPunchGround, MarioMoves.SuperJumpPunchAir),
    "Tornado": MarioMoves.TornadoAir,
})

marth_explorable_moves = ExplorableActions(base_moves | {
    "Counter": (MarthMoves.CounterAirHit, MarthMoves.CounterGroundHit),
    "Shield Breaker": (MarthMoves.ShieldBreakerGroundStartCharge, MarthMoves.ShieldBreakerGroundChargeLoop, MarthMoves.ShieldBreakerGroundEarlyRelease,
    MarthMoves.ShieldBreakerGroundFullyCharged, MarthMoves.ShieldBreakerAirStartCharge, MarthMoves.ShieldBreakerAirChargeLoop,
    MarthMoves.ShieldBreakerAirEarlyRelease, MarthMoves.ShieldBreakerAirFullyCharged),
    "Dancing Blade 1-2": (MarthMoves.DancingBlade1Ground,
    MarthMoves.DancingBlade2UpGround,
    MarthMoves.DancingBlade2SideGround,
    MarthMoves.DancingBlade1Air,
    MarthMoves.DancingBlade2UpAir,
    MarthMoves.DancingBlade2SideAir),
    "Dancing Blade 3-4": (
                          MarthMoves.DancingBlade3UpGround,
                          MarthMoves.DancingBlade3SideGround,
                          MarthMoves.DancingBlade3DownGround,
                          MarthMoves.DancingBlade4UpGround,
                          MarthMoves.DancingBlade4SideGround,
                          MarthMoves.DancingBlade4DownGround,
                          MarthMoves.DancingBlade3UpAir,
                          MarthMoves.DancingBlade3SideAir,
                          MarthMoves.DancingBlade3DownAir,
                          MarthMoves.DancingBlade4UpAir,
                          MarthMoves.DancingBlade4SideAir,
                          MarthMoves.DancingBlade4DownAir),
    "Dolphin Slash": (MarthMoves.DolphinSlashAir, MarthMoves.DolphinSlashGround),
})

mewtwo_explorable_moves = ExplorableActions(base_moves | {
    "Shadow Ball Charge": (
        MewtwoMoves.ShadowBallGroundStartCharge,
        MewtwoMoves.ShadowBallGroundChargeLoop,
        MewtwoMoves.ShadowBallGroundFullyCharged,
        MewtwoMoves.ShadowBallGroundEndCharge,
        MewtwoMoves.ShadowBallAirStartCharge,
        MewtwoMoves.ShadowBallAirChargeLoop,
        MewtwoMoves.ShadowBallAirFullyCharged,
        MewtwoMoves.ShadowBallAirEndCharge,
    ),
    "Shadow Ball": (MewtwoMoves.ShadowBallGroundFire, MewtwoMoves.ShadowBallAirFire),
    "Confusion": (MewtwoMoves.ConfusionAir, MewtwoMoves.ConfusionGround),
    "Disable": (MewtwoMoves.DisableAir, MewtwoMoves.DisableGround)
})

ness_explorable_moves = ExplorableActions(base_moves | {
    "Pk Flash": (NessMoves.PkFlashGroundStartup, NessMoves.PkFlashGroundCharge, NessMoves.PkFlashGroundExplode, NessMoves.PkFlashGroundEnd,
     NessMoves.PkFlashAirStartup, NessMoves.PkFlashAirCharge, NessMoves.PkFlashAirExplode, NessMoves.PkFlashAirEnd),
    "Pk Fire": (NessMoves.PkFireAir, NessMoves.PkFireGround),
    "Pk Thunder": (NessMoves.PkThunderAirHit, NessMoves.PkThunderGroundHit),
    "Psi Magnet": (NessMoves.PsiMagnetAirAbsorb, NessMoves.PsiMagnetGroundAbsorb)
})

peach_explorable_moves = ExplorableActions( base_moves | {
    "Bair": (PeachMoves.FloatBair, Moves.Bair),
    "Fair": (PeachMoves.FloatDair, Moves.Dair),
    "Dair": (PeachMoves.FloatFair, Moves.Fair),
    "Uair": (PeachMoves.FloatUair, Moves.Uair),
    "Vegetable": PeachMoves.VegetableGround,
    "Bomber": (PeachMoves.BomberGroundStartup, PeachMoves.BomberGroundEnd, PeachMoves.BomberAirStartup,
    PeachMoves.BomberAirEnd, PeachMoves.BomberAirHit, PeachMoves.BomberAir),
    "Parasol": (PeachMoves.ParasolOpen, PeachMoves.ParasolOpening, PeachMoves.ParasolAirStart, PeachMoves.ParasolGroundStart),
    "Toad": (PeachMoves.ToadGroundAttack, PeachMoves.ToadAirAttack)
})

pichu_explorable_moves = ExplorableActions(base_moves | {
    "Thunder Jolt": (PichuMoves.ThunderJoltAir, PichuMoves.ThunderJoltGround),
    "Skull Bash": (PichuMoves.SkullBashGroundStartup, PichuMoves.SkullBashGroundCharge, PichuMoves.SkullBashGroundLanding, PichuMoves.SkullBashGroundTakeoff,
    PichuMoves.SkullBashAirStartup, PichuMoves.SkullBashAirCharge, PichuMoves.SkullBashAir, PichuMoves.SkullBashAirEnd, PichuMoves.SkullBashAirTakeoff),
    "Agility": (PichuMoves.AgilityGroundStartup, PichuMoves.AgilityGround, PichuMoves.AgilityGroundEnd, PichuMoves.AgilityAirStartup,
    PichuMoves.AgilityAir, PichuMoves.AgilityAirEnd),
    "Thunder": (PichuMoves.ThunderGroundStartup, PichuMoves.ThunderGround, PichuMoves.ThunderGroundHit, PichuMoves.ThunderGroundEnd, PichuMoves.ThunderAirStartup,
    PichuMoves.ThunderAir, PichuMoves.ThunderAirHit, PichuMoves.ThunderAirEnd)
})

pikachu_explorable_moves = ExplorableActions(base_moves | {
    "Thunder Jolt": (PikachuMoves.ThunderJoltAir, PikachuMoves.ThunderJoltGround),
    "Skull Bash": (PikachuMoves.SkullBashGroundStartup, PikachuMoves.SkullBashGroundCharge, PikachuMoves.SkullBashGroundLanding,
     PikachuMoves.SkullBashGroundTakeoff,
     PikachuMoves.SkullBashAirStartup, PikachuMoves.SkullBashAirCharge, PikachuMoves.SkullBashAir, PikachuMoves.SkullBashAirEnd,
     PikachuMoves.SkullBashAirTakeoff),
    "Quick Attack": (PikachuMoves.QuickAttackGroundStartup, PikachuMoves.QuickAttackGround, PikachuMoves.QuickAttackGroundEnd,
     PikachuMoves.QuickAttackAirStartup,
     PikachuMoves.QuickAttackAir, PikachuMoves.QuickAttackAirEnd),
    "Thunder": (
    PikachuMoves.ThunderGroundStartup, PikachuMoves.ThunderGround, PikachuMoves.ThunderGroundHit, PikachuMoves.ThunderGroundEnd,
    PikachuMoves.ThunderAirStartup,
    PikachuMoves.ThunderAir, PikachuMoves.ThunderAirHit, PikachuMoves.ThunderAirEnd)
})

popo_explorable_moves = ExplorableActions(base_moves | {
    "Ice Shot": (PopoMoves.IceShotAir, PopoMoves.IceShotAir),
    "Squall Hammer": (PopoMoves.SquallHammerGroundSolo, PopoMoves.SquallHammerGroundTogether, PopoMoves.SquallHammerAirSolo, PopoMoves.SquallHammerAirTogether),
    "Belay": (PopoMoves.BelayGroundStartup, PopoMoves.BelayGroundCatapultingNana, PopoMoves.BelayAirStartup, PopoMoves.BelayAirCatapultingNana,
     PopoMoves.BelayCatapulting, PopoMoves.BelayAirFailedCatapulting, PopoMoves.BelayAirFailedCatapultingEnd),
    "Blizzard": (PopoMoves.BlizzardAir, PopoMoves.BlizzardGround)
})

roy_explorable_moves = ExplorableActions(base_moves | {
    "Flare Blade": (RoyMoves.FlareBladeGroundStartCharge, RoyMoves.FlareBladeGroundChargeLoop, RoyMoves.FlareBladeGroundEarlyRelease,
    RoyMoves.FlareBladeGroundFullyCharged, RoyMoves.FlareBladeAirStartCharge, RoyMoves.FlareBladeAirChargeLoop,
    RoyMoves.FlareBladeAirEarlyRelease, RoyMoves.FlareBladeAirFullyCharged),
    "Double Edge Dance 1-2": (RoyMoves.DoubleEdgeDance1Ground, RoyMoves.DoubleEdgeDance2UpGround, RoyMoves.DoubleEdgeDance2SideGround,
                              RoyMoves.DoubleEdgeDance1Air, RoyMoves.DoubleEdgeDance2UpAir,
                              RoyMoves.DoubleEdgeDance2SideAir,
                              ),
    "Double Edge Dance 3-4": (
    RoyMoves.DoubleEdgeDance3UpGround, RoyMoves.DoubleEdgeDance3SideGround, RoyMoves.DoubleEdgeDance3DownGround,
    RoyMoves.DoubleEdgeDance4UpGround, RoyMoves.DoubleEdgeDance4SideGround, RoyMoves.DoubleEdgeDance4DownGround,
    RoyMoves.DoubleEdgeDance3UpAir, RoyMoves.DoubleEdgeDance3SideAir, RoyMoves.DoubleEdgeDance3DownAir,
    RoyMoves.DoubleEdgeDance4UpAir, RoyMoves.DoubleEdgeDance4SideAir, RoyMoves.DoubleEdgeDance4DownAir),
    "Blazer": (RoyMoves.BlazerAir, RoyMoves.BlazerGround),
    "Counter": (RoyMoves.CounterGroundHit, RoyMoves.CounterAirHit)
})

samus_explorable_moves = ExplorableActions(base_moves | {
    "Bomb": (SamusMoves.BombAir, SamusMoves.BombEndGround, SamusMoves.BombJumpAir, SamusMoves.BombJumpGround),
    "Charge Shot": (SamusMoves.ChargeShotGroundFire, SamusMoves.ChargeShotAirFire),
    "Missile": (SamusMoves.MissileGround, SamusMoves.MissileAir),
    "Missile Smash": (SamusMoves.MissileSmashAir, SamusMoves.MissileSmashGround),
    "Screw Attack": (SamusMoves.ScrewAttackAir, SamusMoves.ScrewAttackGround),
    "Edge Tether": SamusMoves.ZairCatch,
})

sheik_explorable_moves = ExplorableActions(base_moves | {
    "Needle Storm": (SheikMoves.NeedleStormAirFire, SheikMoves.NeedleStormGroundFire),
    "Chain": (SheikMoves.ChainGroundStartup, SheikMoves.ChainGroundLoop, SheikMoves.ChainGroundEnd,
    SheikMoves.ChainAirStartup, SheikMoves.ChainAirLoop, SheikMoves.ChainAirEnd),
})

yoshi_explorable_moves = ExplorableActions(base_moves | {
    "Shield": YoshiMoves.ShieldDamage,
    "Egg Lay": (YoshiMoves.EggLayGround, YoshiMoves.EggLayAir),
    "Bomb": (YoshiMoves.BombGround, YoshiMoves.BombAir)
})

young_link_explorable_moves = ExplorableActions(base_moves | {
    "FireBow": (YoungLinkMoves.FireBowAirFire, YoungLinkMoves.FireBowGroundFire),
    "Boomerang": (YoungLinkMoves.BoomerangAirThrow, YoungLinkMoves.BoomerangGroundThrow),
    "Spin Attack": (YoungLinkMoves.SpinAttackGround, YoungLinkMoves.SpinAttackAir),
    "Bomb": (YoungLinkMoves.BombGround, YoungLinkMoves.BombAir),
    "Wall Hook": YoungLinkMoves.ZairCatch,
})

zelda_explorable_moves = ExplorableActions(base_moves | {
    "Nayrus Love": (ZeldaMoves.NayrusLoveAir, ZeldaMoves.NayrusLoveGround),
    "Dins Fire": (ZeldaMoves.DinsFireAirExplode, ZeldaMoves.DinsFireGroundExplode),
})

explorables_dict = {
    Character.BOWSER: bowser_explorable_moves,
    Character.CPTFALCON: captain_falcon_explorable_moves,
    Character.DK: dk_explorable_moves,
    Character.DOC: doc_explorable_moves,
    Character.FALCO: falco_explorable_moves,
    Character.FOX: fox_explorable_moves,
    Character.GAMEANDWATCH: game_and_watch_explorable_moves,
    Character.GANONDORF: ganon_explorable_moves,
    Character.JIGGLYPUFF: jigglypuff_explorable_moves,
    Character.KIRBY: kirby_explorable_moves,
    Character.LINK: link_explorable_moves,
    Character.LUIGI: luigi_explorable_moves,
    Character.MARIO: mario_explorable_moves,
    Character.MARTH: marth_explorable_moves,
    Character.MEWTWO: mewtwo_explorable_moves,
    Character.NESS: ness_explorable_moves,
    Character.PEACH: peach_explorable_moves,
    Character.PICHU: pichu_explorable_moves,
    Character.PIKACHU: pikachu_explorable_moves,
    Character.POPO: popo_explorable_moves,
    Character.ROY: roy_explorable_moves,
    Character.SAMUS: samus_explorable_moves,
    Character.SHEIK: sheik_explorable_moves,
    Character.YOSHI: yoshi_explorable_moves,
    Character.YLINK: young_link_explorable_moves,
    Character.ZELDA: zelda_explorable_moves,
}


def initialise_probs_for(tracked: ExplorableActions):
    return {
        s: 1 / len(tracked) for s in tracked.explorable_names
    }



class ExploratoryActions:
    # TODO: punish move spams ?

    def __init__(self, tracked: ExplorableActions, decay=0.95, preferred: Action = None,
                 ):
        # Encourage the bot to increase in action_state entropy
        self.tracked = tracked
        self.probs = initialise_probs_for(tracked)
        self.explorables = set()
        self.decay = decay
        self.preferred_name = self.tracked[preferred]

    def update(self, counts: dict):

        total_counts = np.maximum(sum(counts.values()), 1)

        for action, count in counts.items():
            if action not in self.explorables and count > 0:
                self.explorables.add(set)

        probs = {}
        for action, prob in self.probs.items():

            probs[action] = prob * self.decay + counts[action] * (1 - self.decay) / total_counts

        # tree.map_structure(
        #     lambda o, n: o * self.decay + n  * (1 - self.decay) / total_counts,
        #     self.probs,
        #     counts
        # )
        self.probs = probs

    def get_reward(self, action: Action):
        key = self.tracked[action]
        if key not in self.probs:
            return 0.

        logp = np.log(self.probs[key] + 1e-8)

        if key == self.preferred_name: # TODO: update frontend for preferred moves
            logp -= 1.5

        target_logp = -np.log(np.maximum(len(self.explorables), 10.))

        r = target_logp - logp

        return np.clip(r, -10, 10.)

    def get_top_k(self, k=8):

        interpreted = {}
        for action_name, v in sorted(self.probs.items(), key=lambda item: item[1])[-k:]:
            interpreted[action_name] = round(v * 100)  # get the rounded up percent usage of the move
        return interpreted