import math

from melee import FrameData, stages
from melee.enums import Action, AttackState, Character
from collections import defaultdict

class CompiledFrameData:
    def __init__(self):
        self.FD = FrameData()

        self.attack = defaultdict(
            dict
        )
        self.first_hitbox_frame = defaultdict(
            dict
        )
        self.last_hitbox_frame = defaultdict(
            dict
        )

        self.iasa = defaultdict(
            dict
        )

        self.hitbox_frames = defaultdict(
            lambda: defaultdict(list)
        )

        self.ranges = defaultdict(
            lambda: defaultdict(list)
        )

        self.is_grab = self.FD.is_grab
        self.is_roll = self.FD.is_roll
        self.is_bmove = self.FD.is_bmove
        self.is_shield = self.FD.is_shield

        for char in Character:
            for action in Action:
                self.attack[char][action] = self.FD.is_attack(char, action)
                self.first_hitbox_frame[char][action] = self.FD.first_hitbox_frame(char, action)
                self.last_hitbox_frame[char][action] = self.FD.last_hitbox_frame(char, action)
                self.iasa[char][action] = self.FD.iasa(char, action)

                for frame in range(0, len(self.FD.framedata[char][action])):
                    attackingframe = self.FD._getframe(char, action, frame)
                    if attackingframe is None:
                        continue

                    if attackingframe['hitbox_1_status']:
                        self.ranges[char][action].append((frame, attackingframe["hitbox_1_x"],
                                                          attackingframe["hitbox_1_y"],
                                                          attackingframe["hitbox_1_size"]))
                    if attackingframe['hitbox_2_status']:
                        self.ranges[char][action].append((frame, attackingframe["hitbox_2_x"],
                                                          attackingframe["hitbox_2_y"],
                                                          attackingframe["hitbox_2_size"]))
                    if attackingframe['hitbox_3_status']:
                        self.ranges[char][action].append((frame, attackingframe["hitbox_3_x"],
                                                          attackingframe["hitbox_3_y"],
                                                          attackingframe["hitbox_3_size"]))
                    if attackingframe['hitbox_4_status']:
                        self.ranges[char][action].append((frame,
                                                          attackingframe["hitbox_4_x"],
                                                          attackingframe["hitbox_4_y"],
                                                          attackingframe["hitbox_4_size"]))

                    if attackingframe['hitbox_1_status'] or attackingframe['hitbox_2_status'] \
                            or attackingframe['hitbox_3_status'] or attackingframe['hitbox_4_status'] or \
                            attackingframe['projectile']:
                        self.hitbox_frames[char][action].append(frame)

    def is_attack(self, character, action):
        return self.attack[character][action]

    def attack_state(self, character, action, action_frame):
        if not self.is_attack(character, action):
            return AttackState.NOT_ATTACKING

        if action_frame < self.first_hitbox_frame[character][action]:
            return AttackState.WINDUP

        if action_frame > self.last_hitbox_frame[character][action]:
            return AttackState.COOLDOWN

        return AttackState.ATTACKING

    def in_range(self, attacker, defender, stage):
        """Calculates if an attack is in range of a given defender

               Args:
                   attacker (gamestate.PlayerState): The attacking player
                   defender (gamestate.PlayerState): The defending player
                   stage (enums.Stage): The stage being played on

               Returns:
                   integer with in how many frames the specified attack will hit the defender
                   -1 if it won't hit

               Note:
                   This considers the defending character to have a single hurtbox, centered
                   at the x,y coordinates of the player (adjusted up a little to be centered)
               """
        lastframe = self.last_hitbox_frame[attacker.character][attacker.action]

        # Adjust the defender's hurtbox up a little, to be more centered.
        #   the game keeps y coordinates based on the bottom of a character, not
        #   their center. So we need to move up by one radius of the character's size
        defender_size = float(self.FD.characterdata[defender.character]["size"])
        defender_y = defender.y + defender_size

        # Running totals of how far the attacker will travel each frame
        attacker_x = attacker.position.x
        attacker_y = attacker.position.y

        onground = attacker.on_ground

        attacker_speed_x = 0
        if onground:
            attacker_speed_x = attacker.speed_ground_x_self
        else:
            attacker_speed_x = attacker.speed_air_x_self
        attacker_speed_y = attacker.speed_y_self

        friction = self.FD.characterdata[attacker.character]["Friction"]
        gravity = self.FD.characterdata[attacker.character]["Gravity"]
        termvelocity = self.FD.characterdata[attacker.character]["TerminalVelocity"]

        for i in range(attacker.action_frame + 1, lastframe + 1):
            attackingframe = self.FD._getframe(attacker.character, attacker.action, i)
            if attackingframe is None:
                continue

            # Figure out how much the attacker will be moving this frame
            #   Is there any locomotion in the animation? If so, use that
            locomotion_x = float(attackingframe["locomotion_x"])
            locomotion_y = float(attackingframe["locomotion_y"])
            if locomotion_y == 0 and locomotion_x == 0:
                # There's no locomotion, so let's figure out how the attacker will be moving...
                #   Are they on the ground or in the air?
                if onground:
                    attacker_speed_y = 0
                    # Slow down the speed by the character's friction, then apply it
                    if attacker_speed_x > 0:
                        attacker_speed_x = max(0, attacker_speed_x - friction)
                    else:
                        attacker_speed_x = min(0, attacker_speed_x + friction)
                    attacker_x += attacker_speed_x
                # If attacker is in tha air...
                else:
                    # First consider vertical movement. They will decelerate towards the stage
                    attacker_speed_y = max(-termvelocity, attacker_speed_y - gravity)
                    # NOTE Assume that the attacker will keep moving how they currently are
                    # If they do move halfway, then this will re-calculate later runs

                    attacker_y += attacker_speed_y
                    # Did we hit the ground this frame? If so, let's make some changes
                    if attacker_y <= 0 and abs(attacker_x) < stages.EDGE_GROUND_POSITION[stage]:
                        # TODO: Let's consider A moves that cancel when landing
                        attacker_y = 0
                        attacker_speed_y = 0
                        onground = True

                    attacker_x += attacker_speed_x
            else:
                attacker_x += locomotion_x
                attacker_y += locomotion_y

            if attackingframe['hitbox_1_status'] or attackingframe['hitbox_2_status'] or \
                    attackingframe['hitbox_3_status'] or attackingframe['hitbox_4_status']:
                # Calculate the x and y positions of all 4 hitboxes for this frame
                hitbox_1_x = float(attackingframe["hitbox_1_x"])
                hitbox_1_y = float(attackingframe["hitbox_1_y"]) + attacker_y
                hitbox_2_x = float(attackingframe["hitbox_2_x"])
                hitbox_2_y = float(attackingframe["hitbox_2_y"]) + attacker_y
                hitbox_3_x = float(attackingframe["hitbox_3_x"])
                hitbox_3_y = float(attackingframe["hitbox_3_y"]) + attacker_y
                hitbox_4_x = float(attackingframe["hitbox_4_x"])
                hitbox_4_y = float(attackingframe["hitbox_4_y"]) + attacker_y

                # Flip the horizontal hitboxes around if we're facing left
                if not attacker.facing:
                    hitbox_1_x *= -1
                    hitbox_2_x *= -1
                    hitbox_3_x *= -1
                    hitbox_4_x *= -1

                hitbox_1_x += attacker_x
                hitbox_2_x += attacker_x
                hitbox_3_x += attacker_x
                hitbox_4_x += attacker_x

                # Now see if any of the hitboxes are in range
                distance1 = math.sqrt((hitbox_1_x - defender.position.x) ** 2 + (hitbox_1_y - defender_y) ** 2)
                distance2 = math.sqrt((hitbox_2_x - defender.position.x) ** 2 + (hitbox_2_y - defender_y) ** 2)
                distance3 = math.sqrt((hitbox_3_x - defender.position.x) ** 2 + (hitbox_3_y - defender_y) ** 2)
                distance4 = math.sqrt((hitbox_4_x - defender.position.x) ** 2 + (hitbox_4_y - defender_y) ** 2)

                if distance1 < defender_size + float(attackingframe["hitbox_1_size"]):
                    return i-attacker.action_frame
                if distance2 < defender_size + float(attackingframe["hitbox_2_size"]):
                    return i-attacker.action_frame
                if distance3 < defender_size + float(attackingframe["hitbox_3_size"]):
                    return i-attacker.action_frame
                if distance4 < defender_size + float(attackingframe["hitbox_4_size"]):
                    return i-attacker.action_frame
        return -1.

    def frames_before_next_hitbox(self, character, action, curr_frame):

        frames_before_next_hitbox = 180
        for frame in self.hitbox_frames[character][action]:
            if frame > curr_frame:
                frames_before_next_hitbox = frame - curr_frame
                break

        return frames_before_next_hitbox

    def get_move_next_hitboxes(self, character, action, curr_frame):
        infos = []
        for frame, x, y, r in self.ranges[character][action]:
            if frame > curr_frame:
                infos.append((x, y, r))
        return infos




