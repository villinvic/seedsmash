import os

from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage, ChatSub, ChatCommand
import asyncio
from seedsmash.twitch_app import APP_ID, APP_SECRET
import zmq

USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
TARGET_CHANNEL = 'seedsmash'


class TwitchBotPUSH:
    def __init__(self):

        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.PUSH)
        self.socket.bind("ipc://twitchbot")

    def push(self, x):
        self.socket.send_pyobj(x)


class TwitchBotPULL:
    def __init__(self):
        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.PULL)
        self.socket.connect("ipc://twitchbot")

    def pull(self):
        rcved = []
        try:
            while True:
                rcved.append(self.socket.recv_pyobj(zmq.NOBLOCK))
        except zmq.error.Again as e:
            pass
        return rcved


class SSTwitchBot(TwitchBotPUSH):

    def __init__(self):
        super().__init__()


    # this will be called when the event READY is triggered, which will be on bot start
    async def on_ready(self, ready_event: EventData):
        print('Bot is ready for work, joining channels')
        # join our target channel, if you want to join multiple, either call join for each individually
        # or even better pass a list of channels as the argument
        await ready_event.chat.join_room(TARGET_CHANNEL)
        # you can do other bot initialization things in here


    async def info_command(self, cmd: ChatCommand):
        await cmd.reply("🌱 Welcome to Seedsmash!🌱 Seedsmash is a pure Reinforcement Learning project where bots learn"
                  "to battle each other in real time! We’re currently in the alpha phase, testing and refining the system."
                  "🚀 In the future, you'll be able to submit your own bot configurations to join the fight! For now, the "
                  "bots you see are custom examples showcasing what's possible.Please note: I'm not always available to "
                  "answer questions, but feel free to enjoy the action and stay tuned for updates!")

    async def progress_command(self, cmd: ChatCommand):
        await cmd.reply("1. Random phase (0 -> 10k games) | 2. Begins to recognise where are the ledges and the opponent "
                        "(10k -> 20k games) | 3. Beginner human level (20k -> 40k games) | 4. Stops move spamming, learns shields"
                        " and to play around ledges (40k -> ???)")

    async def get_bot_list(self, cmd: ChatCommand):
        _, _, botnames = next(os.walk("bot_configs"))
        botnames = [bn.rstrip(".txt") for bn in botnames if bn.endswith(".txt")]

        string = 'Here is the current list of bots:'
        for b in botnames:
            string += f"\n- {b}"
        await cmd.reply(string)


    async def push_game_command(self, cmd: ChatCommand):

        _, _, botnames = next(os.walk("bot_configs"))
        botnames = [bn.rstrip(".txt") for bn in botnames if bn.endswith(".txt")]


        if len(cmd.parameter) == 0:
            await cmd.reply('syntax: !play bot_tag1 bot_tag2.')

        requested_bots = cmd.parameter.split()
        if len(requested_bots) != 2:
            await cmd.reply('syntax: !play bot_tag1 bot_tag2.')

        for b in requested_bots:
            if b not in botnames:
                await cmd.reply(f'Cannot find {b} in the current list of bots!')
                return
        if requested_bots[0] == requested_bots[1]:
            await cmd.reply(f'Mirror matches are unsupported!')
            return

        self.push({"requested_matchup": requested_bots})
        await cmd.reply(f'Requested next game: {requested_bots[0]} vs {requested_bots[1]}')

    async def run(self):

        twitch = await Twitch(APP_ID, APP_SECRET)
        auth = UserAuthenticator(twitch, USER_SCOPE)
        token, refresh_token = await auth.authenticate()
        await twitch.set_user_authentication(token, USER_SCOPE, refresh_token)

        # create chat instance
        chat = await Chat(twitch)

        # register the handlers for the events you want

        # listen to when the bot is done starting up and ready to join channels
        chat.register_event(ChatEvent.READY, self.on_ready)

        # there are more events, you can view them all in this documentation

        # you can directly register commands and their handlers, this will register the !reply command

        commands = (
            ("play", self.push_game_command, "Requests a matchup for the next game. The bot tags should be in the current list of playing bots."),
            ("info", self.info_command, "Shows an introductory message to the channel."),
            ("progress", self.progress_command,
             "Shows an approximation of how good a bot should be in function of how many games it played."),
            ("botlist", self.get_bot_list, "Shows the list of currently playing bots."),

        )

        for c in commands:
            chat.register_command(*c[:2])

        async def help(cmd: ChatCommand):
            string = ""
            for i, (c, _, desc) in enumerate(commands, 1):
                string += f"{i}. !{c} ({desc}) | "

            await cmd.reply(string[:-3])

        chat.register_command(
            "help", help
        )

        # we are done with our setup, lets start this bot up!
        chat.start()

        # lets run till we press enter in the console
        try:
            input('press ENTER to stop\n')
        finally:
            # now we can close the chat bot and the twitch api client
            chat.stop()
            await twitch.close()

if __name__ == '__main__':
    sstwitch_bot = SSTwitchBot()
    # lets run our setup
    asyncio.run(sstwitch_bot.run())